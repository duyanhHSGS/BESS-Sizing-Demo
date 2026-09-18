"""bess.agents.ppo2_agent.py — PPO with reward-decomposed critics for the BESS CMDP.

Architecture mirrors the archived senior reference in git-plz-ignore/senior-ppo2-reference/:
  - 2×128 Tanh MLPs (larger capacity than original PPO's 2×64)
  - Two value heads: critic_energy + critic_peak with PopArt normalisers
  - Per-component GAE lambda (energy ~0.97, peak ~0.5)
  - Separate actor/critic learning rates with linear annealing
  - KL-based early stopping
  - Squashed Gaussian policy (tanh on sample, not on mean)
  - Orthogonal weight initialisation
  - log_std clamped to [-3, 0]
  - Clip penalty on unexecuted kW (training-only)

Deployment:
  PPO2InferenceAgent — lightweight actor-only loader for dispatch/benchmarking.
  Loads only actor.* / log_std keys, drops all critic tensors.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# The senior trainer uses two Torch CPU threads. In this multi-agent repo that
# setting is applied by the PPO2 training entrypoint instead of at import time,
# so merely importing PPO2 cannot perturb PPO numerical execution.

# ---------------------------------------------------------------------------
# Squashed Gaussian helpers (vendored from bess-drl engine/squashed_gaussian.py)
# ---------------------------------------------------------------------------
TANH_EPS = 1e-6
LOG_STD_MIN, LOG_STD_MAX = -3.0, 0.0
TANH_GAIN = 5.0 / 3.0   # nn.init.calculate_gain("tanh")


def resolve_ppo2_device(device: str = "auto") -> str:
    requested = str(device).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("PPO2 device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda":
        raise RuntimeError("PPO2 senior-reference mode is CPU-only, matching the senior trainer")
    return "cpu"


def _squashed_log_prob_from_latent(
    distribution: torch.distributions.Normal,
    latent: torch.Tensor,
) -> torch.Tensor:
    """log pi(tanh(latent)) with the exact change-of-variables correction.

    Uses log(1 - tanh(u)^2) = 2*(log 2 - u - softplus(-2u)), which stays finite
    for large |u| where the naive log(1 - a^2) underflows.
    """
    correction = 2.0 * (
        torch.log(torch.tensor(2.0, dtype=latent.dtype, device=latent.device))
        - latent
        - torch.nn.functional.softplus(-2.0 * latent)
    )
    return (distribution.log_prob(latent) - correction).sum(-1)


def _sample_squashed(
    distribution: torch.distributions.Normal,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (action, log_prob, latent); keep the latent for the PPO update."""
    latent = distribution.mean if deterministic else distribution.rsample()
    action = torch.tanh(latent)
    return action, _squashed_log_prob_from_latent(distribution, latent), latent


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------
def _mlp(inp, out, hidden=128):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )


def _orthogonal_init(block: nn.Sequential, head_gain: float) -> None:
    layers = [m for m in block if isinstance(m, nn.Linear)]
    for layer in layers[:-1]:
        nn.init.orthogonal_(layer.weight, gain=TANH_GAIN)
        nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(layers[-1].weight, gain=head_gain)
    nn.init.zeros_(layers[-1].bias)


def _last_linear(block: nn.Sequential) -> nn.Linear:
    return [m for m in block if isinstance(m, nn.Linear)][-1]


# ---------------------------------------------------------------------------
# PopArt normaliser (van Hasselt et al. 2016)
# ---------------------------------------------------------------------------
class PopArtNormalizer:
    """Running (mean, std) of value targets with output-preserving rescale."""

    def __init__(self, head: nn.Sequential, beta: float = 0.1, eps: float = 1e-4):
        self._head = head
        self._beta = beta
        self._eps = eps
        self._mean_acc = 0.0
        self._sq_acc = 0.0
        self._debias = 0.0
        self.mean = 0.0
        self.std = 1.0

    def normalize(self, target: np.ndarray) -> np.ndarray:
        return (target - self.mean) / self.std

    def denormalize(self, value):
        return value * self.std + self.mean

    def update(self, target: np.ndarray) -> None:
        old_mean, old_std = self.mean, self.std
        beta = self._beta
        self._mean_acc = (1 - beta) * self._mean_acc + beta * float(target.mean())
        self._sq_acc = (1 - beta) * self._sq_acc + beta * float((target ** 2).mean())
        self._debias = (1 - beta) * self._debias + beta
        mean = self._mean_acc / self._debias
        variance = self._sq_acc / self._debias - mean ** 2
        self.mean = mean
        self.std = max(float(np.sqrt(max(variance, 0.0))), self._eps)
        self._rescale(old_mean, old_std)

    @torch.no_grad()
    def _rescale(self, old_mean: float, old_std: float) -> None:
        layer = _last_linear(self._head)
        ratio = old_std / self.std
        layer.weight.mul_(ratio)
        layer.bias.mul_(ratio)
        layer.bias.add_((old_mean - self.mean) / self.std)

    def state(self) -> dict:
        return {
            "mean": self.mean, "std": self.std, "mean_acc": self._mean_acc,
            "sq_acc": self._sq_acc, "debias": self._debias,
        }

    def load_state(self, state: dict) -> None:
        self.mean = float(state["mean"])
        self.std = float(state["std"])
        self._mean_acc = float(state["mean_acc"])
        self._sq_acc = float(state["sq_acc"])
        self._debias = float(state["debias"])


# ---------------------------------------------------------------------------
# Actor-Critic with decomposed value heads
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Actor plus one critic head per reward component.

    actor / log_std keep their names so PPO2InferenceAgent can load them
    and drop the critics — checkpoint layout stays decoupled from inference.
    """

    def __init__(self, obs_dim: int, hidden: int = 128, log_std_init: float = -0.5):
        super().__init__()
        self.actor = _mlp(obs_dim, 1, hidden)
        self.critic_energy = _mlp(obs_dim, 1, hidden)
        self.critic_peak = _mlp(obs_dim, 1, hidden)
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))
        _orthogonal_init(self.actor, head_gain=0.01)       # start near a_t ~ 0
        _orthogonal_init(self.critic_energy, head_gain=1.0)
        _orthogonal_init(self.critic_peak, head_gain=1.0)

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def normalized_values(self, obs):
        return (
            self.critic_energy(obs).squeeze(-1),
            self.critic_peak(obs).squeeze(-1),
        )


class RecurrentActorCritic(nn.Module):
    """PPO2 actor and two PopArt critics with separate chronological GRU memory."""

    def __init__(self, obs_dim: int, hidden: int = 128, log_std_init: float = -0.5):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden)
        self.actor_encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.actor_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.actor = nn.Linear(hidden, 1)
        self.critic_encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh())
        self.critic_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.critic_energy = nn.Sequential(nn.Linear(hidden, 1))
        self.critic_peak = nn.Sequential(nn.Linear(hidden, 1))
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))

        nn.init.orthogonal_(self.actor_encoder[0].weight, gain=TANH_GAIN)
        nn.init.zeros_(self.actor_encoder[0].bias)
        nn.init.orthogonal_(self.critic_encoder[0].weight, gain=TANH_GAIN)
        nn.init.zeros_(self.critic_encoder[0].bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic_energy[0].weight, gain=1.0)
        nn.init.zeros_(self.critic_energy[0].bias)
        nn.init.orthogonal_(self.critic_peak[0].weight, gain=1.0)
        nn.init.zeros_(self.critic_peak[0].bias)

    def zero_hidden(self, batch_size: int, *, device=None) -> torch.Tensor:
        target = device if device is not None else next(self.parameters()).device
        return torch.zeros(1, int(batch_size), self.hidden_size, device=target)

    def actor_sequence(self, obs: torch.Tensor, hidden=None):
        if obs.ndim != 3:
            raise ValueError("recurrent PPO2 actor expects [batch, time, obs]")
        if hidden is None:
            hidden = self.zero_hidden(obs.shape[0], device=obs.device)
        encoded = self.actor_encoder(obs)
        features, next_hidden = self.actor_gru(encoded, hidden)
        return self.actor(features), next_hidden

    def normalized_values_sequence(self, obs: torch.Tensor, hidden=None):
        if obs.ndim != 3:
            raise ValueError("recurrent PPO2 critic expects [batch, time, obs]")
        if hidden is None:
            hidden = self.zero_hidden(obs.shape[0], device=obs.device)
        encoded = self.critic_encoder(obs)
        features, next_hidden = self.critic_gru(encoded, hidden)
        return (
            self.critic_energy(features).squeeze(-1),
            self.critic_peak(features).squeeze(-1),
        ), next_hidden

    def dist_sequence(self, obs: torch.Tensor, hidden=None):
        mean, next_hidden = self.actor_sequence(obs, hidden)
        return torch.distributions.Normal(mean, self.log_std.exp()), next_hidden

    def dist_step(self, obs: torch.Tensor, hidden=None):
        distribution, next_hidden = self.dist_sequence(obs.unsqueeze(1), hidden)
        return torch.distributions.Normal(
            distribution.loc[:, 0], distribution.scale[:, 0]
        ), next_hidden

    def normalized_values_step(self, obs: torch.Tensor, hidden=None):
        values, next_hidden = self.normalized_values_sequence(obs.unsqueeze(1), hidden)
        return (values[0][:, 0], values[1][:, 0]), next_hidden


# ---------------------------------------------------------------------------
# Rollout buffer (decomposed rewards)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores decomposed rewards plus recurrent state consumed by each action."""

    def __init__(
        self,
        size: int,
        obs_dim: int,
        *,
        recurrent_hidden_size: int | None = None,
    ):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.latent = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew_e = np.zeros(size, np.float32)
        self.rew_p = np.zeros(size, np.float32)
        self.val_e = np.zeros(size, np.float32)
        self.val_p = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.actor_hidden = (
            np.zeros((size, recurrent_hidden_size), np.float32)
            if recurrent_hidden_size is not None
            else None
        )
        self.critic_hidden = (
            np.zeros((size, recurrent_hidden_size), np.float32)
            if recurrent_hidden_size is not None
            else None
        )
        self.ptr = 0
        self.size = size

    def add(
        self,
        o,
        a,
        latent,
        lp,
        r_e,
        r_p,
        v_e,
        v_p,
        d,
        *,
        actor_hidden=None,
        critic_hidden=None,
    ):
        i = self.ptr
        self.obs[i], self.act[i], self.latent[i] = o, a, latent
        self.logp[i] = lp
        self.rew_e[i], self.rew_p[i] = r_e, r_p
        self.val_e[i], self.val_p[i] = v_e, v_p
        self.done[i] = d
        if self.actor_hidden is not None:
            if actor_hidden is None or critic_hidden is None:
                raise ValueError("recurrent PPO2 rollout requires actor and critic hidden state")
            self.actor_hidden[i] = actor_hidden
            self.critic_hidden[i] = critic_hidden
        elif actor_hidden is not None or critic_hidden is not None:
            raise ValueError("feed-forward PPO2 rollout cannot store recurrent hidden state")
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size


# ---------------------------------------------------------------------------
# Generalised Advantage Estimation
# ---------------------------------------------------------------------------
def compute_gae(
    rew: np.ndarray,
    val: np.ndarray,
    done: np.ndarray,
    last_val: float,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """GAE: A_t = δ_t + γλ(1-done_t)*A_{t+1}  where δ_t = r_t + γ*V(s_{t+1})(1-done_t) - V(s_t)"""
    n = len(rew)
    adv = np.zeros(n, np.float32)
    gae = 0.0
    for i in reversed(range(n)):
        next_val = last_val if i == n - 1 else val[i + 1]
        nonterminal = 1.0 - done[i]
        delta = rew[i] + gamma * next_val * nonterminal - val[i]
        gae = delta + gamma * lam * nonterminal * gae
        adv[i] = gae
    return adv


def _adv_share_of_return(returns: np.ndarray, values: np.ndarray) -> float:
    """Return Var(advantage) / Var(return), matching the senior PPO diagnostic."""
    variance = float(np.var(returns))
    if variance < 1e-12:
        return 0.0
    return float(np.var(returns - values) / variance)


# ---------------------------------------------------------------------------
# PPO2Agent (training)
# ---------------------------------------------------------------------------
class PPO2Agent:
    """PPO with reward-decomposed critics, PopArt, and squashed Gaussian policy."""

    def __init__(self, obs_dim: int, lr=1e-4, gamma=1.0,
                 lam_energy=0.97, lam_peak=0.5,
                 clip=0.2, epochs=6, minibatch=256, ent_coef=0.01,
                 vf_coef=0.5, target_kl=0.01, seed=0,
                 actor_lr: float | None = None,
                 critic_lr: float | None = None,
                 log_std_init: float = -0.5,
                 device: str = "auto",
                 hidden_size: int = 128,
                 recurrent_enabled: bool = False,
                 recurrent_sequence_length: int = 96):
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.device = torch.device(resolve_ppo2_device(device))
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.recurrent_enabled = bool(recurrent_enabled)
        self.recurrent_sequence_length = int(recurrent_sequence_length)
        if self.recurrent_sequence_length < 1:
            raise ValueError("recurrent_sequence_length must be >= 1")
        self.actor_lr = lr if actor_lr is None else float(actor_lr)
        self.critic_lr = lr if critic_lr is None else float(critic_lr)
        self.lr = lr
        self.gamma = gamma
        self.lam_energy, self.lam_peak = lam_energy, lam_peak
        self.clip = clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.target_kl = target_kl
        self.meta = {}
        self.diagnostics = {}
        self._build_network(log_std_init=log_std_init)

    def _build_network(self, *, log_std_init: float) -> None:
        if self.recurrent_enabled:
            self.net = RecurrentActorCritic(
                self.obs_dim,
                hidden=self.hidden_size,
                log_std_init=log_std_init,
            )
            actor_params = (
                list(self.net.actor_encoder.parameters())
                + list(self.net.actor_gru.parameters())
                + list(self.net.actor.parameters())
                + [self.net.log_std]
            )
            critic_params = (
                list(self.net.critic_encoder.parameters())
                + list(self.net.critic_gru.parameters())
                + list(self.net.critic_energy.parameters())
                + list(self.net.critic_peak.parameters())
            )
        else:
            self.net = ActorCritic(
                self.obs_dim,
                hidden=self.hidden_size,
                log_std_init=log_std_init,
            )
            actor_params = list(self.net.actor.parameters()) + [self.net.log_std]
            critic_params = (
                list(self.net.critic_energy.parameters())
                + list(self.net.critic_peak.parameters())
            )
        self.net = self.net.to(self.device)
        self.opt = torch.optim.Adam([
            {"params": actor_params, "lr": self.actor_lr},
            {"params": critic_params, "lr": self.critic_lr},
        ])
        self._base_lrs = [self.actor_lr, self.critic_lr]
        self.norm_energy = PopArtNormalizer(self.net.critic_energy)
        self.norm_peak = PopArtNormalizer(self.net.critic_peak)
        self.reset_recurrent_state()

    def reset_recurrent_state(self) -> None:
        self._actor_hidden = None
        self._critic_hidden = None
        self.last_actor_hidden_input = None
        self.last_critic_hidden_input = None

    def _hidden_numpy(self, hidden) -> np.ndarray:
        if hidden is None:
            return np.zeros(self.hidden_size, dtype=np.float32)
        return hidden.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)

    def recurrent_rollout_inputs(self):
        if not self.recurrent_enabled:
            return None, None
        return self.last_actor_hidden_input, self.last_critic_hidden_input

    def snapshot_recurrent_state(self) -> dict:
        return {
            "actor": None if self._actor_hidden is None else self._actor_hidden.detach().clone(),
            "critic": None if self._critic_hidden is None else self._critic_hidden.detach().clone(),
            "last_actor": None if self.last_actor_hidden_input is None else self.last_actor_hidden_input.copy(),
            "last_critic": None if self.last_critic_hidden_input is None else self.last_critic_hidden_input.copy(),
        }

    def restore_recurrent_state(self, state: dict) -> None:
        self._actor_hidden = state["actor"]
        self._critic_hidden = state["critic"]
        self.last_actor_hidden_input = state["last_actor"]
        self.last_critic_hidden_input = state["last_critic"]

    def anneal_lr(self, progress: float) -> None:
        """Linearly decay each group's learning rate; progress runs 0 -> 1."""
        decay = max(0.0, 1.0 - float(progress))
        for group, base in zip(self.opt.param_groups, self._base_lrs, strict=True):
            group["lr"] = base * decay

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Return action/log-prob/latent plus denormalized component values."""
        o = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.recurrent_enabled:
            self.last_actor_hidden_input = self._hidden_numpy(self._actor_hidden)
            self.last_critic_hidden_input = self._hidden_numpy(self._critic_hidden)
            dist, self._actor_hidden = self.net.dist_step(o, self._actor_hidden)
            (v_e, v_p), self._critic_hidden = self.net.normalized_values_step(
                o, self._critic_hidden
            )
            self._actor_hidden = self._actor_hidden.detach()
            self._critic_hidden = self._critic_hidden.detach()
        else:
            dist = self.net.dist(o)
            v_e, v_p = self.net.normalized_values(o)
        a, logp, latent = _sample_squashed(dist, deterministic=deterministic)
        return (
            float(a.item()),
            float(logp.item()),
            float(latent.item()),
            float(self.norm_energy.denormalize(v_e.item())),
            float(self.norm_peak.denormalize(v_p.item())),
        )

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference while preserving recurrent history."""
        o = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.recurrent_enabled:
            dist, self._actor_hidden = self.net.dist_step(o, self._actor_hidden)
            self._actor_hidden = self._actor_hidden.detach()
        else:
            dist = self.net.dist(o)
        a, _, _ = _sample_squashed(dist, deterministic=True)
        return float(a.item())

    @torch.no_grad()
    def estimate_values(self, obs: np.ndarray) -> tuple[float, float]:
        """Bootstrap critic values without advancing recurrent memory."""
        o = torch.as_tensor(
            obs, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if self.recurrent_enabled:
            (v_e, v_p), _ = self.net.normalized_values_step(o, self._critic_hidden)
        else:
            v_e, v_p = self.net.normalized_values(o)
        return (
            float(self.norm_energy.denormalize(v_e.item())),
            float(self.norm_peak.denormalize(v_p.item())),
        )

    def _recurrent_chunks(self, done: np.ndarray) -> list[tuple[int, int]]:
        """Build TBPTT chunks that never cross a calendar-month boundary."""
        chunks: list[tuple[int, int]] = []
        episode_start = 0
        terminal_indexes = list(np.flatnonzero(done > 0.5) + 1)
        if not terminal_indexes or terminal_indexes[-1] < len(done):
            terminal_indexes.append(len(done))
        for episode_stop in terminal_indexes:
            for start in range(
                episode_start,
                int(episode_stop),
                self.recurrent_sequence_length,
            ):
                chunks.append((
                    start,
                    min(start + self.recurrent_sequence_length, int(episode_stop)),
                ))
            episode_start = int(episode_stop)
        return chunks

    def _update_recurrent(
        self,
        buf: RolloutBuffer,
        last_val_energy: float,
        last_val_peak: float,
    ) -> None:
        if buf.actor_hidden is None or buf.critic_hidden is None:
            raise ValueError("recurrent PPO2 update requires hidden states in RolloutBuffer")
        n = buf.ptr
        done = buf.done[:n]
        adv_e = compute_gae(
            buf.rew_e[:n], buf.val_e[:n], done,
            last_val_energy, self.gamma, self.lam_energy,
        )
        adv_p = compute_gae(
            buf.rew_p[:n], buf.val_p[:n], done,
            last_val_peak, self.gamma, self.lam_peak,
        )
        ret_e = adv_e + buf.val_e[:n]
        ret_p = adv_p + buf.val_p[:n]
        adv = adv_e + adv_p
        adv_raw_std = float(adv.std())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.norm_energy.update(ret_e)
        self.norm_peak.update(ret_p)
        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_e_t = torch.as_tensor(
            self.norm_energy.normalize(ret_e), device=self.device
        )
        ret_p_t = torch.as_tensor(
            self.norm_peak.normalize(ret_p), device=self.device
        )

        chunks = self._recurrent_chunks(done)
        chunks_per_minibatch = max(
            1, self.minibatch // self.recurrent_sequence_length
        )
        approx_kl = 0.0
        stop_epoch = self.epochs
        for epoch in range(self.epochs):
            order = self._rng.permutation(len(chunks))
            kl_batches: list[float] = []
            for offset in range(0, len(order), chunks_per_minibatch):
                selected = [
                    chunks[int(index)]
                    for index in order[offset:offset + chunks_per_minibatch]
                ]
                index_parts = []
                logp_parts = []
                value_e_parts = []
                value_p_parts = []
                entropy_parts = []
                for start, stop in selected:
                    seq_obs = obs[start:stop].unsqueeze(0)
                    actor_hidden = torch.as_tensor(
                        buf.actor_hidden[start],
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(1, 1, -1)
                    critic_hidden = torch.as_tensor(
                        buf.critic_hidden[start],
                        dtype=torch.float32,
                        device=self.device,
                    ).reshape(1, 1, -1)
                    dist, _ = self.net.dist_sequence(seq_obs, actor_hidden)
                    flat_dist = torch.distributions.Normal(
                        dist.loc.squeeze(0), dist.scale.squeeze(0)
                    )
                    logp_parts.append(_squashed_log_prob_from_latent(
                        flat_dist, latent[start:stop]
                    ))
                    (v_e, v_p), _ = self.net.normalized_values_sequence(
                        seq_obs, critic_hidden
                    )
                    value_e_parts.append(v_e.squeeze(0))
                    value_p_parts.append(v_p.squeeze(0))
                    _, entropy_logp, _ = _sample_squashed(
                        flat_dist, deterministic=False
                    )
                    entropy_parts.append(-entropy_logp)
                    index_parts.append(torch.arange(
                        start, stop, dtype=torch.long, device=self.device
                    ))

                indexes = torch.cat(index_parts)
                logp = torch.cat(logp_parts)
                log_ratio = logp - logp_old[indexes]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    kl_batches.append(float(
                        ((ratio - 1.0) - log_ratio).mean()
                    ))
                surr1 = ratio * adv_t[indexes]
                surr2 = torch.clamp(
                    ratio, 1 - self.clip, 1 + self.clip
                ) * adv_t[indexes]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_e = torch.cat(value_e_parts)
                v_p = torch.cat(value_p_parts)
                v_loss = ((v_e - ret_e_t[indexes]) ** 2).mean() + (
                    (v_p - ret_p_t[indexes]) ** 2
                ).mean()
                ent = torch.cat(entropy_parts).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

            approx_kl = float(np.mean(kl_batches)) if kl_batches else 0.0
            if approx_kl > 1.5 * self.target_kl:
                stop_epoch = epoch + 1
                break

        self.diagnostics = {
            "adv_share_energy": _adv_share_of_return(ret_e, buf.val_e[:n]),
            "adv_share_peak": _adv_share_of_return(ret_p, buf.val_p[:n]),
            "adv_raw_std": adv_raw_std,
            "adv_near_zero_pct": float(
                100.0 * np.mean(np.abs(adv_e + adv_p) < 1e-3)
            ),
            "approx_kl": approx_kl,
            "epochs_run": stop_epoch,
            "log_std": float(self.net.log_std.item()),
            "value_std_energy": self.norm_energy.std,
            "value_std_peak": self.norm_peak.std,
            "recurrent_sequence_length": self.recurrent_sequence_length,
            "recurrent_chunk_count": len(chunks),
        }
        buf.ptr = 0

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, last_val_energy: float,
               last_val_peak: float):
        if buf.ptr <= 0:
            raise ValueError("PPO2 update requires at least one rollout transition")
        if self.recurrent_enabled:
            self._update_recurrent(buf, last_val_energy, last_val_peak)
            return
        n = buf.ptr
        done = buf.done[:n]

        # Per-component GAE
        adv_e = compute_gae(buf.rew_e[:n], buf.val_e[:n], done,
                            last_val_energy, self.gamma, self.lam_energy)
        adv_p = compute_gae(buf.rew_p[:n], buf.val_p[:n], done,
                            last_val_peak, self.gamma, self.lam_peak)
        ret_e = adv_e + buf.val_e[:n]
        ret_p = adv_p + buf.val_p[:n]

        # Sum in raw reward units, THEN normalise once
        adv = adv_e + adv_p
        adv_raw_std = float(adv.std())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Refresh value normalisers (rescales each head, predictions intact)
        self.norm_energy.update(ret_e)
        self.norm_peak.update(ret_p)

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_e_t = torch.as_tensor(
            self.norm_energy.normalize(ret_e), device=self.device
        )
        ret_p_t = torch.as_tensor(
            self.norm_peak.normalize(ret_p), device=self.device
        )

        idx = np.arange(n)
        approx_kl = 0.0
        stop_epoch = self.epochs
        for epoch in range(self.epochs):
            self._rng.shuffle(idx)
            kl_batches: list[float] = []
            for s in range(0, n, self.minibatch):
                mb = torch.as_tensor(
                    idx[s:s + self.minibatch],
                    dtype=torch.long,
                    device=self.device,
                )
                dist = self.net.dist(obs[mb])
                logp = _squashed_log_prob_from_latent(dist, latent[mb])
                log_ratio = logp - logp_old[mb]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    # Schulman's k3 estimator
                    kl_batches.append(float(((ratio - 1.0) - log_ratio).mean()))

                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()

                v_e, v_p = self.net.normalized_values(obs[mb])
                v_loss = ((v_e - ret_e_t[mb]) ** 2).mean() \
                    + ((v_p - ret_p_t[mb]) ** 2).mean()

                _, entropy_logp, _ = _sample_squashed(dist, deterministic=False)
                ent = -entropy_logp.mean()

                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

            approx_kl = float(np.mean(kl_batches)) if kl_batches else 0.0
            if approx_kl > 1.5 * self.target_kl:
                stop_epoch = epoch + 1
                break

        self.diagnostics = {
            "adv_share_energy": _adv_share_of_return(ret_e, buf.val_e[:n]),
            "adv_share_peak": _adv_share_of_return(ret_p, buf.val_p[:n]),
            "adv_raw_std": adv_raw_std,
            "adv_near_zero_pct": float(100.0 * np.mean(np.abs(adv_e + adv_p) < 1e-3)),
            "approx_kl": approx_kl,
            "epochs_run": stop_epoch,
            "log_std": float(self.net.log_std.item()),
            "value_std_energy": self.norm_energy.std,
            "value_std_peak": self.norm_peak.std,
        }
        buf.ptr = 0

    # ------------------------------------------------------------------
    def save(self, path):
        state_dict = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        torch.save({
            "algo": "ppo2",
            "state_dict": state_dict,
            "value_normalizers": {
                "energy": self.norm_energy.state(),
                "peak": self.norm_peak.state(),
            },
            "meta": dict(self.meta),
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if (
            not isinstance(ck, dict)
            or ck.get("algo") != "ppo2"
            or "state_dict" not in ck
        ):
            raise ValueError("Unsupported checkpoint; retrain the PPO2 policy")
        meta = ck.get("meta", {})
        checkpoint_recurrent = bool(meta.get("recurrent_enabled", False))
        checkpoint_hidden = int(meta.get("hidden_size", self.hidden_size))
        self.recurrent_sequence_length = int(
            meta.get("recurrent_sequence_length", self.recurrent_sequence_length)
        )
        if (
            checkpoint_recurrent != self.recurrent_enabled
            or checkpoint_hidden != self.hidden_size
        ):
            self.recurrent_enabled = checkpoint_recurrent
            self.hidden_size = checkpoint_hidden
            self._build_network(log_std_init=-0.5)
        self.net.load_state_dict(ck["state_dict"], strict=True)
        normalizers = ck["value_normalizers"]
        self.norm_energy.load_state(normalizers["energy"])
        self.norm_peak.load_state(normalizers["peak"])
        self.meta = meta
        self.reset_recurrent_state()
        self.net.eval()


# ---------------------------------------------------------------------------
# PPO2InferenceAgent — senior-style actor-only deployment wrapper
# ---------------------------------------------------------------------------
class PPO2InferenceActor(nn.Module):
    """Feed-forward actor-only inference shell for IQ1-IQ3 checkpoints."""

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()
        self.actor = _mlp(obs_dim, 1, hidden_size)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def dist(self, obs: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(self.actor(obs), self.log_std.exp())


class PPO2RecurrentInferenceActor(nn.Module):
    """Actor-only GRU shell matching IQ4+ recurrent PPO2 checkpoints."""

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.actor_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.Tanh(),
        )
        self.actor_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def dist_step(self, obs: torch.Tensor, hidden=None):
        if hidden is None:
            hidden = torch.zeros(
                1, obs.shape[0], self.hidden_size, device=obs.device
            )
        encoded = self.actor_encoder(obs).unsqueeze(1)
        features, next_hidden = self.actor_gru(encoded, hidden)
        mean = self.actor(features[:, 0])
        return torch.distributions.Normal(mean, self.log_std.exp()), next_hidden


class PPO2InferenceAgent:
    """Actor-only loader supporting feed-forward and recurrent PPO2 checkpoints."""

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        self._obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.recurrent_enabled = False
        self.net = PPO2InferenceActor(self._obs_dim, self.hidden_size)
        self.meta: dict = {}
        self.reset_recurrent_state()

    def reset_recurrent_state(self) -> None:
        self._actor_hidden = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = True) -> float:
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.recurrent_enabled:
            distribution, self._actor_hidden = self.net.dist_step(
                o, self._actor_hidden
            )
            self._actor_hidden = self._actor_hidden.detach()
        else:
            distribution = self.net.dist(o)
        action, _, _ = _sample_squashed(
            distribution, deterministic=deterministic
        )
        return float(action.item())

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray) -> float:
        return self.act(obs, deterministic=True)

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("algo") != "ppo2"
            or "state_dict" not in checkpoint
        ):
            raise ValueError("Unsupported checkpoint; retrain the PPO2 policy")
        if "meta" not in checkpoint:
            raise ValueError("Checkpoint carries no meta; retraining is required")
        self.meta = checkpoint["meta"]
        self.recurrent_enabled = bool(self.meta.get("recurrent_enabled", False))
        self.hidden_size = int(self.meta.get("hidden_size", self.hidden_size))
        self.net = (
            PPO2RecurrentInferenceActor(self._obs_dim, self.hidden_size)
            if self.recurrent_enabled
            else PPO2InferenceActor(self._obs_dim, self.hidden_size)
        )
        state_dict = checkpoint["state_dict"]
        expected = set(self.net.state_dict())
        missing = expected - set(state_dict)
        if missing:
            raise ValueError(
                "Checkpoint is missing policy tensors: " + ", ".join(sorted(missing))
            )
        self.net.load_state_dict(
            {key: value for key, value in state_dict.items() if key in expected}
        )
        self.reset_recurrent_state()
        self.net.eval()
        return self.meta