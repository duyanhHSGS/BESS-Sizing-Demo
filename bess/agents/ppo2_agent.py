"""bess.agents.ppo2_agent.py — PPO with reward-decomposed critics for the BESS CMDP.

Architecture mirrors bess-drl (other-project/bess-drl):
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
import torch.nn as nn

from bess.core.settings import PPO2_GAMMA

torch.set_num_threads(6)

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
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PPO2 device='cuda' requested, but PyTorch reports that CUDA is unavailable"
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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


# ---------------------------------------------------------------------------
# Rollout buffer (decomposed rewards)
# ---------------------------------------------------------------------------
class RolloutBuffer:
    """Stores two reward/value components plus the pre-tanh latent."""

    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.latent = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew_e = np.zeros(size, np.float32)
        self.rew_p = np.zeros(size, np.float32)
        self.val_e = np.zeros(size, np.float32)
        self.val_p = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0
        self.size = size

    def add(self, o, a, latent, lp, r_e, r_p, v_e, v_p, d):
        i = self.ptr
        self.obs[i], self.act[i], self.latent[i] = o, a, latent
        self.logp[i] = lp
        self.rew_e[i], self.rew_p[i] = r_e, r_p
        self.val_e[i], self.val_p[i] = v_e, v_p
        self.done[i] = d
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


# ---------------------------------------------------------------------------
# PPO2Agent (training)
# ---------------------------------------------------------------------------
class PPO2Agent:
    """PPO with reward-decomposed critics, PopArt, and squashed Gaussian policy."""

    def __init__(self, obs_dim: int, lr=1e-4, gamma=PPO2_GAMMA,
                 lam_energy=0.97, lam_peak=0.97,
                 clip=0.2, epochs=6, minibatch=256, ent_coef=0.01,
                 vf_coef=0.5, target_kl=0.01, seed=0,
                 actor_lr: float | None = None,
                 critic_lr: float | None = None,
                 log_std_init: float = float(np.log(0.15)),
                 device: str = "auto"):
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.device = torch.device(resolve_ppo2_device(device))
        self.net = ActorCritic(obs_dim, log_std_init=log_std_init).to(self.device)
        self.actor_lr = lr if actor_lr is None else float(actor_lr)
        self.critic_lr = lr if critic_lr is None else float(critic_lr)
        self.lr = lr
        actor_params = list(self.net.actor.parameters()) + [self.net.log_std]
        critic_params = (
            list(self.net.critic_energy.parameters())
            + list(self.net.critic_peak.parameters())
        )
        self.opt = torch.optim.Adam([
            {"params": actor_params, "lr": self.actor_lr},
            {"params": critic_params, "lr": self.critic_lr},
        ])
        self._base_lrs = [self.actor_lr, self.critic_lr]
        self.gamma = gamma
        self.lam_energy, self.lam_peak = lam_energy, lam_peak
        self.clip = clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.target_kl = target_kl
        self.norm_energy = PopArtNormalizer(self.net.critic_energy)
        self.norm_peak = PopArtNormalizer(self.net.critic_peak)
        with torch.random.fork_rng(devices=[]):
            self.collector_net = ActorCritic(obs_dim, log_std_init=log_std_init).cpu()
        self._sync_collector()
        self.meta = {}
        self.diagnostics = {}

    @torch.inference_mode()
    def _sync_collector(self) -> None:
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        self.collector_net.load_state_dict(state)
        self.collector_net.eval()

    def anneal_lr(self, progress: float) -> None:
        """Linearly decay each group's learning rate; progress runs 0 -> 1."""
        decay = max(0.0, 1.0 - float(progress))
        for group, base in zip(self.opt.param_groups, self._base_lrs, strict=True):
            group["lr"] = base * decay

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Return (action, log_prob, latent, value_energy, value_peak).

        Values are DENORMALISED so GAE runs on the same scale as rewards.
        """
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = self.collector_net.dist(o)
        a, logp, latent = _sample_squashed(dist, deterministic=deterministic)
        v_e, v_p = self.collector_net.normalized_values(o)
        return (
            float(a.item()),
            float(logp.item()),
            float(latent.item()),
            float(self.norm_energy.denormalize(v_e.item())),
            float(self.norm_peak.denormalize(v_p.item())),
        )

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference for evaluation rollouts."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = self.collector_net.dist(o)
        a, _, _ = _sample_squashed(dist, deterministic=True)
        return float(a.item())

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, last_val_energy: float,
               last_val_peak: float):
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

        self._sync_collector()
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
        self.net.load_state_dict(ck["state_dict"], strict=True)
        normalizers = ck["value_normalizers"]
        self.norm_energy.load_state(normalizers["energy"])
        self.norm_peak.load_state(normalizers["peak"])
        self.meta = ck["meta"]
        self.net.eval()
        self._sync_collector()


# ---------------------------------------------------------------------------
# PPO2InferenceAgent — lightweight actor-only for dispatch/benchmarking
# ---------------------------------------------------------------------------
class PPO2InferenceAgent:
    """Loads a trained PPO2 checkpoint and runs deterministic inference.

    Deliberately holds NO critic tensors — loads only actor.* / log_std keys.
    Compatible with the same act() signature as PPOAgent from ppo_agent.py:
    act(obs, deterministic=False) -> (action, log_prob, value).
    """

    def __init__(self, obs_dim: int, hidden_size: int = 128):
        super().__init__()
        self._obs_dim = obs_dim
        self.net = ActorCritic(obs_dim, hidden_size)
        self.meta: dict = {}

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[float, float, float]:
        """Return (action, log_prob, value_sum) for dispatch compatibility.

        The single value is the sum of denormalised energy + peak critics,
        reconstructed from the saved normaliser state in self.meta.
        """
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = self.net.dist(o)
        a, logp, _ = _sample_squashed(dist, deterministic=deterministic)
        # Denormalise using saved normaliser state for dispatch compatibility
        n_e = self.meta.get("norm_energy_mean", 0.0)
        n_e_std = self.meta.get("norm_energy_std", 1.0)
        n_p = self.meta.get("norm_peak_mean", 0.0)
        n_p_std = self.meta.get("norm_peak_std", 1.0)
        v_e_raw, v_p_raw = self.net.normalized_values(o)
        v_total = float(
            (v_e_raw.item() * n_e_std + n_e)
            + (v_p_raw.item() * n_p_std + n_p)
        )
        return (float(a.item()), float(logp.item()), v_total)

    @torch.no_grad()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference — single float output."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        dist = self.net.dist(o)
        a, _, _ = _sample_squashed(dist, deterministic=True)
        return float(a.item())

    def load(self, path: str) -> dict:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(checkpoint, dict)
            or checkpoint.get("algo") != "ppo2"
            or "state_dict" not in checkpoint
        ):
            raise ValueError("Unsupported checkpoint; retrain the PPO2 policy")
        state_dict = checkpoint["state_dict"]
        expected = set(self.net.state_dict())
        missing = expected - set(state_dict)
        if missing:
            raise ValueError(
                "Checkpoint is missing policy tensors: " + ", ".join(sorted(missing))
            )
        # Drop critic tensors — only actor and log_std are needed for inference
        self.net.load_state_dict(
            {key: value for key, value in state_dict.items() if key in expected}
        )
        if "meta" not in checkpoint:
            raise ValueError("Checkpoint carries no meta; retraining is required")
        self.meta = checkpoint["meta"]
        # Store normaliser state so act() can denormalise values
        if "value_normalizers" in checkpoint:
            self.meta["norm_energy_mean"] = checkpoint["value_normalizers"]["energy"]["mean"]
            self.meta["norm_energy_std"] = checkpoint["value_normalizers"]["energy"]["std"]
            self.meta["norm_peak_mean"] = checkpoint["value_normalizers"]["peak"]["mean"]
            self.meta["norm_peak_std"] = checkpoint["value_normalizers"]["peak"]["std"]
        self.net.eval()
        return self.meta