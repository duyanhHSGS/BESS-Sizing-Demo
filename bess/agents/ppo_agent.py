"""bess.agents.ppo_agent.py  compact PPO (clipped surrogate, GAE) for the BESS CMDP.

The network is deliberately small (2x64 MLP) so a policy step stays far
below the 500 ms edge cycle-time budget on a Raspberry Pi class CPU.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bess.core.settings import PPO_GAMMA, PPO_LAMBDA

torch.set_num_threads(6)

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _squashed_log_prob_from_latent(
    distribution: torch.distributions.Normal,
    latent: torch.Tensor,
) -> torch.Tensor:
    """Log-probability of ``tanh(latent)`` under a squashed Gaussian policy."""
    correction = 2.0 * (
        np.log(2.0)
        - latent
        - torch.nn.functional.softplus(-2.0 * latent)
    )
    return (distribution.log_prob(latent) - correction).sum(-1)


def _sample_squashed(
    distribution: torch.distributions.Normal,
    *,
    deterministic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return bounded action, corrected log-probability, and pre-tanh latent."""
    latent = distribution.mean if deterministic else distribution.rsample()
    action = torch.tanh(latent)
    log_probability = _squashed_log_prob_from_latent(distribution, latent)
    return action, log_probability, latent


def resolve_ppo_device(device: str = "auto") -> str:
    requested = str(device).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("PPO device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "PPO device='cuda' requested, but PyTorch reports that CUDA is unavailable"
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _mlp(inp, out, hidden=64):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int):
        super().__init__()
        self.actor = _mlp(obs_dim, 1)
        self.critic = _mlp(obs_dim, 1)
        # Start deterministic deployment at physical idle. Exploration still
        # comes from log_std, but random final-layer bias can no longer make
        # the first checkpoint an always-charge policy.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((1,), -0.5))

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.latent = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew = np.zeros(size, np.float32)
        self.val = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0
        self.size = size

    def add(self, o, a, lp, r, v, d, latent):
        i = self.ptr
        self.obs[i], self.act[i], self.latent[i] = o, a, latent
        self.logp[i], self.rew[i], self.val[i], self.done[i] = lp, r, v, d
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size


class PPOAgent:
    def __init__(self, obs_dim: int, lr=3e-4, gamma=PPO_GAMMA, lam=PPO_LAMBDA,
                 clip=0.2, epochs=8, minibatch=256, ent_coef=3e-3,
                 vf_coef=0.5, target_kl=0.01, seed=0, device="auto"):
        torch.manual_seed(seed)
        self._rng = np.random.default_rng(seed)
        self.device = torch.device(resolve_ppo_device(device))
        self.net = ActorCritic(obs_dim).to(self.device)
        self._actor_parameters = list(self.net.actor.parameters()) + [self.net.log_std]
        self._critic_parameters = list(self.net.critic.parameters())
        self.opt = torch.optim.Adam([
            {"params": self._actor_parameters, "lr": lr},
            {"params": self._critic_parameters, "lr": lr},
        ])
        self._base_lrs = [float(lr), float(lr)]
        # Keep tiny step-by-step environment inference on CPU. Only large PPO
        # minibatches cross to CUDA, avoiding thousands of tiny PCIe transfers.
        with torch.random.fork_rng(devices=[]):
            self.collector_net = ActorCritic(obs_dim).cpu()
        self._sync_collector()
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.target_kl = float(target_kl)
        self.meta = {}          # deployment context (p_ref_kw, obs_variant)
        self.forecast_bundle = None
        self.diagnostics = {}

    @torch.inference_mode()
    def _sync_collector(self):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        self.collector_net.load_state_dict(state)
        self.collector_net.eval()

    # ------------------------------------------------------------------
    def anneal_lr(self, progress: float) -> None:
        """Linearly decay learning rates as training progress moves 0 -> 1."""
        decay = max(0.0, 1.0 - float(progress))
        for group, base_lr in zip(self.opt.param_groups, self._base_lrs, strict=True):
            group["lr"] = base_lr * decay

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def act_with_latent(self, obs: np.ndarray, deterministic: bool = False):
        """Return bounded action plus its pre-tanh latent for exact PPO updates."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        distribution = self.collector_net.dist(o)
        action, log_probability, latent = _sample_squashed(
            distribution,
            deterministic=deterministic,
        )
        value = self.collector_net.value(o)
        return (
            float(action.item()),
            float(log_probability.item()),
            float(latent.item()),
            float(value.item()),
        )

    @torch.inference_mode()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        action, log_probability, _latent, value = self.act_with_latent(
            obs,
            deterministic=deterministic,
        )
        return action, log_probability, value

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference for evaluation rollouts."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = torch.tanh(self.collector_net.actor(o))
        return float(action.item())

    @torch.inference_mode()
    def predict_value(self, obs: np.ndarray) -> float:
        """Critic-only bootstrap value without sampling and perturbing policy RNG."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        return float(self.collector_net.value(o).item())

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, last_val: float):
        n = buf.ptr
        adv = np.zeros(n, np.float32)
        gae = 0.0
        next_val, next_nonterm = last_val, 1.0
        for i in reversed(range(n)):
            delta = buf.rew[i] + self.gamma * next_val * next_nonterm - buf.val[i]
            gae = delta + self.gamma * self.lam * next_nonterm * gae
            adv[i] = gae
            next_val = buf.val[i]
            next_nonterm = 1.0 - buf.done[i]
        ret = adv + buf.val[:n]
        adv_raw_std = float(adv.std())
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        value_variance = float(np.var(ret))
        explained_variance = (
            0.0
            if value_variance < 1e-12
            else float(1.0 - np.var(ret - buf.val[:n]) / value_variance)
        )

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)

        idx = np.arange(n)
        approx_kl = 0.0
        stop_epoch = self.epochs
        policy_losses = []
        value_losses = []
        entropies = []
        clip_fractions = []
        actor_grad_norms = []
        critic_grad_norms = []
        for epoch in range(self.epochs):
            self._rng.shuffle(idx)
            kl_batches = []
            for s in range(0, n, self.minibatch):
                mb = torch.as_tensor(
                    idx[s:s + self.minibatch], dtype=torch.long,
                    device=self.device,
                )
                dist = self.net.dist(obs[mb])
                logp = _squashed_log_prob_from_latent(dist, latent[mb])
                log_ratio = logp - logp_old[mb]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    kl_batches.append(float(((ratio - 1.0) - log_ratio).mean()))
                    clip_fractions.append(
                        float((torch.abs(ratio - 1.0) > self.clip).float().mean())
                    )
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((self.net.value(obs[mb]) - ret_t[mb]) ** 2).mean()
                _, entropy_log_probability, _ = _sample_squashed(
                    dist,
                    deterministic=False,
                )
                ent = -entropy_log_probability.mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                # Actor and critic do not share a trunk. Clip them separately so
                # a large sparse demand-charge value error cannot crush the actor
                # gradient merely by dominating one global gradient norm.
                actor_grad_norms.append(float(nn.utils.clip_grad_norm_(self._actor_parameters, 0.5)))
                critic_grad_norms.append(float(nn.utils.clip_grad_norm_(self._critic_parameters, 0.5)))
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)
                policy_losses.append(float(pi_loss.detach()))
                value_losses.append(float(v_loss.detach()))
                entropies.append(float(ent.detach()))

            approx_kl = float(np.mean(kl_batches)) if kl_batches else 0.0
            if approx_kl > 1.5 * self.target_kl:
                stop_epoch = epoch + 1
                break

        self.diagnostics = {
            "approx_kl": approx_kl,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "epochs_run": stop_epoch,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "log_std": float(self.net.log_std.item()),
            "actor_grad_norm": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            "critic_grad_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
            "adv_raw_std": adv_raw_std,
            "explained_variance": explained_variance,
            "learning_rate": float(self.opt.param_groups[0]["lr"]),
        }
        self._sync_collector()
        buf.ptr = 0
        return dict(self.diagnostics)

    # ------------------------------------------------------------------
    # self.meta carries deployment context (e.g. p_ref_kw the observation
    # normalisation was trained with) so loaders can reconstruct the env.
    def save(self, path):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        payload = {"algo": "ppo", "state_dict": state,
                   "meta": dict(self.meta)}
        if self.forecast_bundle is not None:
            payload["forecast_bundle"] = {
                **self.forecast_bundle,
                "values": torch.as_tensor(self.forecast_bundle["values"]).cpu(),
            }
        torch.save(payload, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict) and "state_dict" in ck:
            self.net.load_state_dict(ck["state_dict"])
            self.meta = ck.get("meta", {}) or {}
            self.forecast_bundle = ck.get("forecast_bundle")
        else:                                   # legacy raw state_dict
            self.net.load_state_dict(ck)
            self.meta = {}
            self.forecast_bundle = None
        self._sync_collector()
        self.net.eval()
