"""bess.agents.ppo_agent.py  compact PPO (clipped surrogate, GAE) for the BESS CMDP.

The network is deliberately small (2x64 MLP) so a policy step stays far
below the 500 ms edge cycle-time budget on a Raspberry Pi class CPU.
"""
from __future__ import annotations

import copy
import os
import random

import numpy as np

# Required by deterministic CUDA matrix multiplication.  It must be set before
# PPO creates a CUDA context; keeping it here also covers direct PPOAgent users.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import nn

from bess.core.settings import PPO_GAMMA, PPO_LAMBDA

torch.set_num_threads(6)


def configure_ppo_determinism(seed: int) -> None:
    """Seed PPO randomness and reject nondeterministic PyTorch operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Parallel floating-point reductions can change their summation order.
    # PPO prioritizes reproducibility over learner throughput.
    torch.set_num_threads(1)


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


def _gae_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    last_val: float,
    gamma: float,
    lam: float,
) -> np.ndarray:
    """Compute GAE without allowing value/advantage to cross done boundaries."""
    n = len(rewards)
    if len(values) != n or len(dones) != n:
        raise ValueError("rewards, values, and dones must have equal length")
    adv = np.zeros(n, np.float32)
    gae = 0.0
    next_val = float(last_val)
    for i in reversed(range(n)):
        nonterminal = 1.0 - float(dones[i])
        delta = rewards[i] + gamma * next_val * nonterminal - values[i]
        gae = delta + gamma * lam * nonterminal * gae
        adv[i] = gae
        next_val = float(values[i])
    return adv


class PPOAgent:
    def __init__(self, obs_dim: int, lr=1e-4, gamma=PPO_GAMMA, lam=PPO_LAMBDA,
                 clip=0.2, epochs=4, minibatch=256, ent_coef=0.0,
                 vf_coef=0.5, target_kl=0.02, seed=0, device="auto"):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(resolve_ppo_device(device))
        self.net = ActorCritic(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Keep tiny step-by-step environment inference on CPU. Only large PPO
        # minibatches cross to CUDA, avoiding thousands of tiny PCIe transfers.
        with torch.random.fork_rng(devices=[]):
            self.collector_net = ActorCritic(obs_dim).cpu()
        self._sync_collector()
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.target_kl = float(target_kl)
        self.last_update_stats = {}
        self.meta = {}          # deployment context (p_ref_kw, obs_variant)

    @torch.inference_mode()
    def _sync_collector(self):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        self.collector_net.load_state_dict(state)
        self.collector_net.eval()

    def snapshot_training_state(self) -> dict:
        """Copy learned PPO + Adam state without freezing any random generator."""
        return {
            "network": copy.deepcopy(self.net.state_dict()),
            "optimizer": copy.deepcopy(self.opt.state_dict()),
        }

    def restore_training_state(self, snapshot: dict) -> None:
        """Restore an in-memory learner snapshot and make collection use it immediately."""
        self.net.load_state_dict(snapshot["network"])
        self.opt.load_state_dict(snapshot["optimizer"])
        self._sync_collector()

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

    # ------------------------------------------------------------------
    def update(self, buf: RolloutBuffer, last_val: float):
        n = buf.ptr
        if n <= 0:
            raise ValueError("PPO update requires at least one rollout transition")

        # GAE must mask the bootstrap with the *current transition's* done flag.
        # Using the previous loop iteration's flag leaks value/advantage across
        # episode boundaries whenever one rollout contains multiple episodes.
        adv = _gae_advantages(
            buf.rew[:n],
            buf.val[:n],
            buf.done[:n],
            last_val=last_val,
            gamma=self.gamma,
            lam=self.lam,
        )
        ret = adv + buf.val[:n]
        raw_adv_mean = float(adv.mean())
        raw_adv_std = float(adv.std())
        ret_mean = float(ret.mean())
        ret_std = float(ret.std())
        reward_mean = float(buf.rew[:n].mean())
        reward_std = float(buf.rew[:n].std())
        return_variance = float(np.var(ret))
        explained_variance = (
            1.0 - float(np.var(ret - buf.val[:n])) / return_variance
            if return_variance > 1e-12
            else 0.0
        )
        adv = (adv - raw_adv_mean) / (raw_adv_std + 1e-8)

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)

        pi_losses = []
        value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []
        actor_grad_norms = []
        critic_grad_norms = []
        actor_params = [*self.net.actor.parameters(), self.net.log_std]
        critic_params = list(self.net.critic.parameters())
        epochs_completed = 0
        early_stopped = False

        for epoch in range(self.epochs):
            idx = self.rng.permutation(n)
            epoch_kls = []
            for s in range(0, n, self.minibatch):
                mb = torch.as_tensor(
                    idx[s:s + self.minibatch], dtype=torch.long,
                    device=self.device,
                )
                dist = self.net.dist(obs[mb])
                logp = _squashed_log_prob_from_latent(dist, latent[mb])
                log_ratio = logp - logp_old[mb]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((self.net.value(obs[mb]) - ret_t[mb]) ** 2).mean()
                _, entropy_log_probability, _ = _sample_squashed(
                    dist,
                    deterministic=False,
                )
                ent = -entropy_log_probability.mean()
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
                clip_fraction = ((ratio - 1.0).abs() > self.clip).float().mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                actor_grad_norm = nn.utils.clip_grad_norm_(actor_params, 0.5)
                critic_grad_norm = nn.utils.clip_grad_norm_(critic_params, 0.5)
                self.opt.step()

                pi_losses.append(float(pi_loss.detach().cpu()))
                value_losses.append(float(v_loss.detach().cpu()))
                entropies.append(float(ent.detach().cpu()))
                kl_value = float(approx_kl.detach().cpu())
                approx_kls.append(kl_value)
                epoch_kls.append(kl_value)
                clip_fractions.append(float(clip_fraction.detach().cpu()))
                actor_grad_norm_value = float(actor_grad_norm.detach().cpu())
                critic_grad_norm_value = float(critic_grad_norm.detach().cpu())
                actor_grad_norms.append(actor_grad_norm_value)
                critic_grad_norms.append(critic_grad_norm_value)
                grad_norms.append(max(actor_grad_norm_value, critic_grad_norm_value))

            epochs_completed = epoch + 1
            if epoch_kls and float(np.mean(epoch_kls)) > self.target_kl:
                early_stopped = True
                break

        def _mean(values):
            return float(np.mean(values)) if values else 0.0

        self.last_update_stats = {
            "policy_loss": _mean(pi_losses),
            "value_loss": _mean(value_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(approx_kls),
            "clip_fraction": _mean(clip_fractions),
            "grad_norm": _mean(grad_norms),
            "actor_grad_norm": _mean(actor_grad_norms),
            "critic_grad_norm": _mean(critic_grad_norms),
            "advantage_mean_raw": raw_adv_mean,
            "advantage_std_raw": raw_adv_std,
            "return_mean": ret_mean,
            "return_std": ret_std,
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "explained_variance": explained_variance,
            "log_std": float(self.net.log_std.detach().cpu().item()),
            "epochs_completed": epochs_completed,
            "kl_early_stop": early_stopped,
        }
        self._sync_collector()
        buf.ptr = 0
        return dict(self.last_update_stats)

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
        torch.save(payload, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict) and "state_dict" in ck:
            algo = str(ck.get("algo") or "ppo").lower()
            if algo != "ppo":
                raise ValueError(f"checkpoint algorithm {algo!r} is not PPO")
            self.net.load_state_dict(ck["state_dict"])
            self.meta = ck.get("meta", {}) or {}
        else:                                   # legacy raw PPO state_dict
            self.net.load_state_dict(ck)
            self.meta = {}
        self._sync_collector()
        self.net.eval()
