"""pro_agent.py  PPO with Oracle Regularization (PRO) for the BESS CMDP.

Extension of the standard PPO clipped-surrogate objective with an auxiliary
behavioural-cloning loss that guides the policy toward the perfect-foresight
Oracle LP's implied action at each step.  The oracle weight decays linearly
over training, so early iterations benefit from a strong imitation signal
while later iterations let the RL objective take over.

Architecture is identical to ppo_agent.py (2×64 Tanh MLP, learnable log_std,
GAE, entropy bonus) so checkpoints are cross-loadable and inference latency
stays unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from settings import PPO_GAMMA, PPO_LAMBDA

torch.set_num_threads(6)
_LOG_2PI = float(np.log(2.0 * np.pi))


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
        mean = torch.tanh(self.actor(obs))
        return torch.distributions.Normal(mean, self.log_std.exp())

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


class PROBuffer:
    """Rollout buffer with an extra slot for the oracle target action."""

    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew = np.zeros(size, np.float32)
        self.val = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.a_oracle = np.zeros((size, 1), np.float32)   # oracle target
        self.ptr = 0
        self.size = size

    def add(self, o, a, lp, r, v, d, a_oracle=0.0):
        i = self.ptr
        self.obs[i], self.act[i] = o, a
        self.logp[i], self.rew[i], self.val[i], self.done[i] = lp, r, v, d
        self.a_oracle[i] = a_oracle
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size

    def clear(self):
        self.ptr = 0


class PROAgent:
    """PPO agent with decaying oracle-regularization auxiliary loss.

    Parameters
    ----------
    obs_dim : int
        Observation dimension (13 or 17).
    oracle_coef : float
        Initial weight on the oracle imitation loss L_oracle.
    oracle_coef_decay : float
        Linear decay subtracted from oracle_coef after each update().
        Reaches zero after oracle_coef / oracle_coef_decay updates.
        Set to 0 to disable decay (constant oracle weight).
    lr, gamma, lam, clip, epochs, minibatch, ent_coef, vf_coef, seed, device :
        Identical to PPOAgent.
    """

    def __init__(self, obs_dim: int,
                 oracle_coef: float = 1.0,
                 oracle_coef_decay: float = 0.0,
                 lr=3e-4, gamma=PPO_GAMMA, lam=PPO_LAMBDA,
                 clip=0.2, epochs=8, minibatch=256, ent_coef=3e-3,
                 vf_coef=0.5, seed=0, device="auto"):
        import torch

        from ppo_agent import resolve_ppo_device

        torch.manual_seed(seed)
        self.device = torch.device(resolve_ppo_device(device))
        self.net = ActorCritic(obs_dim).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        with torch.random.fork_rng(devices=[]):
            self.collector_net = ActorCritic(obs_dim).cpu()
        self._sync_collector()
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.oracle_coef = float(oracle_coef)
        self.oracle_coef_decay = float(oracle_coef_decay)
        self.meta = {}
        self.forecast_bundle = None

    # ------------------------------------------------------------------
    # Inference (identical to PPOAgent)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _sync_collector(self):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        self.collector_net.load_state_dict(state)
        self.collector_net.eval()

    @torch.inference_mode()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.collector_net.actor(o))
        std = self.collector_net.log_std.exp()
        a = mean if deterministic else torch.normal(mean, std)
        logp = (
            -0.5 * ((a - mean) / std).square()
            - self.collector_net.log_std
            - 0.5 * _LOG_2PI
        ).sum(-1)
        v = self.collector_net.value(o)
        return (float(np.clip(a.item(), -1.0, 1.0)),
                float(logp.item()), float(v.item()))

    @torch.inference_mode()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference for evaluation rollouts."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.collector_net.actor(o))
        return float(np.clip(mean.item(), -1.0, 1.0))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def update(self, buf: PROBuffer, last_val: float) -> dict:
        """One PPO epoch with oracle-regularized policy loss.

        Returns a dict with pi_loss, vf_loss, ent, oracle_loss, oracle_coef
        for logging.
        """
        n = buf.ptr

        # --- GAE -----------------------------------------------------------
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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        act = torch.as_tensor(buf.act[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        a_oracle_t = torch.as_tensor(buf.a_oracle[:n], device=self.device)

        idx = np.arange(n)
        pi_loss_sum = 0.0
        vf_loss_sum = 0.0
        ent_sum = 0.0
        oracle_loss_sum = 0.0
        update_count = 0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = torch.as_tensor(
                    idx[s:s + self.minibatch], dtype=torch.long,
                    device=self.device,
                )
                dist = self.net.dist(obs[mb])
                logp = dist.log_prob(act[mb]).sum(-1)
                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip,
                                    1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                vf_loss = ((self.net.value(obs[mb]) - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()

                # Oracle auxiliary loss: MSE between actor mean and oracle target
                actor_mean = torch.tanh(self.net.actor(obs[mb]))
                oracle_loss = ((actor_mean - a_oracle_t[mb]) ** 2).mean()

                loss = (pi_loss + self.vf_coef * vf_loss
                        - self.ent_coef * ent
                        + self.oracle_coef * oracle_loss)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                pi_loss_sum += pi_loss.detach().item()
                vf_loss_sum += vf_loss.detach().item()
                ent_sum += ent.detach().item()
                oracle_loss_sum += oracle_loss.detach().item()
                update_count += 1

        # Decay oracle weight
        current_oracle_coef = self.oracle_coef
        if self.oracle_coef_decay > 0 and self.oracle_coef > 0:
            self.oracle_coef = max(0.0,
                                   self.oracle_coef - self.oracle_coef_decay)

        self._sync_collector()
        buf.clear()

        return {
            "pi_loss": pi_loss_sum / max(1, update_count),
            "vf_loss": vf_loss_sum / max(1, update_count),
            "ent": ent_sum / max(1, update_count),
            "oracle_loss": oracle_loss_sum / max(1, update_count),
            "oracle_coef": current_oracle_coef,
        }

    # ------------------------------------------------------------------
    # Persistence (compatible with PPOAgent checkpoint format)
    # ------------------------------------------------------------------
    def save(self, path):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        payload = {"algo": "pro", "state_dict": state,
                   "meta": dict(self.meta),
                   "oracle_coef": self.oracle_coef,
                   "oracle_coef_decay": self.oracle_coef_decay}
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
            # Restore oracle schedule if the checkpoint came from PRO
            self.oracle_coef = float(ck.get("oracle_coef", self.oracle_coef))
            self.oracle_coef_decay = float(
                ck.get("oracle_coef_decay", self.oracle_coef_decay)
            )
        else:                                   # legacy raw state_dict
            self.net.load_state_dict(ck)
            self.meta = {}
            self.forecast_bundle = None
        self._sync_collector()
        self.net.eval()