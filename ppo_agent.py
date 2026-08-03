"""ppo_agent.py  compact PPO (clipped surrogate, GAE) for the BESS CMDP.

The network is deliberately small (2x64 MLP) so a policy step stays far
below the 500 ms edge cycle-time budget on a Raspberry Pi class CPU.
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


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew = np.zeros(size, np.float32)
        self.val = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.ptr = 0
        self.size = size

    def add(self, o, a, lp, r, v, d):
        i = self.ptr
        self.obs[i], self.act[i] = o, a
        self.logp[i], self.rew[i], self.val[i], self.done[i] = lp, r, v, d
        self.ptr += 1

    def full(self):
        return self.ptr >= self.size


class PPOAgent:
    def __init__(self, obs_dim: int, lr=3e-4, gamma=PPO_GAMMA, lam=PPO_LAMBDA,
                 clip=0.2, epochs=8, minibatch=256, ent_coef=3e-3,
                 vf_coef=0.5, seed=0):
        torch.manual_seed(seed)
        self.net = ActorCritic(obs_dim)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.meta = {}          # deployment context (p_ref_kw, obs_variant)
        self.forecast_bundle = None

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.net.actor(o))
        std = self.net.log_std.exp()
        a = mean if deterministic else torch.normal(mean, std)
        logp = (
            -0.5 * ((a - mean) / std).square()
            - self.net.log_std
            - 0.5 * _LOG_2PI
        ).sum(-1)
        v = self.net.value(o)
        return (float(np.clip(a.item(), -1.0, 1.0)),
                float(logp.item()), float(v.item()))

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference for evaluation rollouts."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.net.actor(o))
        return float(np.clip(mean.item(), -1.0, 1.0))

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
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(buf.obs[:n])
        act = torch.as_tensor(buf.act[:n])
        logp_old = torch.as_tensor(buf.logp[:n])
        adv_t = torch.as_tensor(adv)
        ret_t = torch.as_tensor(ret)

        idx = np.arange(n)
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                dist = self.net.dist(obs[mb])
                logp = dist.log_prob(act[mb]).sum(-1)
                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                v_loss = ((self.net.value(obs[mb]) - ret_t[mb]) ** 2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + self.vf_coef * v_loss - self.ent_coef * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
        buf.ptr = 0

    # ------------------------------------------------------------------
    # self.meta carries deployment context (e.g. p_ref_kw the observation
    # normalisation was trained with) so loaders can reconstruct the env.
    def save(self, path):
        payload = {"algo": "ppo", "state_dict": self.net.state_dict(),
                   "meta": dict(self.meta)}
        if self.forecast_bundle is not None:
            payload["forecast_bundle"] = {
                **self.forecast_bundle,
                "values": torch.as_tensor(self.forecast_bundle["values"]).cpu(),
            }
        torch.save(payload, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        if isinstance(ck, dict) and "state_dict" in ck:
            self.net.load_state_dict(ck["state_dict"])
            self.meta = ck.get("meta", {}) or {}
            self.forecast_bundle = ck.get("forecast_bundle")
        else:                                   # legacy raw state_dict
            self.net.load_state_dict(ck)
            self.meta = {}
            self.forecast_bundle = None
        self.net.eval()
