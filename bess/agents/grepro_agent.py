"""GrePRO: Group-relative Progressive-horizon Optimization.

This is a repo-local experimental method, not the published GREPO method.
It preserves chronological multi-day experience and mixes two advantages:
an absolute globally-normalized return advantage and a same-time group-
relative advantage. GREPO remains implemented separately in grepo_agent.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from bess.agents.grepo_agent import GREPOAgent


class GREPROAgent(GREPOAgent):
    def __init__(self, *args, epochs: int = 4, minibatch: int = 2048, **kwargs):
        super().__init__(*args, epochs=epochs, minibatch=minibatch, **kwargs)
        # Residual policy starts by trusting SADRBC exactly.  Exploration can
        # still discover improvements, but deterministic inference begins at
        # a zero correction instead of a random battery command.
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self._sync_collector_actor()

    def update(self, obs_g, act_g, logp_g, rew_g):
        """Blend absolute and group-relative learning signals."""
        return_started = time.perf_counter()
        ng, horizon = rew_g.shape
        eps = 1e-8
        returns = self._discounted_returns(rew_g)

        # Absolute signal: one normalization across the whole group/month.
        absolute = (returns - returns.mean()) / (returns.std() + eps)

        # Relative signal: which sibling made the better decision at time t?
        group_mean = returns.mean(axis=0, keepdims=True)
        group_std = returns.std(axis=0, keepdims=True)
        relative = (returns - group_mean) / (group_std + eps)
        return_seconds = time.perf_counter() - return_started

        transfer_started = time.perf_counter()
        count = ng * horizon
        obs = torch.as_tensor(
            obs_g.reshape(count, -1), dtype=torch.float32, device=self.device
        )
        act = torch.as_tensor(
            act_g.reshape(count, 1), dtype=torch.float32, device=self.device
        )
        old_logp = torch.as_tensor(
            logp_g.reshape(count), dtype=torch.float32, device=self.device
        )
        absolute_t = torch.as_tensor(
            absolute.reshape(count), dtype=torch.float32, device=self.device
        )
        relative_t = torch.as_tensor(
            relative.reshape(count), dtype=torch.float32, device=self.device
        )
        self._sync_device()
        transfer_seconds = time.perf_counter() - transfer_started

        update_started = time.perf_counter()
        indices = np.arange(count)
        policy_loss = value_loss = None
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            for start in range(0, count, self.minibatch):
                batch = torch.as_tensor(
                    indices[start:start + self.minibatch],
                    dtype=torch.long,
                    device=self.device,
                )
                mean = torch.tanh(self.actor(obs[batch]))
                logp = self._logp(act[batch], mean)
                value = self.critic(obs[batch]).squeeze(-1)

                absolute_adv = absolute_t[batch] - value.detach()
                advantage = (
                    (1.0 - self.beta) * absolute_adv
                    + self.beta * relative_t[batch]
                )
                advantage = (
                    advantage - advantage.mean()
                ) / (advantage.std(unbiased=False) + eps)

                ratio = torch.exp(logp - old_logp[batch])
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip, 1.0 + self.clip
                ) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = ((value - absolute_t[batch]) ** 2).mean()
                loss = policy_loss + self.vf_coef * value_loss

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, 0.5)
                self.opt.step()
        self._sync_device()
        update_seconds = time.perf_counter() - update_started

        sync_started = time.perf_counter()
        self._sync_collector_actor()
        sync_seconds = time.perf_counter() - sync_started
        self.last_update_stats = {
            "return_preparation_seconds": return_seconds,
            "batch_transfer_seconds": transfer_seconds,
            "actor_critic_update_seconds": update_seconds,
            "cpu_actor_sync_seconds": sync_seconds,
        }
        return {
            "pi_loss": float(policy_loss.detach().cpu().item()),
            "vf_loss": float(value_loss.detach().cpu().item()),
            "absolute_return_std": float(returns.std()),
            "relative_adv_std": float(relative.std()),
        }

    def save(self, path):
        actor_state = {
            key: value.detach().cpu()
            for key, value in self.actor.state_dict().items()
        }
        critic_state = {
            key: value.detach().cpu()
            for key, value in self.critic.state_dict().items()
        }
        payload = {
            "actor": actor_state,
            "critic": critic_state,
            "obs_dim": self.obs_dim,
            "std": self.std,
            "algo": "grepro",
            "meta": dict(self.meta),
        }
        if self.forecast_bundle is not None:
            payload["forecast_bundle"] = {
                **self.forecast_bundle,
                "values": torch.as_tensor(
                    self.forecast_bundle["values"]
                ).cpu(),
            }
        torch.save(payload, path)
