"""PPO with Oracle Regularization (PRO) for the BESS CMDP.

PRO reuses the standard PPO implementation and changes only what is unique:
its rollout buffer stores an Oracle target action and its policy update adds a
decaying Oracle-imitation loss. Network construction, device handling,
inference, collector synchronization, and ordinary checkpoint compatibility
remain owned by :mod:`ppo_agent`.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from bess.agents.ppo_agent import (
    LOG_STD_MAX,
    LOG_STD_MIN,
    PPOAgent,
    RolloutBuffer,
    _sample_squashed,
    _squashed_log_prob_from_latent,
)
from bess.core.settings import PPO_LAMBDA, PRO_GAMMA


class PROBuffer(RolloutBuffer):
    """Standard PPO rollout data plus one Oracle action target per step."""

    def __init__(self, size: int, obs_dim: int):
        super().__init__(size, obs_dim)
        self.a_oracle = np.zeros((size, 1), np.float32)

    def add(self, o, a, lp, r, v, d, latent, a_oracle=0.0):
        index = self.ptr
        super().add(o, a, lp, r, v, d, latent)
        self.a_oracle[index] = a_oracle

    def clear(self):
        self.ptr = 0


class PROAgent(PPOAgent):
    """PPO agent with a decaying Oracle-imitation auxiliary loss."""

    def __init__(
        self,
        obs_dim: int,
        oracle_coef: float = 1.0,
        oracle_coef_decay: float = 0.0,
        lr=3e-4,
        gamma=PRO_GAMMA,
        lam=PPO_LAMBDA,
        clip=0.2,
        epochs=8,
        minibatch=256,
        ent_coef=3e-3,
        vf_coef=0.5,
        seed=0,
        device="auto",
    ):
        super().__init__(
            obs_dim,
            lr=lr,
            gamma=gamma,
            lam=lam,
            clip=clip,
            epochs=epochs,
            minibatch=minibatch,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            seed=seed,
            device=device,
        )
        self.oracle_coef = float(oracle_coef)
        self.oracle_coef_decay = float(oracle_coef_decay)

    def update(self, buf: PROBuffer, last_val: float) -> dict:
        """Run exact squashed-PPO updates plus Oracle-action imitation."""
        n = buf.ptr
        adv = np.zeros(n, np.float32)
        gae = 0.0
        next_val, next_nonterm = last_val, 1.0
        for i in reversed(range(n)):
            delta = (
                buf.rew[i]
                + self.gamma * next_val * next_nonterm
                - buf.val[i]
            )
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
        oracle_t = torch.as_tensor(buf.a_oracle[:n], device=self.device)

        indices = np.arange(n)
        policy_losses = []
        value_losses = []
        entropies = []
        oracle_losses = []
        clip_fractions = []
        actor_grad_norms = []
        critic_grad_norms = []
        approx_kl = 0.0
        epochs_run = self.epochs

        for epoch in range(self.epochs):
            self._rng.shuffle(indices)
            kl_batches = []
            for start in range(0, n, self.minibatch):
                batch = torch.as_tensor(
                    indices[start:start + self.minibatch],
                    dtype=torch.long,
                    device=self.device,
                )
                dist = self.net.dist(obs[batch])
                logp = _squashed_log_prob_from_latent(dist, latent[batch])
                log_ratio = logp - logp_old[batch]
                ratio = torch.exp(log_ratio)
                with torch.no_grad():
                    kl_batches.append(float(((ratio - 1.0) - log_ratio).mean()))
                    clip_fractions.append(
                        float((torch.abs(ratio - 1.0) > self.clip).float().mean())
                    )

                unclipped = ratio * adv_t[batch]
                clipped = torch.clamp(
                    ratio, 1 - self.clip, 1 + self.clip
                ) * adv_t[batch]
                pi_loss = -torch.min(unclipped, clipped).mean()
                vf_loss = (
                    (self.net.value(obs[batch]) - ret_t[batch]) ** 2
                ).mean()
                _, entropy_log_probability, _ = _sample_squashed(
                    dist, deterministic=False
                )
                entropy = -entropy_log_probability.mean()
                actor_mean = torch.tanh(self.net.actor(obs[batch]))
                oracle_loss = ((actor_mean - oracle_t[batch]) ** 2).mean()
                loss = (
                    pi_loss
                    + self.vf_coef * vf_loss
                    - self.ent_coef * entropy
                    + self.oracle_coef * oracle_loss
                )

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                actor_grad_norms.append(
                    float(nn.utils.clip_grad_norm_(self._actor_parameters, 0.5))
                )
                critic_grad_norms.append(
                    float(nn.utils.clip_grad_norm_(self._critic_parameters, 0.5))
                )
                self.opt.step()
                with torch.no_grad():
                    self.net.log_std.clamp_(LOG_STD_MIN, LOG_STD_MAX)

                policy_losses.append(float(pi_loss.detach()))
                value_losses.append(float(vf_loss.detach()))
                entropies.append(float(entropy.detach()))
                oracle_losses.append(float(oracle_loss.detach()))

            approx_kl = float(np.mean(kl_batches)) if kl_batches else 0.0
            if approx_kl > 1.5 * self.target_kl:
                epochs_run = epoch + 1
                break

        current_oracle_coef = self.oracle_coef
        if self.oracle_coef_decay > 0.0 and self.oracle_coef > 0.0:
            self.oracle_coef = max(
                0.0, self.oracle_coef - self.oracle_coef_decay
            )

        self.diagnostics = {
            "pi_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "vf_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "ent": float(np.mean(entropies)) if entropies else 0.0,
            "oracle_loss": float(np.mean(oracle_losses)) if oracle_losses else 0.0,
            "oracle_coef": current_oracle_coef,
            "approx_kl": approx_kl,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "epochs_run": epochs_run,
            "log_std": float(self.net.log_std.item()),
            "actor_grad_norm": float(np.mean(actor_grad_norms)) if actor_grad_norms else 0.0,
            "critic_grad_norm": float(np.mean(critic_grad_norms)) if critic_grad_norms else 0.0,
            "adv_raw_std": adv_raw_std,
            "explained_variance": explained_variance,
            "learning_rate": float(self.opt.param_groups[0]["lr"]),
        }
        self._sync_collector()
        buf.clear()
        return dict(self.diagnostics)

    def save(self, path):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        payload = {
            "algo": "pro",
            "state_dict": state,
            "meta": dict(self.meta),
            "oracle_coef": self.oracle_coef,
            "oracle_coef_decay": self.oracle_coef_decay,
        }
        if self.forecast_bundle is not None:
            payload["forecast_bundle"] = {
                **self.forecast_bundle,
                "values": torch.as_tensor(
                    self.forecast_bundle["values"]
                ).cpu(),
            }
        torch.save(payload, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.net.load_state_dict(checkpoint["state_dict"])
            self.meta = checkpoint.get("meta", {}) or {}
            self.forecast_bundle = checkpoint.get("forecast_bundle")
            self.oracle_coef = float(
                checkpoint.get("oracle_coef", self.oracle_coef)
            )
            self.oracle_coef_decay = float(
                checkpoint.get(
                    "oracle_coef_decay", self.oracle_coef_decay
                )
            )
        else:
            self.net.load_state_dict(checkpoint)
            self.meta = {}
            self.forecast_bundle = None
        self._sync_collector()
        self.net.eval()
