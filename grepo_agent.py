"""grepo_agent.py  Group Relative Enhancement Policy Optimization.

Faithful implementation of Hu et al. 2026 (Applied Energy 417:128017),
Section 2.2.2, Eq. (24)-(31):

  * Group-enhanced value estimation: N_g parallel environments share the
    SAME exogenous trajectory (load/PV month) and initial state; trajectory
    differences arise only from stochastic action sampling (Sec 2.2.2.2).
  * Monte-Carlo discounted returns G_t (25), per-trajectory z-normalised
    G_hat (26); critic fits G_hat by MSE (27).
  * Hybrid baseline B = (1-beta)*V + beta*mu_tilde where mu_tilde is the
    normalised group-mean return sequence (28)-(29).
  * Advantage A_hat = G_hat - detach(B)  (30)  no GAE.
  * PPO clipped surrogate (24) + c * value loss, joint Adam update over
    K epochs (31). Gaussian policy with FIXED std lambda (Sec 2.2.2.4),
    actor/critic MLPs 256-128 with Tanh activations, actor mean squashed
    by Tanh into [-1, 1].
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(2)


def _mlp(inp, out, h1=256, h2=128):
    return nn.Sequential(
        nn.Linear(inp, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
        nn.Linear(h2, out),
    )


class GREPOAgent:
    def __init__(self, obs_dim: int, n_group: int = 6, gamma: float = 0.995,
                 std: float = 0.30, beta: float = 0.5, clip: float = 0.2,
                 lr: float = 3e-4, epochs: int = 8, minibatch: int = 512,
                 vf_coef: float = 0.5, seed: int = 0):
        torch.manual_seed(seed)
        self.actor = _mlp(obs_dim, 1)
        self.critic = _mlp(obs_dim, 1)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)
        self.n_group = n_group
        self.gamma, self.std, self.beta = gamma, std, beta
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.vf_coef = vf_coef
        self.obs_dim = obs_dim
        self.meta = {}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Same call signature as PPOAgent so baselines.run_drl_policy and
        the latency tests work unchanged. Returns (action, logp, value)."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.actor(o))
        if deterministic:
            a = mean
        else:
            a = mean + self.std * torch.randn_like(mean)
        logp = self._logp(a, mean).item()
        v = self.critic(o).squeeze(-1).item()
        return float(np.clip(a.item(), -1.0, 1.0)), float(logp), float(v)

    def _logp(self, a, mean):
        d = torch.distributions.Normal(mean, self.std)
        return d.log_prob(a).sum(-1)

    # ------------------------------------------------------------------
    def collect_group(self, make_env, month, soc_init=None,
                      d_run_init=None):
        """Rollout N_g episodes on the SAME episode data (identical exogenous
        trajectory + initial state; Sec 2.2.2.2). Per the paper, an episode
        is ONE DAY (their T=1440 at 1-min; ours T=96 at 15-min)  pass a
        1-day MonthData. Returns stacked arrays [N_g, T, ...]."""
        obs_g, act_g, logp_g, rew_g = [], [], [], []
        for i in range(self.n_group):
            env = make_env()
            if d_run_init is not None:
                env.d_run_init = float(d_run_init)
            obs = env.reset(month, soc_init=soc_init)
            done = False
            O, A, L, R = [], [], [], []
            while not done:
                o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    mean = torch.tanh(self.actor(o))
                    a = mean + self.std * torch.randn_like(mean)
                    lp = self._logp(a, mean).item()
                O.append(obs)
                A.append(float(a.item()))
                L.append(lp)
                obs, r, done, _ = env.step(float(np.clip(a.item(), -1, 1)))
                R.append(r)
            obs_g.append(np.array(O, np.float32))
            act_g.append(np.array(A, np.float32))
            logp_g.append(np.array(L, np.float32))
            rew_g.append(np.array(R, np.float32))
        return (np.stack(obs_g), np.stack(act_g),
                np.stack(logp_g), np.stack(rew_g))

    # ------------------------------------------------------------------
    def update(self, obs_g, act_g, logp_g, rew_g):
        """One GREPO training phase on a collected group batch."""
        ng, T = rew_g.shape
        eps = 1e-8
        # Eq (25): discounted MC returns per env
        G = np.zeros_like(rew_g)
        for i in range(ng):
            acc = 0.0
            for t in reversed(range(T)):
                acc = rew_g[i, t] + self.gamma * acc
                G[i, t] = acc
        # Eq (26): per-trajectory normalisation
        mu_i = G.mean(axis=1, keepdims=True)
        sd_i = G.std(axis=1, keepdims=True)
        G_hat = (G - mu_i) / (sd_i + eps)
        # Eq (29): normalised group-mean return sequence
        mu_t = G_hat.mean(axis=0)                       # [T]
        mu_tilde = (mu_t - mu_t.mean()) / (mu_t.std() + eps)
        mu_tilde_g = np.broadcast_to(mu_tilde, (ng, T))

        obs = torch.as_tensor(obs_g.reshape(ng * T, -1))
        act = torch.as_tensor(act_g.reshape(ng * T, 1))
        logp_old = torch.as_tensor(logp_g.reshape(ng * T))
        g_hat = torch.as_tensor(G_hat.reshape(ng * T).astype(np.float32))
        mu_tl = torch.as_tensor(
            np.ascontiguousarray(mu_tilde_g).reshape(ng * T).astype(np.float32))

        n = ng * T
        idx = np.arange(n)
        pi_l = vf_l = 0.0
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = idx[s:s + self.minibatch]
                mean = torch.tanh(self.actor(obs[mb]))
                logp = self._logp(act[mb], mean)
                v = self.critic(obs[mb]).squeeze(-1)
                # Eq (28)+(30): hybrid baseline, gradient-free
                B = (1 - self.beta) * v.detach() + self.beta * mu_tl[mb]
                adv = g_hat[mb] - B
                ratio = torch.exp(logp - logp_old[mb])
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                l_clip = -torch.min(surr1, surr2).mean()      # Eq (24)
                l_vf = ((v - g_hat[mb]) ** 2).mean()          # Eq (27)
                loss = l_clip + self.vf_coef * l_vf           # Eq (31)
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters())
                    + list(self.critic.parameters()), 0.5)
                self.opt.step()
                pi_l, vf_l = float(l_clip.item()), float(l_vf.item())
        return {"pi_loss": pi_l, "vf_loss": vf_l}

    # ------------------------------------------------------------------
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                    "obs_dim": self.obs_dim, "std": self.std,
                    "algo": "grepo", "meta": dict(self.meta)}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.meta = ck.get("meta", {}) or {}
        self.actor.eval()
        self.critic.eval()
