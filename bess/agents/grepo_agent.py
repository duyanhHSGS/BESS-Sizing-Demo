"""bess.agents.grepo_agent.py  Group Relative Enhancement Policy Optimization.

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

import time

import numpy as np
import torch
import torch.nn as nn

from bess.core.brain_runtime import observation_array, step_brain_control

torch.set_num_threads(2)

def _mlp(inp, out, h1=256, h2=128):
    return nn.Sequential(
        nn.Linear(inp, h1), nn.Tanh(),
        nn.Linear(h1, h2), nn.Tanh(),
        nn.Linear(h2, out),
    )


def resolve_grepo_device(device: str = "auto") -> str:
    """Resolve a safe learner device without moving rollout inference."""
    requested = str(device).lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("GREPO device must be 'auto', 'cpu', or 'cuda'")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GREPO device='cuda' requested, but PyTorch reports that "
                "CUDA is unavailable"
            )
        return "cuda"
    if requested == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


class GREPOAgent:
    def __init__(self, obs_dim: int, n_group: int = 6, gamma: float = 0.995,
                 std: float = 0.30, beta: float = 0.5, clip: float = 0.2,
                 lr: float = 3e-4, epochs: int = 8, minibatch: int = 512,
                 vf_coef: float = 0.5, seed: int = 0,
                 device: str = "auto"):
        torch.manual_seed(seed)
        self.device = torch.device(resolve_grepo_device(device))
        self.actor = _mlp(obs_dim, 1).to(self.device)
        self.critic = _mlp(obs_dim, 1).to(self.device)
        self._params = (
            list(self.actor.parameters()) + list(self.critic.parameters())
        )
        self.opt = torch.optim.Adam(self._params, lr=lr)
        # Building the CPU mirror must not consume samples from the policy's
        # seeded exploration RNG stream.
        with torch.random.fork_rng(devices=[]):
            self.collector_actor = _mlp(obs_dim, 1).cpu()
        self._sync_collector_actor()
        self._scalar_reference_envs = []
        self.n_group = n_group
        self.gamma, self.std, self.beta = gamma, std, beta
        self.clip, self.epochs, self.minibatch = clip, epochs, minibatch
        self.vf_coef = vf_coef
        self.obs_dim = obs_dim
        self.meta = {}
        self.last_collect_stats = {}
        self.last_update_stats = {}

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def _sync_collector_actor(self):
        state = {
            key: value.detach().cpu()
            for key, value in self.actor.state_dict().items()
        }
        self.collector_actor.load_state_dict(state)
        self.collector_actor.eval()

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def predict_action(self, obs: np.ndarray) -> float:
        """Deterministic actor-only inference for validation and Dispatch."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = torch.tanh(self.collector_actor(o))
        return float(np.clip(mean.item(), -1.0, 1.0))

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False):
        """Same call signature as PPOAgent so baselines.run_drl_policy and
        the latency tests work unchanged. Returns (action, logp, value)."""
        if deterministic:
            action = self.predict_action(obs)
            o = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            a = torch.tensor(
                [[action]], dtype=torch.float32, device=self.device
            )
            mean = torch.tanh(self.actor(o))
        else:
            o = torch.as_tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            mean = torch.tanh(self.actor(o))
            a = mean + self.std * torch.randn_like(mean)
            action = float(np.clip(a.item(), -1.0, 1.0))
        logp = self._logp(a, mean).item()
        v = self.critic(o).squeeze(-1).item()
        return action, float(logp), float(v)

    def _logp(self, a, mean):
        # Keep this implementation until an analytical replacement passes
        # numerical parity tests on every supported device.
        d = torch.distributions.Normal(mean, self.std)
        return d.log_prob(a).sum(-1)

    # ------------------------------------------------------------------
    @staticmethod
    def _decision_count(month, native_steps: int) -> int:
        native_rows = sum(len(day.load) for day in month.days)
        interval = int(native_steps)
        if interval <= 0:
            raise ValueError("native_steps must be greater than 0")
        if native_rows % interval:
            raise ValueError(
                "episode native rows must be divisible by the control interval"
            )
        return native_rows // interval

    def _prepare_noise(self, decisions: int, noise_g=None) -> torch.Tensor:
        if noise_g is None:
            # Group-major layout preserves the old group-then-time RNG order.
            return torch.randn(
                (self.n_group, decisions), dtype=torch.float32
            )
        noise = torch.as_tensor(noise_g, dtype=torch.float32, device="cpu")
        expected = (self.n_group, decisions)
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"noise_g must have shape {expected}, got {tuple(noise.shape)}"
            )
        return noise

    def collect_group(
        self,
        make_env,
        month,
        soc_init=None,
        noise_g=None,
        *,
        native_steps: int = 1,
        baseline_actions=None,
        baseline_costs=None,
        residual_limit: float = 1.0,
    ):
        """Roll out a lockstep group on canonical episode-owned ``BrainEnv`` values.

        ``make_env(month, soc_init)`` must return a fresh configured ``BrainEnv``.
        Optional baseline actions/costs keep GrePRO residual composition outside
        the environment instead of subclassing or extending the seven-eye contract.
        """
        started = time.perf_counter()
        decisions = self._decision_count(month, native_steps)
        envs = [make_env(month, soc_init) for _ in range(self.n_group)]
        observations = [observation_array(env.reset()) for env in envs]

        baseline_action_values = None
        if baseline_actions is not None:
            baseline_action_values = np.asarray(baseline_actions, dtype=np.float64)
            if baseline_action_values.shape != (decisions,):
                raise ValueError(
                    f"baseline_actions must have shape ({decisions},), "
                    f"got {baseline_action_values.shape}"
                )
        baseline_cost_values = None
        if baseline_costs is not None:
            baseline_cost_values = np.asarray(baseline_costs, dtype=np.float64)
            if baseline_cost_values.shape != (decisions,):
                raise ValueError(
                    f"baseline_costs must have shape ({decisions},), "
                    f"got {baseline_cost_values.shape}"
                )
        if baseline_cost_values is not None and baseline_action_values is None:
            raise ValueError("baseline_costs require baseline_actions")

        noise = self._prepare_noise(decisions, noise_g=noise_g)
        obs_g = np.empty(
            (self.n_group, decisions, self.obs_dim), dtype=np.float32
        )
        act_g = np.empty((self.n_group, decisions), dtype=np.float32)
        logp_g = np.empty((self.n_group, decisions), dtype=np.float32)
        rew_g = np.empty((self.n_group, decisions), dtype=np.float32)
        obs_batch = np.empty((self.n_group, self.obs_dim), dtype=np.float32)
        native_rows = 0

        for t in range(decisions):
            for group_index, observation in enumerate(observations):
                if observation is None:
                    raise RuntimeError("GREPO environment completed before its decision horizon")
                obs_batch[group_index] = observation
            obs_g[:, t, :] = obs_batch
            with torch.inference_mode():
                obs_tensor = torch.from_numpy(obs_batch)
                mean = torch.tanh(self.collector_actor(obs_tensor))
                raw = mean.squeeze(-1) + self.std * noise[:, t]
                logp = self._logp(raw.unsqueeze(-1), mean)
            raw_np = raw.numpy()
            act_g[:, t] = raw_np
            logp_g[:, t] = logp.numpy()

            completed = 0
            for group_index, env in enumerate(envs):
                policy_action = float(np.clip(raw_np[group_index], -1.0, 1.0))
                if baseline_action_values is None:
                    final_action = policy_action
                else:
                    final_action = float(np.clip(
                        baseline_action_values[t]
                        + float(residual_limit) * policy_action,
                        -1.0,
                        1.0,
                    ))

                transition = step_brain_control(
                    env,
                    final_action,
                    native_steps=native_steps,
                )
                native_rows += len(transition.native_results)
                if baseline_cost_values is None:
                    reward = transition.reward_million_vnd
                else:
                    hybrid_cost_vnd = sum(
                        result.bess.cost.operating_cost_vnd
                        for result in transition.native_results
                    )
                    reward = (baseline_cost_values[t] - hybrid_cost_vnd) / 1_000_000.0
                rew_g[group_index, t] = float(reward)
                observations[group_index] = (
                    None
                    if transition.done
                    else observation_array(transition.next_observation)
                )
                completed += int(transition.done)

            if completed and (completed != self.n_group or t != decisions - 1):
                raise RuntimeError(
                    "lockstep GREPO environments completed inconsistently"
                )

        if any(observation is not None for observation in observations):
            raise RuntimeError("GREPO episode did not complete at its horizon")

        elapsed = time.perf_counter() - started
        self.last_collect_stats = {
            "group_rollout_seconds": elapsed,
            "native_rows": native_rows,
            "decisions": decisions,
            "samples": self.n_group * decisions,
        }
        return obs_g, act_g, logp_g, rew_g

    def collect_group_scalar_reference(
        self,
        make_env,
        month,
        soc_init=None,
        noise_g=None,
        *,
        native_steps: int = 1,
        baseline_actions=None,
        baseline_costs=None,
        residual_limit: float = 1.0,
    ):
        """Unoptimized parity reference using the same canonical BrainEnv contract."""
        envs = [make_env(month, soc_init) for _ in range(self.n_group)]
        self._scalar_reference_envs = envs
        decisions = self._decision_count(month, native_steps)
        noise = self._prepare_noise(decisions, noise_g=noise_g)
        baseline_action_values = (
            None
            if baseline_actions is None
            else np.asarray(baseline_actions, dtype=np.float64)
        )
        baseline_cost_values = (
            None
            if baseline_costs is None
            else np.asarray(baseline_costs, dtype=np.float64)
        )
        if baseline_action_values is not None and baseline_action_values.shape != (decisions,):
            raise ValueError("baseline_actions length does not match decision horizon")
        if baseline_cost_values is not None and baseline_cost_values.shape != (decisions,):
            raise ValueError("baseline_costs length does not match decision horizon")
        if baseline_cost_values is not None and baseline_action_values is None:
            raise ValueError("baseline_costs require baseline_actions")

        obs_g = np.empty(
            (self.n_group, decisions, self.obs_dim), dtype=np.float32
        )
        act_g = np.empty((self.n_group, decisions), dtype=np.float32)
        logp_g = np.empty((self.n_group, decisions), dtype=np.float32)
        rew_g = np.empty((self.n_group, decisions), dtype=np.float32)
        for group_index, env in enumerate(envs):
            obs = observation_array(env.reset())
            for t in range(decisions):
                obs_g[group_index, t] = obs
                with torch.inference_mode():
                    o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    mean = torch.tanh(self.collector_actor(o))
                    raw = mean.squeeze() + self.std * noise[group_index, t]
                    logp = self._logp(raw.reshape(1, 1), mean)
                raw_value = float(raw.item())
                act_g[group_index, t] = raw_value
                logp_g[group_index, t] = float(logp.item())
                policy_action = float(np.clip(raw_value, -1.0, 1.0))
                if baseline_action_values is None:
                    final_action = policy_action
                else:
                    final_action = float(np.clip(
                        baseline_action_values[t]
                        + float(residual_limit) * policy_action,
                        -1.0,
                        1.0,
                    ))
                transition = step_brain_control(
                    env,
                    final_action,
                    native_steps=native_steps,
                )
                if baseline_cost_values is None:
                    reward = transition.reward_million_vnd
                else:
                    hybrid_cost_vnd = sum(
                        result.bess.cost.operating_cost_vnd
                        for result in transition.native_results
                    )
                    reward = (baseline_cost_values[t] - hybrid_cost_vnd) / 1_000_000.0
                rew_g[group_index, t] = float(reward)
                done = transition.done
                if done != (t == decisions - 1):
                    raise RuntimeError(
                        "scalar GREPO reference completed unexpectedly"
                    )
                if not done:
                    obs = observation_array(transition.next_observation)
        return obs_g, act_g, logp_g, rew_g

    # ------------------------------------------------------------------
    def _discounted_returns(self, rew_g):
        """Discount over time while vectorizing across group members."""
        returns = np.empty_like(rew_g)
        # The scalar implementation accumulated in Python/NumPy float64 and
        # stored into the float32 return buffer after each step.
        accumulator = np.zeros(rew_g.shape[0], dtype=np.float64)
        for t in range(rew_g.shape[1] - 1, -1, -1):
            accumulator = rew_g[:, t] + self.gamma * accumulator
            returns[:, t] = accumulator
        return returns

    def update(self, obs_g, act_g, logp_g, rew_g):
        """One GREPO training phase on a collected group batch."""
        return_started = time.perf_counter()
        ng, T = rew_g.shape
        eps = 1e-8
        # Eq (25): discounted MC returns per env
        G = self._discounted_returns(rew_g)
        # Eq (26): per-trajectory normalisation
        mu_i = G.mean(axis=1, keepdims=True)
        sd_i = G.std(axis=1, keepdims=True)
        G_hat = (G - mu_i) / (sd_i + eps)
        # Eq (29): normalised group-mean return sequence
        mu_t = G_hat.mean(axis=0)                       # [T]
        mu_tilde = (mu_t - mu_t.mean()) / (mu_t.std() + eps)
        mu_tilde_g = np.broadcast_to(mu_tilde, (ng, T))
        return_seconds = time.perf_counter() - return_started

        transfer_started = time.perf_counter()
        obs = torch.as_tensor(
            obs_g.reshape(ng * T, -1), dtype=torch.float32,
            device=self.device
        )
        act = torch.as_tensor(
            act_g.reshape(ng * T, 1), dtype=torch.float32,
            device=self.device
        )
        logp_old = torch.as_tensor(
            logp_g.reshape(ng * T), dtype=torch.float32,
            device=self.device
        )
        g_hat = torch.as_tensor(
            G_hat.reshape(ng * T), dtype=torch.float32,
            device=self.device
        )
        mu_tl = torch.as_tensor(
            np.ascontiguousarray(mu_tilde_g).reshape(ng * T),
            dtype=torch.float32, device=self.device
        )
        self._sync_device()
        transfer_seconds = time.perf_counter() - transfer_started

        update_started = time.perf_counter()
        n = ng * T
        idx = np.arange(n)
        pi_l = vf_l = None
        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for s in range(0, n, self.minibatch):
                mb = torch.as_tensor(
                    idx[s:s + self.minibatch], dtype=torch.long,
                    device=self.device
                )
                obs_mb = obs[mb]
                act_mb = act[mb]
                logp_old_mb = logp_old[mb]
                g_hat_mb = g_hat[mb]
                mu_tl_mb = mu_tl[mb]
                mean = torch.tanh(self.actor(obs_mb))
                logp = self._logp(act_mb, mean)
                v = self.critic(obs_mb).squeeze(-1)
                # Eq (28)+(30): hybrid baseline, gradient-free
                B = (1 - self.beta) * v.detach() + self.beta * mu_tl_mb
                adv = g_hat_mb - B
                ratio = torch.exp(logp - logp_old_mb)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv
                l_clip = -torch.min(surr1, surr2).mean()      # Eq (24)
                l_vf = ((v - g_hat_mb) ** 2).mean()           # Eq (27)
                loss = l_clip + self.vf_coef * l_vf           # Eq (31)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self._params, 0.5)
                self.opt.step()
                pi_l, vf_l = l_clip.detach(), l_vf.detach()
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
            "pi_loss": float(pi_l.cpu().item()),
            "vf_loss": float(vf_l.cpu().item()),
        }

    # ------------------------------------------------------------------
    def save(self, path):
        actor_state = {
            key: value.detach().cpu()
            for key, value in self.actor.state_dict().items()
        }
        critic_state = {
            key: value.detach().cpu()
            for key, value in self.critic.state_dict().items()
        }
        payload = {"actor": actor_state,
                   "critic": critic_state,
                   "obs_dim": self.obs_dim, "std": self.std,
                   "algo": "grepo", "meta": dict(self.meta)}
        torch.save(payload, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.meta = ck.get("meta", {}) or {}
        self.actor.eval()
        self.critic.eval()
        self._sync_collector_actor()
