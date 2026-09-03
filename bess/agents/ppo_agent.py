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

from bess.core.settings import (
    PPO_ACTOR_GRAD_CLIP,
    PPO_CRITIC_GRAD_CLIP,
    PPO_EXPLORATION_LR_MULTIPLIER,
    PPO_GAMMA,
    PPO_HIDDEN_SIZE,
    PPO_INITIAL_LOG_STD,
    PPO_LAMBDA,
    PPO_RECURRENT_SEQUENCE_LENGTH,
    PPO_SOC_EDGE_LOG_STD_PENALTY,
    PPO_TORCH_THREADS,
)

torch.set_num_threads(PPO_TORCH_THREADS)


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


def _mlp(inp, out, hidden=PPO_HIDDEN_SIZE):
    return nn.Sequential(
        nn.Linear(inp, hidden), nn.Tanh(),
        nn.Linear(hidden, hidden), nn.Tanh(),
        nn.Linear(hidden, out),
    )


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        *,
        hidden_size: int = PPO_HIDDEN_SIZE,
        initial_log_std: float = PPO_INITIAL_LOG_STD,
        soc_edge_log_std_penalty: float = 0.0,
        decomposed_critic: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.decomposed_critic = bool(decomposed_critic)
        self.soc_edge_log_std_penalty = float(soc_edge_log_std_penalty)
        self.actor = _mlp(obs_dim, 1, hidden=hidden_size)
        # Keep the historical `critic` key as the energy/scalar head so old
        # checkpoints stay loadable. Demand/wear heads exist only for the new
        # generic-PPO decomposed-critic experiment.
        self.critic = _mlp(obs_dim, 1, hidden=hidden_size)
        if self.decomposed_critic:
            self.critic_demand = copy.deepcopy(self.critic)
            self.critic_wear = copy.deepcopy(self.critic)
            for head in (self.critic_demand, self.critic_wear):
                nn.init.zeros_(head[-1].weight)
                nn.init.zeros_(head[-1].bias)
        self.log_std = nn.Parameter(torch.full((1,), float(initial_log_std)))

        # IQ-29 adaptive exploration: keep the proven scalar log_std as the
        # baseline and learn only a small observation-conditioned delta.  The
        # final layer starts at exact zero, so the initial policy distribution is
        # bit-for-bit the old scalar-std policy.  Fork RNG so constructing this
        # helper cannot perturb the deterministic PPO sampling stream.
        self.exploration_hidden_size = max(8, int(hidden_size) // 4)
        with torch.random.fork_rng(devices=[]):
            self.log_std_delta = nn.Sequential(
                nn.Linear(obs_dim, self.exploration_hidden_size),
                nn.Tanh(),
                nn.Linear(self.exploration_hidden_size, 1, bias=False),
            )
            nn.init.zeros_(self.log_std_delta[0].bias)
            nn.init.zeros_(self.log_std_delta[-1].weight)

    def effective_log_std(self, obs):
        delta = self.log_std_delta(obs)
        if self.obs_dim > 3 and self.soc_edge_log_std_penalty > 0.0:
            soc = obs[..., 3:4]
            edge_strength = (2.0 * soc - 1.0).square()
            delta = delta - self.soc_edge_log_std_penalty * edge_strength
        return self.log_std + delta

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.effective_log_std(obs).exp())

    def value_components(self, obs):
        energy = self.critic(obs).squeeze(-1)
        if not self.decomposed_critic:
            zero = torch.zeros_like(energy)
            return energy, zero, zero
        demand = self.critic_demand(obs).squeeze(-1)
        wear = self.critic_wear(obs).squeeze(-1)
        return energy, demand, wear

    def value(self, obs):
        energy, demand, wear = self.value_components(obs)
        return energy + demand - wear


class RecurrentActorCritic(nn.Module):
    """Brain7 actor/critic with separate GRU memories for policy and value."""

    def __init__(
        self,
        obs_dim: int,
        *,
        hidden_size: int = PPO_HIDDEN_SIZE,
        initial_log_std: float = PPO_INITIAL_LOG_STD,
        soc_edge_log_std_penalty: float = 0.0,
        decomposed_critic: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.decomposed_critic = bool(decomposed_critic)
        self.soc_edge_log_std_penalty = float(soc_edge_log_std_penalty)
        self.actor_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.actor_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.actor = nn.Linear(hidden_size, 1)
        # One recurrent critic memory, three economically distinct output heads.
        # The shared GRU keeps rollout runtime tiny while preventing one scalar
        # value target from mixing energy, monthly demand, and wear semantics.
        self.critic_encoder = nn.Sequential(nn.Linear(obs_dim, hidden_size), nn.Tanh())
        self.critic_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.critic = nn.Linear(hidden_size, 1)
        if self.decomposed_critic:
            self.critic_demand = copy.deepcopy(self.critic)
            self.critic_wear = copy.deepcopy(self.critic)
            nn.init.zeros_(self.critic_demand.weight)
            nn.init.zeros_(self.critic_demand.bias)
            nn.init.zeros_(self.critic_wear.weight)
            nn.init.zeros_(self.critic_wear.bias)
        self.log_std = nn.Parameter(torch.full((1,), float(initial_log_std)))

        self.exploration_hidden_size = max(8, int(hidden_size) // 4)
        with torch.random.fork_rng(devices=[]):
            self.log_std_delta = nn.Sequential(
                nn.Linear(hidden_size, self.exploration_hidden_size),
                nn.Tanh(),
                nn.Linear(self.exploration_hidden_size, 1, bias=False),
            )
            nn.init.zeros_(self.log_std_delta[0].bias)
            nn.init.zeros_(self.log_std_delta[-1].weight)

    def zero_hidden(self, batch_size: int, *, device=None):
        target = device if device is not None else next(self.parameters()).device
        return torch.zeros(1, int(batch_size), self.hidden_size, device=target)

    def actor_sequence(self, obs, hidden=None):
        if obs.ndim != 3:
            raise ValueError("recurrent actor expects observations shaped [batch, time, obs]")
        if hidden is None:
            hidden = self.zero_hidden(obs.shape[0], device=obs.device)
        encoded = self.actor_encoder(obs)
        features, next_hidden = self.actor_gru(encoded, hidden)
        return self.actor(features), features, next_hidden

    def critic_components_sequence(self, obs, hidden=None):
        if obs.ndim != 3:
            raise ValueError("recurrent critic expects observations shaped [batch, time, obs]")
        if hidden is None:
            hidden = self.zero_hidden(obs.shape[0], device=obs.device)
        encoded = self.critic_encoder(obs)
        features, next_hidden = self.critic_gru(encoded, hidden)
        energy = self.critic(features).squeeze(-1)
        if not self.decomposed_critic:
            zero = torch.zeros_like(energy)
            return (energy, zero, zero), next_hidden
        demand = self.critic_demand(features).squeeze(-1)
        wear = self.critic_wear(features).squeeze(-1)
        return (energy, demand, wear), next_hidden

    def critic_sequence(self, obs, hidden=None):
        (energy, demand, wear), next_hidden = self.critic_components_sequence(obs, hidden)
        return energy + demand - wear, next_hidden

    def effective_log_std(self, actor_features, obs):
        delta = self.log_std_delta(actor_features)
        if self.obs_dim > 3 and self.soc_edge_log_std_penalty > 0.0:
            soc = obs[..., 3:4]
            edge_strength = (2.0 * soc - 1.0).square()
            delta = delta - self.soc_edge_log_std_penalty * edge_strength
        return self.log_std + delta

    def dist_sequence(self, obs, hidden=None):
        mean, features, next_hidden = self.actor_sequence(obs, hidden)
        std = self.effective_log_std(features, obs).exp()
        return torch.distributions.Normal(mean, std), next_hidden

    def value_components_sequence(self, obs, hidden=None):
        return self.critic_components_sequence(obs, hidden)

    def value_sequence(self, obs, hidden=None):
        return self.critic_sequence(obs, hidden)

    def dist_step(self, obs, hidden=None):
        dist, next_hidden = self.dist_sequence(obs.unsqueeze(1), hidden)
        return torch.distributions.Normal(dist.loc[:, 0], dist.scale[:, 0]), next_hidden

    def value_components_step(self, obs, hidden=None):
        components, next_hidden = self.value_components_sequence(obs.unsqueeze(1), hidden)
        return tuple(component[:, 0] for component in components), next_hidden

    def value_step(self, obs, hidden=None):
        value, next_hidden = self.value_sequence(obs.unsqueeze(1), hidden)
        return value[:, 0], next_hidden


class RolloutBuffer:
    def __init__(
        self,
        size: int,
        obs_dim: int,
        recurrent_hidden_size: int = 0,
        *,
        decomposed_rewards: bool = False,
    ):
        self.obs = np.zeros((size, obs_dim), np.float32)
        self.act = np.zeros((size, 1), np.float32)
        self.latent = np.zeros((size, 1), np.float32)
        self.logp = np.zeros(size, np.float32)
        self.rew = np.zeros(size, np.float32)
        self.val = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.decomposed_rewards = bool(decomposed_rewards)
        if self.decomposed_rewards:
            self.rew_energy = np.zeros(size, np.float32)
            self.rew_demand = np.zeros(size, np.float32)
            self.rew_wear = np.zeros(size, np.float32)
            self.val_energy = np.zeros(size, np.float32)
            self.val_demand = np.zeros(size, np.float32)
            self.val_wear = np.zeros(size, np.float32)
        else:
            self.rew_energy = self.rew_demand = self.rew_wear = None
            self.val_energy = self.val_demand = self.val_wear = None
        self.recurrent_hidden_size = int(recurrent_hidden_size)
        self.actor_hidden = (
            np.zeros((size, self.recurrent_hidden_size), np.float32)
            if self.recurrent_hidden_size > 0 else None
        )
        self.critic_hidden = (
            np.zeros((size, self.recurrent_hidden_size), np.float32)
            if self.recurrent_hidden_size > 0 else None
        )
        self.ptr = 0
        self.size = size

    def add(
        self,
        o,
        a,
        lp,
        r,
        v,
        d,
        latent,
        actor_hidden=None,
        critic_hidden=None,
        *,
        reward_components=None,
        value_components=None,
    ):
        i = self.ptr
        self.obs[i], self.act[i], self.latent[i] = o, a, latent
        if self.decomposed_rewards:
            if reward_components is None or value_components is None:
                raise ValueError(
                    "decomposed rollout transition requires energy/demand/wear reward and value components"
                )
            reward_energy, reward_demand, reward_wear = reward_components
            value_energy, value_demand, value_wear = value_components
            self.rew_energy[i] = reward_energy
            self.rew_demand[i] = reward_demand
            self.rew_wear[i] = reward_wear
            self.val_energy[i] = value_energy
            self.val_demand[i] = value_demand
            self.val_wear[i] = value_wear
            # Derive the scalar compatibility views from components. This keeps
            # the actor objective exactly E + D - W without a second arithmetic path.
            r = reward_energy + reward_demand - reward_wear
            v = value_energy + value_demand - value_wear
        self.logp[i], self.rew[i], self.val[i], self.done[i] = lp, r, v, d
        if self.recurrent_hidden_size > 0:
            if actor_hidden is None or critic_hidden is None:
                raise ValueError("recurrent rollout transition requires actor and critic hidden state")
            self.actor_hidden[i] = actor_hidden
            self.critic_hidden[i] = critic_hidden
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


def _critic_value_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Scale-robust critic loss using the rollout return spread as Huber's transition."""
    with torch.no_grad():
        beta = max(float(target.std(unbiased=False).item()), 1.0)
    return nn.functional.smooth_l1_loss(prediction, target, beta=beta)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        lr=1e-4,
        gamma=PPO_GAMMA,
        lam=PPO_LAMBDA,
        clip=0.2,
        epochs=4,
        minibatch=256,
        ent_coef=0.0,
        vf_coef=0.5,
        target_kl=0.02,
        seed=0,
        device="auto",
        hidden_size=PPO_HIDDEN_SIZE,
        initial_log_std=PPO_INITIAL_LOG_STD,
        exploration_lr_multiplier=PPO_EXPLORATION_LR_MULTIPLIER,
        soc_edge_log_std_penalty=PPO_SOC_EDGE_LOG_STD_PENALTY,
        recurrent_enabled=False,
        recurrent_sequence_length=PPO_RECURRENT_SEQUENCE_LENGTH,
        decomposed_critic=False,
        actor_grad_clip=PPO_ACTOR_GRAD_CLIP,
        critic_grad_clip=PPO_CRITIC_GRAD_CLIP,
    ):
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.device = torch.device(resolve_ppo_device(device))
        self.obs_dim = int(obs_dim)
        self.hidden_size = int(hidden_size)
        self.initial_log_std = float(initial_log_std)
        self.learning_rate = float(lr)
        self.exploration_lr_multiplier = float(exploration_lr_multiplier)
        self.exploration_learning_rate = (
            self.learning_rate * self.exploration_lr_multiplier
        )
        self.soc_edge_log_std_penalty = float(soc_edge_log_std_penalty)
        self.recurrent_enabled = bool(recurrent_enabled)
        self.recurrent_sequence_length = int(recurrent_sequence_length)
        self.decomposed_critic = bool(decomposed_critic)
        if self.recurrent_sequence_length < 1:
            raise ValueError("recurrent_sequence_length must be >= 1")
        self.actor_grad_clip = float(actor_grad_clip)
        self.critic_grad_clip = float(critic_grad_clip)
        self.net = self._make_network().to(self.device)
        self.opt = self._build_optimizer()
        # Keep tiny step-by-step environment inference on CPU. Only large PPO
        # sequence minibatches cross to CUDA, avoiding thousands of tiny PCIe transfers.
        with torch.random.fork_rng(devices=[]):
            self.collector_net = self._make_network().cpu()
        self._sync_collector()
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatch = epochs, minibatch
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.target_kl = float(target_kl)
        self.last_update_stats = {}
        self.meta = {}          # deployment context (p_ref_kw, obs_variant)

    def _make_network(self):
        network_cls = RecurrentActorCritic if self.recurrent_enabled else ActorCritic
        return network_cls(
            self.obs_dim,
            hidden_size=self.hidden_size,
            initial_log_std=self.initial_log_std,
            soc_edge_log_std_penalty=self.soc_edge_log_std_penalty,
            decomposed_critic=self.decomposed_critic,
        )

    def _actor_parameters(self):
        if not self.recurrent_enabled:
            return [
                *self.net.actor.parameters(),
                self.net.log_std,
                *self.net.log_std_delta.parameters(),
            ]
        return [
            *self.net.actor_encoder.parameters(),
            *self.net.actor_gru.parameters(),
            *self.net.actor.parameters(),
            self.net.log_std,
            *self.net.log_std_delta.parameters(),
        ]

    def _critic_parameters(self):
        parameters = []
        if self.recurrent_enabled:
            parameters.extend(self.net.critic_encoder.parameters())
            parameters.extend(self.net.critic_gru.parameters())
        parameters.extend(self.net.critic.parameters())
        if self.decomposed_critic:
            parameters.extend(self.net.critic_demand.parameters())
            parameters.extend(self.net.critic_wear.parameters())
        return list(parameters)

    def critic_state_prefixes(self) -> tuple[str, ...]:
        """State-dict prefixes owned by the critic, including all economic heads."""
        prefixes = ["critic."]
        if self.decomposed_critic:
            prefixes.extend(("critic_demand.", "critic_wear."))
        if self.recurrent_enabled:
            prefixes.extend(("critic_encoder.", "critic_gru."))
        return tuple(prefixes)

    def policy_architecture_name(self) -> str:
        if self.recurrent_enabled:
            return (
                "brain7_actor_gru_shared_critic_gru_3head_v2"
                if self.decomposed_critic
                else "brain7_separate_actor_critic_gru_v1"
            )
        return (
            "brain7_feedforward_3head_critic_v2"
            if self.decomposed_critic
            else "brain7_feedforward_mlp_v1"
        )

    def _build_optimizer(self):
        exploration_params = list(self.net.log_std_delta.parameters())
        exploration_ids = {id(parameter) for parameter in exploration_params}
        base_params = [
            parameter
            for parameter in self.net.parameters()
            if id(parameter) not in exploration_ids
        ]
        return torch.optim.Adam(
            [
                {"params": base_params, "lr": self.learning_rate},
                {"params": exploration_params, "lr": self.exploration_learning_rate},
            ]
        )

    @torch.inference_mode()
    def _sync_collector(self):
        state = {
            key: value.detach().cpu()
            for key, value in self.net.state_dict().items()
        }
        self.collector_net.load_state_dict(state)
        self.collector_net.eval()
        self.reset_recurrent_state()

    def reset_recurrent_state(self) -> None:
        """Forget episode history without touching learned weights."""
        self._actor_hidden = None
        self._critic_hidden = None
        self.last_actor_hidden_input = None
        self.last_critic_hidden_input = None
        self.last_value_components = None

    def _hidden_numpy(self, hidden):
        if hidden is None:
            return np.zeros(self.hidden_size, dtype=np.float32)
        return hidden.detach().cpu().numpy().reshape(-1).astype(np.float32, copy=True)

    def recurrent_rollout_inputs(self):
        """Hidden states consumed by the most recent recurrent action/value step."""
        if not self.recurrent_enabled:
            return None, None
        return self.last_actor_hidden_input, self.last_critic_hidden_input

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
        if self.recurrent_enabled:
            self.last_actor_hidden_input = self._hidden_numpy(self._actor_hidden)
            self.last_critic_hidden_input = self._hidden_numpy(self._critic_hidden)
            distribution, self._actor_hidden = self.collector_net.dist_step(
                o,
                self._actor_hidden,
            )
            value_components, self._critic_hidden = self.collector_net.value_components_step(
                o,
                self._critic_hidden,
            )
            self._actor_hidden = self._actor_hidden.detach()
            self._critic_hidden = self._critic_hidden.detach()
        else:
            distribution = self.collector_net.dist(o)
            value_components = self.collector_net.value_components(o)
        energy_value, demand_value, wear_value = value_components
        value = energy_value + demand_value - wear_value
        self.last_value_components = tuple(
            float(component.item()) for component in value_components
        )
        action, log_probability, latent = _sample_squashed(
            distribution,
            deterministic=deterministic,
        )
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
        if self.recurrent_enabled:
            distribution, self._actor_hidden = self.collector_net.dist_step(
                o,
                self._actor_hidden,
            )
            self._actor_hidden = self._actor_hidden.detach()
            action, _, _ = _sample_squashed(distribution, deterministic=True)
        else:
            action = torch.tanh(self.collector_net.actor(o))
        return float(action.item())

    @torch.inference_mode()
    def estimate_value_components(self, obs: np.ndarray) -> tuple[float, float, float]:
        """Bootstrap economic values without advancing recurrent memory."""
        o = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if self.recurrent_enabled:
            components, _ = self.collector_net.value_components_step(o, self._critic_hidden)
        else:
            components = self.collector_net.value_components(o)
        return tuple(float(component.item()) for component in components)

    @torch.inference_mode()
    def estimate_value(self, obs: np.ndarray) -> float:
        """Bootstrap aggregate value without advancing recurrent memory."""
        energy, demand, wear = self.estimate_value_components(obs)
        return energy + demand - wear

    def _prepare_rollout_targets(
        self,
        buf: RolloutBuffer,
        last_val: float,
        last_val_components: tuple[float, float, float] | None = None,
    ):
        """Build PPO advantages, keeping economic components separate until the actor sum."""
        n = buf.ptr

        def explained(target, prediction):
            variance = float(np.var(target))
            if variance <= 1e-12:
                return 0.0
            return 1.0 - float(np.var(target - prediction)) / variance

        if not self.decomposed_critic:
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
            stats = {
                "advantage_mean_raw": raw_adv_mean,
                "advantage_std_raw": raw_adv_std,
                "return_mean": float(ret.mean()),
                "return_std": float(ret.std()),
                "reward_mean": float(buf.rew[:n].mean()),
                "reward_std": float(buf.rew[:n].std()),
                "explained_variance": explained(ret, buf.val[:n]),
            }
            return (
                (adv - raw_adv_mean) / (raw_adv_std + 1e-8),
                ret,
                None,
                stats,
            )

        if not buf.decomposed_rewards:
            raise ValueError("decomposed critic requires a decomposed RolloutBuffer")
        if last_val_components is None:
            if abs(float(last_val)) > 1e-12:
                raise ValueError(
                    "non-terminal decomposed PPO update requires component bootstrap values"
                )
            last_val_components = (0.0, 0.0, 0.0)
        last_energy, last_demand, last_wear = last_val_components
        adv_energy = _gae_advantages(
            buf.rew_energy[:n],
            buf.val_energy[:n],
            buf.done[:n],
            last_val=last_energy,
            gamma=self.gamma,
            lam=self.lam,
        )
        adv_demand = _gae_advantages(
            buf.rew_demand[:n],
            buf.val_demand[:n],
            buf.done[:n],
            last_val=last_demand,
            gamma=self.gamma,
            lam=self.lam,
        )
        adv_wear = _gae_advantages(
            buf.rew_wear[:n],
            buf.val_wear[:n],
            buf.done[:n],
            last_val=last_wear,
            gamma=self.gamma,
            lam=self.lam,
        )
        ret_energy = adv_energy + buf.val_energy[:n]
        ret_demand = adv_demand + buf.val_demand[:n]
        ret_wear = adv_wear + buf.val_wear[:n]

        # IQ-61: combine in raw million-VND units, normalize exactly once after
        # the economic signs are applied. Never normalize heads independently.
        adv = adv_energy + adv_demand - adv_wear
        ret = ret_energy + ret_demand - ret_wear
        value = buf.val_energy[:n] + buf.val_demand[:n] - buf.val_wear[:n]
        raw_adv_mean = float(adv.mean())
        raw_adv_std = float(adv.std())
        stats = {
            "advantage_mean_raw": raw_adv_mean,
            "advantage_std_raw": raw_adv_std,
            "return_mean": float(ret.mean()),
            "return_std": float(ret.std()),
            "reward_mean": float(buf.rew[:n].mean()),
            "reward_std": float(buf.rew[:n].std()),
            "explained_variance": explained(ret, value),
            "energy_advantage_mean_raw": float(adv_energy.mean()),
            "energy_advantage_std_raw": float(adv_energy.std()),
            "demand_advantage_mean_raw": float(adv_demand.mean()),
            "demand_advantage_std_raw": float(adv_demand.std()),
            "wear_advantage_mean_raw": float(adv_wear.mean()),
            "wear_advantage_std_raw": float(adv_wear.std()),
            "energy_explained_variance": explained(ret_energy, buf.val_energy[:n]),
            "demand_explained_variance": explained(ret_demand, buf.val_demand[:n]),
            "wear_explained_variance": explained(ret_wear, buf.val_wear[:n]),
        }
        # TODO(IQ-61): keep these heads only if unseen-bucket economics improve;
        # per-head EV is diagnostic, not a Champion-selection objective.
        return (
            (adv - raw_adv_mean) / (raw_adv_std + 1e-8),
            ret,
            (ret_energy, ret_demand, ret_wear),
            stats,
        )

    def _update_recurrent(
        self,
        buf: RolloutBuffer,
        last_val: float,
        last_val_components: tuple[float, float, float] | None = None,
    ):
        n = buf.ptr
        if n <= 0:
            raise ValueError("PPO update requires at least one rollout transition")
        if buf.actor_hidden is None or buf.critic_hidden is None:
            raise ValueError("recurrent PPO requires hidden states in the rollout buffer")

        adv, ret, component_returns, rollout_stats = self._prepare_rollout_targets(
            buf,
            last_val,
            last_val_components,
        )

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        component_ret_t = (
            tuple(torch.as_tensor(values, device=self.device) for values in component_returns)
            if component_returns is not None
            else None
        )

        pi_losses = []
        value_losses = []
        energy_value_losses = []
        demand_value_losses = []
        wear_value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []
        actor_grad_norms = []
        critic_grad_norms = []
        actor_params = self._actor_parameters()
        critic_params = self._critic_parameters()
        epochs_completed = 0
        early_stopped = False

        sequence_length = min(self.recurrent_sequence_length, n)
        chunk_starts = np.arange(0, n, sequence_length, dtype=np.int64)
        chunks_per_minibatch = max(1, self.minibatch // sequence_length)

        for epoch in range(self.epochs):
            shuffled_starts = self.rng.permutation(chunk_starts)
            epoch_kls = []
            for batch_start in range(0, len(shuffled_starts), chunks_per_minibatch):
                selected_starts = shuffled_starts[
                    batch_start:batch_start + chunks_per_minibatch
                ]
                chunk_ranges = [
                    (int(raw_start), min(int(raw_start) + sequence_length, n))
                    for raw_start in selected_starts
                ]
                chunk_lengths = [stop - start for start, stop in chunk_ranges]
                index_parts = [
                    torch.arange(start, stop, dtype=torch.long, device=self.device)
                    for start, stop in chunk_ranges
                ]

                if len(set(chunk_lengths)) == 1:
                    # Fast path used by the 1440 = 30 x 48 IQ-34 fit rollout:
                    # batch independent TBPTT chunks into one GRU launch.
                    obs_batch = torch.stack(
                        [obs[start:stop] for start, stop in chunk_ranges],
                        dim=0,
                    )
                    actor_hidden = torch.as_tensor(
                        np.stack([buf.actor_hidden[start] for start, _ in chunk_ranges]),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    critic_hidden = torch.as_tensor(
                        np.stack([buf.critic_hidden[start] for start, _ in chunk_ranges]),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)
                    dist_batch, _ = self.net.dist_sequence(obs_batch, actor_hidden)
                    component_batch, _ = self.net.value_components_sequence(
                        obs_batch,
                        critic_hidden,
                    )
                    dist = torch.distributions.Normal(
                        dist_batch.loc.reshape(-1, 1),
                        dist_batch.scale.reshape(-1, 1),
                    )
                    value_components = tuple(
                        component.reshape(-1) for component in component_batch
                    )
                    values = (
                        value_components[0]
                        + value_components[1]
                        - value_components[2]
                    )
                else:
                    # Only the final partial chunk can reach this path.
                    loc_parts = []
                    scale_parts = []
                    value_component_parts = [[], [], []]
                    for start, stop in chunk_ranges:
                        obs_chunk = obs[start:stop].unsqueeze(0)
                        actor_hidden = torch.as_tensor(
                            buf.actor_hidden[start],
                            dtype=torch.float32,
                            device=self.device,
                        ).view(1, 1, self.hidden_size)
                        critic_hidden = torch.as_tensor(
                            buf.critic_hidden[start],
                            dtype=torch.float32,
                            device=self.device,
                        ).view(1, 1, self.hidden_size)
                        dist_chunk, _ = self.net.dist_sequence(obs_chunk, actor_hidden)
                        component_chunk, _ = self.net.value_components_sequence(
                            obs_chunk,
                            critic_hidden,
                        )
                        loc_parts.append(dist_chunk.loc.reshape(-1, 1))
                        scale_parts.append(dist_chunk.scale.reshape(-1, 1))
                        for component_index, component in enumerate(component_chunk):
                            value_component_parts[component_index].append(
                                component.reshape(-1)
                            )
                    dist = torch.distributions.Normal(
                        torch.cat(loc_parts, dim=0),
                        torch.cat(scale_parts, dim=0),
                    )
                    value_components = tuple(
                        torch.cat(parts, dim=0) for parts in value_component_parts
                    )
                    values = (
                        value_components[0]
                        + value_components[1]
                        - value_components[2]
                    )

                mb = torch.cat(index_parts)
                logp = _squashed_log_prob_from_latent(dist, latent[mb])
                log_ratio = logp - logp_old[mb]
                ratio = torch.exp(log_ratio)
                surr1 = ratio * adv_t[mb]
                surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[mb]
                pi_loss = -torch.min(surr1, surr2).mean()
                if component_ret_t is None:
                    v_loss_energy = _critic_value_loss(values, ret_t[mb])
                    v_loss_demand = torch.zeros((), device=self.device)
                    v_loss_wear = torch.zeros((), device=self.device)
                    v_loss = v_loss_energy
                else:
                    v_loss_energy = _critic_value_loss(
                        value_components[0],
                        component_ret_t[0][mb],
                    )
                    v_loss_demand = _critic_value_loss(
                        value_components[1],
                        component_ret_t[1][mb],
                    )
                    v_loss_wear = _critic_value_loss(
                        value_components[2],
                        component_ret_t[2][mb],
                    )
                    # Keep nominal critic-loss scale close to the scalar baseline.
                    v_loss = (v_loss_energy + v_loss_demand + v_loss_wear) / 3.0
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
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    actor_params,
                    self.actor_grad_clip,
                )
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    critic_params,
                    self.critic_grad_clip,
                )
                self.opt.step()

                pi_losses.append(float(pi_loss.detach().cpu()))
                value_losses.append(float(v_loss.detach().cpu()))
                energy_value_losses.append(float(v_loss_energy.detach().cpu()))
                demand_value_losses.append(float(v_loss_demand.detach().cpu()))
                wear_value_losses.append(float(v_loss_wear.detach().cpu()))
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

        with torch.no_grad():
            effective_parts = []
            segment_start = 0
            for index in range(n):
                if not bool(buf.done[index]) and index != n - 1:
                    continue
                segment_stop = index + 1
                segment_obs = obs[segment_start:segment_stop].unsqueeze(0)
                segment_dist, _ = self.net.dist_sequence(segment_obs, None)
                effective_parts.append(segment_dist.scale.log().reshape(-1))
                segment_start = segment_stop
            effective_log_std = torch.cat(effective_parts, dim=0)
            effective_std = effective_log_std.exp()
            exploration_stats = {
                "effective_log_std_mean": float(effective_log_std.mean().cpu()),
                "effective_log_std_min": float(effective_log_std.min().cpu()),
                "effective_log_std_max": float(effective_log_std.max().cpu()),
                "effective_action_std_mean": float(effective_std.mean().cpu()),
            }
            if self.obs_dim > 3:
                soc = obs[:, 3]
                low = soc < (1.0 / 3.0)
                middle = (soc >= (1.0 / 3.0)) & (soc <= (2.0 / 3.0))
                high = soc > (2.0 / 3.0)

                def _masked_log_std(mask):
                    if not bool(mask.any().item()):
                        return 0.0
                    return float(effective_log_std[mask].mean().cpu())

                exploration_stats.update({
                    "effective_log_std_soc_low": _masked_log_std(low),
                    "effective_log_std_soc_middle": _masked_log_std(middle),
                    "effective_log_std_soc_high": _masked_log_std(high),
                })

        self.last_update_stats = {
            "policy_loss": _mean(pi_losses),
            "value_loss": _mean(value_losses),
            "energy_value_loss": _mean(energy_value_losses),
            "demand_value_loss": _mean(demand_value_losses),
            "wear_value_loss": _mean(wear_value_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(approx_kls),
            "clip_fraction": _mean(clip_fractions),
            "grad_norm": _mean(grad_norms),
            "actor_grad_norm": _mean(actor_grad_norms),
            "critic_grad_norm": _mean(critic_grad_norms),
            **rollout_stats,
            "log_std": float(self.net.log_std.detach().cpu().item()),
            "recurrent_sequence_length": int(sequence_length),
            "recurrent_chunk_count": len(chunk_starts),
            **exploration_stats,
            "epochs_completed": epochs_completed,
            "kl_early_stop": early_stopped,
        }
        self._sync_collector()
        buf.ptr = 0
        return dict(self.last_update_stats)

    # ------------------------------------------------------------------
    def update(
        self,
        buf: RolloutBuffer,
        last_val: float,
        last_val_components: tuple[float, float, float] | None = None,
    ):
        if self.recurrent_enabled:
            return self._update_recurrent(buf, last_val, last_val_components)
        n = buf.ptr
        if n <= 0:
            raise ValueError("PPO update requires at least one rollout transition")

        # GAE still masks with each transition's own done flag; decomposed mode
        # applies the same boundary independently to energy, demand, and wear.
        adv, ret, component_returns, rollout_stats = self._prepare_rollout_targets(
            buf,
            last_val,
            last_val_components,
        )

        obs = torch.as_tensor(buf.obs[:n], device=self.device)
        latent = torch.as_tensor(buf.latent[:n], device=self.device)
        logp_old = torch.as_tensor(buf.logp[:n], device=self.device)
        adv_t = torch.as_tensor(adv, device=self.device)
        ret_t = torch.as_tensor(ret, device=self.device)
        component_ret_t = (
            tuple(torch.as_tensor(values, device=self.device) for values in component_returns)
            if component_returns is not None
            else None
        )

        pi_losses = []
        value_losses = []
        energy_value_losses = []
        demand_value_losses = []
        wear_value_losses = []
        entropies = []
        approx_kls = []
        clip_fractions = []
        grad_norms = []
        actor_grad_norms = []
        critic_grad_norms = []
        actor_params = self._actor_parameters()
        critic_params = self._critic_parameters()
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
                value_components = self.net.value_components(obs[mb])
                values = value_components[0] + value_components[1] - value_components[2]
                if component_ret_t is None:
                    v_loss_energy = _critic_value_loss(values, ret_t[mb])
                    v_loss_demand = torch.zeros((), device=self.device)
                    v_loss_wear = torch.zeros((), device=self.device)
                    v_loss = v_loss_energy
                else:
                    v_loss_energy = _critic_value_loss(
                        value_components[0],
                        component_ret_t[0][mb],
                    )
                    v_loss_demand = _critic_value_loss(
                        value_components[1],
                        component_ret_t[1][mb],
                    )
                    v_loss_wear = _critic_value_loss(
                        value_components[2],
                        component_ret_t[2][mb],
                    )
                    v_loss = (v_loss_energy + v_loss_demand + v_loss_wear) / 3.0
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
                actor_grad_norm = nn.utils.clip_grad_norm_(
                    actor_params,
                    self.actor_grad_clip,
                )
                critic_grad_norm = nn.utils.clip_grad_norm_(
                    critic_params,
                    self.critic_grad_clip,
                )
                self.opt.step()

                pi_losses.append(float(pi_loss.detach().cpu()))
                value_losses.append(float(v_loss.detach().cpu()))
                energy_value_losses.append(float(v_loss_energy.detach().cpu()))
                demand_value_losses.append(float(v_loss_demand.detach().cpu()))
                wear_value_losses.append(float(v_loss_wear.detach().cpu()))
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

        with torch.no_grad():
            effective_log_std = self.net.effective_log_std(obs).squeeze(-1)
            effective_std = effective_log_std.exp()
            exploration_stats = {
                "effective_log_std_mean": float(effective_log_std.mean().cpu()),
                "effective_log_std_min": float(effective_log_std.min().cpu()),
                "effective_log_std_max": float(effective_log_std.max().cpu()),
                "effective_action_std_mean": float(effective_std.mean().cpu()),
            }
            if self.obs_dim > 3:
                soc = obs[:, 3]
                low = soc < (1.0 / 3.0)
                middle = (soc >= (1.0 / 3.0)) & (soc <= (2.0 / 3.0))
                high = soc > (2.0 / 3.0)

                def _masked_log_std(mask):
                    if not bool(mask.any().item()):
                        return 0.0
                    return float(effective_log_std[mask].mean().cpu())

                exploration_stats.update({
                    "effective_log_std_soc_low": _masked_log_std(low),
                    "effective_log_std_soc_middle": _masked_log_std(middle),
                    "effective_log_std_soc_high": _masked_log_std(high),
                })

        self.last_update_stats = {
            "policy_loss": _mean(pi_losses),
            "value_loss": _mean(value_losses),
            "energy_value_loss": _mean(energy_value_losses),
            "demand_value_loss": _mean(demand_value_losses),
            "wear_value_loss": _mean(wear_value_losses),
            "entropy": _mean(entropies),
            "approx_kl": _mean(approx_kls),
            "clip_fraction": _mean(clip_fractions),
            "grad_norm": _mean(grad_norms),
            "actor_grad_norm": _mean(actor_grad_norms),
            "critic_grad_norm": _mean(critic_grad_norms),
            **rollout_stats,
            "log_std": float(self.net.log_std.detach().cpu().item()),
            **exploration_stats,
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
        meta = {
            **dict(self.meta),
            "hidden_size": self.hidden_size,
            "initial_log_std": self.initial_log_std,
            "recurrent_enabled": self.recurrent_enabled,
            "recurrent_sequence_length": self.recurrent_sequence_length,
            "decomposed_critic": self.decomposed_critic,
            "critic_components": (
                ["energy", "demand", "wear"] if self.decomposed_critic else ["total"]
            ),
            "policy_architecture": self.policy_architecture_name(),
            "exploration_mode": "state_dependent_log_std_delta_soc_edge_v2",
            "exploration_hidden_size": self.net.exploration_hidden_size,
            "exploration_lr_multiplier": self.exploration_lr_multiplier,
            "exploration_learning_rate": self.exploration_learning_rate,
            "soc_edge_log_std_penalty": self.soc_edge_log_std_penalty,
        }
        self.meta = meta
        payload = {"algo": "ppo", "state_dict": state, "meta": meta}
        torch.save(payload, path)

    def _rebuild_network_for_hidden_size(
        self,
        hidden_size: int,
        *,
        recurrent_enabled: bool | None = None,
        decomposed_critic: bool | None = None,
    ) -> None:
        """Recreate actor/critic shells before loading a different checkpoint architecture."""
        self.hidden_size = int(hidden_size)
        if recurrent_enabled is not None:
            self.recurrent_enabled = bool(recurrent_enabled)
        if decomposed_critic is not None:
            self.decomposed_critic = bool(decomposed_critic)
        self.net = self._make_network().to(self.device)
        self.opt = self._build_optimizer()
        with torch.random.fork_rng(devices=[]):
            self.collector_net = self._make_network().cpu()
        self._sync_collector()

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        if isinstance(ck, dict) and "state_dict" in ck:
            algo = str(ck.get("algo") or "ppo").lower()
            if algo != "ppo":
                raise ValueError(f"checkpoint algorithm {algo!r} is not PPO")
            state_dict = ck["state_dict"]
            self.meta = ck.get("meta", {}) or {}
        else:                                   # legacy raw PPO state_dict
            state_dict = ck
            self.meta = {}

        checkpoint_recurrent = bool(self.meta.get("recurrent_enabled", False))
        inferred_decomposed_critic = any(
            key.startswith(("critic_demand.", "critic_wear."))
            for key in state_dict
        )
        checkpoint_decomposed_critic = bool(
            self.meta.get("decomposed_critic", inferred_decomposed_critic)
        )
        actor_input = state_dict.get("actor.0.weight")
        inferred_hidden_size = (
            int(actor_input.shape[0])
            if actor_input is not None
            else self.hidden_size
        )
        checkpoint_hidden_size = int(
            self.meta.get("hidden_size", inferred_hidden_size)
        )
        self.recurrent_sequence_length = int(
            self.meta.get("recurrent_sequence_length", self.recurrent_sequence_length)
        )
        self.initial_log_std = float(
            self.meta.get("initial_log_std", self.initial_log_std)
        )
        self.exploration_lr_multiplier = float(
            self.meta.get("exploration_lr_multiplier", self.exploration_lr_multiplier)
        )
        self.exploration_learning_rate = (
            self.learning_rate * self.exploration_lr_multiplier
        )
        self.soc_edge_log_std_penalty = float(
            self.meta.get("soc_edge_log_std_penalty", 0.0)
        )
        if (
            checkpoint_hidden_size != self.hidden_size
            or checkpoint_recurrent != self.recurrent_enabled
            or checkpoint_decomposed_critic != self.decomposed_critic
        ):
            self._rebuild_network_for_hidden_size(
                checkpoint_hidden_size,
                recurrent_enabled=checkpoint_recurrent,
                decomposed_critic=checkpoint_decomposed_critic,
            )

        # Pre-IQ-29 checkpoints have no adaptive-exploration head.  Seed those
        # missing tensors from this model's zero-output initialization so old PPO
        # checkpoints retain exactly their original scalar-log_std behavior.
        current_state = self.net.state_dict()
        missing_exploration = [
            key
            for key in current_state
            if key.startswith("log_std_delta.") and key not in state_dict
        ]
        if missing_exploration:
            state_dict = dict(state_dict)
            for key in missing_exploration:
                state_dict[key] = current_state[key]
        self.net.load_state_dict(state_dict)
        self.net.soc_edge_log_std_penalty = self.soc_edge_log_std_penalty
        self.collector_net.soc_edge_log_std_penalty = self.soc_edge_log_std_penalty
        self._sync_collector()
        self.net.eval()
