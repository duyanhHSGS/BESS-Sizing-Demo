from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from EXPERIMENT_FIELD.brain_env import (
    OBSERVATION_DIM,
    BrainEnv,
    BrainEnvironmentStepResult,
    BrainObservation,
)


# BrainEnv's real battery-side sign convention. Do not reverse this:
#   -1 = maximum charge, 0 = idle, +1 = maximum discharge.
BRAIN3_ACTIONS: tuple[float, float, float] = (-1.0, 0.0, 1.0)
BRAIN3_ACTION_LABELS: tuple[str, str, str] = ("CHARGE", "IDLE", "DISCHARGE")
BRAIN3_ACTION_DIM = len(BRAIN3_ACTIONS)


@dataclass(frozen=True, slots=True)
class Brain3Decision:
    """One explainable DQN decision over Brain 3's three discrete buttons."""

    action_index: int
    action: float
    label: str
    q_values: tuple[float, float, float]
    epsilon: float
    explored: bool


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards_vnd: np.ndarray
    next_observations: np.ndarray
    dones: np.ndarray


class Brain3ReplayBuffer:
    """Preallocated ring buffer for fast, allocation-light DQN replay sampling."""

    def __init__(self, capacity: int, observation_dim: int = OBSERVATION_DIM, *, seed: int = 0):
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("Brain3 replay capacity must be a positive integer")
        if observation_dim != OBSERVATION_DIM:
            raise ValueError(f"Brain3 replay expects exactly {OBSERVATION_DIM} observation eyes")

        self.capacity = capacity
        self.observation_dim = observation_dim
        self._observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards_vnd = np.empty(capacity, dtype=np.float32)
        self._next_observations = np.empty((capacity, observation_dim), dtype=np.float32)
        self._dones = np.empty(capacity, dtype=np.float32)
        self._size = 0
        self._write_index = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        observation: BrainObservation | tuple[float, ...] | np.ndarray,
        action_index: int,
        reward_vnd: float,
        next_observation: BrainObservation | tuple[float, ...] | np.ndarray | None,
        done: bool,
    ) -> None:
        observation_array = _validated_observation_array(observation)
        if isinstance(action_index, bool) or not isinstance(action_index, (int, np.integer)):
            raise TypeError("Brain3 replay action_index must be an integer")
        action_index = int(action_index)
        if action_index < 0 or action_index >= BRAIN3_ACTION_DIM:
            raise ValueError("Brain3 replay action_index must be one of 0, 1, 2")

        reward_value = float(reward_vnd)
        if not math.isfinite(reward_value):
            raise ValueError("Brain3 replay reward_vnd must be finite")
        if not isinstance(done, (bool, np.bool_)):
            raise TypeError("Brain3 replay done must be boolean")

        if done:
            if next_observation is not None:
                next_array = _validated_observation_array(next_observation)
            else:
                next_array = np.zeros(self.observation_dim, dtype=np.float32)
        else:
            if next_observation is None:
                raise ValueError("Brain3 non-terminal replay entries require next_observation")
            next_array = _validated_observation_array(next_observation)

        index = self._write_index
        self._observations[index] = observation_array
        self._actions[index] = action_index
        self._rewards_vnd[index] = reward_value
        self._next_observations[index] = next_array
        self._dones[index] = 1.0 if done else 0.0

        self._write_index = (index + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int) -> ReplayBatch:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Brain3 replay batch_size must be a positive integer")
        if batch_size > self._size:
            raise ValueError("Brain3 replay cannot sample more transitions than it currently stores")

        indexes = self._rng.choice(self._size, size=batch_size, replace=False)
        return ReplayBatch(
            observations=self._observations[indexes],
            actions=self._actions[indexes],
            rewards_vnd=self._rewards_vnd[indexes],
            next_observations=self._next_observations[indexes],
            dones=self._dones[indexes],
        )


class Brain3QNetwork(nn.Module):
    """Seven BrainEnv eyes -> one Q-value for each of the three discrete actions."""

    def __init__(self, observation_dim: int = OBSERVATION_DIM, hidden_dim: int = 128):
        super().__init__()
        if observation_dim != OBSERVATION_DIM:
            raise ValueError(f"Brain3 expects exactly {OBSERVATION_DIM} observation eyes")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("Brain3 hidden_dim must be a positive integer")

        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, BRAIN3_ACTION_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


@dataclass(frozen=True, slots=True)
class Brain3EpisodeStats:
    steps: int
    monthly_savings_vnd: float
    mean_loss: float | None
    ending_epsilon: float


class Brain3Agent:
    """Standalone classic DQN brain for BrainEnv's three discrete battery actions.

    The optimization target is unchanged from BrainEnv: maximize the sum of
    timestep savings, where each timestep reward is RawWorld operating cost minus
    BessWorld operating cost. ``reward_divisor_vnd`` only rescales numbers used by
    gradient descent; any positive divisor preserves the optimal policy.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        replay_capacity: int = 100_000,
        learning_starts: int = 1_024,
        target_sync_interval: int = 1_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100_000,
        reward_divisor_vnd: float = 1_000_000.0,
        gradient_clip_norm: float = 10.0,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.gamma = _finite_between("gamma", gamma, 0.0, 1.0, inclusive_low=True, inclusive_high=True)
        self.learning_rate = _finite_positive("learning_rate", learning_rate)
        self.reward_divisor_vnd = _finite_positive("reward_divisor_vnd", reward_divisor_vnd)
        self.gradient_clip_norm = _finite_positive("gradient_clip_norm", gradient_clip_norm)

        self.batch_size = _positive_int("batch_size", batch_size)
        self.learning_starts = _nonnegative_int("learning_starts", learning_starts)
        self.target_sync_interval = _positive_int("target_sync_interval", target_sync_interval)
        self.epsilon_decay_steps = _positive_int("epsilon_decay_steps", epsilon_decay_steps)
        self.epsilon_start = _finite_between(
            "epsilon_start", epsilon_start, 0.0, 1.0, inclusive_low=True, inclusive_high=True
        )
        self.epsilon_end = _finite_between(
            "epsilon_end", epsilon_end, 0.0, 1.0, inclusive_low=True, inclusive_high=True
        )
        if self.epsilon_end > self.epsilon_start:
            raise ValueError("Brain3 epsilon_end must be <= epsilon_start")

        self.device = torch.device(device)
        self._rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        self.online_network = Brain3QNetwork(hidden_dim=hidden_dim).to(self.device)
        self.target_network = Brain3QNetwork(hidden_dim=hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        for parameter in self.target_network.parameters():
            parameter.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=self.learning_rate)
        self.replay = Brain3ReplayBuffer(replay_capacity, seed=seed)
        self.environment_steps = 0
        self.gradient_steps = 0

    @property
    def epsilon(self) -> float:
        fraction = min(1.0, self.environment_steps / self.epsilon_decay_steps)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)

    @staticmethod
    def action_from_index(action_index: int) -> float:
        if isinstance(action_index, bool) or not isinstance(action_index, (int, np.integer)):
            raise TypeError("Brain3 action_index must be an integer")
        action_index = int(action_index)
        if action_index < 0 or action_index >= BRAIN3_ACTION_DIM:
            raise ValueError("Brain3 action_index must be one of 0, 1, 2")
        return BRAIN3_ACTIONS[action_index]

    def q_values(self, observation: BrainObservation | tuple[float, ...] | np.ndarray) -> tuple[float, float, float]:
        observation_array = _validated_observation_array(observation)
        tensor = torch.from_numpy(observation_array).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            q = self.online_network(tensor)[0].detach().cpu().tolist()
        return (float(q[0]), float(q[1]), float(q[2]))

    def decide(
        self,
        observation: BrainObservation | tuple[float, ...] | np.ndarray,
        *,
        explore: bool = False,
    ) -> Brain3Decision:
        q = self.q_values(observation)
        epsilon = self.epsilon if explore else 0.0
        explored = bool(explore and self._rng.random() < epsilon)
        if explored:
            action_index = int(self._rng.integers(0, BRAIN3_ACTION_DIM))
        else:
            action_index = int(np.argmax(q))
        return Brain3Decision(
            action_index=action_index,
            action=BRAIN3_ACTIONS[action_index],
            label=BRAIN3_ACTION_LABELS[action_index],
            q_values=q,
            epsilon=epsilon,
            explored=explored,
        )

    def act(
        self,
        observation: BrainObservation | tuple[float, ...] | np.ndarray,
        *,
        explore: bool = False,
    ) -> float:
        return self.decide(observation, explore=explore).action

    def remember(
        self,
        observation: BrainObservation | tuple[float, ...] | np.ndarray,
        action_index: int,
        reward_vnd: float,
        next_observation: BrainObservation | tuple[float, ...] | np.ndarray | None,
        done: bool,
    ) -> None:
        self.replay.add(observation, action_index, reward_vnd, next_observation, done)
        self.environment_steps += 1

    def _targets_from_batch(self, batch: ReplayBatch) -> torch.Tensor:
        rewards = torch.from_numpy(batch.rewards_vnd).to(self.device)
        next_observations = torch.from_numpy(batch.next_observations).to(self.device)
        dones = torch.from_numpy(batch.dones).to(self.device)
        with torch.no_grad():
            best_future_q = self.target_network(next_observations).max(dim=1).values
            return rewards / self.reward_divisor_vnd + self.gamma * (1.0 - dones) * best_future_q

    def learn(self) -> float | None:
        minimum_replay = max(self.batch_size, self.learning_starts)
        if len(self.replay) < minimum_replay:
            return None

        batch = self.replay.sample(self.batch_size)
        observations = torch.from_numpy(batch.observations).to(self.device)
        actions = torch.from_numpy(batch.actions).to(self.device)
        targets = self._targets_from_batch(batch)

        predicted_q = self.online_network(observations).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(predicted_q, targets)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_network.parameters(), self.gradient_clip_norm)
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_sync_interval == 0:
            self.sync_target_network()
        return float(loss.detach().cpu().item())

    def sync_target_network(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def train_episode(self, env: BrainEnv) -> Brain3EpisodeStats:
        """Run one complete owned BrainEpisode with epsilon-greedy DQN learning."""
        observation = env.reset()
        losses: list[float] = []
        final_result: BrainEnvironmentStepResult | None = None
        steps = 0

        while True:
            decision = self.decide(observation, explore=True)
            result = env.step(decision.action)
            if not isinstance(result, BrainEnvironmentStepResult):
                raise RuntimeError("Brain3 training requires BrainEnv configured with a BrainEpisode")

            reward_vnd = result.reward.timestep_savings_vnd
            self.remember(
                observation,
                decision.action_index,
                reward_vnd,
                result.next_observation,
                result.done,
            )
            loss = self.learn()
            if loss is not None:
                losses.append(loss)

            steps += 1
            final_result = result
            if result.done:
                break
            assert result.next_observation is not None
            observation = result.next_observation

        assert final_result is not None
        mean_loss = float(np.mean(losses)) if losses else None
        return Brain3EpisodeStats(
            steps=steps,
            monthly_savings_vnd=final_result.reward.monthly_savings_vnd,
            mean_loss=mean_loss,
            ending_epsilon=self.epsilon,
        )

    def evaluate_episode(self, env: BrainEnv) -> Brain3EpisodeStats:
        """Run one complete greedy episode with no exploration and no learning."""
        observation = env.reset()
        final_result: BrainEnvironmentStepResult | None = None
        steps = 0

        while True:
            decision = self.decide(observation, explore=False)
            result = env.step(decision.action)
            if not isinstance(result, BrainEnvironmentStepResult):
                raise RuntimeError("Brain3 evaluation requires BrainEnv configured with a BrainEpisode")
            steps += 1
            final_result = result
            if result.done:
                break
            assert result.next_observation is not None
            observation = result.next_observation

        assert final_result is not None
        return Brain3EpisodeStats(
            steps=steps,
            monthly_savings_vnd=final_result.reward.monthly_savings_vnd,
            mean_loss=None,
            ending_epsilon=self.epsilon,
        )

    def checkpoint(self) -> dict[str, Any]:
        """Return enough state to resume Brain 3 without losing DQN counters/optimizer state."""
        return {
            "algorithm": "dqn",
            "action_values": BRAIN3_ACTIONS,
            "observation_dim": OBSERVATION_DIM,
            "online_network": self.online_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "environment_steps": self.environment_steps,
            "gradient_steps": self.gradient_steps,
            "gamma": self.gamma,
            "reward_divisor_vnd": self.reward_divisor_vnd,
        }


def _validated_observation_array(
    observation: BrainObservation | tuple[float, ...] | np.ndarray,
) -> np.ndarray:
    try:
        array = np.asarray(observation, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise TypeError("Brain3 observation must be a numeric seven-value sequence") from exc
    if array.shape != (OBSERVATION_DIM,):
        raise ValueError(f"Brain3 observation must contain exactly {OBSERVATION_DIM} values")
    if not np.isfinite(array).all():
        raise ValueError("Brain3 observation values must all be finite")
    return array


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Brain3 {name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Brain3 {name} must be a non-negative integer")
    return value


def _finite_positive(name: str, value: float) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"Brain3 {name} must be finite and greater than 0")
    return numeric


def _finite_between(
    name: str,
    value: float,
    low: float,
    high: float,
    *,
    inclusive_low: bool,
    inclusive_high: bool,
) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"Brain3 {name} must be finite")
    low_ok = numeric >= low if inclusive_low else numeric > low
    high_ok = numeric <= high if inclusive_high else numeric < high
    if not (low_ok and high_ok):
        raise ValueError(f"Brain3 {name} must be inside [{low}, {high}]")
    return numeric
