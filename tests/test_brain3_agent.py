import ast
from pathlib import Path

import numpy as np
import pytest
import torch

from bess.brain.brain3_agent import (
    BRAIN3_ACTIONS,
    BRAIN3_ACTION_DIM,
    BRAIN3_ACTION_LABELS,
    Brain3Agent,
    Brain3QNetwork,
    Brain3ReplayBuffer,
    ReplayBatch,
)
from bess.brain.brain_env import BrainEnv, BrainEpisode, BrainTimestepInput


def observation(value: float = 0.0):
    return (value, 1.0, 0.2, 0.5, 0.4, 0.1, 1.0)


def set_constant_q(network: Brain3QNetwork, q_values: tuple[float, float, float]) -> None:
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
        network.net[-1].bias.copy_(torch.tensor(q_values, dtype=torch.float32))


def make_tiny_env() -> BrainEnv:
    episode = BrainEpisode(
        timesteps=(
            BrainTimestepInput(net_load_kw=400.0, tariff_vnd_per_kwh=1.0, is_working_day=True),
            BrainTimestepInput(net_load_kw=400.0, tariff_vnd_per_kwh=1.0, is_working_day=True),
            BrainTimestepInput(net_load_kw=400.0, tariff_vnd_per_kwh=100.0, is_working_day=True),
            BrainTimestepInput(net_load_kw=400.0, tariff_vnd_per_kwh=100.0, is_working_day=True),
        ),
        steps_per_day=4,
    )
    return BrainEnv(
        initial_state_of_charge=0.50,
        minimum_state_of_charge=0.0,
        maximum_state_of_charge=1.0,
        battery_capacity_kwh=100.0,
        battery_power_kw=100.0,
        timestep_hours=0.25,
        charge_efficiency=1.0,
        discharge_efficiency=1.0,
        demand_charge_vnd_per_kw=0.0,
        battery_wear_vnd_per_kwh=0.0,
        episode=episode,
    )


def test_brain3_has_exactly_three_actions_and_preserves_brain_env_sign_convention():
    assert BRAIN3_ACTION_DIM == 3
    assert BRAIN3_ACTIONS == (-1.0, 0.0, 1.0)
    assert BRAIN3_ACTION_LABELS == ("CHARGE", "IDLE", "DISCHARGE")
    assert Brain3Agent.action_from_index(0) == -1.0
    assert Brain3Agent.action_from_index(1) == 0.0
    assert Brain3Agent.action_from_index(2) == 1.0


def test_q_network_maps_seven_eyes_to_three_q_values():
    network = Brain3QNetwork(hidden_dim=32)
    output = network(torch.zeros((5, 7), dtype=torch.float32))
    assert output.shape == (5, 3)


def test_greedy_decision_chooses_biggest_q_and_returns_real_battery_action():
    agent = Brain3Agent(hidden_dim=16, epsilon_start=0.0, epsilon_end=0.0, seed=1)
    set_constant_q(agent.online_network, (8.0, 2.0, -5.0))

    charge = agent.decide(observation())
    assert charge.action_index == 0
    assert charge.action == -1.0
    assert charge.label == "CHARGE"
    assert charge.q_values == pytest.approx((8.0, 2.0, -5.0))

    set_constant_q(agent.online_network, (-4.0, 3.0, 12.0))
    discharge = agent.decide(observation())
    assert discharge.action_index == 2
    assert discharge.action == 1.0
    assert discharge.label == "DISCHARGE"


def test_replay_buffer_is_fixed_capacity_and_returns_vectorized_batches():
    replay = Brain3ReplayBuffer(capacity=3, seed=3)
    for index in range(5):
        replay.add(observation(index / 10.0), index % 3, float(index), observation(), False)

    assert len(replay) == 3
    batch = replay.sample(2)
    assert batch.observations.shape == (2, 7)
    assert batch.actions.shape == (2,)
    assert batch.rewards_vnd.shape == (2,)
    assert batch.next_observations.shape == (2, 7)
    assert batch.dones.shape == (2,)
    assert batch.observations.dtype == np.float32
    assert batch.actions.dtype == np.int64


def test_dqn_target_is_reward_plus_discounted_best_future_q_and_terminal_does_not_bootstrap():
    agent = Brain3Agent(
        hidden_dim=16,
        gamma=0.5,
        reward_divisor_vnd=1_000_000.0,
        epsilon_start=0.0,
        epsilon_end=0.0,
    )
    set_constant_q(agent.target_network, (1.0, 2.0, 3.0))

    batch = ReplayBatch(
        observations=np.zeros((2, 7), dtype=np.float32),
        actions=np.array([0, 2], dtype=np.int64),
        rewards_vnd=np.array([1_000_000.0, 2_000_000.0], dtype=np.float32),
        next_observations=np.zeros((2, 7), dtype=np.float32),
        dones=np.array([0.0, 1.0], dtype=np.float32),
    )

    targets = agent._targets_from_batch(batch).cpu().numpy()
    assert targets == pytest.approx([2.5, 2.0])


def test_learning_updates_online_network_but_target_waits_for_sync_interval():
    agent = Brain3Agent(
        hidden_dim=16,
        batch_size=1,
        learning_starts=1,
        target_sync_interval=2,
        epsilon_start=0.0,
        epsilon_end=0.0,
        seed=2,
    )
    before_online = [parameter.detach().clone() for parameter in agent.online_network.parameters()]
    before_target = [parameter.detach().clone() for parameter in agent.target_network.parameters()]

    agent.remember(observation(), 0, 1_000_000.0, observation(0.1), False)
    first_loss = agent.learn()

    assert first_loss is not None and np.isfinite(first_loss)
    assert agent.gradient_steps == 1
    assert any(
        not torch.equal(before, after)
        for before, after in zip(before_online, agent.online_network.parameters())
    )
    assert all(
        torch.equal(before, after)
        for before, after in zip(before_target, agent.target_network.parameters())
    )

    agent.remember(observation(0.1), 1, -500_000.0, None, True)
    second_loss = agent.learn()
    assert second_loss is not None and np.isfinite(second_loss)
    assert agent.gradient_steps == 2
    assert all(
        torch.equal(online, target)
        for online, target in zip(agent.online_network.parameters(), agent.target_network.parameters())
    )


def test_train_episode_uses_brainenv_savings_reward_and_learns_each_step_after_warmup():
    agent = Brain3Agent(
        hidden_dim=16,
        batch_size=1,
        learning_starts=1,
        replay_capacity=32,
        target_sync_interval=2,
        epsilon_start=1.0,
        epsilon_end=1.0,
        epsilon_decay_steps=10,
        reward_divisor_vnd=1_000.0,
        seed=4,
    )
    env = make_tiny_env()

    stats = agent.train_episode(env)

    assert stats.steps == 4
    assert stats.monthly_savings_vnd == pytest.approx(env.net_battery_savings_vnd)
    assert stats.mean_loss is not None and np.isfinite(stats.mean_loss)
    assert stats.ending_epsilon == pytest.approx(1.0)
    assert agent.environment_steps == 4
    assert agent.gradient_steps == 4
    assert len(agent.replay) == 4


def test_evaluation_is_greedy_and_does_not_mutate_learning_counters_or_replay():
    agent = Brain3Agent(hidden_dim=16, epsilon_start=1.0, epsilon_end=0.1, seed=5)
    set_constant_q(agent.online_network, (0.0, 10.0, 0.0))
    env = make_tiny_env()

    before_steps = agent.environment_steps
    before_gradients = agent.gradient_steps
    before_replay = len(agent.replay)
    stats = agent.evaluate_episode(env)

    assert stats.steps == 4
    assert stats.monthly_savings_vnd == pytest.approx(0.0)
    assert agent.environment_steps == before_steps
    assert agent.gradient_steps == before_gradients
    assert len(agent.replay) == before_replay


def test_checkpoint_declares_dqn_and_three_action_contract():
    agent = Brain3Agent(hidden_dim=16)
    checkpoint = agent.checkpoint()
    assert checkpoint["algorithm"] == "dqn"
    assert checkpoint["action_values"] == (-1.0, 0.0, 1.0)
    assert checkpoint["observation_dim"] == 7


def test_invalid_observations_and_action_indexes_fail_loudly():
    agent = Brain3Agent(hidden_dim=16)
    with pytest.raises(ValueError, match="exactly 7"):
        agent.act((0.0,) * 6)
    with pytest.raises(ValueError, match="finite"):
        agent.act((0.0, 1.0, np.nan, 0.5, 0.4, 0.1, 1.0))
    with pytest.raises(ValueError, match="0, 1, 2"):
        Brain3Agent.action_from_index(3)


def test_brain3_imports_only_canonical_env_not_training_stack():
    path = Path(__file__).resolve().parents[1] / "bess" / "brain" / "brain3_agent.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported_modules.append(node.module)

    assert "bess.brain.brain_env" in imported_modules
    assert all(not module.startswith("bess.training") for module in imported_modules)
    assert all("legacy" not in module.lower() for module in imported_modules)
