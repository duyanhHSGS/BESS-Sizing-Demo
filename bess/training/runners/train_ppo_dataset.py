from __future__ import annotations

import argparse
import calendar
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bess.agents.ppo_agent import (
    PPOAgent,
    RolloutBuffer,
    configure_ppo_determinism,
    resolve_ppo_device,
)
from bess.core.bess_env import OBSERVATION_DIM
from bess.core.brain_runtime import (
    REWARD_SCALE_VND,
    make_brain_env,
    native_steps_per_action,
    observation_array,
    step_brain_control,
)
from bess.core.common import RESULTS_DIR, score_month
from bess.core.scenario_gen import MonthData
from bess.core.settings import (
    PPO_ACTION_MISMATCH_SHAPING_SCALE,
    PPO_ACTOR_GRAD_CLIP,
    PPO_BC_FINE_TUNE_LOG_STD,
    PPO_CHALLENGER_RESET_PATIENCE,
    PPO_CHALLENGER_RESETS_ENABLED,
    PPO_CLIP,
    PPO_CRITIC_GRAD_CLIP,
    PPO_ENTROPY_COEF,
    PPO_EXPLORATION_LR_MULTIPLIER,
    PPO_FINE_TUNE_EPOCHS,
    PPO_FIT_CONTROL_DT_MINUTES,
    PPO_GAMMA,
    PPO_HIDDEN_SIZE,
    PPO_INITIAL_LOG_STD,
    PPO_LAMBDA,
    PPO_LEARNING_RATE,
    PPO_LOG_EVERY_UPDATES,
    PPO_MINIBATCH,
    PPO_ORACLE_ACTOR_BC_MAX_EPOCHS,
    PPO_ORACLE_BC_ENABLED,
    PPO_ORACLE_BC_LEARNING_RATE,
    PPO_ORACLE_BC_MAX_EPOCHS,
    PPO_ORACLE_BC_MINIBATCH,
    PPO_ORACLE_BC_TARGET_MSE,
    PPO_PRESERVE_CRITIC_ON_REANCHOR,
    PPO_RECURRENT_ENABLED,
    PPO_RECURRENT_SEQUENCE_LENGTH,
    PPO_RESET_OPTIMIZER_ON_REANCHOR,
    PPO_SEED,
    PPO_SOC_EDGE_LOG_STD_PENALTY,
    PPO_STEPS,
    PPO_TARGET_KL,
    PPO_TORCH_THREADS,
    PPO_VALIDATE_EVERY_UPDATES,
    PPO_VALUE_COEF,
)
from bess.evaluation.baselines import run_drl_policy, run_no_bess
from bess.evaluation.oracle.oracle_cache import (
    load_cached_training_dispatch,
    load_cached_training_grids,
)
from bess.training.training_common import (
    build_training_bess_config,
    load_training_days,
    month_blocks,
)
from bess.training.training_reports import (
    PPO_CHAMPION_CURVE_FIELDS,
    write_curve,
    write_report,
)

REANCHOR_SCOPE = "full_state"


def _action_mismatch_penalty_vnd(transition, *, timestep_hours: float, wear_vnd_per_kwh: float) -> float:
    """Penalize requested battery energy that physics refuses to execute.

    PPO's sampled action and stored log-probability stay untouched, preserving
    on-policy correctness. This is trainer-only reward shaping: Champion/test
    scoring still uses the real economic objective with no shaping penalty.
    """
    mismatch_kwh = sum(
        abs(result.requested_battery_kw - result.bess.physics.final_battery_kw)
        * timestep_hours
        for result in transition.native_results
    )
    return mismatch_kwh * wear_vnd_per_kwh


def _load_ui_wear_cost(training_config_path: str | Path) -> float:
    config = json.loads(Path(training_config_path).read_text(encoding="utf-8"))
    if "battery_wear_cost" not in config:
        raise SystemExit("PPO training config requires UI battery_wear_cost")
    wear_cost = float(config["battery_wear_cost"])
    if not math.isfinite(wear_cost) or wear_cost < 0.0:
        raise SystemExit("UI battery_wear_cost must be finite and >= 0")
    return wear_cost


def _score_ppo_operating_month(
    p_grid_days: list[np.ndarray],
    p_bess_days: list[np.ndarray],
    cfg,
    wear_cost_vnd_per_kwh: float,
    days: list,
) -> dict:
    utility = score_month(p_grid_days, cfg, days=days)
    throughput_kwh = sum(
        float(np.sum(np.abs(np.asarray(day, dtype=np.float64))) * cfg.dt)
        for day in p_bess_days
    )
    wear_cost_vnd = throughput_kwh * wear_cost_vnd_per_kwh
    return {
        **utility,
        "throughput_kwh": throughput_kwh,
        "wear_cost_vnd": wear_cost_vnd,
        "total_operating_cost_vnd": utility["total_cost_vnd"] + wear_cost_vnd,
    }


def _score_grid_calendar_months(
    p_grid_days: list[np.ndarray],
    months: list[MonthData],
    cfg,
) -> float:
    """Sum utility bills without merging demand peaks across calendar months."""
    cursor = 0
    total_cost = 0.0
    for month in months:
        stop = cursor + len(month.days)
        total_cost += float(
            score_month(
                p_grid_days[cursor:stop],
                cfg,
                days=month.days,
            )["total_cost_vnd"]
        )
        cursor = stop
    if cursor != len(p_grid_days):
        raise ValueError("Grid traces do not exactly match calendar-month holdouts")
    return total_cost


def _oracle_dispatch_wear_cost_vnd(
    dispatch_days: list[dict[str, list[float]]],
    *,
    timestep_hours: float,
    wear_vnd_per_kwh: float,
) -> float:
    """Score Oracle charge/discharge throughput with the same LP wear convention."""
    throughput_kwh = float(timestep_hours) * sum(
        sum(
            discharge + grid_charge + solar_charge
            for discharge, grid_charge, solar_charge in zip(
                day["discharge"],
                day["grid_charge"],
                day["solar_charge"],
                strict=True,
            )
        )
        for day in dispatch_days
    )
    return throughput_kwh * float(wear_vnd_per_kwh)


def _oracle_teacher_action(
    dispatch: dict[str, list[float]],
    start: int,
    stop: int,
    cfg,
) -> float:
    """Map Oracle outside-world flows into one held BrainEnv battery-side action."""
    battery_side_kw = []
    for step in range(start, stop):
        discharge_kw = float(dispatch["discharge"][step]) / float(cfg.eta_dis)
        charge_kw = (
            float(dispatch["grid_charge"][step])
            + float(dispatch["solar_charge"][step])
        ) * float(cfg.eta_ch)
        battery_side_kw.append(discharge_kw - charge_kw)
    mean_battery_kw = float(np.mean(battery_side_kw)) if battery_side_kw else 0.0
    return float(np.clip(mean_battery_kw / float(cfg.P_rated_nominal), -1.0, 1.0))


def _collect_oracle_teacher_samples(
    month: MonthData,
    oracle_dispatch: list[dict[str, list[float]]],
    cfg,
    *,
    power_scale_kw: float,
    battery_wear_cost: float,
    native_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replay Oracle dispatch and collect brain7 actions plus real rewards."""
    if len(oracle_dispatch) != len(month.days):
        raise ValueError("Oracle dispatch day count must match the teacher month")
    env = make_brain_env(
        month,
        cfg,
        power_scale_kw=power_scale_kw,
        battery_wear_vnd_per_kwh=battery_wear_cost,
    )
    observation = env.reset()
    observations: list[np.ndarray] = []
    targets: list[float] = []
    rewards: list[float] = []

    for day, dispatch in zip(month.days, oracle_dispatch, strict=True):
        native_rows = len(day.load)
        if any(len(values) != native_rows for values in dispatch.values()):
            raise ValueError(
                f"Oracle dispatch length does not match dataset day {day.day_index}"
            )
        for start in range(0, native_rows, native_steps):
            stop = min(start + native_steps, native_rows)
            action = _oracle_teacher_action(dispatch, start, stop, cfg)
            observations.append(observation_array(observation))
            targets.append(action)
            transition = step_brain_control(
                env,
                action,
                native_steps=stop - start,
            )
            rewards.append(transition.reward_million_vnd)
            if transition.next_observation is not None:
                observation = transition.next_observation

    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(rewards, dtype=np.float32),
    )


def _behavior_clone_actor(
    agent: PPOAgent,
    observations: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    max_epochs: int = PPO_ORACLE_ACTOR_BC_MAX_EPOCHS,
    learning_rate: float = PPO_ORACLE_BC_LEARNING_RATE,
    minibatch: int = PPO_ORACLE_BC_MINIBATCH,
    target_mse: float = PPO_ORACLE_BC_TARGET_MSE,
    score_policy_cost=None,
    episode_lengths: list[int] | None = None,
) -> dict[str, float | int]:
    """Supervised-fit the existing generic PPO actor to Oracle teacher actions."""
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("Oracle BC observations must have shape (N, OBSERVATION_DIM)")
    if targets.ndim != 1 or len(targets) != len(observations) or len(targets) == 0:
        raise ValueError("Oracle BC targets must be one non-empty action per observation")
    if episode_lengths is None:
        episode_lengths = [len(observations)]
    if any(length <= 0 for length in episode_lengths) or sum(episode_lengths) != len(observations):
        raise ValueError("Oracle BC episode lengths must exactly partition observations")
    episode_ranges = []
    episode_start = 0
    for episode_length in episode_lengths:
        episode_stop = episode_start + episode_length
        episode_ranges.append((episode_start, episode_stop))
        episode_start = episode_stop

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    target_t = torch.as_tensor(targets, dtype=torch.float32, device=agent.device)
    if agent.recurrent_enabled:
        actor_parameters = [
            *agent.net.actor_encoder.parameters(),
            *agent.net.actor_gru.parameters(),
            *agent.net.actor.parameters(),
        ]
    else:
        actor_parameters = list(agent.net.actor.parameters())
    optimizer = torch.optim.Adam(actor_parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def full_mse() -> float:
        if agent.recurrent_enabled:
            prediction = torch.cat([
                torch.tanh(
                    agent.net.actor_sequence(obs_t[start:stop].unsqueeze(0), None)[0].squeeze(0)
                ).squeeze(-1)
                for start, stop in episode_ranges
            ])
        else:
            prediction = torch.tanh(agent.net.actor(obs_t)).squeeze(-1)
        return float(torch.mean((prediction - target_t) ** 2).cpu())

    initial_mse = full_mse()
    epochs_completed = 0
    best_validation_cost = float("inf")
    best_validation_epoch = 0
    best_validation_mse = float("inf")
    best_actor_state = None
    for epoch in range(max_epochs):
        if agent.recurrent_enabled:
            # Preserve chronology inside each real billing month, but never carry
            # recurrent memory across month boundaries.
            for episode_start, episode_stop in episode_ranges:
                hidden = None
                for start in range(
                    episode_start,
                    episode_stop,
                    agent.recurrent_sequence_length,
                ):
                    stop = min(start + agent.recurrent_sequence_length, episode_stop)
                    mean, _, hidden = agent.net.actor_sequence(
                        obs_t[start:stop].unsqueeze(0),
                        hidden,
                    )
                    prediction = torch.tanh(mean.squeeze(0)).squeeze(-1)
                    loss = torch.mean((prediction - target_t[start:stop]) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
        else:
            indexes = rng.permutation(len(observations))
            for start in range(0, len(indexes), minibatch):
                mb = torch.as_tensor(
                    indexes[start:start + minibatch],
                    dtype=torch.long,
                    device=agent.device,
                )
                prediction = torch.tanh(agent.net.actor(obs_t[mb])).squeeze(-1)
                loss = torch.mean((prediction - target_t[mb]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epochs_completed = epoch + 1
        epoch_mse = full_mse()
        if score_policy_cost is not None:
            # Teacher-forced MSE is not the deployment objective. Re-score the
            # actual closed-loop policy every epoch and remember the cheapest one.
            agent._sync_collector()
            validation_cost = float(score_policy_cost())
            if validation_cost < best_validation_cost:
                best_validation_cost = validation_cost
                best_validation_epoch = epochs_completed
                best_validation_mse = epoch_mse
                best_actor_state = {
                    key: value.detach().clone()
                    for key, value in agent.net.state_dict().items()
                }
        if epoch_mse <= target_mse:
            break

    last_epoch_mse = full_mse()
    if best_actor_state is not None:
        agent.net.load_state_dict(best_actor_state)
    final_mse = full_mse()
    agent._sync_collector()
    stats = {
        "samples": len(observations),
        "charge_samples": int(np.count_nonzero(targets < -1e-6)),
        "idle_samples": int(np.count_nonzero(np.abs(targets) <= 1e-6)),
        "discharge_samples": int(np.count_nonzero(targets > 1e-6)),
        "epochs_completed": int(epochs_completed),
        "initial_mse": initial_mse,
        "final_mse": final_mse,
        "last_epoch_mse": last_epoch_mse,
    }
    if best_actor_state is not None:
        stats.update(
            {
                "economic_best_epoch": int(best_validation_epoch),
                "economic_best_validation_cost_vnd": best_validation_cost,
                "economic_best_mse": best_validation_mse,
            }
        )
    return stats


def _behavior_clone_critic(
    agent: PPOAgent,
    observations: np.ndarray,
    rewards: np.ndarray,
    *,
    gamma: float,
    seed: int,
    max_epochs: int = PPO_ORACLE_BC_MAX_EPOCHS,
    learning_rate: float = PPO_ORACLE_BC_LEARNING_RATE,
    minibatch: int = PPO_ORACLE_BC_MINIBATCH,
    target_mse: float = PPO_ORACLE_BC_TARGET_MSE,
    episode_lengths: list[int] | None = None,
) -> dict[str, float | int]:
    """Warm-start the existing critic on Oracle-path discounted reward-to-go."""
    if observations.ndim != 2 or observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("Oracle critic observations must have shape (N, OBSERVATION_DIM)")
    if rewards.ndim != 1 or len(rewards) != len(observations) or len(rewards) == 0:
        raise ValueError("Oracle critic rewards must be one non-empty reward per observation")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("Oracle critic gamma must be in [0, 1]")
    if episode_lengths is None:
        episode_lengths = [len(observations)]
    if any(length <= 0 for length in episode_lengths) or sum(episode_lengths) != len(observations):
        raise ValueError("Oracle critic episode lengths must exactly partition observations")
    episode_ranges = []
    episode_start = 0
    for episode_length in episode_lengths:
        episode_stop = episode_start + episode_length
        episode_ranges.append((episode_start, episode_stop))
        episode_start = episode_stop

    returns = np.empty_like(rewards, dtype=np.float32)
    for episode_start, episode_stop in episode_ranges:
        running_return = 0.0
        for index in range(episode_stop - 1, episode_start - 1, -1):
            running_return = float(rewards[index]) + gamma * running_return
            returns[index] = running_return

    obs_t = torch.as_tensor(observations, dtype=torch.float32, device=agent.device)
    return_t = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
    if agent.recurrent_enabled:
        critic_parameters = [
            *agent.net.critic_encoder.parameters(),
            *agent.net.critic_gru.parameters(),
            *agent.net.critic.parameters(),
        ]
    else:
        critic_parameters = list(agent.net.critic.parameters())
    optimizer = torch.optim.Adam(critic_parameters, lr=learning_rate)
    rng = np.random.default_rng(seed)

    @torch.inference_mode()
    def full_mse() -> float:
        if agent.recurrent_enabled:
            prediction = torch.cat([
                agent.net.value_sequence(obs_t[start:stop].unsqueeze(0), None)[0].squeeze(0)
                for start, stop in episode_ranges
            ])
        else:
            prediction = agent.net.value(obs_t)
        return float(torch.mean((prediction - return_t) ** 2).cpu())

    initial_mse = full_mse()
    epochs_completed = 0
    best_mse = initial_mse
    best_epoch = 0
    critic_state_prefixes = (
        ("critic_encoder.", "critic_gru.", "critic.")
        if agent.recurrent_enabled
        else ("critic.",)
    )
    best_critic_state = {
        key: value.detach().clone()
        for key, value in agent.net.state_dict().items()
        if key.startswith(critic_state_prefixes)
    }
    last_epoch_mse = initial_mse
    # TODO(IQ-54): compare best-epoch critic restoration against IQ-53's
    # degraded final critic before changing any PPO/environment contract.
    for epoch in range(max_epochs):
        if agent.recurrent_enabled:
            for episode_start, episode_stop in episode_ranges:
                hidden = None
                for start in range(
                    episode_start,
                    episode_stop,
                    agent.recurrent_sequence_length,
                ):
                    stop = min(start + agent.recurrent_sequence_length, episode_stop)
                    prediction, hidden = agent.net.value_sequence(
                        obs_t[start:stop].unsqueeze(0),
                        hidden,
                    )
                    prediction = prediction.squeeze(0)
                    loss = torch.mean((prediction - return_t[start:stop]) ** 2)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    hidden = hidden.detach()
        else:
            indexes = rng.permutation(len(observations))
            for start in range(0, len(indexes), minibatch):
                mb = torch.as_tensor(
                    indexes[start:start + minibatch],
                    dtype=torch.long,
                    device=agent.device,
                )
                prediction = agent.net.value(obs_t[mb])
                loss = torch.mean((prediction - return_t[mb]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        epochs_completed = epoch + 1
        epoch_mse = full_mse()
        last_epoch_mse = epoch_mse
        if epoch_mse < best_mse:
            best_mse = epoch_mse
            best_epoch = epochs_completed
            best_critic_state = {
                key: value.detach().clone()
                for key, value in agent.net.state_dict().items()
                if key.startswith(critic_state_prefixes)
            }
        if epoch_mse <= target_mse:
            break

    current_state = agent.net.state_dict()
    current_state.update(best_critic_state)
    agent.net.load_state_dict(current_state)
    final_mse = full_mse()
    agent._sync_collector()
    return {
        "critic_epochs_completed": int(epochs_completed),
        "critic_initial_mse": initial_mse,
        "critic_final_mse": final_mse,
        "critic_last_epoch_mse": last_epoch_mse,
        "critic_best_epoch": int(best_epoch),
        "critic_best_mse": best_mse,
        "critic_target_mean": float(np.mean(returns)),
        "critic_target_std": float(np.std(returns)),
    }


def _initialize_champion(agent: PPOAgent, validate_cost, checkpoint_path: Path) -> float:
    """Validate and persist Champion #0 before any PPO update can run."""
    champion_cost = float(validate_cost())
    agent.save(checkpoint_path)
    return champion_cost


def _resolve_challenger(
    champion_cost: float,
    candidate_cost: float,
) -> tuple[float, bool]:
    """Keep the trusted Champion monotonic without mutating the live learner."""
    if candidate_cost < champion_cost:
        return candidate_cost, True
    return champion_cost, False


def _calendar_month_is_complete(month: MonthData) -> bool:
    """Return True only when a month contains every calendar day exactly once."""
    if not month.days:
        return False
    dates = [datetime.fromisoformat(str(day.date_iso)).date() for day in month.days]
    year = dates[0].year
    month_number = dates[0].month
    if any(date.year != year or date.month != month_number for date in dates):
        return False
    expected_days = calendar.monthrange(year, month_number)[1]
    return (
        len(dates) == expected_days
        and {date.day for date in dates} == set(range(1, expected_days + 1))
    )


def _complete_calendar_month_blocks(days):
    """Return complete calendar months and separately retain every incomplete month."""
    calendar_months = month_blocks(list(days), minimum_days=1)
    complete_months = []
    incomplete_months = []
    for month in calendar_months:
        target = complete_months if _calendar_month_is_complete(month) else incomplete_months
        target.append(month)
    if not complete_months:
        raise ValueError("Training CSV contains no complete calendar month")
    return complete_months, incomplete_months


def _chronological_month_holdout_split(
    days,
    train_months: int,
    val_months: int,
    test_months: int,
):
    """Dynamically use the latest requested complete months for train/val/test."""
    if train_months < 1 or val_months < 1 or test_months < 1:
        raise ValueError("Training, validation, and test splits must each contain at least 1 month")
    complete_months, incomplete_months = _complete_calendar_month_blocks(days)
    required_months = train_months + val_months + test_months
    if len(complete_months) < required_months:
        raise ValueError(
            "Training CSV must contain at least "
            f"{required_months} complete calendar months for {train_months} training + "
            f"{val_months} validation + {test_months} test; got {len(complete_months)}"
        )

    # IQ-53: real telemetry may have holes inside otherwise useful history. Keep
    # billing math honest by skipping incomplete months, then slide the requested
    # train/validation/test window to the newest complete months automatically.
    # TODO(IQ-53): re-audit the dynamic 5/1/1 split whenever new Youngone months
    # accumulate; never fabricate missing dates merely to make a month complete.
    selected = complete_months[-required_months:]
    older_complete_months = complete_months[:-required_months]
    train_stop = train_months
    val_stop = train_months + val_months

    def _flatten(blocks):
        return [day for month in blocks for day in month.days]

    ignored_months = sorted(
        [*incomplete_months, *older_complete_months],
        key=lambda month: month.source,
    )
    return (
        _flatten(selected[:train_stop]),
        _flatten(selected[train_stop:val_stop]),
        _flatten(selected[val_stop:]),
        _flatten(ignored_months),
    )


def _training_reference_power_kw(train_days) -> float:
    """Derive the observation/reference scale from training data only."""
    if not train_days:
        raise ValueError("Training split must contain at least one day")
    peak = max(float(day.load.max()) for day in train_days)
    return math.ceil(peak / 500.0) * 500.0


def _backtracked_challenger_state(
    champion_state: dict,
    candidate_state: dict,
    *,
    fraction: float,
) -> dict:
    """Interpolate a rejected challenger toward Champion and restart Adam cleanly."""
    if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
        raise ValueError("Backtracking fraction must be finite and strictly between 0 and 1")

    champion_network = champion_state["network"]
    candidate_network = candidate_state["network"]
    if champion_network.keys() != candidate_network.keys():
        raise ValueError("Champion/challenger network states do not match")

    backtracked_network = {}
    for key, champion_value in champion_network.items():
        candidate_value = candidate_network[key]
        if torch.is_tensor(champion_value) and torch.is_tensor(candidate_value):
            if champion_value.shape != candidate_value.shape:
                raise ValueError(f"Champion/challenger tensor shape mismatch for {key}")
            if torch.is_floating_point(champion_value):
                backtracked_network[key] = torch.lerp(
                    champion_value,
                    candidate_value,
                    fraction,
                )
            else:
                backtracked_network[key] = candidate_value.detach().clone()
        else:
            backtracked_network[key] = candidate_value

    candidate_optimizer = candidate_state["optimizer"]
    fresh_optimizer = {
        "state": {},
        "param_groups": [
            {**group, "params": list(group["params"])}
            for group in candidate_optimizer["param_groups"]
        ],
    }
    return {
        "network": backtracked_network,
        # IQ-51 keeps IQ-48's proven fresh-Adam rescue baseline. Quarter-step
        # geometry may start probation, but only a later full PPO update can crown.
        # TODO(IQ-51): keep quarter probation only if remote complete-month evidence
        # generalizes better than IQ-50 without sacrificing the IQ-48 baseline.
        "optimizer": fresh_optimizer,
    }


def _midpoint_challenger_state(champion_state: dict, candidate_state: dict) -> dict:
    """Build the exact 50% Champion/challenger state with fresh Adam."""
    return _backtracked_challenger_state(
        champion_state,
        candidate_state,
        fraction=0.5,
    )


def _should_try_quarter_step(full_candidate_cost: float, midpoint_cost: float) -> bool:
    """Spend a quarter-step validation only when halving improved the direction."""
    return (
        math.isfinite(full_candidate_cost)
        and math.isfinite(midpoint_cost)
        and midpoint_cost < full_candidate_cost
    )


def _should_start_quarter_probation(champion_cost: float, quarter_cost: float) -> bool:
    """Let a quarter-step continue learning only when it already beats Champion."""
    return (
        math.isfinite(champion_cost)
        and math.isfinite(quarter_cost)
        and quarter_cost < champion_cost
    )


def _should_validate_candidate(
    updates: int,
    validate_every_updates: int,
    *,
    quarter_probation_active: bool,
) -> bool:
    """Force the one-update probation exam even outside normal validation cadence."""
    if validate_every_updates < 1:
        raise ValueError("Validation cadence must be at least one update")
    return quarter_probation_active or updates % validate_every_updates == 0


def _should_reanchor_rejected_candidate(
    *,
    resets_enabled: bool,
    allow_reset: bool,
    consecutive_rejections: int,
    reset_patience: int,
) -> bool:
    """Apply the configured Champion re-anchor rule after rejected updates."""
    return (
        resets_enabled
        and allow_reset
        and consecutive_rejections >= reset_patience
    )


def _save_accepted_champion(agent: PPOAgent, checkpoint_path: Path, accepted: bool) -> bool:
    """Persist only accepted learner state; rejected challengers never touch disk."""
    if not accepted:
        return False
    agent.save(checkpoint_path)
    return True


def _restore_reanchor_state(
    agent: PPOAgent,
    champion_state: dict,
    *,
    preserve_critic: bool,
    reset_optimizer: bool,
) -> None:
    """Restore Champion policy while optionally carrying live critic homework forward."""
    live_critic_state = None
    live_critic_encoder_state = None
    live_critic_gru_state = None
    if preserve_critic:
        live_critic_state = {
            key: value.detach().clone()
            for key, value in agent.net.critic.state_dict().items()
        }
        if agent.recurrent_enabled:
            live_critic_encoder_state = {
                key: value.detach().clone()
                for key, value in agent.net.critic_encoder.state_dict().items()
            }
            live_critic_gru_state = {
                key: value.detach().clone()
                for key, value in agent.net.critic_gru.state_dict().items()
            }
    agent.restore_training_state(champion_state)
    if live_critic_state is not None:
        agent.net.critic.load_state_dict(live_critic_state)
        if live_critic_encoder_state is not None:
            agent.net.critic_encoder.load_state_dict(live_critic_encoder_state)
        if live_critic_gru_state is not None:
            agent.net.critic_gru.load_state_dict(live_critic_gru_state)
        agent._sync_collector()
    if reset_optimizer:
        agent.opt.state.clear()


def _champion_curve_point(
    *,
    steps: int,
    candidate_cost: float,
    champion_cost: float,
    accepted: bool,
    val_base: float,
    val_oracle: float,
) -> dict:
    saving = (val_base - champion_cost) / val_base * 100
    gap = (champion_cost - val_oracle) / val_oracle * 100
    return {
        "steps": steps,
        "candidate_val_cost_vnd": candidate_cost,
        "champion_val_cost_vnd": champion_cost,
        "val_cost_vnd": champion_cost,
        "accepted": accepted,
        "oracle_gap_pct": gap,
        "saving_vs_nobess_pct": saving,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--e-cap", type=float, required=True)
    parser.add_argument("--p-rated", type=float, required=True)
    parser.add_argument("--steps", type=int, default=PPO_STEPS)
    parser.add_argument("--seed", type=int, default=PPO_SEED)
    parser.add_argument("--gamma", type=float, default=PPO_GAMMA)
    parser.add_argument("--lambda", dest="lambda_value", type=float, default=PPO_LAMBDA)
    parser.add_argument("--learning-rate", type=float, default=PPO_LEARNING_RATE)
    parser.add_argument(
        "--exploration-lr-multiplier",
        type=float,
        default=PPO_EXPLORATION_LR_MULTIPLIER,
    )
    parser.add_argument(
        "--soc-edge-log-std-penalty",
        type=float,
        default=PPO_SOC_EDGE_LOG_STD_PENALTY,
    )
    parser.add_argument("--ppo-clip", type=float, default=PPO_CLIP)
    parser.add_argument("--ppo-epochs", type=int, default=PPO_FINE_TUNE_EPOCHS)
    parser.add_argument("--minibatch", type=int, default=PPO_MINIBATCH)
    parser.add_argument("--entropy-coef", type=float, default=PPO_ENTROPY_COEF)
    parser.add_argument("--value-coef", type=float, default=PPO_VALUE_COEF)
    parser.add_argument("--target-kl", type=float, default=PPO_TARGET_KL)
    parser.add_argument("--actor-grad-clip", type=float, default=PPO_ACTOR_GRAD_CLIP)
    parser.add_argument("--critic-grad-clip", type=float, default=PPO_CRITIC_GRAD_CLIP)
    parser.add_argument("--hidden-size", type=int, default=PPO_HIDDEN_SIZE)
    parser.add_argument(
        "--recurrent-enabled",
        action=argparse.BooleanOptionalAction,
        default=PPO_RECURRENT_ENABLED,
    )
    parser.add_argument(
        "--recurrent-sequence-length",
        type=int,
        default=PPO_RECURRENT_SEQUENCE_LENGTH,
    )
    parser.add_argument("--initial-log-std", type=float, default=PPO_INITIAL_LOG_STD)
    parser.add_argument("--ppo-start-log-std", type=float, default=PPO_BC_FINE_TUNE_LOG_STD)
    parser.add_argument(
        "--validate-every-updates",
        type=int,
        default=PPO_VALIDATE_EVERY_UPDATES,
    )
    parser.add_argument(
        "--challenger-reset-patience",
        type=int,
        default=PPO_CHALLENGER_RESET_PATIENCE,
    )
    parser.add_argument(
        "--challenger-resets-enabled",
        action=argparse.BooleanOptionalAction,
        default=PPO_CHALLENGER_RESETS_ENABLED,
    )
    parser.add_argument(
        "--reset-optimizer-on-reanchor",
        action=argparse.BooleanOptionalAction,
        default=PPO_RESET_OPTIMIZER_ON_REANCHOR,
    )
    parser.add_argument(
        "--preserve-critic-on-reanchor",
        action=argparse.BooleanOptionalAction,
        default=PPO_PRESERVE_CRITIC_ON_REANCHOR,
    )
    parser.add_argument(
        "--action-mismatch-shaping-scale",
        type=float,
        default=PPO_ACTION_MISMATCH_SHAPING_SCALE,
    )
    parser.add_argument(
        "--oracle-bc-enabled",
        action=argparse.BooleanOptionalAction,
        default=PPO_ORACLE_BC_ENABLED,
    )
    parser.add_argument(
        "--oracle-actor-bc-max-epochs",
        type=int,
        default=PPO_ORACLE_ACTOR_BC_MAX_EPOCHS,
    )
    parser.add_argument(
        "--oracle-bc-max-epochs",
        type=int,
        default=PPO_ORACLE_BC_MAX_EPOCHS,
    )
    parser.add_argument(
        "--oracle-bc-learning-rate",
        type=float,
        default=PPO_ORACLE_BC_LEARNING_RATE,
    )
    parser.add_argument(
        "--oracle-bc-minibatch",
        type=int,
        default=PPO_ORACLE_BC_MINIBATCH,
    )
    parser.add_argument(
        "--oracle-bc-target-mse",
        type=float,
        default=PPO_ORACLE_BC_TARGET_MSE,
    )
    parser.add_argument("--log-every-updates", type=int, default=PPO_LOG_EVERY_UPDATES)
    parser.add_argument("--torch-threads", type=int, default=PPO_TORCH_THREADS)
    parser.add_argument("--control-dt-minutes", type=float, required=True)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--billing", choices=("2tc", "tou"), default="2tc")
    parser.add_argument("--training-config", type=str, required=True)
    parser.add_argument("--oracle-cache", required=True)
    # IQ-53 defaults to a dynamic latest-complete-month 5/1/1 split. This changes
    # data selection only; Brain7, BrainEnv, recurrent architecture, and PPO scalar
    # settings remain untouched.
    parser.add_argument("--train-months", type=int, default=5)
    parser.add_argument("--val-months", type=int, default=1)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--obs-variant", choices=("brain7",), default="brain7")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    def require_float(
        name: str,
        value: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
        minimum_inclusive: bool = True,
    ) -> None:
        if not math.isfinite(value):
            raise SystemExit(f"{name} must be finite")
        if minimum is not None:
            below = value < minimum if minimum_inclusive else value <= minimum
            if below:
                bracket = "[" if minimum_inclusive else "("
                raise SystemExit(f"{name} must be in {bracket}{minimum}, ...")
        if maximum is not None and value > maximum:
            raise SystemExit(f"{name} must be <= {maximum}")

    if args.steps < 1:
        raise SystemExit("steps must be >= 1")
    if args.ppo_epochs < 1:
        raise SystemExit("ppo-epochs must be >= 1")
    if args.minibatch < 1:
        raise SystemExit("minibatch must be >= 1")
    if args.hidden_size < 1:
        raise SystemExit("hidden-size must be >= 1")
    if args.recurrent_sequence_length < 1:
        raise SystemExit("recurrent-sequence-length must be >= 1")
    if args.validate_every_updates < 1:
        raise SystemExit("validate-every-updates must be >= 1")
    if args.challenger_reset_patience < 1:
        raise SystemExit("challenger-reset-patience must be >= 1")
    if args.oracle_actor_bc_max_epochs < 0:
        raise SystemExit("oracle-actor-bc-max-epochs must be >= 0")
    if args.oracle_bc_max_epochs < 0:
        raise SystemExit("oracle-bc-max-epochs must be >= 0")
    if args.oracle_bc_minibatch < 1:
        raise SystemExit("oracle-bc-minibatch must be >= 1")
    if args.log_every_updates < 1:
        raise SystemExit("log-every-updates must be >= 1")
    if not 1 <= args.torch_threads <= 128:
        raise SystemExit("torch-threads must be in [1, 128]")
    if args.preserve_critic_on_reanchor and not args.reset_optimizer_on_reanchor:
        raise SystemExit(
            "preserve-critic-on-reanchor requires reset-optimizer-on-reanchor so "
            "live critic weights never inherit stale Champion Adam moments"
        )

    require_float("gamma", args.gamma, minimum=0.0, maximum=1.0, minimum_inclusive=False)
    require_float("lambda", args.lambda_value, minimum=0.0, maximum=1.0)
    require_float(
        "learning-rate",
        args.learning_rate,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    require_float(
        "exploration-lr-multiplier",
        args.exploration_lr_multiplier,
        minimum=0.0,
        maximum=1000.0,
        minimum_inclusive=False,
    )
    require_float(
        "soc-edge-log-std-penalty",
        args.soc_edge_log_std_penalty,
        minimum=0.0,
        maximum=5.0,
    )
    require_float("ppo-clip", args.ppo_clip, minimum=0.0, maximum=1.0, minimum_inclusive=False)
    require_float("entropy-coef", args.entropy_coef, minimum=0.0, maximum=100.0)
    require_float("value-coef", args.value_coef, minimum=0.0, maximum=100.0)
    require_float("target-kl", args.target_kl, minimum=0.0, maximum=100.0, minimum_inclusive=False)
    require_float(
        "actor-grad-clip",
        args.actor_grad_clip,
        minimum=0.0,
        maximum=1e6,
        minimum_inclusive=False,
    )
    require_float(
        "critic-grad-clip",
        args.critic_grad_clip,
        minimum=0.0,
        maximum=1e6,
        minimum_inclusive=False,
    )
    require_float("initial-log-std", args.initial_log_std, minimum=-20.0, maximum=5.0)
    require_float("ppo-start-log-std", args.ppo_start_log_std, minimum=-20.0, maximum=5.0)
    require_float(
        "action-mismatch-shaping-scale",
        args.action_mismatch_shaping_scale,
        minimum=0.0,
        maximum=100.0,
    )
    require_float(
        "oracle-bc-learning-rate",
        args.oracle_bc_learning_rate,
        minimum=0.0,
        maximum=1.0,
        minimum_inclusive=False,
    )
    require_float(
        "oracle-bc-target-mse",
        args.oracle_bc_target_mse,
        minimum=0.0,
        maximum=1e9,
    )
    if not math.isclose(
        args.control_dt_minutes,
        PPO_FIT_CONTROL_DT_MINUTES,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise SystemExit(
            "Generic PPO fit mode requires 30-minute control aligned to demand-meter blocks"
        )

    torch.set_num_threads(args.torch_threads)
    configure_ppo_determinism(args.seed)

    days = load_training_days(args.csv, weather="csv")
    if not days:
        raise SystemExit("Training CSV contains no usable days")
    csv_dt = 24.0 / len(days[0].load)

    try:
        train_days, val_days, test_days, ignored_split_days = (
            _chronological_month_holdout_split(
                days,
                args.train_months,
                args.val_months,
                args.test_months,
            )
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    ignored_split_months = sorted({str(day.date_iso)[:7] for day in ignored_split_days})
    # Training-only scale: validation/test peaks must not leak into the policy's
    # observation normalization or reference power.
    p_ref = _training_reference_power_kw(train_days)

    cfg, billing = build_training_bess_config(
        args.e_cap,
        args.p_rated,
        csv_dt,
        args.training_config,
        default_billing=args.billing,
    )
    battery_wear_cost = _load_ui_wear_cost(args.training_config)

    tag = args.tag or f"ds_{args.e_cap:.0f}kwh_{args.p_rated:.0f}kw"
    if billing == "tou" and not tag.endswith("_tou"):
        tag += "_tou"

    # IQ-46 real-data mode: training days stay chronological but reset BrainEnv at
    # calendar-month boundaries. Validation/test are newer, disjoint holdouts.
    train_months = month_blocks(train_days, minimum_days=1)
    if not train_months:
        raise SystemExit("Training split contains no usable calendar-month episode")
    validation_months = month_blocks(val_days, minimum_days=1)
    test_months = month_blocks(test_days, minimum_days=1)
    if (
        len(train_months) != args.train_months
        or len(validation_months) != args.val_months
        or len(test_months) != args.test_months
    ):
        raise RuntimeError("Dynamic complete-month split lost a calendar block")
    gamma = args.gamma
    native_steps = native_steps_per_action(cfg.dt, args.control_dt_minutes)
    decisions_per_day = len(days[0].load) // native_steps
    rollout_decisions = decisions_per_day * max(
        len(month.days) for month in train_months
    )
    learner_device = resolve_ppo_device(args.device)
    agent = PPOAgent(
        OBSERVATION_DIM,
        lr=args.learning_rate,
        gamma=gamma,
        lam=args.lambda_value,
        clip=args.ppo_clip,
        epochs=args.ppo_epochs,
        minibatch=args.minibatch,
        ent_coef=args.entropy_coef,
        vf_coef=args.value_coef,
        target_kl=args.target_kl,
        seed=args.seed,
        device=learner_device,
        hidden_size=args.hidden_size,
        initial_log_std=args.initial_log_std,
        exploration_lr_multiplier=args.exploration_lr_multiplier,
        recurrent_enabled=args.recurrent_enabled,
        recurrent_sequence_length=args.recurrent_sequence_length,
        soc_edge_log_std_penalty=args.soc_edge_log_std_penalty,
        actor_grad_clip=args.actor_grad_clip,
        critic_grad_clip=args.critic_grad_clip,
    )
    reanchor_scope = (
        "champion_actor_log_std_keep_live_critic_fresh_adam"
        if args.preserve_critic_on_reanchor
        else REANCHOR_SCOPE
    )
    agent.meta = {
        "p_ref_kw": p_ref,
        "e_cap_kwh": args.e_cap,
        "p_rated_kw": args.p_rated,
        "obs_variant": "brain7",
        "obs_dim": OBSERVATION_DIM,
        "battery_wear_cost": battery_wear_cost,
        "reward_mode": "brain_savings_vnd_v1",
        "training_reward_shaping": "infeasible_request_phantom_wear_scaled_v2",
        "training_reward_shaping_scale": args.action_mismatch_shaping_scale,
        "initial_soc": float(cfg.SOC_min),
        "gamma": gamma,
        "lambda": args.lambda_value,
        "learning_rate": float(agent.opt.param_groups[0]["lr"]),
        "exploration_lr_multiplier": agent.exploration_lr_multiplier,
        "exploration_learning_rate": agent.exploration_learning_rate,
        "soc_edge_log_std_penalty": agent.soc_edge_log_std_penalty,
        "ppo_clip": agent.clip,
        "ppo_epochs": agent.epochs,
        "ppo_minibatch": agent.minibatch,
        "entropy_coef": agent.ent_coef,
        "value_coef": agent.vf_coef,
        "target_kl": agent.target_kl,
        "actor_grad_clip": agent.actor_grad_clip,
        "critic_grad_clip": agent.critic_grad_clip,
        "hidden_size": agent.hidden_size,
        "recurrent_enabled": agent.recurrent_enabled,
        "recurrent_sequence_length": agent.recurrent_sequence_length,
        "policy_architecture": (
            "brain7_separate_actor_critic_gru_v1"
            if agent.recurrent_enabled
            else "brain7_feedforward_mlp_v1"
        ),
        "initial_log_std": agent.initial_log_std,
        "exploration_mode": "state_dependent_log_std_delta_soc_edge_v2",
        "exploration_hidden_size": agent.net.exploration_hidden_size,
        "native_dt_minutes": csv_dt * 60.0,
        "control_dt_minutes": args.control_dt_minutes,
        "native_steps_per_action": native_steps,
        "billing_mode": billing,
        "device_requested": args.device,
        "device": learner_device,
        "seed": args.seed,
        "deterministic_training": True,
        "train_csv": str(args.csv),
        "train_range": [train_days[0].date_iso, train_days[-1].date_iso],
        "validation_range": [val_days[0].date_iso, val_days[-1].date_iso],
        "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
        "temporary_full_dataset_overlap": False,
        "data_overlap_note": "dynamic latest complete calendar-month split; incomplete months are skipped",
        "ignored_unselected_days": len(ignored_split_days),
        "ignored_unselected_months": ignored_split_months,
        "validation_month_count": len(validation_months),
        "test_month_count": len(test_months),
        "training_augmentation_enabled": False,
        "rollout_decisions": rollout_decisions,
        "rollout_mode": "one_calendar_month_per_update",
        "training_month_count": len(train_months),
        "validation_every_updates": args.validate_every_updates,
        "challenger_reset_patience": args.challenger_reset_patience,
        "challenger_resets_enabled": args.challenger_resets_enabled,
        "reset_optimizer_on_reanchor": args.reset_optimizer_on_reanchor,
        "preserve_critic_on_reanchor": args.preserve_critic_on_reanchor,
        "reanchor_scope": reanchor_scope,
        "rejected_midpoint_rescue": "conditional_quarter_probation_fresh_adam_v5",
        "oracle_behavior_cloning_enabled": args.oracle_bc_enabled,
        "oracle_actor_behavior_cloning_max_epochs": args.oracle_actor_bc_max_epochs,
        "oracle_behavior_cloning_max_epochs": args.oracle_bc_max_epochs,
        "oracle_behavior_cloning_learning_rate": args.oracle_bc_learning_rate,
        "oracle_behavior_cloning_minibatch": args.oracle_bc_minibatch,
        "oracle_behavior_cloning_target_mse": args.oracle_bc_target_mse,
        "ppo_start_log_std_configured": args.ppo_start_log_std,
        "log_every_updates": args.log_every_updates,
        "torch_threads": args.torch_threads,
    }
    buffer = RolloutBuffer(
        rollout_decisions,
        OBSERVATION_DIM,
        recurrent_hidden_size=agent.hidden_size if agent.recurrent_enabled else 0,
    )

    val_base = sum(
        score_month(
            run_no_bess(month, cfg)["p_grid_days"],
            cfg,
            days=month.days,
        )["total_cost_vnd"]
        for month in validation_months
    )
    val_day_indexes = [day.day_index for day in val_days]
    train_day_indexes = [day.day_index for day in train_days]
    oracle_grids = load_cached_training_grids(args.oracle_cache, val_day_indexes)
    oracle_dispatch = load_cached_training_dispatch(args.oracle_cache, train_day_indexes)
    val_oracle_dispatch = (
        oracle_dispatch
        if train_day_indexes == val_day_indexes
        else load_cached_training_dispatch(args.oracle_cache, val_day_indexes)
    )
    val_oracle_utility = _score_grid_calendar_months(
        oracle_grids,
        validation_months,
        cfg,
    )
    val_oracle_wear = _oracle_dispatch_wear_cost_vnd(
        val_oracle_dispatch,
        timestep_hours=cfg.dt,
        wear_vnd_per_kwh=battery_wear_cost,
    )
    val_oracle = val_oracle_utility + val_oracle_wear
    curve_path = RESULTS_DIR / f"training_curve_{tag}.csv"
    report_path = RESULTS_DIR / f"training_report_{tag}.json"
    report = {
        "version": 1,
        "status": "running",
        "algorithm": "ppo",
        "tag": tag,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "source": str(args.csv),
            "total_days": len(days),
            "train_days": len(train_days),
            "validation_days": len(val_days),
            "test_days": len(test_days),
            "train_range": [train_days[0].date_iso, train_days[-1].date_iso],
            "validation_range": [val_days[0].date_iso, val_days[-1].date_iso],
            "test_range": [test_days[0].date_iso, test_days[-1].date_iso],
            "temporary_full_dataset_overlap": False,
            "split_mode": "dynamic_latest_complete_calendar_months",
            "ignored_unselected_days": len(ignored_split_days),
            "ignored_unselected_months": ignored_split_months,
            "training_months": len(train_months),
            "validation_months": len(validation_months),
            "test_months": len(test_months),
        },
        "battery": {"e_cap_kwh": args.e_cap, "p_rated_kw": args.p_rated},
        "economics": {
            "battery_wear_cost": battery_wear_cost,
            "reward_mode": "brain_savings_vnd_v1",
            "training_reward_shaping": "infeasible_request_phantom_wear_scaled_v2",
            "training_reward_shaping_scale": args.action_mismatch_shaping_scale,
            "champion_scoring_uses_shaping": False,
        },
        "training": {
            "requested_steps": args.steps,
            "seed": args.seed,
            "gamma": gamma,
            "lambda": args.lambda_value,
            "learning_rate": float(agent.opt.param_groups[0]["lr"]),
            "exploration_lr_multiplier": agent.exploration_lr_multiplier,
            "exploration_learning_rate": agent.exploration_learning_rate,
            "soc_edge_log_std_penalty": agent.soc_edge_log_std_penalty,
            "ppo_clip": agent.clip,
            "ppo_epochs": agent.epochs,
            "ppo_minibatch": agent.minibatch,
            "entropy_coef": agent.ent_coef,
            "value_coef": agent.vf_coef,
            "target_kl": agent.target_kl,
            "actor_grad_clip": agent.actor_grad_clip,
            "critic_grad_clip": agent.critic_grad_clip,
            "hidden_size": agent.hidden_size,
            "recurrent_enabled": agent.recurrent_enabled,
            "recurrent_sequence_length": agent.recurrent_sequence_length,
            "policy_architecture": (
                "brain7_separate_actor_critic_gru_v1"
                if agent.recurrent_enabled
                else "brain7_feedforward_mlp_v1"
            ),
            "initial_log_std": agent.initial_log_std,
            "exploration_mode": "state_dependent_log_std_delta_soc_edge_v2",
            "exploration_hidden_size": agent.net.exploration_hidden_size,
            "initial_soc": float(cfg.SOC_min),
            "rollout_days": max(len(month.days) for month in train_months),
            "rollout_decisions": rollout_decisions,
            "rollout_mode": "one_calendar_month_per_update",
            "training_month_count": len(train_months),
            "augmentation_enabled": False,
            "validation_every_updates": args.validate_every_updates,
            "challenger_reset_patience": args.challenger_reset_patience,
            "challenger_resets_enabled": args.challenger_resets_enabled,
            "reset_optimizer_on_reanchor": args.reset_optimizer_on_reanchor,
            "preserve_critic_on_reanchor": args.preserve_critic_on_reanchor,
            "reanchor_scope": reanchor_scope,
            "rejected_midpoint_rescue": "conditional_quarter_probation_fresh_adam_v5",
            "midpoint_rescue_evaluations": 0,
            "midpoint_rescues": 0,
            "quarter_step_evaluations": 0,
            "quarter_step_rescues": 0,
            "quarter_step_probations_started": 0,
            "quarter_step_probations_passed": 0,
            "quarter_step_probations_failed": 0,
            "quarter_step_probations_expired": 0,
            "quarter_step_probation_active": False,
            "oracle_behavior_cloning_enabled": args.oracle_bc_enabled,
            "oracle_actor_behavior_cloning_max_epochs": args.oracle_actor_bc_max_epochs,
            "oracle_behavior_cloning_max_epochs": args.oracle_bc_max_epochs,
            "oracle_behavior_cloning_learning_rate": args.oracle_bc_learning_rate,
            "oracle_behavior_cloning_minibatch": args.oracle_bc_minibatch,
            "oracle_behavior_cloning_target_mse": args.oracle_bc_target_mse,
            "ppo_start_log_std_configured": args.ppo_start_log_std,
            "log_every_updates": args.log_every_updates,
            "torch_threads": args.torch_threads,
            "native_dt_minutes": csv_dt * 60.0,
            "control_dt_minutes": args.control_dt_minutes,
            "native_steps_per_action": native_steps,
            "device_requested": args.device,
            "device": learner_device,
            "updates": 0,
            "candidate_evaluations": 0,
            "accepted_updates": 0,
            "rejected_updates": 0,
            "learner_resets": 0,
            "acceptance_rate_pct": 0.0,
        },
        "billing_mode": billing,
        "p_ref_kw": p_ref,
        "validation": {
            "no_bess_vnd": val_base,
            "oracle_vnd": val_oracle,
            "oracle_utility_vnd": val_oracle_utility,
            "oracle_wear_vnd": val_oracle_wear,
        },
    }
    write_curve(curve_path, [], fields=PPO_CHAMPION_CURVE_FIELDS)
    write_report(report_path, report)
    print(
        f"[train-ds] COMPLETE CALENDAR-MONTH HOLDOUT | {len(days)} source days | "
        f"train {len(train_months)}mo/{len(train_days)}d | "
        f"val {len(validation_months)}mo/{len(val_days)}d | "
        f"test {len(test_months)}mo/{len(test_days)}d | "
        f"ignored/unselected {len(ignored_split_days)}d | "
        f"gamma {gamma:g} | lambda {args.lambda_value:g} | "
        f"UI wear {battery_wear_cost:g} VND/kWh | "
        f"learner {learner_device} (requested {args.device}) | BrainEnv eyes={OBSERVATION_DIM} | "
        f"native dt {csv_dt * 60:g}m | control dt {args.control_dt_minutes:g}m | "
        f"p_ref {p_ref:.0f} | val no-BESS {val_base/1e6:.0f}M, oracle {val_oracle/1e6:.0f}M",
        flush=True,
    )

    curve = []
    perf = {
        "rollout": 0.0,
        "update": 0.0,
        "validation": 0.0,
        "scoring": 0.0,
        "io": 0.0,
        "checkpoint": 0.0,
        "decisions": 0,
        "native_rows": 0,
    }
    checkpoint_path = RESULTS_DIR / f"policy_{tag}.pt"

    def validate_policy_cost() -> float:
        validation_started = time.perf_counter()
        results = [
            (month, run_drl_policy(month, cfg, agent, p_ref_kw=p_ref))
            for month in validation_months
        ]
        perf["validation"] += time.perf_counter() - validation_started
        scoring_started = time.perf_counter()
        val_cost = sum(
            _score_ppo_operating_month(
                result["p_grid_days"],
                result["p_bess_days"],
                cfg,
                battery_wear_cost,
                month.days,
            )["total_operating_cost_vnd"]
            for month, result in results
        )
        perf["scoring"] += time.perf_counter() - scoring_started
        return val_cost

    def persist_progress() -> None:
        started_io = time.perf_counter()
        write_curve(curve_path, curve, fields=PPO_CHAMPION_CURVE_FIELDS)
        if curve:
            report["latest"] = curve[-1]
            if (
                "best" not in report
                or curve[-1]["val_cost_vnd"] < report["best"]["val_cost_vnd"]
            ):
                report["best"] = curve[-1]
        write_report(report_path, report)
        perf["io"] += time.perf_counter() - started_io

    def print_performance() -> None:
        rollout_seconds = max(perf["rollout"], 1e-9)
        print(
            f"  perf rollout {perf['rollout']:.2f}s | "
            f"ppo {perf['update']:.2f}s | validate {perf['validation']:.2f}s | "
            f"score {perf['scoring']:.2f}s | io {perf['io']:.2f}s | "
            f"checkpoint {perf['checkpoint']:.2f}s | "
            f"{perf['native_rows']/rollout_seconds:,.0f} native rows/s | "
            f"{perf['decisions']/rollout_seconds:,.0f} decisions/s",
            flush=True,
        )
        for key in perf:
            perf[key] = 0 if key in ("decisions", "native_rows") else 0.0

    def make_training_env(month):
        return make_brain_env(
            month,
            cfg,
            power_scale_kw=p_ref,
            battery_wear_vnd_per_kwh=battery_wear_cost,
        )

    raw_initial_cost = validate_policy_cost()
    raw_initial_state = agent.snapshot_training_state()
    bc_stats: dict[str, float | int | bool] = {
        "enabled": bool(args.oracle_bc_enabled),
    }
    post_bc_cost = raw_initial_cost
    use_bc = False

    if args.oracle_bc_enabled:
        observation_parts = []
        target_parts = []
        reward_parts = []
        teacher_episode_lengths = []
        dispatch_cursor = 0
        for train_month in train_months:
            month_day_count = len(train_month.days)
            month_dispatch = oracle_dispatch[
                dispatch_cursor:dispatch_cursor + month_day_count
            ]
            dispatch_cursor += month_day_count
            month_observations, month_targets, month_rewards = _collect_oracle_teacher_samples(
                train_month,
                month_dispatch,
                cfg,
                power_scale_kw=p_ref,
                battery_wear_cost=battery_wear_cost,
                native_steps=native_steps,
            )
            observation_parts.append(month_observations)
            target_parts.append(month_targets)
            reward_parts.append(month_rewards)
            teacher_episode_lengths.append(len(month_observations))
        if dispatch_cursor != len(oracle_dispatch):
            raise RuntimeError("Oracle teacher dispatch was not consumed exactly once")
        teacher_observations = np.concatenate(observation_parts, axis=0)
        teacher_targets = np.concatenate(target_parts, axis=0)
        teacher_rewards = np.concatenate(reward_parts, axis=0)
        bc_started = time.perf_counter()
        bc_stats.update(
            _behavior_clone_actor(
                agent,
                teacher_observations,
                teacher_targets,
                seed=args.seed,
                max_epochs=args.oracle_actor_bc_max_epochs,
                learning_rate=args.oracle_bc_learning_rate,
                minibatch=args.oracle_bc_minibatch,
                target_mse=args.oracle_bc_target_mse,
                score_policy_cost=validate_policy_cost if agent.recurrent_enabled else None,
                episode_lengths=teacher_episode_lengths,
            )
        )
        bc_stats.update(
            _behavior_clone_critic(
                agent,
                teacher_observations,
                teacher_rewards,
                gamma=args.gamma,
                seed=args.seed,
                max_epochs=args.oracle_bc_max_epochs,
                learning_rate=args.oracle_bc_learning_rate,
                minibatch=args.oracle_bc_minibatch,
                target_mse=args.oracle_bc_target_mse,
                episode_lengths=teacher_episode_lengths,
            )
        )
        bc_stats["seconds"] = time.perf_counter() - bc_started
        bc_stats["teacher_mean_abs_action"] = float(np.mean(np.abs(teacher_targets)))
        bc_stats["teacher_nonzero_action_pct"] = float(
            np.mean(np.abs(teacher_targets) > 1e-6) * 100.0
        )
        post_bc_cost = validate_policy_cost()
        use_bc = post_bc_cost < raw_initial_cost
        if use_bc:
            # Keep the strong deterministic teacher actor, but let the UI own the
            # stochastic fine-tuning radius instead of hiding it in source code.
            with torch.no_grad():
                agent.net.log_std.fill_(args.ppo_start_log_std)
            agent._sync_collector()
        else:
            agent.restore_training_state(raw_initial_state)
    else:
        agent.restore_training_state(raw_initial_state)

    bc_stats["ppo_start_log_std"] = float(agent.net.log_std.detach().cpu().item())
    bc_stats["ppo_start_action_std"] = float(agent.net.log_std.detach().exp().cpu().item())

    agent.meta["oracle_behavior_cloning"] = {
        **bc_stats,
        "raw_initial_validation_cost_vnd": raw_initial_cost,
        "post_bc_validation_cost_vnd": post_bc_cost,
        "selected_for_ppo": use_bc,
    }
    report["training"]["oracle_behavior_cloning"] = dict(
        agent.meta["oracle_behavior_cloning"]
    )
    write_report(report_path, report)
    if args.oracle_bc_enabled:
        print(
            f"[train-ds] ORACLE TEACHER | {bc_stats['samples']} lessons | "
            f"actor MSE {bc_stats['initial_mse']:.5f}->{bc_stats['final_mse']:.5f} | "
            f"critic MSE {bc_stats['critic_initial_mse']:.3f}->{bc_stats['critic_final_mse']:.3f} | "
            f"raw {raw_initial_cost/1e6:.1f}M -> BC {post_bc_cost/1e6:.1f}M | "
            f"{'USE BC' if use_bc else 'KEEP RAW'}",
            flush=True,
        )
    else:
        print(
            f"[train-ds] ORACLE TEACHER DISABLED | raw {raw_initial_cost/1e6:.1f}M | KEEP RAW",
            flush=True,
        )

    champion_cost = _initialize_champion(agent, validate_policy_cost, checkpoint_path)
    champion_state = agent.snapshot_training_state()
    initial_point = _champion_curve_point(
        steps=0,
        candidate_cost=champion_cost,
        champion_cost=champion_cost,
        accepted=True,
        val_base=val_base,
        val_oracle=val_oracle,
    )
    curve.append(initial_point)
    persist_progress()
    print(
        f"  champion #0 | step {0:>7} | val {champion_cost/1e6:8.1f}M | "
        f"saving {initial_point['saving_vs_nobess_pct']:5.1f}% | "
        f"gap {initial_point['oracle_gap_pct']:6.1f}% | initial checkpoint",
        flush=True,
    )

    # One PPO rollout/update is one calendar billing month. Cycle training months
    # chronologically and reset BrainEnv/recurrent memory at every month boundary.
    train_month_cursor = 0
    env = make_training_env(train_months[train_month_cursor])
    obs = observation_array(env.reset())
    agent.reset_recurrent_state()
    steps = 0
    updates = 0
    candidate_evaluations = 0
    consecutive_rejections = 0
    last_validated_update = 0
    quarter_probation_active = False
    rollout_action_sum = 0.0
    rollout_action_abs_sum = 0.0
    rollout_action_saturation = 0
    rollout_projected = 0
    rollout_soc_counts = [0, 0, 0]
    rollout_soc_projected = [0, 0, 0]
    rollout_mismatch_penalty_vnd = 0.0
    rollout_mismatch_kwh = 0.0
    rollout_count = 0
    started = time.time()
    rollout_started = time.perf_counter()

    def evaluate_live_candidate(*, allow_reset: bool) -> tuple[dict, float, bool, bool]:
        nonlocal champion_cost
        nonlocal champion_state
        nonlocal candidate_evaluations
        nonlocal consecutive_rejections
        nonlocal last_validated_update
        nonlocal quarter_probation_active

        candidate_evaluations += 1
        probation_follow_up = quarter_probation_active
        if probation_follow_up:
            quarter_probation_active = False
            report["training"]["quarter_step_probation_active"] = False

        full_candidate_cost = validate_policy_cost()
        report["training"]["last_full_candidate_cost_vnd"] = full_candidate_cost
        champion_cost, accepted = _resolve_challenger(
            champion_cost,
            full_candidate_cost,
        )
        candidate_cost = full_candidate_cost
        reset_to_champion = False
        should_reanchor = False

        if probation_follow_up:
            if accepted:
                report["training"]["quarter_step_probations_passed"] += 1
                report["training"]["quarter_step_rescues"] += 1
                report["training"]["last_quarter_probation_outcome"] = "passed"
            else:
                # IQ-51 probation gets exactly one ordinary PPO update. A failed
                # follow-up cannot line-search again; the sacred Champion wins.
                should_reanchor = True
                report["training"]["quarter_step_probations_failed"] += 1
                report["training"]["last_quarter_probation_outcome"] = "failed"
        elif not accepted:
            next_rejection_count = consecutive_rejections + 1
            should_reanchor = _should_reanchor_rejected_candidate(
                resets_enabled=args.challenger_resets_enabled,
                allow_reset=allow_reset,
                consecutive_rejections=next_rejection_count,
                reset_patience=args.challenger_reset_patience,
            )
            if should_reanchor:
                # IQ-45: before throwing a rejected PPO direction away, test the
                # exact midpoint. Recurrent closed-loop cost is extremely sensitive,
                # so a useful direction can still overshoot at the full Adam step.
                candidate_state = agent.snapshot_training_state()
                midpoint_state = _midpoint_challenger_state(
                    champion_state,
                    candidate_state,
                )
                agent.restore_training_state(midpoint_state)
                report["training"]["midpoint_rescue_evaluations"] += 1
                midpoint_cost = validate_policy_cost()
                report["training"]["last_midpoint_candidate_cost_vnd"] = midpoint_cost
                rescued_cost, rescued = _resolve_challenger(
                    champion_cost,
                    midpoint_cost,
                )
                if rescued:
                    champion_cost = rescued_cost
                    candidate_cost = midpoint_cost
                    accepted = True
                    report["training"]["midpoint_rescues"] += 1
                elif _should_try_quarter_step(full_candidate_cost, midpoint_cost):
                    quarter_state = _backtracked_challenger_state(
                        champion_state,
                        candidate_state,
                        fraction=0.25,
                    )
                    agent.restore_training_state(quarter_state)
                    report["training"]["quarter_step_evaluations"] += 1
                    quarter_cost = validate_policy_cost()
                    report["training"]["last_quarter_candidate_cost_vnd"] = quarter_cost
                    if _should_start_quarter_probation(champion_cost, quarter_cost):
                        # IQ-51: the quarter-step may borrow the learner for exactly
                        # one PPO update, but it cannot touch Champion/checkpoint yet.
                        quarter_probation_active = True
                        should_reanchor = False
                        report["training"]["quarter_step_probations_started"] += 1
                        report["training"]["quarter_step_probation_active"] = True
                        report["training"]["last_quarter_probation_outcome"] = "started"
                        report["training"]["last_quarter_probation_entry_cost_vnd"] = quarter_cost

        if accepted:
            report["training"]["accepted_updates"] += 1
            consecutive_rejections = 0
            champion_state = agent.snapshot_training_state()
            checkpoint_started = time.perf_counter()
            if _save_accepted_champion(agent, checkpoint_path, accepted=True):
                perf["checkpoint"] += time.perf_counter() - checkpoint_started
        else:
            report["training"]["rejected_updates"] += 1
            consecutive_rejections += 1
            if should_reanchor:
                _restore_reanchor_state(
                    agent,
                    champion_state,
                    preserve_critic=args.preserve_critic_on_reanchor,
                    reset_optimizer=args.reset_optimizer_on_reanchor,
                )
                report["training"]["learner_resets"] += 1
                consecutive_rejections = 0
                reset_to_champion = True

        last_validated_update = updates
        report["training"]["candidate_evaluations"] = candidate_evaluations
        report["training"]["consecutive_rejections"] = consecutive_rejections
        report["training"]["acceptance_rate_pct"] = (
            report["training"]["accepted_updates"] / candidate_evaluations * 100.0
        )
        point = _champion_curve_point(
            steps=steps,
            candidate_cost=candidate_cost,
            champion_cost=champion_cost,
            accepted=accepted,
            val_base=val_base,
            val_oracle=val_oracle,
        )
        curve.append(point)
        persist_progress()
        return point, candidate_cost, accepted, reset_to_champion

    while steps < args.steps:
        action, logp, latent, value = agent.act_with_latent(obs)
        transition = step_brain_control(
            env,
            action,
            native_steps=native_steps,
        )
        done = transition.done
        mismatch_penalty_vnd = _action_mismatch_penalty_vnd(
            transition,
            timestep_hours=cfg.dt,
            wear_vnd_per_kwh=battery_wear_cost,
        )
        mismatch_kwh = (
            mismatch_penalty_vnd / battery_wear_cost
            if battery_wear_cost > 0.0
            else 0.0
        )
        # Keep the PPO action/log-prob pair exact. We shape only the learning
        # reward so repeatedly requesting battery power that physics rejects is
        # no longer free. Champion/test evaluation never includes this penalty.
        reward = (
            transition.reward_million_vnd
            - args.action_mismatch_shaping_scale * mismatch_penalty_vnd / REWARD_SCALE_VND
        )
        actor_hidden, critic_hidden = agent.recurrent_rollout_inputs()
        buffer.add(
            obs,
            action,
            logp,
            reward,
            value,
            float(done),
            latent=latent,
            actor_hidden=actor_hidden,
            critic_hidden=critic_hidden,
        )
        steps += 1
        perf["decisions"] += 1
        perf["native_rows"] += len(transition.native_results)
        rollout_action_sum += action
        rollout_action_abs_sum += abs(action)
        rollout_action_saturation += int(abs(action) >= 0.98)
        projected = int(transition.adjusted_action)
        rollout_projected += projected
        soc_bin = 0 if obs[3] < (1.0 / 3.0) else (2 if obs[3] > (2.0 / 3.0) else 1)
        rollout_soc_counts[soc_bin] += 1
        rollout_soc_projected[soc_bin] += projected
        rollout_mismatch_penalty_vnd += mismatch_penalty_vnd
        rollout_mismatch_kwh += mismatch_kwh
        rollout_count += 1

        if done:
            agent.reset_recurrent_state()
            train_month_cursor = (train_month_cursor + 1) % len(train_months)
            env = make_training_env(train_months[train_month_cursor])
            next_obs = observation_array(env.reset())
        else:
            if transition.next_observation is None:
                raise RuntimeError("BrainEnv omitted next observation before episode end")
            next_obs = observation_array(transition.next_observation)
        obs = next_obs
        if not done and not buffer.full():
            continue

        perf["rollout"] += time.perf_counter() - rollout_started
        updates += 1
        last_value = agent.estimate_value(obs)
        update_started = time.perf_counter()
        update_stats = agent.update(buffer, 0.0 if done else last_value)
        perf["update"] += time.perf_counter() - update_started
        soc_projection_pct = [
            rollout_soc_projected[index] / count * 100.0 if count else 0.0
            for index, count in enumerate(rollout_soc_counts)
        ]
        rollout_stats = {
            "mean_action": rollout_action_sum / rollout_count,
            "mean_abs_action": rollout_action_abs_sum / rollout_count,
            "action_saturation_pct": rollout_action_saturation / rollout_count * 100.0,
            "projected_action_pct": rollout_projected / rollout_count * 100.0,
            "projected_action_pct_soc_low": soc_projection_pct[0],
            "projected_action_pct_soc_middle": soc_projection_pct[1],
            "projected_action_pct_soc_high": soc_projection_pct[2],
            "action_mismatch_kwh": rollout_mismatch_kwh,
            "action_mismatch_penalty_vnd": rollout_mismatch_penalty_vnd,
            "action_mismatch_shaping_penalty_vnd": (
                args.action_mismatch_shaping_scale * rollout_mismatch_penalty_vnd
            ),
            "mean_action_mismatch_penalty_vnd": rollout_mismatch_penalty_vnd / rollout_count,
        }
        report["training"]["updates"] = updates
        report["training"]["last_update"] = {**update_stats, **rollout_stats}

        should_validate = _should_validate_candidate(
            updates,
            args.validate_every_updates,
            quarter_probation_active=quarter_probation_active,
        )
        should_log = updates == 1 or updates % args.log_every_updates == 0
        if should_validate:
            point, candidate_cost, accepted, reset_to_champion = evaluate_live_candidate(
                allow_reset=True
            )
            if should_log:
                if report["training"]["quarter_step_probation_active"]:
                    verdict = "PROBATION 🧪"
                else:
                    verdict = "ACCEPTED 👑" if accepted else "REJECTED 💀"
                reset_note = " | RESET→CHAMPION" if reset_to_champion else ""
                print(
                    f"  update {updates:>4} | step {steps:>7} | "
                    f"candidate {candidate_cost/1e6:8.1f}M | champion {champion_cost/1e6:8.1f}M | "
                    f"{verdict}{reset_note} | saving {point['saving_vs_nobess_pct']:5.1f}% | "
                    f"gap {point['oracle_gap_pct']:6.1f}% | {steps/(time.time()-started):,.0f} sps",
                    flush=True,
                )
        elif should_log:
            print(
                f"  update {updates:>4} | step {steps:>7} | learner continues | "
                f"KL {update_stats['approx_kl']:.4f} | clip {update_stats['clip_fraction']*100:4.1f}% | "
                f"EV {update_stats['explained_variance']:+.3f} | "
                f"{steps/(time.time()-started):,.0f} sps",
                flush=True,
            )

        if should_log:
            print(
                f"  champion stats | accepted {report['training']['accepted_updates']} / "
                f"{candidate_evaluations} evals | rejected {report['training']['rejected_updates']} / "
                f"{candidate_evaluations} | resets {report['training']['learner_resets']} | "
                f"probation {report['training']['quarter_step_probations_started']} start / "
                f"{report['training']['quarter_step_probations_passed']} pass / "
                f"{report['training']['quarter_step_probations_failed']} fail / "
                f"{report['training']['quarter_step_probations_expired']} expire | "
                f"rate {report['training']['acceptance_rate_pct']:.1f}%",
                flush=True,
            )
            print_performance()

        rollout_action_sum = 0.0
        rollout_action_abs_sum = 0.0
        rollout_action_saturation = 0
        rollout_projected = 0
        rollout_soc_counts = [0, 0, 0]
        rollout_soc_projected = [0, 0, 0]
        rollout_mismatch_penalty_vnd = 0.0
        rollout_mismatch_kwh = 0.0
        rollout_count = 0
        rollout_started = time.perf_counter()

    if quarter_probation_active:
        # A quarter-step cannot become the final learner just because the step
        # budget ended before its mandatory one-update probation exam.
        _restore_reanchor_state(
            agent,
            champion_state,
            preserve_critic=args.preserve_critic_on_reanchor,
            reset_optimizer=args.reset_optimizer_on_reanchor,
        )
        quarter_probation_active = False
        report["training"]["quarter_step_probation_active"] = False
        report["training"]["quarter_step_probations_expired"] += 1
        report["training"]["last_quarter_probation_outcome"] = "expired"
        report["training"]["learner_resets"] += 1
        consecutive_rejections = 0
        report["training"]["consecutive_rejections"] = 0

    # The requested step budget may end after 1-3 unvalidated full PPO updates.
    # Score that final learner once so a late improvement is not thrown away.
    if updates > last_validated_update:
        point, candidate_cost, accepted, _reset_to_champion = evaluate_live_candidate(
            allow_reset=False
        )
        verdict = "ACCEPTED 👑" if accepted else "REJECTED 💀"
        print(
            f"  final candidate | update {updates:>4} | step {steps:>7} | "
            f"candidate {candidate_cost/1e6:8.1f}M | champion {champion_cost/1e6:8.1f}M | "
            f"{verdict} | saving {point['saving_vs_nobess_pct']:5.1f}% | "
            f"gap {point['oracle_gap_pct']:6.1f}%",
            flush=True,
        )

    persist_progress()

    best_agent = PPOAgent(OBSERVATION_DIM, device=learner_device)
    best_agent.load(RESULTS_DIR / f"policy_{tag}.pt")
    test_results = [
        (month, run_drl_policy(month, cfg, best_agent, p_ref_kw=p_ref))
        for month in test_months
    ]
    test_operating_months = [
        _score_ppo_operating_month(
            result["p_grid_days"],
            result["p_bess_days"],
            cfg,
            battery_wear_cost,
            month.days,
        )
        for month, result in test_results
    ]
    test_cost = sum(
        operating["total_operating_cost_vnd"]
        for operating in test_operating_months
    )
    test_wear_cost = sum(
        operating["wear_cost_vnd"]
        for operating in test_operating_months
    )
    test_throughput_kwh = sum(
        operating["throughput_kwh"]
        for operating in test_operating_months
    )
    no_bess_cost = sum(
        score_month(
            run_no_bess(month, cfg)["p_grid_days"],
            cfg,
            days=month.days,
        )["total_cost_vnd"]
        for month in test_months
    )
    test_saving = (no_bess_cost - test_cost) / no_bess_cost * 100
    final_test_result = test_results[-1][1]
    best_agent.meta = {
        **agent.meta,
        "test_saving_pct": round(test_saving, 2),
        "trained": datetime.now(timezone.utc).astimezone().date().isoformat(),
    }
    best_agent.save(RESULTS_DIR / f"policy_{tag}.pt")
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["test"] = {
        "policy_cost_vnd": test_cost,
        "no_bess_vnd": no_bess_cost,
        "saving_pct": test_saving,
        "wear_cost_vnd": test_wear_cost,
        "throughput_kwh": test_throughput_kwh,
        "month_count": len(test_months),
        "initial_soc": float(cfg.SOC_min),
        "final_soc": float(final_test_result["soc_days"][-1][-1]),
    }
    write_report(report_path, report)
    print(
        f"[train-ds] TEST {test_days[0].date_iso}->{test_days[-1].date_iso}: "
        f"{test_cost/1e6:.1f}M vs no-BESS {no_bess_cost/1e6:.1f}M -> saving {test_saving:.2f}%",
        flush=True,
    )


if __name__ == "__main__":
    main()
