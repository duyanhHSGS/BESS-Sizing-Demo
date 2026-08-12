"""Train the only supported learner: Brain 3 DQN on canonical BrainEnv periods."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from bess.brain.brain3_agent import BRAIN3_ACTIONS, Brain3Agent
from bess.brain.brain_env import BrainEnvironmentStepResult
from bess.brain.runtime import episode_for_period, load_csv_days, make_env, split_billing_periods
from bess.core.config import BrainConfig


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--eval-every", type=int, default=10_000)
    parser.add_argument("--validation-periods", type=int, default=1)
    parser.add_argument("--test-periods", type=int, default=1)
    parser.add_argument("--control-dt-minutes", type=float, required=True)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=1_024)
    parser.add_argument("--target-sync-interval", type=int, default=1_000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=100_000)
    parser.add_argument("--reward-divisor-vnd", type=float, default=1_000_000.0)
    parser.add_argument("--gradient-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def _decision_rollout(agent: Brain3Agent, period, config: BrainConfig, held_steps: int, *, learn: bool) -> tuple[int, float, list[float]]:
    episode, _ = episode_for_period(period, config)
    env = make_env(episode, config)
    observation = env.reset()
    decisions = 0
    losses: list[float] = []
    while True:
        decision = agent.decide(observation, explore=learn)
        decision_observation = observation
        reward = 0.0
        result: BrainEnvironmentStepResult | None = None
        for _ in range(held_steps):
            step_result = env.step(decision.action)
            if not isinstance(step_result, BrainEnvironmentStepResult):
                raise RuntimeError("Brain 3 training requires owned-episode BrainEnv")
            result = step_result
            reward += step_result.reward.timestep_savings_vnd
            if step_result.done:
                break
        assert result is not None
        if learn:
            agent.remember(
                decision_observation,
                decision.action_index,
                reward,
                result.next_observation,
                result.done,
            )
            loss = agent.learn()
            if loss is not None:
                losses.append(loss)
        decisions += 1
        if result.done:
            break
        assert result.next_observation is not None
        observation = result.next_observation
    return decisions, env.net_battery_savings_vnd, losses


def _evaluate(agent: Brain3Agent, periods, config: BrainConfig, held_steps: int) -> dict:
    savings = []
    for period in periods:
        _, value, _ = _decision_rollout(agent, period, config, held_steps, learn=False)
        savings.append(value)
    return {
        "periods": len(periods),
        "savings_vnd": float(sum(savings)),
        "mean_period_savings_vnd": float(np.mean(savings)),
    }


def _deployment(agent: Brain3Agent, config: BrainConfig, args: argparse.Namespace, meta: dict) -> dict:
    return {
        "schema_version": 1,
        "algorithm": "brain3_dqn",
        "observation_dim": 7,
        "action_values": BRAIN3_ACTIONS,
        "online_network": agent.online_network.state_dict(),
        "meta": {
            **meta,
            "hidden_dim": args.hidden_dim,
            "native_dt_minutes": config.timestep_hours * 60.0,
            "control_dt_minutes": args.control_dt_minutes,
            "native_steps_per_action": round(args.control_dt_minutes / (config.timestep_hours * 60.0)),
            "environment_fingerprint": config.fingerprint(),
            "environment": asdict(config),
        },
    }


def main() -> None:
    args = _args()
    if args.steps <= 0 or args.eval_every <= 0:
        raise SystemExit("steps and eval-every must be positive")
    parameters = json.loads(args.parameters.read_text(encoding="utf-8"))
    config = BrainConfig.from_parameters(parameters)
    period_warnings: list[str] = []
    periods = split_billing_periods(
        load_csv_days(args.csv), reject_leftover=True, warnings=period_warnings
    )
    for warning in period_warnings:
        print(json.dumps({"type": "warning", "message": warning}), flush=True)
    held_ratio = args.control_dt_minutes / (config.timestep_hours * 60.0)
    held_steps = round(held_ratio)
    decisions_per_day = 1440.0 / args.control_dt_minutes
    if (
        held_steps <= 0
        or not np.isclose(held_ratio, held_steps)
        or not np.isclose(decisions_per_day, round(decisions_per_day))
    ):
        raise SystemExit("control interval must be a native multiple that divides 24 hours")
    dataset_fingerprint = hashlib.sha256(args.csv.read_bytes()).hexdigest()
    training_contract = {
        "tag": args.tag,
        "control_dt_minutes": args.control_dt_minutes,
        "eval_every": args.eval_every,
        "validation_periods": args.validation_periods,
        "test_periods": args.test_periods,
        "hidden_dim": args.hidden_dim,
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "replay_capacity": args.replay_capacity,
        "learning_starts": args.learning_starts,
        "target_sync_interval": args.target_sync_interval,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "epsilon_decay_steps": args.epsilon_decay_steps,
        "reward_divisor_vnd": args.reward_divisor_vnd,
        "gradient_clip_norm": args.gradient_clip_norm,
        "seed": args.seed,
    }
    training_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "environment": config.fingerprint(),
                "dataset": dataset_fingerprint,
                "training": training_contract,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    reserve = args.validation_periods + args.test_periods
    if args.validation_periods <= 0 or args.test_periods <= 0 or len(periods) <= reserve:
        raise SystemExit("dataset needs training periods plus positive validation and test periods")
    train_periods = periods[:-reserve]
    validation_periods = periods[-reserve:-args.test_periods]
    test_periods = periods[-args.test_periods:]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    agent = Brain3Agent(
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        replay_capacity=args.replay_capacity,
        learning_starts=args.learning_starts,
        target_sync_interval=args.target_sync_interval,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        reward_divisor_vnd=args.reward_divisor_vnd,
        gradient_clip_norm=args.gradient_clip_norm,
        seed=args.seed,
        device=args.device,
    )
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=args.device, weights_only=False)
        if resume_payload.get("training_fingerprint") != training_fingerprint:
            raise SystemExit("resume checkpoint environment, dataset, or training contract does not match")
        agent.load_checkpoint(resume_payload["agent"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    deployment_path = args.output_dir / f"brain3_{args.tag}.pt"
    resume_path = args.output_dir / f"brain3_resume_{args.tag}.pt"
    report_path = args.output_dir / f"brain3_report_{args.tag}.json"
    if resume_payload:
        random.setstate(resume_payload["python_rng_state"])
        np.random.set_state(resume_payload["numpy_rng_state"])
        torch.set_rng_state(resume_payload["torch_rng_state"].cpu())
        if args.device == "cuda" and resume_payload.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng_state"])
        period_order = [int(value) for value in resume_payload["period_order"]]
        order_cursor = int(resume_payload["order_cursor"])
        if sorted(period_order) != list(range(len(train_periods))) or not 0 <= order_cursor < len(period_order):
            raise SystemExit("resume checkpoint training-period order is incompatible")
    else:
        period_order = list(range(len(train_periods)))
        order_cursor = 0
    next_evaluation = (
        int(resume_payload["next_evaluation"])
        if resume_payload
        else args.eval_every
    )
    best_validation = float(resume_payload["best_validation"]) if resume_payload else float("-inf")
    best_step = int(resume_payload.get("best_step", 0)) if resume_payload else 0
    curve = list(resume_payload["curve"]) if resume_payload else []
    best_deployment = resume_payload.get("best_deployment") if resume_payload else None
    if best_deployment is not None:
        torch.save(best_deployment, deployment_path)
    losses: list[float] = []

    while agent.environment_steps < args.steps:
        period = train_periods[period_order[order_cursor]]
        order_cursor += 1
        if order_cursor >= len(period_order):
            order_cursor = 0
        _, _, period_losses = _decision_rollout(agent, period, config, held_steps, learn=True)
        losses.extend(period_losses)
        if agent.environment_steps >= next_evaluation or agent.environment_steps >= args.steps:
            interval_evaluation_due = agent.environment_steps >= next_evaluation
            validation = _evaluate(agent, validation_periods, config, held_steps)
            is_new_best = validation["savings_vnd"] > best_validation
            if is_new_best:
                best_validation = validation["savings_vnd"]
                best_step = agent.environment_steps
            point = {
                "type": "validation",
                "steps": agent.environment_steps,
                "gradient_steps": agent.gradient_steps,
                "replay_size": len(agent.replay),
                "replay_capacity": agent.replay.capacity,
                "effective_learning_start": max(agent.batch_size, agent.learning_starts),
                "validation_savings_vnd": validation["savings_vnd"],
                "best_validation_savings_vnd": best_validation,
                "best_step": best_step,
                "is_new_best": is_new_best,
                "mean_loss": float(np.mean(losses)) if losses else None,
                "epsilon": agent.epsilon,
                "training_epsilon": agent.epsilon,
                "validation_epsilon": 0.0,
                "validation_exploration": False,
                "target_sync_interval": agent.target_sync_interval,
                "target_sync_progress": agent.gradient_steps % agent.target_sync_interval,
                "evaluation_trigger": "interval" if interval_evaluation_due else "final budget",
                "evaluation_target_steps": next_evaluation if interval_evaluation_due else args.steps,
            }
            print(json.dumps(point), flush=True)
            curve.append(point)
            losses.clear()
            if is_new_best:
                best_deployment = copy.deepcopy(
                    _deployment(
                        agent,
                        config,
                        args,
                        {
                            "tag": args.tag,
                            "best_validation_savings_vnd": best_validation,
                            "best_validation_step": best_step,
                            "training_periods": len(train_periods),
                            "validation_periods": len(validation_periods),
                            "test_periods": len(test_periods),
                            "dataset_fingerprint": dataset_fingerprint,
                            "training_contract": training_contract,
                        },
                    )
                )
                torch.save(best_deployment, deployment_path)
            if best_deployment is None:
                raise RuntimeError("validation did not produce a deployment checkpoint")
            next_evaluation += args.eval_every
            torch.save(
                {
                    "schema_version": 1,
                    "algorithm": "brain3_dqn_resume",
                    "environment_fingerprint": config.fingerprint(),
                    "dataset_fingerprint": dataset_fingerprint,
                    "training_fingerprint": training_fingerprint,
                    "agent": agent.checkpoint(),
                    "python_rng_state": random.getstate(),
                    "numpy_rng_state": np.random.get_state(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if args.device == "cuda" else None,
                    "period_order": period_order,
                    "order_cursor": order_cursor,
                    "next_evaluation": next_evaluation,
                    "best_validation": best_validation,
                    "best_step": best_step,
                    "best_deployment": best_deployment,
                    "curve": curve,
                },
                resume_path,
            )

    assert best_deployment is not None
    agent.online_network.load_state_dict(best_deployment["online_network"])
    agent.target_network.load_state_dict(best_deployment["online_network"])
    test = _evaluate(agent, test_periods, config, held_steps)
    report = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "algorithm": "brain3_dqn",
        "steps": agent.environment_steps,
        "best_validation_savings_vnd": best_validation,
        "best_validation_step": best_step,
        "environment_fingerprint": config.fingerprint(),
        "dataset_fingerprint": dataset_fingerprint,
        "training_contract": training_contract,
        "split": {
            "training": [period.key for period in train_periods],
            "validation": [period.key for period in validation_periods],
            "test": [period.key for period in test_periods],
        },
        "test": test,
        "curve": curve,
        "deployment": deployment_path.name,
        "resume": resume_path.name,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"type": "complete", **report}), flush=True)


if __name__ == "__main__":
    main()
