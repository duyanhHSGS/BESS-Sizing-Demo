"""One-file entrypoint for the debloated BESS DRL runner."""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from runtime import DebloatedRuntime, RuntimeOptions
from train import run_training


def _runtime_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Mongo-free BESS DRL: direct PPO training and MQTT runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run PPO from MQTT telemetry")
    run.add_argument("--policy", type=Path, required=True, help="policy_*.pt checkpoint")
    run.add_argument(
        "--config",
        type=Path,
        default=None,
        help="fallback BessDrlConfig JSON for old checkpoints without embedded effective_config",
    )
    run.add_argument("--mode", choices=("shadow", "closed"), default="shadow")
    run.add_argument("--timezone", default="Asia/Ho_Chi_Minh")
    run.add_argument("--interval", type=int, default=15, help="tick interval in minutes")
    run.add_argument("--offset", type=int, default=2, help="seconds after tick boundary")
    run.add_argument("--mqtt-host", default="localhost")
    run.add_argument("--mqtt-port", type=int, default=1883)
    run.add_argument("--mqtt-username", default="")
    run.add_argument("--mqtt-password", default="")
    run.add_argument("--mqtt-client-id", default="bess-drl-debloated")
    run.add_argument("--state", type=Path, default=Path("state/runtime_state.json"))
    run.add_argument("--log", type=Path, default=Path("logs/setpoints.jsonl"))
    run.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="optional planner JSON; absent means a zero/standby base plan",
    )
    run.add_argument("--controller-url", default="http://localhost:8001")
    run.add_argument("--api-key", default="dev-api-key-change-me")
    run.add_argument("--log-level", default="INFO")
    return parser


def _run_runtime(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    options = RuntimeOptions(
        policy_path=args.policy.resolve(),
        config_path=args.config.resolve() if args.config else None,
        mode=args.mode,
        timezone=args.timezone,
        interval_minutes=args.interval,
        tick_offset_seconds=args.offset,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        mqtt_username=args.mqtt_username,
        mqtt_password=args.mqtt_password,
        mqtt_client_id=args.mqtt_client_id,
        state_path=args.state.resolve(),
        log_path=args.log.resolve(),
        plan_path=args.plan.resolve() if args.plan else None,
        controller_url=args.controller_url,
        api_key=args.api_key,
    )
    asyncio.run(DebloatedRuntime(options).run_forever())
    return 0


def main() -> int:
    # UI is imported lazily so normal train/run CLI startup does not import Flask.
    if len(sys.argv) >= 2 and sys.argv[1] == "ui":
        from webui import run_ui

        return run_ui(sys.argv[2:])
    # Training intentionally forwards ALL remaining flags untouched to the proven
    # trainer. This keeps its full CLI without duplicating 15 hyperparameter flags.
    if len(sys.argv) >= 2 and sys.argv[1] == "train":
        return run_training(sys.argv[2:])
    parser = _runtime_parser()
    args = parser.parse_args()
    if args.command == "run":
        return _run_runtime(args)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
