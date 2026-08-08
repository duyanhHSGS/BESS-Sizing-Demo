from __future__ import annotations

import csv
import json
import os
from pathlib import Path


CURVE_FIELDS = [
    "steps",
    "val_cost_vnd",
    "oracle_gap_pct",
    "saving_vs_nobess_pct",
    "saving_vs_sadrbc_pct",
    "throughput_kwh",
    "mean_abs_p_bess_kw",
    "soc_span_pct",
    "blocked_action_pct",
    "residual_limit",
    "active_gate",
    "zero_export_violation_days",
    "soc_violation_days",
    "approx_kl",
    "clip_fraction",
    "ppo_epochs_run",
    "policy_loss",
    "value_loss",
    "entropy",
    "log_std",
    "actor_grad_norm",
    "critic_grad_norm",
    "adv_raw_std",
    "explained_variance",
    "learning_rate",
    "final_soc_forced_charge_kwh",
]


def write_curve(path: Path, points: list[dict]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CURVE_FIELDS)
        writer.writeheader()
        writer.writerows(points)
    os.replace(temp, path)


def write_report(path: Path, report: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temp, path)
