from flask import Flask, Response, jsonify, render_template, request

import dispatch_runner
import dispatch_store
import oracle_cache
from benchmark import (
    build_benchmark,
    detect_dt_hours,
    list_data_csvs,
    selected_data_filename,
    selected_data_path,
)
from oracle_lp import build_oracle_lp
from settings import (
    BILLING_MODE_FIELD,
    BILLING_SUNDAY_FIELD,
    DEFAULT_PARAMETERS,
    FORM_FIELDS,
    GREPO_GAMMA,
    PPO_GAMMA,
    PPO_LAMBDA,
    SAMPLE_BATTERY_CANDIDATES,
)
from training_checkpoints import get_checkpoint_report, list_checkpoints
from training_datasets import DatasetError, list_datasets
from training_jobs import MANAGER
from training_launcher import (
    TrainingLaunchError,
    UnsupportedAlgorithm,
    start_training,
    training_oracle_status,
)


app = Flask(__name__)

PARAMETERS = DEFAULT_PARAMETERS.copy()


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", saved=False, should_calculate=False, **view_context())


@app.route("/set-parameters", methods=["POST"])
def set_parameters():
    PARAMETERS.update(_parameters_from_form())
    should_calculate = request.form.get("form_action") == "calculate"
    return render_template("index.html", saved=True, should_calculate=should_calculate, **view_context())


@app.route("/candidate-oracle/<int:index>", methods=["GET"])
def candidate_oracle(index):
    candidates = _active_battery_candidates(PARAMETERS)
    if index < 0 or index >= len(candidates):
        return jsonify({"error": "Candidate index out of range."}), 404

    candidate = candidates[index]
    force = request.args.get("force") == "1"
    return jsonify(
        {
            "index": index,
            "candidate": _candidate_oracle(PARAMETERS, candidate, force=force),
        }
    )


@app.route("/api/training/datasets", methods=["GET"])
def training_datasets():
    return jsonify(list_datasets())


@app.route("/api/training/checkpoints", methods=["GET"])
def training_checkpoints():
    return jsonify(list_checkpoints())


@app.route("/api/training/checkpoints/<checkpoint_name>/report", methods=["GET"])
def training_checkpoint_report(checkpoint_name):
    try:
        return jsonify(get_checkpoint_report(checkpoint_name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "checkpoint not found"}), 404


@app.route("/api/training/oracle-status", methods=["POST"])
def training_oracle_cache_status():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify(training_oracle_status(payload, PARAMETERS))
    except (DatasetError, TrainingLaunchError, ValueError) as exc:
        return jsonify({"ready": False, "error": str(exc)}), 422


@app.route("/api/training/start", methods=["POST"])
def training_start():
    payload = request.get_json(silent=True) or {}
    try:
        job, details = start_training(payload, PARAMETERS, MANAGER)
    except oracle_cache.OracleCacheRequired as exc:
        return jsonify({"error": str(exc), "code": "oracle_cache_required"}), 422
    except UnsupportedAlgorithm as exc:
        return jsonify({"error": str(exc)}), 422
    except (DatasetError, TrainingLaunchError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify(details | {"status": job.status})


@app.route("/api/training/stop/<job_id>", methods=["POST"])
def training_stop(job_id):
    if not MANAGER.stop(job_id):
        return jsonify({"error": "job not running or not found"}), 404
    return jsonify({"ok": True})


@app.route("/api/training/jobs/<job_id>", methods=["GET"])
def training_job_detail(job_id):
    detail = MANAGER.detail(job_id)
    if detail is None:
        return jsonify({"error": "job not found"}), 404
    return jsonify(detail)


@app.route("/api/training/jobs/<job_id>/events", methods=["GET"])
def training_job_events(job_id):
    return Response(MANAGER.sse_events(job_id), mimetype="text/event-stream")


@app.route("/api/dispatch/policies", methods=["GET"])
def dispatch_policies():
    latest = dispatch_store.latest_runs_by_policy()
    rows = []
    for checkpoint in list_checkpoints():
        run = latest.get(checkpoint["name"])
        rows.append(
            {
                **checkpoint,
                "latest_run": run,
                "latest_status": "saved" if run else "no saved trace",
                "has_trace": bool(run and dispatch_store.get_traces(run["id"])),
                "warning": None if run else "No saved trace exists for this policy yet.",
            }
        )
    return jsonify({"policies": rows})


@app.route("/api/dispatch/runs", methods=["GET"])
def dispatch_runs():
    return jsonify({"runs": dispatch_store.list_runs()})


@app.route("/api/dispatch/policy-traces", methods=["GET"])
def dispatch_policy_traces():
    policy_names = _policy_names_from_query()
    payload = {}
    warnings = []
    for policy_name in policy_names:
        run = dispatch_store.latest_run_for_policy(policy_name)
        if not run:
            warning = f"{policy_name}: no saved trace exists yet."
            warnings.append(warning)
            payload[policy_name] = {"warning": warning, "days": []}
            continue
        traces = dispatch_store.get_traces(run["id"])
        if not traces or policy_name not in traces:
            warning = f"{policy_name}: saved trace is missing or malformed."
            warnings.append(warning)
            payload[policy_name] = {"warning": warning, "run_id": run["id"], "days": []}
            continue
        payload[policy_name] = {
            "run_id": run["id"],
            "days": traces.get(policy_name, []),
            "kpi": run.get("kpi", {}).get(policy_name, {}),
        }
    return jsonify({"policies": payload, "warnings": warnings})


@app.route("/api/dispatch/run", methods=["POST"])
def dispatch_run():
    payload = request.get_json(silent=True) or {}
    policy_names = payload.get("policies") or []
    if not isinstance(policy_names, list) or not policy_names:
        return jsonify({"error": "Select at least one policy."}), 422

    known = {checkpoint["name"] for checkpoint in list_checkpoints()}
    unknown = [name for name in policy_names if name not in known]
    if unknown:
        return jsonify({"error": f"Unknown local policy: {', '.join(unknown)}"}), 422

    results, warnings = dispatch_runner.run_policies(policy_names, PARAMETERS)
    if not results:
        return jsonify({"error": "No selected policy produced a dispatch trace.", "warnings": warnings}), 422
    run_id = dispatch_store.save_run(
        "Sizing Demo dispatch",
        {"policies": list(results), "parameters": PARAMETERS},
        results,
    )
    return jsonify({"run_id": run_id, "policies": list(results), "warnings": warnings})


def _parameters_from_form():
    values = {
        field: request.form.get(field, "")
        for field in FORM_FIELDS
    }
    values["selected_data_csv"] = selected_data_filename(values)
    values["dt"] = str(detect_dt_hours(selected_data_path(values)))
    values[BILLING_MODE_FIELD] = request.form.get(
        BILLING_MODE_FIELD,
        DEFAULT_PARAMETERS[BILLING_MODE_FIELD],
    )
    values[BILLING_SUNDAY_FIELD] = BILLING_SUNDAY_FIELD in request.form
    return values


def _policy_names_from_query():
    raw = request.args.get("policies", "")
    names = []
    for name in raw.split(","):
        clean = name.strip()
        if clean:
            names.append(clean)
    return names


def view_context():
    PARAMETERS["selected_data_csv"] = selected_data_filename(PARAMETERS)
    PARAMETERS["dt"] = str(detect_dt_hours(selected_data_path(PARAMETERS)))
    benchmark = build_benchmark(PARAMETERS)
    candidate_oracles = _cached_candidate_oracles()
    return {
        **PARAMETERS,
        "data_csv_files": list_data_csvs(),
        "benchmark": benchmark,
        "oracle": candidate_oracles[0]["oracle"] if candidate_oracles else _pending_oracle(),
        "sample_battery_candidates": SAMPLE_BATTERY_CANDIDATES,
        "ppo_gamma": PPO_GAMMA,
        "ppo_lambda": PPO_LAMBDA,
        "grepo_gamma": GREPO_GAMMA,
        "candidate_oracles": candidate_oracles,
        "csv_has_oracle_cache": oracle_cache.selected_csv_has_cache(PARAMETERS),
        "exact_oracle_cache_exists": any(
            candidate.get("oracle", {}).get("cache", {}).get("hit")
            for candidate in candidate_oracles
        ),
        "selected_candidate_index": _selected_candidate_index(PARAMETERS),
        "checked_2tc": "checked" if PARAMETERS["billing_mode"] == "2tc" else "",
        "checked_tou": "checked" if PARAMETERS["billing_mode"] == "tou" else "",
        "checked_sunday": "checked" if PARAMETERS["billing_sunday"] else "",
    }


def _candidate_oracle(parameters, candidate, *, force=False):
    candidate_parameters = {
        **parameters,
        "battery_capacity_kWh": str(candidate["battery_capacity_kWh"]),
        "battery_power_limit_kW": str(candidate["battery_power_limit_kW"]),
    }
    return {
        **candidate,
        "oracle": oracle_cache.get_or_build_oracle_lp(
            candidate_parameters,
            build_oracle_lp,
            force=force,
        ),
    }


def _cached_candidate_oracles():
    candidates = []
    for candidate in _active_battery_candidates(PARAMETERS):
        candidate_parameters = {
            **PARAMETERS,
            "battery_capacity_kWh": str(candidate["battery_capacity_kWh"]),
            "battery_power_limit_kW": str(candidate["battery_power_limit_kW"]),
        }
        candidates.append(
            {
                **candidate,
                "oracle": oracle_cache.cached_oracle_lp(candidate_parameters) or _pending_oracle(),
            }
        )
    return candidates


def _active_battery_candidates(parameters):
    capacity = _to_float(parameters.get("battery_capacity_kWh"), 0.0)
    power = _to_float(parameters.get("battery_power_limit_kW"), 0.0)
    custom_candidate = {
        "id": "custom",
        "label": f"Custom {capacity:,.0f} kWh / {power:,.1f} kW",
        "battery_capacity_kWh": capacity,
        "battery_power_limit_kW": power,
        "power_ratio": power / capacity if capacity else 0,
    }
    if parameters.get("use_sample_battery_options") != "yes":
        return (custom_candidate,)
    return (
        custom_candidate,
        *SAMPLE_BATTERY_CANDIDATES,
    )


def _pending_oracle():
    return {
        "available": False,
        "status": "Waiting for battery calculation.",
        "days": [],
        "summary": _pending_summary(),
    }


def _pending_summary():
    return {
        "solved_day_count": 0,
        "total_grid_kWh": 0,
        "total_bill_vnd": 0,
        "peak_grid_kW": 0,
        "oracle_saving_vnd": 0,
        "seer_saving_vnd": 0,
        "oracle_annual_saving_vnd": 0,
        "seer_annual_saving_vnd": 0,
        "month_count": 0,
        "monthly_billing": [],
        "seer_factor": _to_float(PARAMETERS.get("billing_real_saving_factor"), 1.0),
        "sizing_economics": {
            "battery_capacity_kWh": 0,
            "battery_power_limit_kW": 0,
            "oracle_annual_saving_vnd": 0,
            "annual_saving_vnd": 0,
            "annual_saving_million_vnd": 0,
            "npv_vnd": 0,
            "npv_billion_vnd": 0,
            "payback_years": None,
            "recommended_contract_max_kW": 0,
            "oracle_peak_kW": 0,
            "pareto_status": "...",
        },
    }


def _attach_pareto_status(results):
    economics = [
        result.get("oracle", {}).get("summary", {}).get("sizing_economics", {})
        for result in results
    ]
    for index, sizing in enumerate(economics):
        annual_saving = sizing.get("annual_saving_vnd", 0)
        npv = sizing.get("npv_vnd", 0)
        dominated = any(
            other_index != index
            and other.get("annual_saving_vnd", 0) >= annual_saving
            and other.get("npv_vnd", 0) >= npv
            and (
                other.get("annual_saving_vnd", 0) > annual_saving
                or other.get("npv_vnd", 0) > npv
            )
            for other_index, other in enumerate(economics)
        )
        sizing["pareto_status"] = "Yes" if not dominated else "No"


def _selected_candidate_index(parameters):
    return 0


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


if __name__ == "__main__":
    app.run(debug=True)
