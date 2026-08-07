import csv
import io
import json
from datetime import date

from flask import Flask, Response, jsonify, render_template, request

import bess.evaluation.benchmark_jobs as benchmark_jobs
import bess.evaluation.benchmark_store as benchmark_store
import bess.evaluation.benchmarking as benchmarking
import bess.dispatch.dispatch_runner as dispatch_runner
import bess.dispatch.dispatch_store as dispatch_store
import bess.shadow.live_runs as live_runs
import bess.evaluation.oracle.oracle_cache as oracle_cache
import bess.shadow.shadow_jobs as shadow_jobs
import bess.shadow.shadow_runs as shadow_runs
import bess.integrations.thingsboard_connector as thingsboard_connector
import bess.forecasting.shadow_weather as shadow_weather
from bess.evaluation.benchmark import (
    build_benchmark,
    detect_dt_hours,
    list_data_csvs,
    selected_data_filename,
    selected_data_path,
)
from bess.evaluation.oracle.oracle_lp import build_oracle_lp
from bess.core.settings import (
    BILLING_MODE_FIELD,
    BILLING_SUNDAY_FIELD,
    DEFAULT_PARAMETERS,
    FORM_FIELDS,
    GREPO_GAMMA,
    GREPRO_GAMMA,
    PPO_GAMMA,
    PPO_LAMBDA,
    PPO2_GAMMA,
    PPO2_LAM_ENERGY,
    PPO2_LAM_PEAK,
    PRO_GAMMA,
    SAMPLE_BATTERY_CANDIDATES,
)
from bess.training.training_checkpoints import get_checkpoint_report, list_checkpoints
from bess.training.training_datasets import DatasetError, list_datasets
from bess.training.training_jobs import MANAGER
from bess.training.training_launcher import (
    TrainingLaunchError,
    UnsupportedAlgorithm,
    start_training,
    training_oracle_status,
)
from bess.forecasting.weather_forecast import WeatherError, fetch_weather, weather_status


app = Flask(__name__, template_folder="web/templates")

PARAMETERS = DEFAULT_PARAMETERS.copy()


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        saved=False,
        should_calculate=False,
        should_force_calculate=False,
        **view_context(),
    )


@app.route("/set-parameters", methods=["POST"])
def set_parameters():
    PARAMETERS.update(_parameters_from_form())
    form_action = request.form.get("form_action")
    should_calculate = form_action in {"calculate", "recalculate"}
    return render_template(
        "index.html",
        saved=True,
        should_calculate=should_calculate,
        should_force_calculate=form_action == "recalculate",
        **view_context(),
    )


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


@app.route("/api/weather/context", methods=["GET"])
def weather_context():
    rows = []
    for dataset in list_datasets():
        try:
            status = weather_status(dataset["id"])
        except (DatasetError, WeatherError, ValueError) as exc:
            status = {"ready": False, "message": str(exc)}
        rows.append({**dataset, "weather": status})
    return jsonify({"datasets": rows})


@app.route("/api/weather/status", methods=["GET"])
def weather_dataset_status():
    try:
        return jsonify(weather_status(request.args.get("dataset_id", "")))
    except (DatasetError, WeatherError, ValueError) as exc:
        return jsonify({"ready": False, "error": str(exc)}), 422


@app.route("/api/weather/fetch", methods=["POST"])
def weather_fetch():
    try:
        return jsonify(fetch_weather(request.get_json(silent=True) or {}))
    except (DatasetError, WeatherError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422
    except Exception as exc:  # network/provider failures are shown, never replaced
        return jsonify({"error": f"Weather provider failed: {exc}"}), 502


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
    sadrbc_run = latest.get("sadrbc_v13")
    rows = [
        {
            "name": "sadrbc_v13",
            "display_name": "SADRBC v13",
            "algo": "rule-based",
            "e_cap_kwh": _to_float(PARAMETERS.get("battery_capacity_kWh"), 0.0),
            "p_rated_kw": _to_float(PARAMETERS.get("battery_power_limit_kW"), 0.0),
            "billing_mode": PARAMETERS.get("billing_mode"),
            "meta": {"controller": "SADRBC v13", "uses_current_sizing": True},
            "error": None,
            "latest_run": sadrbc_run,
            "latest_status": "saved" if sadrbc_run else "no saved trace",
            "has_trace": bool(sadrbc_run and dispatch_store.get_traces(sadrbc_run["id"])),
            "warning": None if sadrbc_run else "No saved SADRBC v13 trace exists yet.",
        }
    ]
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

    known = {"sadrbc_v13", *{checkpoint["name"] for checkpoint in list_checkpoints()}}
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


@app.route("/api/live-runs", methods=["GET"])
def live_run_list():
    return jsonify({"sessions": live_runs.list_sessions()})


@app.route("/api/live-runs", methods=["POST"])
def live_run_create():
    payload = request.get_json(silent=True) or {}
    policy_name = str(payload.get("policy") or "")
    known = {checkpoint["name"] for checkpoint in list_checkpoints()}
    if policy_name not in known:
        return jsonify({"error": "Select a loadable local policy checkpoint."}), 422
    try:
        session = live_runs.create_session(policy_name, dict(PARAMETERS))
    except (dispatch_runner.DispatchRunWarning, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify(session.status()), 201


def _live_session_or_404(session_id):
    session = live_runs.get_session(session_id)
    if session is None:
        return None, (jsonify({"error": "Live run session not found."}), 404)
    return session, None


@app.route("/api/live-runs/<session_id>", methods=["GET"])
def live_run_detail(session_id):
    session, error = _live_session_or_404(session_id)
    if error:
        return error
    return jsonify({"status": session.status(), "days": list(session.day_log)})


@app.route("/api/live-runs/<session_id>/step", methods=["POST"])
def live_run_step(session_id):
    session, error = _live_session_or_404(session_id)
    if error:
        return error
    try:
        entry = session.step_day()
    except Exception as exc:  # the session retains prior completed days
        session.error = str(exc)[:500]
        return jsonify({"error": str(exc), "status": session.status()}), 422
    return jsonify({
        "done": entry is None,
        "entry": entry,
        "status": session.status(),
    })


@app.route("/api/live-runs/<session_id>/auto", methods=["POST"])
def live_run_auto(session_id):
    session, error = _live_session_or_404(session_id)
    if error:
        return error
    payload = request.get_json(silent=True) or {}
    try:
        interval_s = max(1.0, float(payload.get("interval_s", 3.0)))
    except (TypeError, ValueError):
        return jsonify({"error": "Auto interval must be a number of seconds."}), 422
    session.start_auto(interval_s)
    return jsonify(session.status())


@app.route("/api/live-runs/<session_id>/stop", methods=["POST"])
def live_run_stop(session_id):
    session, error = _live_session_or_404(session_id)
    if error:
        return error
    session.stop_auto()
    return jsonify(session.status())


@app.route("/api/live-runs/<session_id>", methods=["DELETE"])
def live_run_delete(session_id):
    if not live_runs.drop_session(session_id):
        return jsonify({"error": "Live run session not found."}), 404
    return jsonify({"ok": True})


@app.route("/api/shadow/config", methods=["GET"])
def shadow_config():
    return jsonify(shadow_runs.get_config(dict(PARAMETERS)))


@app.route("/api/shadow/config", methods=["POST"])
def shadow_config_save():
    try:
        config = shadow_runs.set_config(request.get_json(silent=True) or {}, dict(PARAMETERS))
    except shadow_runs.ShadowRunError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify(config)


@app.route("/api/shadow/connector", methods=["GET"])
def shadow_connector_config():
    return jsonify(thingsboard_connector.public_config())


@app.route("/api/shadow/connector", methods=["POST"])
def shadow_connector_save():
    try:
        config = thingsboard_connector.save_config(request.get_json(silent=True) or {})
    except thingsboard_connector.ThingsBoardError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify(config)


@app.route("/api/shadow/connector/test", methods=["POST"])
def shadow_connector_test():
    try:
        return jsonify(thingsboard_connector.test_connection())
    except thingsboard_connector.ThingsBoardError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/shadow/weather", methods=["GET"])
def shadow_weather_config():
    return jsonify(shadow_weather.public_config())


@app.route("/api/shadow/weather", methods=["POST"])
def shadow_weather_save():
    try:
        config = shadow_weather.save_config(request.get_json(silent=True) or {})
    except shadow_weather.ShadowWeatherError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify(config)


@app.route("/api/shadow/weather/test", methods=["POST"])
def shadow_weather_test():
    try:
        return jsonify(shadow_weather.test_connection())
    except shadow_weather.ShadowWeatherError as exc:
        return jsonify({"error": str(exc)}), 502


@app.route("/api/shadow/catchup", methods=["POST"])
def shadow_catchup():
    payload = request.get_json(silent=True) or {}
    start_date = payload.get("start_date") or None
    end_date = payload.get("end_date") or None
    try:
        if start_date:
            date.fromisoformat(str(start_date))
        if end_date:
            date.fromisoformat(str(end_date))
    except ValueError:
        return jsonify({"error": "Shadow dates must use YYYY-MM-DD."}), 422
    job = shadow_jobs.MANAGER.start(
        lambda progress, cancelled: shadow_runs.catchup(
            start_date, end_date, progress, cancelled
        )
    )
    return jsonify(job.public()), 202


@app.route("/api/shadow/jobs/<job_id>", methods=["GET"])
def shadow_job(job_id):
    detail = shadow_jobs.MANAGER.get(job_id)
    if detail is None:
        return jsonify({"error": "Shadow job not found."}), 404
    return jsonify(detail)


@app.route("/api/shadow/jobs/<job_id>/cancel", methods=["POST"])
def shadow_job_cancel(job_id):
    if not shadow_jobs.MANAGER.cancel(job_id):
        return jsonify({"error": "Shadow job is not running."}), 404
    return jsonify({"ok": True})


@app.route("/api/shadow/days", methods=["GET"])
def shadow_days():
    return jsonify({"days": shadow_runs.list_days(request.args.get("month"))})


@app.route("/api/shadow/monthly", methods=["GET"])
def shadow_monthly():
    return jsonify({"months": shadow_runs.monthly_report()})


@app.route("/api/shadow/reset", methods=["POST"])
def shadow_reset():
    try:
        shadow_runs.reset_history()
    except shadow_runs.ShadowRunError as exc:
        return jsonify({"error": str(exc)}), 409
    return jsonify({"ok": True})


@app.route("/api/benchmarking/context", methods=["GET"])
def benchmarking_context():
    return jsonify(benchmarking.context(PARAMETERS))


@app.route("/api/benchmarking/cache", methods=["POST"])
def benchmarking_cache():
    payload = request.get_json(silent=True) or {}
    try:
        cached = benchmarking.cached_result(PARAMETERS, payload.get("policies") or [])
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    return jsonify({"cached": cached})


@app.route("/api/benchmarking/jobs", methods=["POST"])
def benchmarking_start():
    payload = request.get_json(silent=True) or {}
    policy_names = payload.get("policies") or []
    parameters = dict(PARAMETERS)
    try:
        benchmarking.fingerprint(parameters, policy_names)
        current = benchmarking.context(parameters)
        if not current["oracle_ready"]:
            return jsonify({"error": "Exact Oracle is missing. Calculate it in Sizing Demo first."}), 422
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422

    job = benchmark_jobs.MANAGER.start(
        lambda progress, cancelled: benchmarking.run_and_save(
            parameters, list(policy_names), progress, cancelled
        )
    )
    return jsonify(job.public())


@app.route("/api/benchmarking/jobs/<job_id>", methods=["GET"])
def benchmarking_job(job_id):
    detail = benchmark_jobs.MANAGER.get(job_id)
    if detail is None:
        return jsonify({"error": "Benchmark job not found."}), 404
    return jsonify(detail)


@app.route("/api/benchmarking/jobs/<job_id>/cancel", methods=["POST"])
def benchmarking_cancel(job_id):
    if not benchmark_jobs.MANAGER.cancel(job_id):
        return jsonify({"error": "Benchmark job is not running."}), 404
    return jsonify({"ok": True})


@app.route("/api/benchmarking/runs", methods=["GET"])
def benchmarking_runs():
    return jsonify({"runs": benchmark_store.list_runs()})


@app.route("/api/benchmarking/runs/<run_id>", methods=["GET"])
def benchmarking_result(run_id):
    result = benchmark_store.get_result(run_id)
    if result is None:
        return jsonify({"error": "Benchmark result not found."}), 404
    return jsonify(result)


@app.route("/api/benchmarking/runs/<run_id>/export.<format_name>", methods=["GET"])
def benchmarking_export(run_id, format_name):
    result = benchmark_store.get_result(run_id)
    if result is None:
        return jsonify({"error": "Benchmark result not found."}), 404
    if format_name == "json":
        return Response(
            json.dumps(result, indent=2),
            mimetype="application/json",
            headers={"Content-Disposition": f'attachment; filename="benchmark-{run_id}.json"'},
        )
    if format_name != "csv":
        return jsonify({"error": "Export format must be csv or json."}), 404

    output = io.StringIO()
    fields = [
        "contestant", "type", "scope", "start_day", "end_day",
        "energy_cost_vnd", "demand_cost_vnd", "wear_cost_vnd",
        "total_operating_cost_vnd", "peak_kw", "oracle_relation",
    ]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for contestant in result.get("contestants", []):
        summary = contestant.get("summary", {})
        common = {
            "contestant": contestant.get("label"),
            "type": contestant.get("type"),
            "oracle_relation": summary.get("oracle_relation"),
        }
        writer.writerow(
            common | {
                "scope": "entire_csv",
                "start_day": "",
                "end_day": "",
                **{field: summary.get(field) for field in fields if field in summary},
            }
        )
        for block in summary.get("blocks", []):
            writer.writerow(common | {"scope": "30_day_block", **block})
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="benchmark-{run_id}.csv"'},
    )


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
    benchmark = build_benchmark(PARAMETERS)
    PARAMETERS["dt"] = str(benchmark["dt"])
    candidate_oracles = _cached_candidate_oracles()
    return {
        **PARAMETERS,
        "data_csv_files": list_data_csvs(),
        "benchmark": benchmark,
        "oracle": candidate_oracles[0]["oracle"] if candidate_oracles else _pending_oracle(),
        "sample_battery_candidates": SAMPLE_BATTERY_CANDIDATES,
        "ppo_gamma": PPO_GAMMA,
        "ppo_lambda": PPO_LAMBDA,
        "ppo2_gamma": PPO2_GAMMA,
        "ppo2_lam_energy": PPO2_LAM_ENERGY,
        "ppo2_lam_peak": PPO2_LAM_PEAK,
        "pro_gamma": PRO_GAMMA,
        "grepo_gamma": GREPO_GAMMA,
        "grepro_gamma": GREPRO_GAMMA,
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
