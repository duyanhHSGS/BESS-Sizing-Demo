"""Brain HQ Flask composition root."""
from __future__ import annotations

import csv
import io
import json
from flask import Flask, Response, jsonify, render_template, request

from bess.paths import PROJECT_ROOT
from bess.brain import brain_env_sessions
from bess.core.settings import DEFAULT_PARAMETERS, FORM_FIELDS, SAMPLE_BATTERY_CANDIDATES
from bess.core.config import BrainConfig
from bess.dispatch import brain_dispatch, dispatch_store
from bess.evaluation import benchmark_jobs, benchmark_store, brain_benchmarking
from bess.evaluation.benchmark import build_benchmark, detect_dt_hours, list_data_csvs, selected_data_filename, selected_data_path
from bess.evaluation.oracle import oracle_cache
from bess.evaluation.oracle.oracle_lp import build_oracle_lp
from bess.integrations import thingsboard_connector
from bess.shadow import brain_live_runs, brain_shadow_runs, shadow_jobs
from bess.training.brain3_checkpoints import get_checkpoint_report, list_checkpoints, list_compatible_checkpoints, list_resume_checkpoints
from bess.training.brain3_launcher import TrainingLaunchError, UnsupportedAlgorithm, start_training
from bess.training.training_datasets import DatasetError, list_datasets
from bess.training.training_jobs import MANAGER

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "web"),
    static_url_path="/assets",
)
PARAMETERS = DEFAULT_PARAMETERS.copy()


def _float(value, fallback=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


@app.get("/")
def home():
    PARAMETERS["selected_data_csv"] = selected_data_filename(PARAMETERS)
    PARAMETERS["dt"] = str(detect_dt_hours(selected_data_path(PARAMETERS)))
    return render_template(
        "index.html",
        parameters=PARAMETERS,
        parameters_json=json.dumps(PARAMETERS),
        datasets=list_data_csvs(),
        sizing=build_benchmark(PARAMETERS),
    )


@app.get("/api/parameters")
def parameters_get():
    return jsonify(PARAMETERS)


@app.post("/api/parameters")
def parameters_save():
    payload = request.get_json(silent=True) or {}
    for key in DEFAULT_PARAMETERS:
        if key in payload:
            PARAMETERS[key] = payload[key]
    PARAMETERS["selected_data_csv"] = selected_data_filename(PARAMETERS)
    PARAMETERS["dt"] = str(detect_dt_hours(selected_data_path(PARAMETERS)))
    return jsonify({"parameters": PARAMETERS, "sizing": build_benchmark(PARAMETERS)})


@app.post("/set-parameters")
def parameters_form_save():
    values = {field: request.form.get(field, "") for field in FORM_FIELDS}
    values["billing_mode"] = request.form.get("billing_mode", DEFAULT_PARAMETERS["billing_mode"])
    values["billing_sunday"] = "billing_sunday" in request.form
    PARAMETERS.update(values)
    PARAMETERS["selected_data_csv"] = selected_data_filename(PARAMETERS)
    PARAMETERS["dt"] = str(detect_dt_hours(selected_data_path(PARAMETERS)))
    return home()


@app.get("/api/sizing")
def sizing_context():
    return jsonify(
        {
            "benchmark": build_benchmark(PARAMETERS),
            "oracle": oracle_cache.cached_oracle_lp(PARAMETERS),
            "candidates": SAMPLE_BATTERY_CANDIDATES,
        }
    )


@app.get("/candidate-oracle/<int:index>")
def candidate_oracle(index: int):
    candidates = SAMPLE_BATTERY_CANDIDATES if PARAMETERS.get("use_sample_battery_options") == "yes" else (
        {
            "id": "selected",
            "label": "Selected battery",
            "battery_capacity_kWh": _float(PARAMETERS.get("battery_capacity_kWh")),
            "battery_power_limit_kW": _float(PARAMETERS.get("battery_power_limit_kW")),
        },
    )
    if index < 0 or index >= len(candidates):
        return jsonify({"error": "candidate index out of range"}), 404
    candidate = candidates[index]
    applied = {
        **PARAMETERS,
        "battery_capacity_kWh": str(candidate["battery_capacity_kWh"]),
        "battery_power_limit_kW": str(candidate["battery_power_limit_kW"]),
    }
    return jsonify(
        {
            "index": index,
            "candidate": {
                **candidate,
                "oracle": oracle_cache.get_or_build_oracle_lp(
                    applied, build_oracle_lp, force=request.args.get("force") == "1"
                ),
            },
        }
    )


@app.get("/api/training/datasets")
def training_datasets():
    return jsonify(list_datasets())


@app.get("/api/training/checkpoints")
def training_checkpoints():
    return jsonify({"deployments": list_checkpoints(), "resume": list_resume_checkpoints()})


@app.get("/api/training/checkpoints/<checkpoint_name>/report")
def training_checkpoint_report(checkpoint_name: str):
    try:
        return jsonify(get_checkpoint_report(checkpoint_name))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "checkpoint not found"}), 404


@app.post("/api/training/start")
def training_start():
    try:
        job, details = start_training(request.get_json(silent=True) or {}, dict(PARAMETERS), MANAGER)
        return jsonify({**details, "status": job.status})
    except (DatasetError, TrainingLaunchError, UnsupportedAlgorithm, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/training/stop/<job_id>")
def training_stop(job_id: str):
    return jsonify({"ok": True}) if MANAGER.stop(job_id) else (jsonify({"error": "job not running"}), 404)


@app.get("/api/training/jobs/<job_id>")
def training_job(job_id: str):
    detail = MANAGER.detail(job_id)
    return jsonify(detail) if detail else (jsonify({"error": "job not found"}), 404)


@app.get("/api/training/jobs/<job_id>/events")
def training_events(job_id: str):
    return Response(MANAGER.sse_events(job_id), mimetype="text/event-stream")


@app.get("/api/brain-env/context")
def brain_env_context():
    try:
        return jsonify(brain_env_sessions.context(dict(PARAMETERS)))
    except (brain_env_sessions.BrainEnvSessionError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/brain-env/sessions")
def brain_env_create():
    payload = request.get_json(silent=True) or {}
    try:
        session = brain_env_sessions.create_session(
            dict(PARAMETERS),
            day_index=payload.get("day_index"),
            starting_peak_kw=payload.get("starting_peak_kw", 0.0),
        )
        return jsonify({"status": session.status(), "trace": []}), 201
    except (brain_env_sessions.BrainEnvSessionError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422


def _human_session(session_id: str):
    try:
        return brain_env_sessions.get_session(session_id), None
    except brain_env_sessions.BrainEnvSessionNotFound as exc:
        return None, (jsonify({"error": str(exc)}), 404)


@app.get("/api/brain-env/sessions/<session_id>")
def brain_env_detail(session_id: str):
    session, error = _human_session(session_id)
    return error or jsonify(session.detail())


@app.post("/api/brain-env/sessions/<session_id>/step")
def brain_env_step(session_id: str):
    session, error = _human_session(session_id)
    if error:
        return error
    try:
        entry, status = session.step((request.get_json(silent=True) or {}).get("action"))
        return jsonify({"entry": entry, "status": status})
    except brain_env_sessions.BrainEnvSessionComplete as exc:
        return jsonify({"error": str(exc), "status": session.status()}), 409
    except (brain_env_sessions.BrainEnvSessionError, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422


@app.delete("/api/brain-env/sessions/<session_id>")
def brain_env_delete(session_id: str):
    return jsonify({"ok": True}) if brain_env_sessions.drop_session(session_id) else (jsonify({"error": "session not found"}), 404)


def _controller_rows() -> list[dict]:
    compatible = list_compatible_checkpoints(BrainConfig.from_parameters(PARAMETERS).fingerprint())
    return [
        {"id": "brain1", "name": "brain1", "display_name": "Brain 1", "algo": "rule", "error": None},
        {"id": "brain2", "name": "brain2", "display_name": "Brain 2", "algo": "schedule", "error": None},
        *compatible,
    ]


@app.get("/api/dispatch/policies")
def dispatch_policies():
    latest = dispatch_store.latest_runs_by_controller()
    rows = []
    for controller in _controller_rows():
        run = latest.get(controller["id"])
        rows.append({**controller, "latest_run": run, "has_trace": bool(run and dispatch_store.get_traces(run["id"]))})
    return jsonify({"controllers": rows})


@app.get("/api/dispatch/runs")
def dispatch_runs():
    return jsonify({"runs": dispatch_store.list_runs()})


@app.get("/api/dispatch/controller-traces")
def dispatch_traces():
    names = [name.strip() for name in request.args.get("controllers", "").split(",") if name.strip()]
    output, warnings = {}, []
    for name in names:
        run = dispatch_store.latest_run_for_controller(name)
        traces = dispatch_store.get_traces(run["id"]) if run else None
        if not run or not traces or name not in traces:
            warnings.append(f"{name}: no saved trace")
            output[name] = {"trace": []}
        else:
            output[name] = {"run_id": run["id"], "trace": traces[name], "kpi": run["kpi"].get(name, {})}
    return jsonify({"controllers": output, "warnings": warnings})


@app.post("/api/dispatch/run")
def dispatch_run():
    payload = request.get_json(silent=True) or {}
    names = payload.get("controllers") or []
    if not isinstance(names, list) or not names or not all(isinstance(name, str) for name in names):
        return jsonify({"error": "controllers must be a non-empty list of controller IDs"}), 422
    known = {row["id"] for row in _controller_rows() if not row.get("error")}
    unknown = sorted(set(names) - known)
    if unknown:
        return jsonify({"error": f"unknown controller: {', '.join(unknown)}"}), 422
    try:
        results, warnings = brain_dispatch.run_dispatch(names, dict(PARAMETERS))
    except (brain_dispatch.DispatchRunWarning, ValueError) as exc:
        return jsonify({"error": str(exc)}), 422
    if not results:
        return jsonify({"error": "no controller completed", "warnings": warnings}), 422
    run_id = dispatch_store.save_run("Brain HQ dispatch", {"controllers": list(results), "parameters": PARAMETERS}, results)
    return jsonify({"run_id": run_id, "controllers": results, "warnings": warnings})


@app.get("/api/live-runs")
def live_list():
    return jsonify({"sessions": brain_live_runs.list_sessions()})


@app.post("/api/live-runs")
def live_create():
    try:
        session = brain_live_runs.create_session(request.get_json(silent=True) or {}, dict(PARAMETERS))
        return jsonify(session.public()), 201
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422


@app.get("/api/live-runs/<session_id>")
def live_detail(session_id: str):
    try:
        return jsonify(brain_live_runs.get_session(session_id).public())
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@app.post("/api/live-runs/<session_id>/step")
def live_step(session_id: str):
    try:
        session = brain_live_runs.get_session(session_id)
        day = session.step()
        return jsonify({"day": day, "session": session.public()})
    except KeyError as exc:
        return jsonify({"error": str(exc)}), 404


@app.delete("/api/live-runs/<session_id>")
def live_delete(session_id: str):
    return jsonify({"ok": True}) if brain_live_runs.drop_session(session_id) else (jsonify({"error": "session not found"}), 404)


@app.get("/api/shadow/config")
def shadow_config():
    return jsonify(brain_shadow_runs.get_config(dict(PARAMETERS)))


@app.post("/api/shadow/config")
def shadow_config_save():
    try:
        return jsonify(brain_shadow_runs.set_config(request.get_json(silent=True) or {}, dict(PARAMETERS)))
    except (brain_shadow_runs.ShadowRunError, ValueError, KeyError) as exc:
        return jsonify({"error": str(exc)}), 409


@app.get("/api/shadow/connector")
def shadow_connector():
    return jsonify(thingsboard_connector.public_config())


@app.post("/api/shadow/connector")
def shadow_connector_save():
    try:
        return jsonify(thingsboard_connector.save_config(request.get_json(silent=True) or {}))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/shadow/connector/test")
def shadow_connector_test():
    try:
        return jsonify(thingsboard_connector.test_connection())
    except Exception as exc:  # noqa: BLE001
        return jsonify({"error": str(exc)}), 502


@app.post("/api/shadow/catchup")
def shadow_catchup():
    payload = request.get_json(silent=True) or {}
    job = shadow_jobs.MANAGER.start(
        lambda progress, cancelled: brain_shadow_runs.catchup(payload, dict(PARAMETERS), progress, cancelled)
    )
    return jsonify(job.public())


@app.get("/api/shadow/jobs/<job_id>")
def shadow_job(job_id: str):
    detail = shadow_jobs.MANAGER.get(job_id)
    return jsonify(detail) if detail else (jsonify({"error": "job not found"}), 404)


@app.post("/api/shadow/jobs/<job_id>/cancel")
def shadow_cancel(job_id: str):
    return jsonify({"ok": True}) if shadow_jobs.MANAGER.cancel(job_id) else (jsonify({"error": "job not running"}), 404)


@app.get("/api/shadow/days")
def shadow_days():
    return jsonify({"days": brain_shadow_runs.list_days(request.args.get("month"))})


@app.get("/api/shadow/monthly")
def shadow_monthly():
    return jsonify({"months": brain_shadow_runs.monthly_report()})


@app.post("/api/shadow/reset")
def shadow_reset():
    brain_shadow_runs.reset_history()
    return jsonify({"ok": True})


@app.get("/api/benchmarking/context")
def benchmarking_context():
    return jsonify(brain_benchmarking.context(dict(PARAMETERS)))


@app.post("/api/benchmarking/cache")
def benchmarking_cache():
    payload = request.get_json(silent=True) or {}
    try:
        return jsonify({"cached": brain_benchmarking.cached_result(dict(PARAMETERS), payload.get("controllers") or [])})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422


@app.post("/api/benchmarking/jobs")
def benchmarking_start():
    payload = request.get_json(silent=True) or {}
    controllers = payload.get("controllers") or []
    try:
        brain_benchmarking.fingerprint(dict(PARAMETERS), controllers)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 422
    job = benchmark_jobs.MANAGER.start(
        lambda progress, cancelled: brain_benchmarking.run_and_save(dict(PARAMETERS), list(controllers), progress, cancelled)
    )
    return jsonify(job.public())


@app.get("/api/benchmarking/jobs/<job_id>")
def benchmark_job(job_id: str):
    detail = benchmark_jobs.MANAGER.get(job_id)
    return jsonify(detail) if detail else (jsonify({"error": "job not found"}), 404)


@app.post("/api/benchmarking/jobs/<job_id>/cancel")
def benchmark_cancel(job_id: str):
    return jsonify({"ok": True}) if benchmark_jobs.MANAGER.cancel(job_id) else (jsonify({"error": "job not running"}), 404)


@app.get("/api/benchmarking/runs")
def benchmark_runs():
    return jsonify({"runs": benchmark_store.list_runs()})


@app.get("/api/benchmarking/runs/<run_id>")
def benchmark_result(run_id: str):
    result = benchmark_store.get_result(run_id)
    return jsonify(result) if result else (jsonify({"error": "result not found"}), 404)


@app.get("/api/benchmarking/runs/<run_id>/export.<format_name>")
def benchmark_export(run_id: str, format_name: str):
    result = benchmark_store.get_result(run_id)
    if not result:
        return jsonify({"error": "result not found"}), 404
    if format_name == "json":
        return Response(json.dumps(result, indent=2), mimetype="application/json")
    if format_name != "csv":
        return jsonify({"error": "format must be csv or json"}), 404
    output = io.StringIO()
    rows = result.get("leaderboard", [])
    fields = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return Response(output.getvalue(), mimetype="text/csv")
