from flask import Flask, jsonify, render_template, request

from benchmark import build_benchmark
from oracle_lp import build_oracle_lp
from settings import (
    BILLING_MODE_FIELD,
    BILLING_SUNDAY_FIELD,
    DEFAULT_PARAMETERS,
    FORM_FIELDS,
    SAMPLE_BATTERY_CANDIDATES,
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
    return jsonify(
        {
            "index": index,
            "candidate": _candidate_oracle(PARAMETERS, candidate),
        }
    )


def _parameters_from_form():
    values = {
        field: request.form.get(field, "")
        for field in FORM_FIELDS
    }
    values[BILLING_MODE_FIELD] = request.form.get(
        BILLING_MODE_FIELD,
        DEFAULT_PARAMETERS[BILLING_MODE_FIELD],
    )
    values[BILLING_SUNDAY_FIELD] = BILLING_SUNDAY_FIELD in request.form
    return values


def view_context():
    benchmark = build_benchmark(PARAMETERS)
    return {
        **PARAMETERS,
        "benchmark": benchmark,
        "oracle": _pending_oracle(),
        "sample_battery_candidates": SAMPLE_BATTERY_CANDIDATES,
        "candidate_oracles": _pending_candidate_oracles(),
        "selected_candidate_index": _selected_candidate_index(PARAMETERS),
        "checked_2tc": "checked" if PARAMETERS["billing_mode"] == "2tc" else "",
        "checked_tou": "checked" if PARAMETERS["billing_mode"] == "tou" else "",
        "checked_sunday": "checked" if PARAMETERS["billing_sunday"] else "",
    }


def _candidate_oracle(parameters, candidate):
    candidate_parameters = {
        **parameters,
        "battery_capacity_kWh": str(candidate["battery_capacity_kWh"]),
        "battery_power_limit_kW": str(candidate["battery_power_limit_kW"]),
    }
    return {
        **candidate,
        "oracle": build_oracle_lp(candidate_parameters),
    }


def _pending_candidate_oracles():
    return [
        {
            **candidate,
            "oracle": _pending_oracle(),
        }
        for candidate in _active_battery_candidates(PARAMETERS)
    ]


def _active_battery_candidates(parameters):
    capacity = _to_float(parameters.get("battery_capacity_kWh"), 0.0)
    power = _to_float(parameters.get("battery_power_limit_kW"), 0.0)
    return (
        {
            "id": "custom",
            "label": f"Custom {capacity:,.0f} kWh / {power:,.1f} kW",
            "battery_capacity_kWh": capacity,
            "battery_power_limit_kW": power,
            "power_ratio": power / capacity if capacity else 0,
        },
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
        "seer_factor": _to_float(PARAMETERS.get("billing_real_saving_factor"), 1.0),
        "sizing_economics": {
            "battery_capacity_kWh": 0,
            "battery_power_limit_kW": 0,
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
