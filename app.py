from flask import Flask, render_template, request

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
    return render_template("index.html", saved=False, **view_context())


@app.route("/set-parameters", methods=["POST"])
def set_parameters():
    PARAMETERS.update(_parameters_from_form())
    return render_template("index.html", saved=True, **view_context())


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
    candidate_oracles = _build_candidate_oracles(PARAMETERS)
    return {
        **PARAMETERS,
        "benchmark": benchmark,
        "oracle": build_oracle_lp(PARAMETERS),
        "sample_battery_candidates": SAMPLE_BATTERY_CANDIDATES,
        "candidate_oracles": candidate_oracles,
        "selected_candidate_index": _selected_candidate_index(PARAMETERS),
        "checked_2tc": "checked" if PARAMETERS["billing_mode"] == "2tc" else "",
        "checked_tou": "checked" if PARAMETERS["billing_mode"] == "tou" else "",
        "checked_sunday": "checked" if PARAMETERS["billing_sunday"] else "",
    }


def _build_candidate_oracles(parameters):
    results = []
    for candidate in SAMPLE_BATTERY_CANDIDATES:
        candidate_parameters = {
            **parameters,
            "battery_capacity_kWh": str(candidate["battery_capacity_kWh"]),
            "battery_power_limit_kW": str(candidate["battery_power_limit_kW"]),
        }
        results.append(
            {
                **candidate,
                "oracle": build_oracle_lp(candidate_parameters),
            }
        )

    _attach_pareto_status(results)
    return results


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
    capacity = _to_float(parameters.get("battery_capacity_kWh"), -1.0)
    power = _to_float(parameters.get("battery_power_limit_kW"), -1.0)
    for index, candidate in enumerate(SAMPLE_BATTERY_CANDIDATES):
        if (
            abs(candidate["battery_capacity_kWh"] - capacity) < 0.001
            and abs(candidate["battery_power_limit_kW"] - power) < 0.001
        ):
            return index
    return 0


def _to_float(value, fallback):
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


if __name__ == "__main__":
    app.run(debug=True)
