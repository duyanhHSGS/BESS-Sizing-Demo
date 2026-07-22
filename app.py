from flask import Flask, render_template, request

from benchmark import build_benchmark
from oracle_lp import build_oracle_lp
from settings import (
    BILLING_MODE_FIELD,
    BILLING_SUNDAY_FIELD,
    DEFAULT_PARAMETERS,
    FORM_FIELDS,
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
    return {
        **PARAMETERS,
        "benchmark": benchmark,
        "oracle": build_oracle_lp(PARAMETERS),
        "checked_2tc": "checked" if PARAMETERS["billing_mode"] == "2tc" else "",
        "checked_tou": "checked" if PARAMETERS["billing_mode"] == "tou" else "",
        "checked_sunday": "checked" if PARAMETERS["billing_sunday"] else "",
    }


if __name__ == "__main__":
    app.run(debug=True)
