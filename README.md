# Sizing Demo — BESS + Solar PV Sizing Tool

A lightweight Flask web application that benchmarks battery energy storage system (BESS) performance against an Oracle Linear Programming (LP) solver. It uses real industrial load and PV data from Youngone to simulate energy cost savings, peak shaving, and battery wear.

---

## Quick Start (Python 3.10)

### 1. Create and activate a virtual environment

**Windows (PowerShell)**

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1

```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate

```

> **Note:** Python 3.10 is recommended. If you do not have Python 3.10 installed, download it from [python.org](https://www.python.org/downloads/release/python-31011/) or manage versions using `pyenv` or `conda`.

---

### 2. Install dependencies

```bash
pip install -r requirements.txt

```

This installs:

* **Flask** (>= 2.0.0) — web framework
* **SciPy** (>= 1.7.0) — LP solver (HiGHS method)

---

### 3. Launch the application

```bash
python app.py

```

Expected output:

```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000

```

---

### 4. Open the UI

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser to access:

* **Benchmark tab** — baseline cost without a battery
* **Oracle LP tab** — theoretical optimum dispatch using perfect foresight
* **Settings form** — configurable battery parameters, tariff, and cost assumptions
* **Save Parameters button** — recalculates both benchmarks based on updated inputs

---

## Project Structure (`Sizing_Demo/`)

```text
Sizing_Demo/
├── app.py                     # Flask entry point
├── benchmark.py               # No-battery baseline calculator
├── oracle_lp.py               # LP optimal dispatch (SciPy)
├── settings.py                # Default parameters & form fields
├── templates/
│   └── index.html             # UI template
├── offline_data_Youngone.csv  # Industrial dataset (required)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation

```

> **Note:** The application expects `offline_data_Youngone.csv` to reside in the same directory as `app.py`. Request this file from the repository maintainer if missing.

---

## Configuration

All configurations can be adjusted through the web form. Default values are defined in `settings.py`:

| Parameter                | Default | Description                              |
| ------------------------ | ------- | ---------------------------------------- |
| `battery_capacity_kWh`   | 1000    | Usable battery capacity (kWh)            |
| `battery_power_limit_kW` | 500     | Max charge/discharge power (kW)          |
| `charge_efficiency`      | 0.95    | Charging efficiency (0–1)                |
| `discharge_efficiency`   | 0.95    | Discharging efficiency (0–1)             |
| `billing_expensive`      | 2759    | Peak energy price (VND/kWh)              |
| `billing_peak_penalty`   | 285414  | Demand charge (VND/kW) for 2-tariff mode |

> **Note:** Monetary values are denominated in Vietnamese Dong (VND). Adjust accordingly for other currencies.

---

## Understanding the Output

* **Benchmark:** Calculates total energy cost without a battery system (grid import only).
* **Oracle LP:** Calculates theoretical minimum cost using an optimal battery dispatch schedule with full future information.
* **Savings Calculation:** `Savings = Benchmark Bill - Oracle Bill - Battery Wear Cost`

The Oracle LP utilizes SciPy's `linprog` (HiGHS solver) to enforce:

* Energy balance (`grid + PV + battery = load`)
* State of Charge (SOC) dynamics via Coulomb counting
* Operational power limits and demand charge windows
* Minimum final SOC constraints

---

## Custom Datasets

To use a custom dataset, update `DATA_PATH` in `benchmark.py`. The input CSV must contain the following columns:

* `day_index`
* `step`
* `P_load_kW`
* `P_pv_kW`
* `day_type`

---

## Python Version Compatibility

| Python Version | Status                                          |
| -------------- | ----------------------------------------------- |
| **3.10**       | Tested                                          |
| **3.11**       | Compatible                                      |
| **3.12**       | Compatible                                      |
| **3.9**        | Compatible (may require earlier SciPy versions) |

---

## Troubleshooting

| Problem                                        | Solution                                                                                  |
| ---------------------------------------------- | ----------------------------------------------------------------------------------------- |
| `ModuleNotFoundError: No module named 'scipy'` | Activate the virtual environment and execute `pip install -r requirements.txt`.           |
| `FileNotFoundError: offline_data_Youngone.csv` | Ensure the CSV file is placed inside the `Sizing_Demo/` directory.                        |
| `ValueError` / Solver failure                  | Verify that `dt > 0` and all battery parameters are set to valid numeric values.          |
| Port conflict ("Address already in use")       | Terminate the process using the port or assign a different port via `app.run(port=5001)`. |

---
