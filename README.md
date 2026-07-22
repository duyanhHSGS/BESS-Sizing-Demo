# 🏭 Sizing Demo — BESS + Solar PV Sizing Tool

A lightweight **Flask** web app that benchmarks battery energy storage system (BESS) performance against an Oracle Linear Programming (LP) solver.  
Uses real industrial load/PV data from **Youngone** to simulate **energy cost savings**, **peak shaving**, and **battery wear**.

---

## 🚀 Quick Start (Python 3.10)

### 1️⃣ Create & activate a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

> ✅ Python **3.10** recommended.  
> ⚠️ If you don't have Python 3.10, install it from [python.org](https://www.python.org/downloads/release/python-31011/) (Windows) or use `pyenv` / `conda`.

---

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **Flask** ≥ 2.0.0 — web framework
- **SciPy** ≥ 1.7.0 — LP solver (HiGHS method)

---

### 3️⃣ Launch the app

```bash
python app.py
```

You'll see output like:

```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

---

### 4️⃣ Open the UI

In your browser, go to:  
👉 [http://127.0.0.1:5000](http://127.0.0.1:5000)

You'll see:

- **Benchmark** tab — baseline without battery
- **Oracle LP** tab — theoretical optimum using perfect foresight
- **Settings** form — tweak battery parameters, tariff, and cost assumptions
- Hit **Save Parameters** to recompute both benchmarks

---

## 📂 Project Structure (Sizing_Demo/)

```
Sizing_Demo/
├── app.py                     # Flask entry point
├── benchmark.py               # No-battery baseline calculator
├── oracle_lp.py               # LP optimal dispatch (SciPy)
├── settings.py                # Default parameters & form fields
├── templates/
│   └── index.html             # UI template (not shipped in repo? check)
├── offline_data_Youngone.csv  # Real industrial data (required)
├── requirements.txt           # Python dependencies
└── README.md                  # This file 🫵
```

> **Note:** The app expects `offline_data_Youngone.csv` to be in the same folder as `app.py`. If you don't have it, obtain it from the project maintainer.

---

## ⚙️ Configuration

All config is handled via the web form. Defaults are stored in `settings.py`:

| Parameter             | Default  | Description                                    |
|-----------------------|----------|------------------------------------------------|
| `battery_capacity_kWh`| 1000     | Usable battery capacity (kWh)                  |
| `battery_power_limit_kW`| 500    | Max charge/discharge power (kW)                |
| `charge_efficiency`   | 0.95     | Charging efficiency (0–1)                      |
| `discharge_efficiency`| 0.95     | Discharging efficiency (0–1)                   |
| `billing_expensive`   | 2759     | Peak energy price (VND/kWh)                    |
| `billing_peak_penalty`| 285414   | Demand charge (VND/kW) for 2‑tariff mode       |

> 💸 Prices are in **Vietnamese Dong (VND)** — adjust for your region.

---

## 🔬 Understanding the Output

- **Benchmark** – shows the cost *without* a battery (pure grid import).
- **Oracle LP** – shows the *theoretically best* battery dispatch using perfect future knowledge.
- **Saving** = Benchmark bill − Oracle bill − battery wear cost.

The Oracle LP uses a **linear programming** model (SciPy `linprog` with HiGHS solver).  
It enforces:
- Energy balance (grid + PV + battery = load)
- SOC dynamics (Coulomb counting)
- Power limits & demand charge windows
- Minimum final SOC constraint

---

## 🧪 Run on Different Data

Modify `DATA_PATH` in `benchmark.py` to point to your own CSV file.  
Expected columns: `day_index`, `step`, `P_load_kW`, `P_pv_kW`, `day_type`.

---

## 🐍 Python Version Compatibility

| Python | Status |
|--------|--------|
| 3.10   | ✅ Tested |
| 3.11   | ✅ Works |
| 3.12   | ✅ Works |
| 3.9    | ⚠️ Likely works, but SciPy may need older version |

---

## 🛑 Troubleshooting

| Problem                     | Fix                                      |
|-----------------------------|------------------------------------------|
| `ModuleNotFoundError: No module named 'scipy'` | Activate venv & run `pip install -r requirements.txt` |
| `FileNotFoundError: offline_data_Youngone.csv` | Place the CSV in the `Sizing_Demo/` folder |
| `ValueError: ` The LP solver fails | Check `dt` is > 0 and battery parameters are valid |
| Flask says "Address already in use" | Kill the old process or use a different port: `app.run(port=5001)` |

---

## 🎯 Next Steps

- Play with different battery sizes and see the saving curve.
- Compare Oracle LP against a real DRL agent (coming soon™).
- Add your own tariff structure by editing `_prices_for_day()` in `benchmark.py`.

---

**LEZ GOO!** 🚀🔥 You've got everything you need to size a battery like a pro.  
If something breaks, scream at your terminal. We'll fix it. 😎