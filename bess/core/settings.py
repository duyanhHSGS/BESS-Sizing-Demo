"""Canonical project defaults.

Keep literal defaults here once. Other modules derive runtime config from these
values instead of carrying their own battery/tariff/timestep copies.
"""

DEFAULT_PARAMETERS = {
    "selected_data_csv": "offline_tande-15.csv",
    "battery_capacity_kWh": "1250",
    "battery_power_limit_kW": "450",
    "charge_efficiency": "0.9",
    "discharge_efficiency": "0.9",
    "dt": "0.25",
    "battery_wear_cost": "500",
    "minimum_soc": "0.20",
    "maximum_soc": "0.90",
    "required_final_soc": "0.50",
    "billing_mode": "2tc",
    "billing_sunday": True,
    "billing_expensive": "2251",
    "billing_normal": "1332",
    "billing_cheap": "904",
    "billing_peak_penalty": "285000",
    "billing_windows_expensive": "17:30-22:30",
    "billing_windows_cheap": "00:00-06:00",
    "billing_battery_per_kWh": "5000000",
    "billing_battery_per_kW": "4000000",
    "billing_yearly_maintain_percentage": "0.02",
    "billing_discount_rate": "0.08",
    "billing_years": "10",
    "billing_real_saving_factor": "0.6",
    "use_sample_battery_options": "no",
}

DEFAULT_DT_HOURS = float(DEFAULT_PARAMETERS["dt"])

# Runtime system defaults are derived from DEFAULT_PARAMETERS so the web UI,
# dispatch, training, and fallback controller cannot silently disagree.
SYSTEM_CONFIG = {
    "BESS": {
        "E_cap_kWh": float(DEFAULT_PARAMETERS["battery_capacity_kWh"]),
        "P_rated_kW": float(DEFAULT_PARAMETERS["battery_power_limit_kW"]),
        "battery_wear_cost_vnd_per_kwh": float(DEFAULT_PARAMETERS["battery_wear_cost"]),
        "eta_ch": float(DEFAULT_PARAMETERS["charge_efficiency"]),
        "eta_dis": float(DEFAULT_PARAMETERS["discharge_efficiency"]),
        "soc_min": float(DEFAULT_PARAMETERS["minimum_soc"]),
        "soc_max": float(DEFAULT_PARAMETERS["maximum_soc"]),
        "soc_safety_buffer": 0.05,
        "soc_eod": float(DEFAULT_PARAMETERS["required_final_soc"]),
        "soc_min_emergency": 0.05,
    },
    "Tariff": {
        "price_peak_VND_per_kWh": float(DEFAULT_PARAMETERS["billing_expensive"]),
        "price_mid_VND_per_kWh": float(DEFAULT_PARAMETERS["billing_normal"]),
        "price_off_VND_per_kWh": float(DEFAULT_PARAMETERS["billing_cheap"]),
        "T_cap_VND_per_kW_per_month": float(DEFAULT_PARAMETERS["billing_peak_penalty"]),
        "peak_windows": DEFAULT_PARAMETERS["billing_windows_expensive"],
        "off_windows": DEFAULT_PARAMETERS["billing_windows_cheap"],
        "sunday_no_peak": bool(DEFAULT_PARAMETERS["billing_sunday"]),
        "FIT_PRICE_VND_per_kWh": 1200.0,
        "ENABLE_EXPORT": False,
    },
    "Time": {
        "dt_hours": DEFAULT_DT_HOURS,
    },
    "Operation": {
        "P_target_user_kW": 350.0,
    },
    "Safety": {
        "V_NOMINAL": 1.0,
        "V_BLACKOUT_TH": 0.85,
        "T_DERATE": [[35, 1.0], [42, 0.7], [45, 0.5], [999, 0.0]],
    },
}

# PPO training defaults. PPO_GAMMA is also used by potential-based SOC shaping.
PPO_GAMMA = 0.995
PPO_LAMBDA = 0.97
PRO_GAMMA = 0.995
GREPO_GAMMA = 0.995
GREPRO_GAMMA = 0.995
PPO2_GAMMA = 1.0
PPO2_LAM_ENERGY = 0.97
PPO2_LAM_PEAK = 0.97

# Live UI data: main.py uses these when optional sample sizing is enabled.
SAMPLE_BATTERY_CANDIDATES = tuple(
    {
        "id": f"{int(capacity)}kwh-{ratio_label}",
        "label": f"{int(capacity):,} kWh / {capacity * ratio:,.1f} kW ({ratio_label})",
        "battery_capacity_kWh": capacity,
        "battery_power_limit_kW": capacity * ratio,
        "power_ratio": ratio,
    }
    for capacity in (250.0, 500.0, 750.0, 1000.0, 1250.0)
    for ratio, ratio_label in ((0.35, "0.35C"), (0.50, "0.50C"), (0.70, "0.70C"))
)

# Form fields are derived from the canonical parameter keys rather than copied
# into a second hand-maintained list. dt is detected from the selected dataset;
# billing_mode and billing_sunday are handled separately by main.py.
FORM_FIELDS = tuple(
    key
    for key in DEFAULT_PARAMETERS
    if key not in {"dt", "billing_mode", "billing_sunday"}
)
