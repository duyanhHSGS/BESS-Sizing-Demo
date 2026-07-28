"""System-wide simulation parameters  merged from config/system_config.json."""

# BESS and tariff defaults (used by every optimizer via load_system_config)
SYSTEM_CONFIG = {
    "BESS": {
        "E_cap_kWh": 750.0,
        "P_rated_kW": 350.0,
        "eta_ch": 0.95,
        "eta_dis": 0.95,
        "soc_min": 0.10,
        "soc_max": 0.93,
        "soc_safety_buffer": 0.05,
        "soc_eod": None,
        "soc_min_emergency": 0.05,
    },
    "PV": {
        "P_installed_kWp": 500.0,
        "tilt_deg": 15,
        "orientation": "south",
    },
    "Tariff": {
        "price_peak_VND_per_kWh": 2759.0,
        "price_mid_VND_per_kWh": 1485.0,
        "price_off_VND_per_kWh": 982.0,
        "T_cap_VND_per_kW_per_month": 235414.0,
        "FIT_PRICE_VND_per_kWh": 1200.0,
        "ENABLE_EXPORT": False,
    },
    "TimeWindows_15min": {
        "dt_hours": 0.25,
        "W1_steps": [38, 39, 40, 41, 42, 43, 44, 45],
        "W2_steps": [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
        "INTER_steps_range": [46, 68],
        "OFF_steps_morning_range": [0, 16],
        "OFF_steps_evening_range": [88, 96],
        "W1_START": 38,
        "W2_START": 68,
        "OFF_PEAK_END_STEP": 16,
    },
    "Operation": {
        "P_target_user_kW": 350.0,
    },
    "Safety": {
        "V_NOMINAL": 1.0,
        "V_BLACKOUT_TH": 0.85,
        "T_DERATE": [[35, 1.0], [42, 0.7], [45, 0.5], [999, 0.0]],
    },
    "LoadProfile": {
        "shift_pattern": "1-ca",
        "weekday_peak_kW": 500.0,
        "weekend_kW": 100.0,
        "holiday_kW": 80.0,
    },
    "Simulation": {
        "horizon_days": 30,
        "month": "2025-03",
        "T_battery_C": 30.0,
        "V_grid_pu": 1.0,
        "blackout_steps": None,
    },
}

DEFAULT_PARAMETERS = {
    "selected_data_csv": "offline_data_Youngone.csv",
    "battery_capacity_kWh": "1000",
    "battery_power_limit_kW": "500",
    "charge_efficiency": "0.95",
    "discharge_efficiency": "0.95",
    "dt": "0.25",
    "battery_wear_cost": "0",
    "minimum_soc": "0.10",
    "maximum_soc": "0.90",
    "required_final_soc": "0.50",
    "billing_mode": "2tc",
    "billing_sunday": True,
    "billing_expensive": "2759",
    "billing_normal": "1485",
    "billing_cheap": "982",
    "billing_peak_penalty": "285414",
    "billing_windows_expensive": "17:30-22:30",
    "billing_windows_cheap": "00:00-06:00",
    "billing_battery_per_kWh": "5000000",
    "billing_battery_per_kW": "4000000",
    "billing_yearly_maintain_percentage": "0.02",
    "billing_discount_rate": "0.08",
    "billing_years": "20",
    "billing_real_saving_factor": "0.6",
    "use_sample_battery_options": "no",
}

# PPO training defaults. One gamma value is shared by PPO return discounting
# and the environment's potential-based SOC reward shaping.
PPO_GAMMA = 0.995
PPO_LAMBDA = 0.97
GREPO_GAMMA = 0.995

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

FORM_FIELDS = (
    "selected_data_csv",
    "battery_capacity_kWh",
    "battery_power_limit_kW",
    "charge_efficiency",
    "discharge_efficiency",
    "battery_wear_cost",
    "minimum_soc",
    "maximum_soc",
    "required_final_soc",
    "billing_expensive",
    "billing_normal",
    "billing_cheap",
    "billing_peak_penalty",
    "billing_windows_expensive",
    "billing_windows_cheap",
    "billing_battery_per_kWh",
    "billing_battery_per_kW",
    "billing_yearly_maintain_percentage",
    "billing_discount_rate",
    "billing_years",
    "billing_real_saving_factor",
    "use_sample_battery_options",
)

BILLING_MODE_FIELD = "billing_mode"
BILLING_SUNDAY_FIELD = "billing_sunday"
