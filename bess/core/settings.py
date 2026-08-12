"""Canonical algorithm-neutral project defaults."""

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
    "required_final_soc": "0.20",
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

FORM_FIELDS = tuple(
    key for key in DEFAULT_PARAMETERS if key not in {"dt", "billing_mode", "billing_sunday"}
)
