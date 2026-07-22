DEFAULT_PARAMETERS = {
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
}

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
    "battery_capacity_kWh",
    "battery_power_limit_kW",
    "charge_efficiency",
    "discharge_efficiency",
    "dt",
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
)

BILLING_MODE_FIELD = "billing_mode"
BILLING_SUNDAY_FIELD = "billing_sunday"
