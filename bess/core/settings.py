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
        "eta_ch": float(DEFAULT_PARAMETERS["charge_efficiency"]),
        "eta_dis": float(DEFAULT_PARAMETERS["discharge_efficiency"]),
        "soc_min": float(DEFAULT_PARAMETERS["minimum_soc"]),
        "soc_max": float(DEFAULT_PARAMETERS["maximum_soc"]),
        "soc_safety_buffer": 0.05,
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

# Generic PPO training defaults. These are the single source of truth for every
# user-tunable numeric PPO control surfaced by the Training UI and forwarded to
# train_ppo_dataset.py. Fixed environment/observation contracts are documented
# separately and are not disguised as tunables.
PPO_GAMMA = 1.0
PPO_LAMBDA = 0.999
PPO_STEPS = 400_000
PPO_SEED = 0
PPO_LEARNING_RATE = 1e-4
PPO_CLIP = 0.2
PPO_FINE_TUNE_EPOCHS = 1
PPO_MINIBATCH = 256
PPO_ENTROPY_COEF = 0.0
PPO_VALUE_COEF = 0.5
PPO_TARGET_KL = 0.02
PPO_ACTOR_GRAD_CLIP = 0.5
PPO_CRITIC_GRAD_CLIP = 0.5
PPO_HIDDEN_SIZE = 64
PPO_INITIAL_LOG_STD = -0.5
PPO_BC_FINE_TUNE_LOG_STD = -1.5
PPO_VALIDATE_EVERY_UPDATES = 1
PPO_CHALLENGER_RESET_PATIENCE = 6
PPO_CHALLENGER_RESETS_ENABLED = True
PPO_RESET_OPTIMIZER_ON_REANCHOR = True
PPO_PRESERVE_CRITIC_ON_REANCHOR = False
PPO_ACTION_MISMATCH_SHAPING_SCALE = 0.10
PPO_ORACLE_BC_ENABLED = True
PPO_ORACLE_BC_MAX_EPOCHS = 100
PPO_ORACLE_BC_LEARNING_RATE = 1e-3
PPO_ORACLE_BC_MINIBATCH = 256
PPO_ORACLE_BC_TARGET_MSE = 1e-4
PPO_LOG_EVERY_UPDATES = 1
PPO_TORCH_THREADS = 6
PPO_FIT_CONTROL_DT_MINUTES = 30.0

PPO_TUNABLE_DEFAULTS = {
    "steps": PPO_STEPS,
    "seed": PPO_SEED,
    "gamma": PPO_GAMMA,
    "lambda": PPO_LAMBDA,
    "learning_rate": PPO_LEARNING_RATE,
    "ppo_clip": PPO_CLIP,
    "ppo_epochs": PPO_FINE_TUNE_EPOCHS,
    "minibatch": PPO_MINIBATCH,
    "entropy_coef": PPO_ENTROPY_COEF,
    "value_coef": PPO_VALUE_COEF,
    "target_kl": PPO_TARGET_KL,
    "actor_grad_clip": PPO_ACTOR_GRAD_CLIP,
    "critic_grad_clip": PPO_CRITIC_GRAD_CLIP,
    "hidden_size": PPO_HIDDEN_SIZE,
    "initial_log_std": PPO_INITIAL_LOG_STD,
    "ppo_start_log_std": PPO_BC_FINE_TUNE_LOG_STD,
    "validate_every_updates": PPO_VALIDATE_EVERY_UPDATES,
    "challenger_reset_patience": PPO_CHALLENGER_RESET_PATIENCE,
    "challenger_resets_enabled": PPO_CHALLENGER_RESETS_ENABLED,
    "reset_optimizer_on_reanchor": PPO_RESET_OPTIMIZER_ON_REANCHOR,
    "preserve_critic_on_reanchor": PPO_PRESERVE_CRITIC_ON_REANCHOR,
    "action_mismatch_shaping_scale": PPO_ACTION_MISMATCH_SHAPING_SCALE,
    "oracle_bc_enabled": PPO_ORACLE_BC_ENABLED,
    "oracle_bc_max_epochs": PPO_ORACLE_BC_MAX_EPOCHS,
    "oracle_bc_learning_rate": PPO_ORACLE_BC_LEARNING_RATE,
    "oracle_bc_minibatch": PPO_ORACLE_BC_MINIBATCH,
    "oracle_bc_target_mse": PPO_ORACLE_BC_TARGET_MSE,
    "log_every_updates": PPO_LOG_EVERY_UPDATES,
    "torch_threads": PPO_TORCH_THREADS,
}

# PPO2 remains separately configured below.
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
