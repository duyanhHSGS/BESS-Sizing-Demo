from __future__ import annotations

from bess.core.settings import SYSTEM_CONFIG
from bess.core.timebase import build_tariff_windows


def _system_default_config() -> dict:
    battery = SYSTEM_CONFIG["BESS"]
    tariff = SYSTEM_CONFIG["Tariff"]
    time_config = SYSTEM_CONFIG["Time"]
    operation = SYSTEM_CONFIG["Operation"]
    safety = SYSTEM_CONFIG["Safety"]
    return {
        "E_cap_kWh": battery["E_cap_kWh"],
        "P_rated_kW": battery["P_rated_kW"],
        "eta_ch": battery["eta_ch"],
        "eta_dis": battery["eta_dis"],
        "soc_min": battery["soc_min"],
        "soc_max": battery["soc_max"],
        "soc_safety_buffer": battery["soc_safety_buffer"],
        "soc_min_emergency": battery["soc_min_emergency"],
        "dt_hours": float(time_config["dt_hours"]),
        "price_peak": tariff["price_peak_VND_per_kWh"],
        "price_mid": tariff["price_mid_VND_per_kWh"],
        "price_off": tariff["price_off_VND_per_kWh"],
        "T_cap": tariff["T_cap_VND_per_kW_per_month"],
        "P_target_user_kW": operation["P_target_user_kW"],
        "ENABLE_EXPORT": tariff["ENABLE_EXPORT"],
        "FIT_PRICE": tariff["FIT_PRICE_VND_per_kWh"],
        "V_NOMINAL": safety["V_NOMINAL"],
        "V_BLACKOUT_TH": safety["V_BLACKOUT_TH"],
        "T_DERATE": list(safety["T_DERATE"]),
        "peak_windows": tariff["peak_windows"],
        "off_windows": tariff["off_windows"],
    }


DEFAULT_CONFIG = _system_default_config()


def _resolve_config(config: dict | None = None) -> dict:
    resolved = dict(DEFAULT_CONFIG)
    if config:
        resolved.update(config)
    return resolved


class BESSConfig:
    """Shared battery, tariff, timing, and safety configuration for PPO/PPO2."""

    def __init__(self, config: dict | None = None):
        cfg = _resolve_config(config)
        self.E_cap = float(cfg["E_cap_kWh"])
        self.P_rated_nominal = float(cfg["P_rated_kW"])
        self.eta_ch = float(cfg["eta_ch"])
        self.eta_dis = float(cfg["eta_dis"])
        self.eta_RT = self.eta_ch * self.eta_dis
        self.SOC_min = float(cfg["soc_min"])
        self.SOC_max = float(cfg["soc_max"])
        self.SOC_chg = self.SOC_max
        self.SOC_safety = float(cfg["soc_safety_buffer"])
        self.SOC_min_emergency = float(cfg["soc_min_emergency"])
        self.SOC_SAFETY_BUFFER = self.SOC_safety
        self.peak_windows = str(cfg.get("peak_windows", ""))
        self.off_windows = str(cfg.get("off_windows", ""))
        self.set_dt(float(cfg["dt_hours"]))
        self.price_peak = float(cfg["price_peak"])
        self.price_mid = float(cfg["price_mid"])
        self.price_off = float(cfg["price_off"])
        self.T_cap = float(cfg["T_cap"])
        self.P_target_user = float(cfg["P_target_user_kW"])
        self.FIT_PRICE = float(cfg["FIT_PRICE"])
        self.ENABLE_EXPORT = bool(cfg["ENABLE_EXPORT"])
        self.V_NOMINAL = float(cfg["V_NOMINAL"])
        self.V_BLACKOUT_TH = float(cfg["V_BLACKOUT_TH"])
        self.T_DERATE = list(cfg["T_DERATE"])
        self.spread_peak_off = self.price_peak - self.price_off / self.eta_RT
        self.ARB_ENABLED = self.spread_peak_off > 0

    def set_dt(self, dt_hours: float) -> None:
        """Set native timestep and rebuild tariff step indices from clock ranges."""
        self.dt = float(dt_hours)
        windows = build_tariff_windows(self.peak_windows, self.off_windows, self.dt)
        self.W1 = windows["W1"]
        self.W2 = windows["W2"]
        self.INTER = windows["INTER"]
        self.OFF = windows["OFF"]
        self.W1_START = windows["W1_START"]
        self.W2_START = windows["W2_START"]
        self.OFF_PEAK_END_STEP = windows["OFF_PEAK_END_STEP"]


_DEFAULTS = BESSConfig()


def tariff_for_step(step: int, cfg: BESSConfig | None = None) -> float:
    cfg = cfg or _DEFAULTS
    if step in cfg.OFF:
        return cfg.price_off
    if step in cfg.W1 or step in cfg.W2:
        return cfg.price_peak
    return cfg.price_mid
