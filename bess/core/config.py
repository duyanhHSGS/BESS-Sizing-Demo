"""Algorithm-neutral configuration consumed by every BrainEnv workflow."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


ENVIRONMENT_CONTRACT_VERSION = 2


def _number(parameters: dict[str, Any], key: str) -> float:
    value = float(parameters[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} must be finite")
    return value


@dataclass(frozen=True, slots=True)
class BrainConfig:
    battery_capacity_kwh: float
    battery_power_kw: float
    charge_efficiency: float
    discharge_efficiency: float
    minimum_soc: float
    maximum_soc: float
    required_final_soc: float
    timestep_hours: float
    battery_wear_vnd_per_kwh: float
    demand_charge_vnd_per_kw: float
    cheap_tariff_vnd_per_kwh: float
    normal_tariff_vnd_per_kwh: float
    expensive_tariff_vnd_per_kwh: float
    cheap_windows: str
    expensive_windows: str
    sunday_no_peak: bool
    billing_mode: str

    @classmethod
    def from_parameters(cls, parameters: dict[str, Any]) -> "BrainConfig":
        mode = str(parameters.get("billing_mode", "2tc")).lower()
        if mode not in {"2tc", "tou"}:
            raise ValueError("billing_mode must be '2tc' or 'tou'")
        config = cls(
            battery_capacity_kwh=_number(parameters, "battery_capacity_kWh"),
            battery_power_kw=_number(parameters, "battery_power_limit_kW"),
            charge_efficiency=_number(parameters, "charge_efficiency"),
            discharge_efficiency=_number(parameters, "discharge_efficiency"),
            minimum_soc=_number(parameters, "minimum_soc"),
            maximum_soc=_number(parameters, "maximum_soc"),
            required_final_soc=_number(parameters, "required_final_soc"),
            timestep_hours=_number(parameters, "dt"),
            battery_wear_vnd_per_kwh=_number(parameters, "battery_wear_cost"),
            demand_charge_vnd_per_kw=(
                _number(parameters, "billing_peak_penalty") if mode == "2tc" else 0.0
            ),
            cheap_tariff_vnd_per_kwh=_number(parameters, "billing_cheap"),
            normal_tariff_vnd_per_kwh=_number(parameters, "billing_normal"),
            expensive_tariff_vnd_per_kwh=_number(parameters, "billing_expensive"),
            cheap_windows=str(parameters.get("billing_windows_cheap", "")),
            expensive_windows=str(parameters.get("billing_windows_expensive", "")),
            sunday_no_peak=bool(parameters.get("billing_sunday", False)),
            billing_mode=mode,
        )
        config.validate()
        return config

    @property
    def initial_soc(self) -> float:
        """Canonical owned-episode starting SOC: always start at the configured minimum."""
        return self.minimum_soc

    def validate(self) -> None:
        if self.battery_capacity_kwh <= 0 or self.battery_power_kw <= 0:
            raise ValueError("battery capacity and power must be greater than zero")
        if not 0 <= self.minimum_soc < self.maximum_soc <= 1:
            raise ValueError("SOC limits must satisfy 0 <= minimum < maximum <= 1")
        if not self.minimum_soc <= self.required_final_soc <= self.maximum_soc:
            raise ValueError("required_final_soc must be inside the SOC limits")
        if not 0 < self.charge_efficiency <= 1 or not 0 < self.discharge_efficiency <= 1:
            raise ValueError("battery efficiencies must be inside (0, 1]")
        if self.timestep_hours <= 0 or self.battery_wear_vnd_per_kwh < 0:
            raise ValueError("timestep must be positive and wear cost non-negative")
        if not (
            0 <= self.cheap_tariff_vnd_per_kwh
            < self.normal_tariff_vnd_per_kwh
            < self.expensive_tariff_vnd_per_kwh
        ):
            raise ValueError("tariffs must satisfy cheap < normal < expensive")

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
