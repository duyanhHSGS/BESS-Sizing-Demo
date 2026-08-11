from __future__ import annotations

import math
from dataclasses import dataclass

from EXPERIMENT_FIELD.brain_env import BrainObservation


@dataclass(frozen=True, slots=True)
class Brain1Decision:
    """One explainable, deterministic Brain 1 answer."""

    action: float
    label: str
    reason_code: str
    reason: str
    normalized_soc: float
    normalized_tariff: float
    cheap_threshold_normalized: float
    expensive_threshold_normalized: float


@dataclass(frozen=True, slots=True)
class Brain1Agent:
    """Deterministic school-answer benchmark for obvious tariff arbitrage.

    Brain1 deliberately uses only two of BrainEnv's seven eyes:
      - eye 4 (index 3): normalized battery state of charge, 0=minimum and 1=maximum
      - eye 5 (index 4): normalized current tariff, 0..1

    It never performs physics, billing, forecasting, learning, or reward shaping.
    BrainEnv remains responsible for clipping requested actions to physical reality.
    """

    cheap_tariff_max_normalized: float
    expensive_tariff_min_normalized: float

    def __post_init__(self) -> None:
        cheap_threshold = float(self.cheap_tariff_max_normalized)
        expensive_threshold = float(self.expensive_tariff_min_normalized)

        if not math.isfinite(cheap_threshold) or not math.isfinite(expensive_threshold):
            raise ValueError("Brain1 tariff thresholds must be finite")
        if cheap_threshold < 0.0 or cheap_threshold > 1.0:
            raise ValueError("cheap_tariff_max_normalized must be inside [0, 1]")
        if expensive_threshold < 0.0 or expensive_threshold > 1.0:
            raise ValueError("expensive_tariff_min_normalized must be inside [0, 1]")
        if cheap_threshold >= expensive_threshold:
            raise ValueError(
                "cheap_tariff_max_normalized must be strictly less than "
                "expensive_tariff_min_normalized"
            )

        object.__setattr__(self, "cheap_tariff_max_normalized", cheap_threshold)
        object.__setattr__(self, "expensive_tariff_min_normalized", expensive_threshold)

    def act(self, observation: BrainObservation) -> float:
        """Return exactly -1 (charge), 0 (idle), or +1 (discharge)."""
        return self.decide(observation).action

    def decide(self, observation: BrainObservation) -> Brain1Decision:
        """Return the action and the exact school-answer reason behind it."""
        try:
            observation_length = len(observation)
        except TypeError as exc:
            raise TypeError("Brain1 observation must be a seven-value sequence") from exc

        if observation_length != 7:
            raise ValueError("Brain1 observation must contain exactly 7 values")

        values: list[float] = []
        for value in observation:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise TypeError("Brain1 observation values must be numeric") from exc
            if not math.isfinite(numeric_value):
                raise ValueError("Brain1 observation values must all be finite")
            values.append(numeric_value)

        normalized_soc = values[3]
        normalized_tariff = values[4]

        if normalized_soc < 0.0 or normalized_soc > 1.0:
            raise ValueError("Brain1 SOC eye must be inside [0, 1]")
        if normalized_tariff < 0.0 or normalized_tariff > 1.0:
            raise ValueError("Brain1 tariff eye must be inside [0, 1]")

        if (
            normalized_tariff <= self.cheap_tariff_max_normalized
            and normalized_soc < 1.0
        ):
            return Brain1Decision(
                action=-1.0,
                label="CHARGE",
                reason_code="cheap_with_room",
                reason="The tariff is cheap and the battery still has room, so Brain 1 requests full charge.",
                normalized_soc=normalized_soc,
                normalized_tariff=normalized_tariff,
                cheap_threshold_normalized=self.cheap_tariff_max_normalized,
                expensive_threshold_normalized=self.expensive_tariff_min_normalized,
            )

        if (
            normalized_tariff >= self.expensive_tariff_min_normalized
            and normalized_soc > 0.0
        ):
            return Brain1Decision(
                action=1.0,
                label="DISCHARGE",
                reason_code="expensive_with_energy",
                reason="The tariff is expensive and the battery has usable energy, so Brain 1 requests full discharge.",
                normalized_soc=normalized_soc,
                normalized_tariff=normalized_tariff,
                cheap_threshold_normalized=self.cheap_tariff_max_normalized,
                expensive_threshold_normalized=self.expensive_tariff_min_normalized,
            )

        if normalized_tariff <= self.cheap_tariff_max_normalized:
            reason_code = "cheap_but_full"
            reason = "The tariff is cheap, but the battery is already full, so Brain 1 idles."
        elif normalized_tariff >= self.expensive_tariff_min_normalized:
            reason_code = "expensive_but_empty"
            reason = "The tariff is expensive, but the battery is empty, so Brain 1 idles."
        else:
            reason_code = "middle_tariff"
            reason = "The tariff is between the cheap and expensive thresholds, so Brain 1 idles."
        return Brain1Decision(
            action=0.0,
            label="IDLE",
            reason_code=reason_code,
            reason=reason,
            normalized_soc=normalized_soc,
            normalized_tariff=normalized_tariff,
            cheap_threshold_normalized=self.cheap_tariff_max_normalized,
            expensive_threshold_normalized=self.expensive_tariff_min_normalized,
        )
