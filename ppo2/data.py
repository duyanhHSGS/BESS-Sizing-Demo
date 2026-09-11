"""PPO2-owned data containers.

These intentionally mirror the tiny DayData/MonthData attribute contract used by
PPO2 so the PPO2 package does not depend on the generic BESS scenario module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DayData:
    load: np.ndarray
    pv: np.ndarray
    day_type: str
    weather: str
    day_index: int = 0
    date_iso: str | None = None
    forecast: np.ndarray | None = None


@dataclass
class MonthData:
    days: list[DayData] = field(default_factory=list)
    source: str = "sim"

    def __len__(self) -> int:
        return len(self.days)


# TODO(PPO2-HOME): keep these containers deliberately small; add fields only when
# PPO2 itself consumes them, never to mirror unrelated GUI/generic-PPO state.
