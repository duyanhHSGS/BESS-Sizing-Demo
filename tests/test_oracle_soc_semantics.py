from __future__ import annotations

import unittest
from unittest.mock import patch

from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from bess.evaluation.benchmark import _demand_windows
from bess.evaluation.oracle.oracle_lp import (
    _build_equalities,
    _build_inequalities,
    _Indexes,
    _soc_deadline_steps,
    _solve_month,
    _variable_bounds,
)


class OracleSocSemanticsTests(unittest.TestCase):
    def test_oracle_starts_at_soc_min_without_terminal_soc_constraint(self):
        steps = 8
        dt = 0.25
        idx = _Indexes(steps)
        variable_count = idx.peak + 1
        initial_soc = 0.20

        a_eq, b_eq = _build_equalities(
            lil_matrix,
            steps,
            variable_count,
            idx,
            [700.0] * steps,
            dt,
            1250.0,
            0.9,
            0.9,
            initial_soc,
        )
        self.assertEqual(a_eq[0, idx.soc(0)], 1.0)
        self.assertEqual(b_eq[0], initial_soc)

        a_ub, b_ub = _build_inequalities(
            lil_matrix,
            steps,
            variable_count,
            idx,
            dt,
        )
        expected_rows = len(_demand_windows(steps, dt))
        self.assertEqual(a_ub.shape[0], expected_rows)
        self.assertEqual(len(b_ub), expected_rows)
        self.assertEqual(a_ub[:, idx.soc(steps)].nnz, 0)

    def test_iq67_oracle_requires_soc_max_at_each_0600_deadline(self):
        days = [{"load": [700.0] * 96}, {"load": [600.0] * 96}]
        deadline_steps = _soc_deadline_steps(days, 0.25)
        self.assertEqual(deadline_steps, (24, 120))

        steps = 192
        idx = _Indexes(steps)
        a_eq, b_eq = _build_equalities(
            lil_matrix,
            steps,
            idx.peak + 1,
            idx,
            [700.0] * steps,
            0.25,
            1250.0,
            0.9,
            0.9,
            0.20,
            soc_deadline_steps=deadline_steps,
            soc_deadline_target=0.90,
        )

        self.assertEqual(a_eq[-2, idx.soc(24)], 1.0)
        self.assertEqual(a_eq[-1, idx.soc(120)], 1.0)
        self.assertEqual(b_eq[-2:], [0.90, 0.90])

    def test_oracle_has_only_grid_charge_and_factory_discharge_variables(self):
        steps = 3
        idx = _Indexes(steps)

        self.assertFalse(hasattr(idx, "solar_charge"))
        self.assertEqual(idx.grid_import(0), steps * 2)
        self.assertEqual(idx.soc(0), steps * 3)
        self.assertEqual(idx.peak + 1, steps * 4 + 2)

    def test_discharge_is_capped_by_current_factory_grid_demand(self):
        bounds = _variable_bounds(
            3,
            450.0,
            [700.0, 125.0, 0.0],
            0.20,
            0.90,
        )

        self.assertEqual(bounds[:3], [(0.0, 450.0), (0.0, 125.0), (0.0, 0.0)])
        self.assertEqual(bounds[3:6], [(0.0, 450.0)] * 3)

    def test_pv_above_factory_is_capped_at_zero_without_solar_charging(self):
        day = {
            "day_index": 1,
            "date_iso": "2026-01-01",
            "day_type": "working",
            "load": [50.0, 100.0],
            "pv": [75.0, 20.0],
            "grid": [0.0, 80.0],
        }
        parameters = {
            "battery_wear_cost": 500.0,
            "billing_mode": "energy",
            "billing_expensive": 2251.0,
            "billing_normal": 1332.0,
            "billing_cheap": 904.0,
            "billing_windows_expensive": "17:30-22:30",
            "billing_windows_cheap": "00:00-06:00",
            "billing_sunday": True,
        }

        with patch("bess.evaluation.oracle.oracle_lp.PPO_SOC_DEADLINE_ENABLED", False):
            result = _solve_month(
                linprog,
                lil_matrix,
                [day],
                parameters,
                0.25,
                1250.0,
                450.0,
                0.9,
                0.9,
                0.20,
                0.20,
            )

        self.assertEqual(result[0]["grid"], [0.0, 80.0])
        self.assertEqual(result[0]["grid_charge"], [0.0, 0.0])
        self.assertEqual(result[0]["discharge"], [0.0, 0.0])
        self.assertNotIn("solar_charge", result[0])


if __name__ == "__main__":
    unittest.main()
