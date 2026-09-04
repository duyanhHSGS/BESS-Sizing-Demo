from __future__ import annotations

import unittest

from scipy.sparse import lil_matrix

from bess.evaluation.benchmark import _demand_windows
from bess.evaluation.oracle.oracle_lp import (
    _build_equalities,
    _build_inequalities,
    _Indexes,
    _soc_deadline_steps,
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
            450.0,
            dt,
        )
        expected_rows = steps + len(_demand_windows(steps, dt))
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


if __name__ == "__main__":
    unittest.main()
