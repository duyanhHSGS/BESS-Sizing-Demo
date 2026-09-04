from __future__ import annotations

import unittest

from scipy.sparse import lil_matrix

from bess.evaluation.benchmark import _demand_windows
from bess.evaluation.oracle.oracle_lp import (
    _build_equalities,
    _build_inequalities,
    _Indexes,
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


if __name__ == "__main__":
    unittest.main()
