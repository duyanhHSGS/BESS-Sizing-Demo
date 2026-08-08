from __future__ import annotations

import unittest

from bess.core.timebase import (
    build_tariff_windows,
    demand_window_steps,
    fixed_demand_block_averages,
    fixed_demand_windows,
    steps_per_day_from_dt,
)


class TimebaseTests(unittest.TestCase):
    def test_resolution_is_derived_from_dt(self):
        cases = (
            (15.0, 96, 2, 24, 70, 20),
            (1.0, 1440, 30, 360, 1050, 300),
            (0.5, 2880, 60, 720, 2100, 600),
        )
        for dt_minutes, expected_day_steps, expected_30min_steps, expected_off_steps, expected_peak_start, expected_peak_steps in cases:
            with self.subTest(dt_minutes=dt_minutes):
                dt_hours = dt_minutes / 60.0
                windows = build_tariff_windows(
                    "17:30-22:30",
                    "00:00-06:00",
                    dt_hours,
                )
                self.assertEqual(steps_per_day_from_dt(dt_hours), expected_day_steps)
                self.assertEqual(demand_window_steps(dt_hours), expected_30min_steps)
                self.assertEqual(len(windows["OFF"]), expected_off_steps)
                self.assertEqual(windows["OFF_PEAK_END_STEP"], expected_off_steps)
                self.assertEqual(windows["W2"][0], expected_peak_start)
                self.assertEqual(len(windows["W2"]), expected_peak_steps)

    def test_dt_must_tile_30_minute_billing_window(self):
        with self.assertRaisesRegex(ValueError, "30 minutes"):
            demand_window_steps(4.0 / 60.0)

    def test_demand_meter_uses_fixed_non_overlapping_blocks(self):
        values = [100.0, 500.0, 500.0, 100.0]
        self.assertEqual(fixed_demand_block_averages(values, 0.25), [300.0, 300.0])
        self.assertEqual(
            fixed_demand_windows(len(values), 0.25),
            [[(0, 0.5), (1, 0.5)], [(2, 0.5), (3, 0.5)]],
        )

    def test_fixed_blocks_scale_with_native_resolution(self):
        values = list(range(60))
        blocks = fixed_demand_block_averages(values, 1.0 / 60.0)
        self.assertEqual(len(blocks), 2)
        self.assertAlmostEqual(blocks[0], 14.5)
        self.assertAlmostEqual(blocks[1], 44.5)


if __name__ == "__main__":
    unittest.main()
