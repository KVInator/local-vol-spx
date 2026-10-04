"""Tests for the Treasury discounting proxy."""

import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np

from discount_curve import TreasuryYieldProxy


class TestTreasuryYieldProxy(unittest.TestCase):
    def setUp(self):
        self.proxy = TreasuryYieldProxy(
            as_of=date(2023, 9, 1),
            tenor_days=[30.0, 60.0],
            bey_rates=[0.0551, 0.0555],
            source_url="https://home.treasury.gov",
        )

    def test_existing_35_day_discount_benchmark(self):
        maturity = 35.0 / 365.0
        expected_yield = (
            0.0551 * 25.0 / 30.0
            + 0.0555 * 5.0 / 30.0
        )

        self.assertAlmostEqual(
            self.proxy.yield_bey(maturity),
            expected_yield,
            places=14,
        )
        self.assertAlmostEqual(
            self.proxy.discount_factor(maturity),
            0.99479528,
            places=8,
        )

    def test_vector_discounting_and_zero_maturity(self):
        times = np.array([0.0, 30.0, 45.0, 60.0]) / 365.0
        yields = np.array([0.0551, 0.0551, 0.0553, 0.0555])

        expected = (1.0 + yields / 2.0) ** (-2.0 * times)

        np.testing.assert_allclose(
            self.proxy.discount_factor(times),
            expected,
            rtol=1e-14,
            atol=0.0,
        )
        self.assertEqual(self.proxy.discount_factor(0.0), 1.0)

    def test_short_end_policy_and_maturity_limits(self):
        self.assertAlmostEqual(
            self.proxy.yield_bey(21.0 / 365.0),
            0.0551,
            places=14,
        )

        for maturity in (-0.01, np.nan, 61.0 / 365.0):
            with self.subTest(maturity=maturity):
                with self.assertRaises(ValueError):
                    self.proxy.discount_factor(maturity)

    def test_invalid_anchor_data(self):
        cases = [
            ([30.0, 30.0], [0.05, 0.06]),
            ([30.0, 60.0], [0.05]),
            ([30.0, 60.0], [0.05, np.nan]),
            ([30.0, 60.0], [0.05, -2.0]),
        ]

        for days, rates in cases:
            with self.subTest(days=days, rates=rates):
                with self.assertRaises(ValueError):
                    TreasuryYieldProxy(
                        as_of=date(2023, 9, 1),
                        tenor_days=days,
                        bey_rates=rates,
                        source_url="https://home.treasury.gov",
                    )

    def test_input_arrays_are_copied(self):
        days = np.array([30.0, 60.0])
        rates = np.array([0.0551, 0.0555])

        proxy = TreasuryYieldProxy(
            as_of=date(2023, 9, 1),
            tenor_days=days,
            bey_rates=rates,
            source_url="https://home.treasury.gov",
        )
        days[0] = 10.0
        rates[0] = 0.50

        np.testing.assert_array_equal(
            proxy.tenor_days,
            [30.0, 60.0],
        )
        self.assertAlmostEqual(proxy.yield_bey(0.0), 0.0551)
        self.assertFalse(proxy.tenor_days.flags.writeable)
        self.assertFalse(proxy.bey_rates.flags.writeable)

    def test_loads_dated_reference_file(self):
        definition = {
            "as_of": "2023-09-01",
            "source_url": "https://home.treasury.gov",
            "tenor_days": [30, 60],
            "bey_percent": [5.51, 5.55],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.json"
            path.write_text(
                json.dumps(definition),
                encoding="utf-8",
            )
            restored = TreasuryYieldProxy.from_json(path)

        self.assertEqual(restored.as_of, date(2023, 9, 1))
        np.testing.assert_allclose(
            restored.bey_rates,
            self.proxy.bey_rates,
            rtol=1e-15,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()