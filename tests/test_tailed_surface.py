"""Independent analytical fixtures for tailed surfaces and day-zero pricing."""

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.special import ndtr

from black import BlackPricer
from coupled_tails import CoupledTailSlice
from forward_pde import ForwardPDESolver
from local_vol import DupireLocalVolatility
from tailed_surface import TailedCallSurface, TailedShortEndSurface


@dataclass(frozen=True)
class AnalyticalCore:
    """Known Black prices and independent analytical strike derivatives."""

    maturity: float
    sigma: float = 0.2

    @property
    def pricer(self):
        return BlackPricer(
            100.0 * np.exp(0.02 * self.maturity),
            np.exp(-0.03 * self.maturity),
            self.maturity,
        )

    @property
    def strike_origin(self):
        return 0.5 * self.pricer.forward

    @property
    def strike_max(self):
        return 1.5 * self.pricer.forward

    def price(self, strikes):
        return self.pricer.price(
            strikes,
            self.sigma,
            "call",
        )

    def strike_slope(self, strikes):
        strikes = np.asarray(strikes, dtype=float)
        root_w = self.sigma * np.sqrt(self.maturity)

        d2 = (
            np.log(self.pricer.forward / strikes) / root_w
            - 0.5 * root_w
        )

        return -self.pricer.discount_factor * ndtr(d2)

    def strike_curvature(self, strikes):
        strikes = np.asarray(strikes, dtype=float)
        root_w = self.sigma * np.sqrt(self.maturity)

        d2 = (
            np.log(self.pricer.forward / strikes) / root_w
            - 0.5 * root_w
        )

        phi = (
            np.exp(-0.5 * d2**2)
            / np.sqrt(2.0 * np.pi)
        )

        return (
            self.pricer.discount_factor
            * phi
            / (strikes * root_w)
        )


class TestTailedSurface(unittest.TestCase):
    def setUp(self):
        self.cores = (
            AnalyticalCore(0.1),
            AnalyticalCore(0.2),
        )

        self.slices = tuple(
            CoupledTailSlice(core, -0.09, 0.09)
            for core in self.cores
        )

        self.base = TailedCallSurface(
            self.slices,
            -0.6,
            0.6,
        )

        self.extended = TailedShortEndSurface(
            self.base,
            100.0,
        )

    def test_pillar_prices_derivatives_and_scalar_evaluation(self):
        z = np.exp(
            np.array([-0.5, -0.04, 0.0, 0.04, 0.5])
        )

        for model in self.slices:
            for derivative in (0, 1, 2):
                np.testing.assert_allclose(
                    self.base.normalized_call(
                        z,
                        model.maturity,
                        derivative,
                    ),
                    model.normalized_call(z, derivative),
                    rtol=0.0,
                    atol=1e-15,
                )

                scalar = self.base.normalized_call(
                    float(z[0]),
                    model.maturity,
                    derivative,
                )

                self.assertEqual(
                    np.asarray(scalar).shape,
                    (),
                )

                self.assertAlmostEqual(
                    float(scalar),
                    model.normalized_call(z, derivative)[0],
                )

    def test_interpolation_and_calendar_derivative(self):
        z = np.exp(np.array([-0.5, 0.0, 0.5]))
        time = 0.15

        expected = 0.5 * (
            self.slices[0].normalized_call(z)
            + self.slices[1].normalized_call(z)
        )

        np.testing.assert_allclose(
            self.base.normalized_call(z, time),
            expected,
            atol=1e-15,
        )

        bump = 1e-5

        fd = (
            self.base.normalized_call(z, time + bump)
            - self.base.normalized_call(z, time - bump)
        ) / (2.0 * bump)

        np.testing.assert_allclose(
            self.base.normalized_time_derivative(z, time),
            fd,
            rtol=1e-7,
            atol=1e-11,
        )

        self.assertTrue(
            np.all(
                self.base.normalized_time_derivative(z, time)
                >= 0.0
            )
        )

        self.assertTrue(
            all(
                row["passed"]
                for row in self.base.calendar_checks()
            )
        )

    def test_otm_iv_reprices_both_wings(self):
        z = np.exp(
            np.array([-0.5, -0.01, 0.0, 0.01, 0.5])
        )
        time = 0.15

        forward = self.base.forward(time)
        discount = self.base.discount_factor(time)

        volatility = self.base.implied_volatility(
            forward * z,
            time,
        )

        pricer = BlackPricer(
            forward,
            discount,
            time,
        )

        expected = discount * forward * np.where(
            z < 1.0,
            self.base.normalized_put(z, time),
            self.base.normalized_call(z, time),
        )

        actual = np.where(
            z < 1.0,
            pricer.price(forward * z, volatility, "put"),
            pricer.price(forward * z, volatility, "call"),
        )

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-7,
            atol=2e-9,
        )

        for strike in (
            forward * z[0],
            forward * z[-1],
        ):
            self.assertEqual(
                np.asarray(
                    self.base.implied_volatility(strike, time)
                ).shape,
                (),
            )

    def test_invalid_domain_time_and_calendar_are_rejected(self):
        with self.assertRaises(ValueError):
            self.base.normalized_call(np.exp(0.7), 0.15)

        with self.assertRaises(ValueError):
            self.base.forward(0.0)

        with self.assertRaises(ValueError):
            self.base.normalized_time_derivative(
                1.0,
                0.15,
                "centre",
            )

        later = CoupledTailSlice(
            AnalyticalCore(0.2, 0.05),
            -0.09,
            0.09,
        )

        with self.assertRaises(ValueError):
            TailedCallSurface(
                (self.slices[0], later)
            )

    def test_broadcast_strike_time_arrays_match_scalar_evaluation(self):
        z = np.exp(
            np.array([-0.03, 0.03])
        )[:, None]

        times = np.array(
            [0.1, 0.15, 0.2]
        )[None, :]

        for derivative in (0, 1, 2):
            expected = np.array(
                [
                    [
                        float(
                            self.base.normalized_call(
                                strike,
                                time,
                                derivative,
                            )
                        )
                        for time in times[0]
                    ]
                    for strike in z[:, 0]
                ]
            )

            np.testing.assert_allclose(
                self.base.normalized_call(
                    z,
                    times,
                    derivative,
                ),
                expected,
                rtol=0.0,
                atol=1e-15,
            )

        expected = np.array(
            [
                [
                    float(
                        self.base.normalized_time_derivative(
                            strike,
                            time,
                        )
                    )
                    for time in times[0]
                ]
                for strike in z[:, 0]
            ]
        )

        np.testing.assert_allclose(
            self.base.normalized_time_derivative(z, times),
            expected,
            rtol=0.0,
            atol=1e-15,
        )

        for derivative in (0, 1, 2):
            parity_adjustment = (
                z - 1.0
                if derivative == 0
                else 1.0
                if derivative == 1
                else 0.0
            )

            np.testing.assert_allclose(
                self.base.normalized_put(
                    z,
                    times,
                    derivative,
                ),
                self.base.normalized_call(
                    z,
                    times,
                    derivative,
                ) + parity_adjustment,
                rtol=1e-12,
                atol=1e-15,
            )

        physical_strikes = self.base.forward(times) * z

        volatility = self.base.implied_volatility(
            physical_strikes,
            times,
        )

        self.assertEqual(volatility.shape, (2, 3))

        expected_iv = np.array(
            [
                [
                    float(
                        self.base.implied_volatility(
                            physical_strikes[row, column],
                            times[0, column],
                        )
                    )
                    for column in range(3)
                ]
                for row in range(2)
            ]
        )

        np.testing.assert_allclose(
            volatility,
            expected_iv,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_dupire_broadcast_and_post_first_expiry_variance(self):
        z = np.exp(
            np.array([-0.03, 0.0, 0.03])
        )
        time = 0.15

        expected = (
            2.0
            * self.base.normalized_time_derivative(z, time)
            / (
                z**2
                * self.base.normalized_call(
                    z,
                    time,
                    derivative=2,
                )
            )
        )

        np.testing.assert_allclose(
            DupireLocalVolatility(
                self.base
            ).normalized_variance(z, time),
            expected,
            rtol=1e-12,
        )

        np.testing.assert_allclose(
            self.extended.normalized_variance(z, time),
            expected,
            rtol=1e-12,
        )

        np.testing.assert_allclose(
            self.base.normalized_call(
                z,
                np.full(z.shape, time),
                derivative=2,
            ),
            self.base.normalized_call(
                z,
                time,
                derivative=2,
            ),
            rtol=0.0,
            atol=1e-15,
        )

    def test_zero_payoff_carry_and_constant_volatility_limit(self):
        np.testing.assert_array_equal(
            self.extended.call_price(
                [90.0, 100.0, 110.0],
                0.0,
            ),
            [10.0, 0.0, 0.0],
        )

        self.assertEqual(
            self.extended.forward(0.0),
            100.0,
        )
        self.assertEqual(
            self.extended.discount_factor(0.0),
            1.0,
        )

        z = np.exp(
            np.array([-0.02, 0.0, 0.02])
        )

        for time in (0.0, 0.025, 0.05, 0.1):
            side = (
                "left"
                if time == 0.1
                else "right"
            )

            np.testing.assert_allclose(
                self.extended.normalized_variance(
                    z,
                    time,
                    side,
                ),
                0.04,
                rtol=5e-8,
                atol=1e-10,
            )

        with self.assertRaises(ValueError):
            self.extended.normalized_variance(
                z,
                0.0,
                "left",
            )

    def test_short_end_pde_identity_and_anchor_join(self):
        z = np.exp(
            np.array([-0.45, -0.025, 0.0, 0.025, 0.45])
        )
        time = 0.04

        variance = self.extended.normalized_variance(
            z,
            time,
        )
        curvature = self.extended.normalized_call(
            z,
            time,
            derivative=2,
        )
        time_derivative = (
            self.extended.normalized_time_derivative(
                z,
                time,
            )
        )

        np.testing.assert_allclose(
            time_derivative,
            0.5 * variance * z**2 * curvature,
            rtol=2e-12,
            atol=1e-14,
        )

        left_variance = self.extended.normalized_variance(
            z,
            0.1,
            "left",
        )
        anchor_time_derivative = (
            self.extended.normalized_time_derivative(
                z,
                0.1,
                "left",
            )
        )
        anchor_curvature = self.base.normalized_call(
            z,
            0.1,
            derivative=2,
        )

        np.testing.assert_allclose(
            anchor_time_derivative,
            0.5 * left_variance * z**2 * anchor_curvature,
            rtol=2e-12,
            atol=1e-14,
        )

        anchor = self.extended.normalized_call(z, 0.1)
        near_left = self.extended.normalized_call(
            z,
            np.nextafter(0.1, 0.0),
        )

        np.testing.assert_allclose(
            near_left,
            anchor,
            rtol=1e-7,
            atol=2e-10,
        )

    def test_loaded_manifest_is_rechecked_against_models(self):
        rows = [
            {
                "core_file": f"{index}.npz",
                **model.metadata(),
            }
            for index, model in enumerate(self.slices)
        ]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tails.json"

            specification = {
                "format_version": 1,
                "model_type": "calendar_ordered_tail_slices",
                "slices": rows,
            }

            path.write_text(
                json.dumps(specification),
                encoding="utf-8",
            )

            with patch(
                "tailed_surface.ConvexCallSpline.load",
                side_effect=self.cores,
            ):
                loaded = TailedCallSurface.load(
                    path,
                    -0.6,
                    0.6,
                )

            np.testing.assert_allclose(
                loaded.normalized_call([0.8, 1.2], 0.15),
                self.base.normalized_call([0.8, 1.2], 0.15),
            )

            rows[0]["maturity_years"] = 0.11

            path.write_text(
                json.dumps(specification),
                encoding="utf-8",
            )

            with patch(
                "tailed_surface.ConvexCallSpline.load",
                return_value=self.cores[0],
            ):
                with self.assertRaises(ValueError):
                    TailedCallSurface.load(path)

    def test_day_zero_black_prices_converge_with_space_refinement(self):
        times = np.array([0.0, 0.025, 0.05, 0.1])
        errors = []

        for count in (80, 160):
            solver = ForwardPDESolver(
                -0.04,
                0.04,
                count,
                max_time_step=1e-4,
                rannacher_steps=2,
            )

            z = solver.normalized_strikes

            result = solver.solve(
                times,
                np.maximum(1.0 - z, 0.0),
                self.extended.normalized_variance,
                lambda time: tuple(
                    self.extended.normalized_call(
                        z[[0, -1]],
                        time,
                    )
                ),
                time_breaks=[0.1],
            )

            reference = BlackPricer(
                1.0,
                1.0,
                0.1,
            ).price(z, 0.2, "call")

            errors.append(
                float(
                    np.max(
                        np.abs(
                            result.normalized_calls[-1, 1:-1]
                            - reference[1:-1]
                        )
                    )
                )
            )

        self.assertLess(errors[-1], 1e-5)
        self.assertLess(errors[-1], 0.4 * errors[0])


if __name__ == "__main__":
    unittest.main()