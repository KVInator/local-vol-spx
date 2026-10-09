from fixtures import surface_fixture

"Analytical, support, derivative-side and saved-input provenance controls."
import unittest
import numpy as np
import pandas as pd
from diagnostics import (
    SurfaceEvidence,
    SurfaceEvidenceSettings,
    TotalVarianceSurface,
    analytical_controls,
    dupire,
)
from pricing import black_call, black_time_value, invert_time_value
from validation import SSVIValidator


class TotalVarianceControls(unittest.TestCase):

    def test_ssvi_validation_grid_is_distinct_and_inputs_are_preserved(self):
        _, quotes, carry = surface_fixture()
        original_quotes, original_carry = quotes.copy(deep=True), carry.copy(deep=True)
        tables = SSVIValidator().run(quotes, carry)
        pd.testing.assert_frame_equal(quotes, original_quotes)
        pd.testing.assert_frame_equal(carry, original_carry)
        constrained = tables["ssvi_shape_checks"].query("route == 'ssvi_constrained'")
        self.assertTrue(constrained.admissible.eq(constrained.supported).all())
        fields = [
            "negative_calendar_derivatives",
            "negative_density_samples",
            "ill_conditioned_denominators",
            "increasing_prices",
            "vertical_spread_violations",
            "negative_price_butterflies",
            "calendar_price_violations",
        ]
        self.assertTrue(constrained[fields].sum().eq(0).all())
        samples = tables["ssvi_surface_samples"]
        first = samples.loc[samples.days.eq(samples.days.min())]
        self.assertTrue((np.diff(np.exp(first.y)) > 0).all())
        self.assertTrue(first.modeled_short_end.all())
        self.assertFalse(first.quote_derivative_supported.any())
        self.assertEqual(len(tables["ssvi_quote_residuals"]), len(quotes))

    def test_otm_inversion_recovers_known_black_variance_in_both_wings(self):
        y = np.array([-0.4, -0.1, 0, 0.1, 0.4])
        expected = np.full(5, 0.008)
        prices = black_time_value(y, expected)
        recovered, status = invert_time_value(y, prices)
        self.assertTrue((status == "ready").all())
        np.testing.assert_allclose(recovered, expected, rtol=2e-11)

    def test_bad_prices_have_status_and_no_invented_variance(self):
        w, status = invert_time_value(np.zeros(5), np.array([-1, 0, 1, 2, np.nan]))
        self.assertTrue(np.isnan(w).all())
        self.assertEqual(
            status.tolist(),
            [
                "at_or_below_intrinsic",
                "at_or_below_intrinsic",
                "at_or_above_upper_bound",
                "at_or_above_upper_bound",
                "nonfinite",
            ],
        )

    def test_logarithmic_time_value_survives_a_deep_wing(self):
        value = float(black_time_value(0.8, 0.002))
        self.assertGreater(value, 0)
        self.assertLess(value, 1e-50)
        recovered, status = invert_time_value(0.8, value)
        self.assertEqual(str(status), "ready")
        self.assertAlmostEqual(float(recovered), 0.002, places=12)

    def test_flat_black_has_known_local_variance_and_density(self):
        y = np.linspace(-0.1, 0.1, 21)
        w = np.full(y.shape, 0.04 * 0.2)
        r = dupire(y, w, np.zeros_like(y), np.zeros_like(y), np.full(y.shape, 0.04))
        np.testing.assert_allclose(r["variance"], 0.04, rtol=1e-14)
        z = np.exp(y)
        d2 = -y / np.sqrt(w) - np.sqrt(w) / 2
        expected = np.exp(-d2 * d2 / 2) / (z * np.sqrt(2 * np.pi * w))
        np.testing.assert_allclose(r["density_z"], expected, rtol=1e-14)

    def test_negative_calendar_and_density_are_retained_and_masked(self):
        r = dupire(
            np.zeros(2),
            np.full(2, 0.02),
            np.zeros(2),
            np.array([0, -3]),
            np.array([-0.01, 0.01]),
        )
        self.assertLess(r["density_z"][1], 0)
        self.assertTrue(np.isnan(r["variance"]).all())
        self.assertFalse(r["admissible"].any())

    def test_small_positive_denominator_is_not_floored_into_a_valid_value(self):
        r = dupire(
            np.array([0.0]),
            np.array([0.02]),
            np.array([0.0]),
            np.array([-1.9999999999]),
            np.array([0.01]),
        )
        self.assertGreater(r["g"][0], 0)
        self.assertTrue(np.isnan(r["variance"][0]))

    def test_independent_black_benchmark_reproduces_price_and_derivatives(self):
        _, q, c = surface_fixture()
        b = TotalVarianceSurface(q, c)
        y = np.array([-0.065, 0.005, 0.075])
        t = 0.02
        r = b.evaluate(y, t)
        self.assertTrue(r["supported"].all())
        np.testing.assert_allclose(r["w"], 0.04 * t, rtol=2e-08, atol=1e-12)
        np.testing.assert_allclose(r["wt"], 0.04, rtol=2e-08)
        np.testing.assert_allclose(r["wy"], 0, atol=1e-09)
        np.testing.assert_allclose(r["wyy"], 0, atol=1e-06)

    def test_no_short_end_or_wing_extrapolation(self):
        _, q, c = surface_fixture()
        b = TotalVarianceSurface(q, c)
        for t in [0, 1 / 365, 1.0]:
            self.assertFalse(b.evaluate(np.array([0]), t)["supported"].any())
        self.assertFalse(b.evaluate(np.array([-0.101, 0.101]), 0.02)["supported"].any())

    def test_endpoint_time_sides_do_not_invent_an_unobserved_interval(self):
        _, q, c = surface_fixture()
        b = TotalVarianceSurface(q, c)
        left = b.evaluate(np.array([0.003]), b.times[0], "left")
        right = b.evaluate(np.array([0.003]), b.times[0], "right")
        self.assertTrue(np.isfinite(left["w"][0]))
        self.assertTrue(np.isnan(left["wt"][0]))
        self.assertFalse(left["supported"][0])
        self.assertTrue(right["supported"][0])
        self.assertFalse(
            b.evaluate(np.array([0.003]), b.times[-1], "right")["supported"][0]
        )

    def test_pillar_roundoff_does_not_lose_the_price_support(self):
        _, q, c = surface_fixture()
        b = TotalVarianceSurface(q, c)
        r = b.evaluate(np.array([0.003]), b.times[-1] + 1e-16, "left")
        self.assertTrue(r["supported"][0])

    def test_invalid_midpoint_splits_quote_support_without_removing_observation(self):
        _, q, c = surface_fixture()
        idx = q.index[
            (q.expire_date == c.expire_date.iloc[0])
            & np.isclose(q.strike / q.forward, 1)
        ][0]
        q.loc[idx, "call_mid"] = -1
        b = TotalVarianceSurface(q, c)
        self.assertEqual(len(b.observations), len(q))
        self.assertEqual(b.observations.loc[idx, "iv_status"], "at_or_below_intrinsic")
        self.assertFalse(b.evaluate(np.array([0]), 0.02)["supported"][0])
        self.assertTrue(b.evaluate(np.array([0.03]), 0.02)["supported"][0])

    def test_time_derivative_sides_are_distinct_at_nonuniform_pillars(self):
        _, q, c = surface_fixture()
        rates = [0.04, 0.05, 0.045]
        for i, row in c.iterrows():
            mask = q.expire_date.eq(row.expire_date)
            y = np.log(q.loc[mask, "strike"] / row.forward)
            q.loc[mask, "call_mid"] = (
                row.discount_factor
                * row.forward
                * black_call(y.to_numpy(), rates[i] * row.maturity_years)
            )
        b = TotalVarianceSurface(q, c)
        t = c.maturity_years.iloc[1]
        left = b.evaluate(np.array([0.003]), t, "left")
        right = b.evaluate(np.array([0.003]), t, "right")
        self.assertAlmostEqual(left["w"][0], right["w"][0], places=12)
        self.assertGreater(abs(left["wt"][0] - right["wt"][0]), 0.01)

    def test_negative_calendar_interpolation_is_not_repaired(self):
        _, q, c = surface_fixture()
        row = c.iloc[1]
        mask = q.expire_date.eq(row.expire_date)
        y = np.log(q.loc[mask, "strike"] / row.forward)
        q.loc[mask, "call_mid"] = (
            row.discount_factor * row.forward * black_call(y.to_numpy(), 0.0001)
        )
        b = TotalVarianceSurface(q, c)
        self.assertLess(b.evaluate(np.array([0.003]), 0.02)["wt"][0], 0)

    def test_pchip_strike_knot_has_explicit_curvature_sides(self):
        _, q, c = surface_fixture()
        for row in c.to_dict("records"):
            mask = q.expire_date.eq(row["expire_date"])
            y = np.log(q.loc[mask, "strike"] / row["forward"]).to_numpy()
            w = row["maturity_years"] * (0.04 - 0.1 * y + 0.8 * y * y + 0.6 * y * y * y)
            q.loc[mask, "call_mid"] = (
                row["discount_factor"] * row["forward"] * black_call(y, w)
            )
        b = TotalVarianceSurface(q, c)
        knot = float(b.slices[0][0].x[18])
        left = b.evaluate(np.array([knot]), 0.02, strike_side="left")
        right = b.evaluate(np.array([knot]), 0.02, strike_side="right")
        self.assertTrue(left["at_strike_knot"][0])
        self.assertAlmostEqual(left["w"][0], right["w"][0], places=12)
        self.assertGreater(abs(left["wyy"][0] - right["wyy"][0]), 1e-06)

    def test_price_density_and_dupire_controls_refine_at_second_order(self):
        controls = analytical_controls()
        for _, g in controls.groupby("control"):
            error = g.max_variance_error.to_numpy()
            self.assertTrue((error[:-1] / error[1:] > 3.8).all())
            self.assertLess(error[-1], 1e-06)


class AHEvidenceControls(unittest.TestCase):

    def test_native_black_chain_rule_recovers_ah_variance(self):
        from surface import AHLocalVariance

        model, _, _ = surface_fixture()
        for side in ["left", "right"]:
            t = float(model.maturities[1])
            n = SurfaceEvidence.ah_variance_derivatives(model, t, side)
            result = dupire(n["y"], n["w"], n["wy"], n["wyy"], n["wt"])
            expected = AHLocalVariance(model).normalized_variance(
                np.exp(n["y"]), t, side
            )
            mask = (abs(n["y"]) < 0.06) & result["admissible"]
            self.assertGreater(mask.sum(), 100)
            np.testing.assert_allclose(
                result["variance"][mask], expected[mask], rtol=2e-08, atol=1e-10
            )

    def test_run_retains_inputs_and_has_separate_smoothed_coefficients(self):
        m, q, c = surface_fixture()
        original = q.copy(deep=True)
        tables = SurfaceEvidence(
            SurfaceEvidenceSettings(y_points=21, time_subdivisions=1)
        ).run(m, q, c, dict(quote_date="2023-10-02", root="UNKNOWN"))
        pd.testing.assert_frame_equal(q, original)
        self.assertEqual(len(tables["quote_inversions"]), len(q))
        sample = tables["surface_samples"]
        self.assertTrue(sample["T"].gt(0).all())
        self.assertFalse(
            sample.loc[sample["T"].lt(m.maturities[0]), "quote_supported"].any()
        )
        self.assertGreater(
            abs(sample.ah_actual_lv - sample.smoothed_actual_lv).max(), 0
        )
        np.testing.assert_allclose(
            sample.loc[sample["T"].ge(m.maturities[0]), "ah_actual_lv"],
            sample.loc[sample["T"].ge(m.maturities[0]), "smoothed_actual_lv"],
        )
        self.assertTrue(
            np.isfinite(tables["density_moments"].finite_domain_nodal_mass).all()
        )

    def test_saved_carry_mismatch_is_rejected(self):
        m, q, c = surface_fixture()
        c.loc[0, "forward"] += 1
        with self.assertRaises(ValueError):
            SurfaceEvidence().run(
                m, q, c, dict(quote_date="2023-10-02", root="UNKNOWN")
            )

    def test_bad_grid_settings_are_rejected(self):
        for kw in [
            dict(y_points=20),
            dict(half_width=0),
            dict(radius=-1),
            dict(denominator_floor=0),
        ]:
            with self.assertRaises(ValueError):
                SurfaceEvidenceSettings(**kw)


if __name__ == "__main__":
    unittest.main()
