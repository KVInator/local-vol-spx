import unittest

import numpy as np
import pandas as pd

from black import BlackPricer
from calibration_data import CalibrationQuoteBuilder
from implied_vol import ImpliedVolSolver


class TestCalibrationData(unittest.TestCase):
    def setUp(self):
        self.pricer = BlackPricer(
            forward=103.0,
            discount_factor=0.97,
            maturity=0.5,
        )
        self.builder = CalibrationQuoteBuilder(
            solver=ImpliedVolSolver(pricer=self.pricer)
        )

    def quotes(self, strikes):
        strike = np.asarray(strikes, dtype=float)

        return pd.DataFrame(
            {
                "source_record": np.arange(1, len(strike) + 1),
                "strike": strike,
                "c_bid": self.pricer.price(
                    strike, 0.18, kind="call"
                ),
                "c_ask": self.pricer.price(
                    strike, 0.22, kind="call"
                ),
                "p_bid": self.pricer.price(
                    strike, 0.18, kind="put"
                ),
                "p_ask": self.pricer.price(
                    strike, 0.22, kind="put"
                ),
                "c_iv": np.nan,
                "p_iv": np.nan,
            }
        )

    def test_otm_selection_and_iv_recovery(self):
        audit = self.builder.build(
            self.quotes([101.0, 103.0, 106.0])
        )

        self.assertTrue(audit["eligible"].all())
        self.assertEqual(
            audit["option_kind"].tolist(),
            ["put", "call", "call"],
        )

        np.testing.assert_allclose(
            audit["iv_bid"], 0.18, rtol=0.0, atol=2e-11
        )
        np.testing.assert_allclose(
            audit["iv_ask"], 0.22, rtol=0.0, atol=2e-11
        )

        self.assertTrue(
            (
                (audit["iv_bid"] <= audit["iv_mid"])
                & (audit["iv_mid"] <= audit["iv_ask"])
            ).all()
        )

    def test_call_translation_and_preserved_interval_width(self):
        strikes = np.array([101.0, 103.0, 106.0])
        audit = self.builder.build(self.quotes(strikes))

        for level, volatility in (("bid", 0.18), ("ask", 0.22)):
            expected = self.pricer.price(
                strikes, volatility, kind="call"
            )

            np.testing.assert_allclose(
                audit[f"call_{level}"],
                expected,
                rtol=0.0,
                atol=1e-12,
            )

            self.assertLess(
                audit[
                    f"call_{level}_repricing_error"
                ].abs().max(),
                1e-9,
            )

        np.testing.assert_allclose(
            audit["call_half_width"],
            (audit["call_ask"] - audit["call_bid"]) / 2.0,
            rtol=0.0,
            atol=1e-12,
        )

    def test_only_selected_leg_controls_eligibility(self):
        quotes = self.quotes([101.0, 105.0])

        # At 101 the put is selected; at 105 the call is selected.
        quotes.loc[0, "c_ask"] = np.nan
        quotes.loc[1, "p_bid"] = np.nan

        audit = self.builder.build(quotes)

        self.assertTrue(audit["eligible"].all())
        self.assertTrue(
            audit[["c_iv", "p_iv"]].isna().all().all()
        )
        self.assertTrue(
            np.isfinite(audit["iv_mid"]).all()
        )

    def test_rejected_quotes_are_audited_and_source_is_preserved(self):
        quotes = self.quotes(
            [99.0, 100.0, 103.0, 104.0, 105.0, 106.0, 130.0]
        )

        quotes.loc[0, "p_bid"] = 0.0
        quotes.loc[1, "p_ask"] = quotes.loc[1, "p_bid"] - 0.01
        quotes.loc[2, "c_ask"] = np.nan
        quotes.loc[3, "c_ask"] = (
            self.pricer.discount_factor * self.pricer.forward
        )
        quotes.loc[5, "c_bid"] = quotes.loc[5, "c_ask"]

        original = quotes.copy(deep=True)
        audit = self.builder.build(quotes)

        self.assertEqual(
            audit["selection_reason"].tolist(),
            [
                "nonpositive_selected_bid",
                "nonpositive_selected_spread",
                "nonfinite_selected_quote",
                "outside_finite_iv_bounds",
                "eligible",
                "nonpositive_selected_spread",
                "outside_domain",
            ],
        )

        pd.testing.assert_frame_equal(quotes, original)

    def test_duplicate_strikes_are_excluded(self):
        audit = self.builder.build(
            self.quotes([101.0, 103.0, 103.0, 105.0])
        )

        duplicates = audit["strike"].eq(103.0)

        self.assertTrue(
            audit.loc[
                duplicates, "selection_reason"
            ].eq("duplicate_strike").all()
        )
        self.assertEqual(int(audit["eligible"].sum()), 2)

    def test_solver_failures_are_recorded(self):
        limited_solver = ImpliedVolSolver(
            pricer=self.pricer,
            initial_upper=0.10,
            max_volatility=0.15,
        )
        builder = CalibrationQuoteBuilder(
            solver=limited_solver
        )

        audit = builder.build(
            self.quotes([101.0, 105.0])
        )

        self.assertFalse(audit["eligible"].any())
        self.assertTrue(
            audit["selection_reason"].eq(
                "iv_solver_failure"
            ).all()
        )
        self.assertTrue(
            audit["iv_solver_message"].str.len().gt(0).all()
        )


if __name__ == "__main__":
    unittest.main()