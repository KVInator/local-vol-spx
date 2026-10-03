import unittest

import numpy as np
import pandas as pd

from carry import ParityCarryEstimator, ParityQuotePolicy


class TestCarry(unittest.TestCase):
    @staticmethod
    def quotes(
        strikes,
        forward=101.0,
        discount_factor=0.99,
        half_width=0.1,
    ):
        strike = np.asarray(strikes, dtype=float)
        width = np.broadcast_to(half_width, strike.shape)

        put_mid = np.full(strike.shape, 20.0)
        call_mid = put_mid + discount_factor * (
            forward - strike
        )

        return pd.DataFrame(
            {
                "strike": strike,
                "c_bid": call_mid - width,
                "c_ask": call_mid + width,
                "p_bid": put_mid - width,
                "p_ask": put_mid + width,
            }
        )

    @staticmethod
    def eligible(quotes, window=0.06):
        audit = ParityQuotePolicy(
            spot=100.0,
            relative_window=window,
        ).select(quotes)

        return audit.loc[audit["eligible"]].copy()

    def test_recovers_forward_and_discount(self):
        for discount in (0.97, 1.01):
            with self.subTest(discount=discount):
                pairs = self.eligible(
                    self.quotes(
                        [95.0, 100.0, 105.0],
                        forward=102.0,
                        discount_factor=discount,
                        half_width=[0.1, 0.3, 0.2],
                    )
                )

                result = ParityCarryEstimator(0.5).fit(pairs)

                self.assertAlmostEqual(
                    result.forward, 102.0, places=10
                )
                self.assertAlmostEqual(
                    result.discount_factor, discount, places=12
                )
                self.assertEqual(
                    result.summary(pairs)["outside_bands"], 0
                )

    def test_fixed_discount_downweights_wide_quotes(self):
        quotes = self.quotes(
            [99.0, 100.0, 101.0],
            forward=101.0,
            discount_factor=0.98,
            half_width=[0.1, 0.1, 2.0],
        )

        noise = np.array([0.1, 0.1, 5.0])
        quotes["c_bid"] += noise
        quotes["c_ask"] += noise

        pairs = self.eligible(quotes)
        result = ParityCarryEstimator(0.5).fit(
            pairs, discount_factor=0.98
        )

        # Relative weights are 1, 1, 1/400.
        expected_forward = (
            101.0
            + (0.1 + 0.1 + 5.0 / 400.0)
            / (2.0 + 1.0 / 400.0)
            / 0.98
        )

        self.assertAlmostEqual(
            result.forward, expected_forward, places=11
        )
        self.assertEqual(result.discount_factor, 0.98)

    def test_selection_records_reasons_and_preserves_source(self):
        quotes = self.quotes(
            [97.0, 98.0, 99.0, 100.0, 101.0, 102.0, 110.0]
        )

        quotes.loc[0, "c_bid"] = 0.0
        quotes.loc[1, "c_ask"] = quotes.loc[1, "c_bid"] - 0.1
        quotes.loc[2, "p_ask"] = np.nan
        quotes.loc[5, "c_ask"] = quotes.loc[5, "c_bid"]
        quotes.loc[5, "p_ask"] = quotes.loc[5, "p_bid"]

        original = quotes.copy(deep=True)

        audit = ParityQuotePolicy(100.0, 0.03).select(quotes)

        self.assertEqual(
            audit["selection_reason"].tolist(),
            [
                "nonpositive_bid",
                "crossed_quote",
                "nonfinite_quote",
                "eligible",
                "eligible",
                "nonpositive_combined_width",
                "outside_window",
            ],
        )
        self.assertEqual(int(audit["eligible"].sum()), 2)
        pd.testing.assert_frame_equal(quotes, original)

    def test_all_duplicate_members_are_excluded(self):
        quotes = self.quotes([99.0, 100.0, 100.0, 101.0])
        audit = ParityQuotePolicy(100.0, 0.03).select(quotes)

        duplicate_rows = audit["strike"].eq(100.0)

        self.assertTrue(
            audit.loc[
                duplicate_rows, "selection_reason"
            ].eq("duplicate_strike").all()
        )
        self.assertEqual(int(audit["eligible"].sum()), 2)

    def test_reports_quote_band_excess(self):
        quotes = self.quotes([99.0, 100.0, 101.0])
        estimator = ParityCarryEstimator(0.5)

        result = estimator.fit(
            self.eligible(quotes),
            discount_factor=0.99,
        )

        quotes.loc[
            quotes["strike"].eq(100.0),
            ["c_bid", "c_ask"],
        ] += 1.0

        changed_pairs = self.eligible(quotes)
        diagnostics = result.quote_diagnostics(changed_pairs)
        middle = diagnostics.loc[
            diagnostics["strike"].eq(100.0)
        ].iloc[0]

        self.assertAlmostEqual(
            middle["band_excess_points"], 0.8, places=12
        )
        self.assertAlmostEqual(
            middle["residual_half_widths"], -5.0, places=12
        )
        self.assertEqual(
            result.summary(changed_pairs)["outside_bands"], 1
        )

    def test_rejects_nonpositive_fitted_discount(self):
        quotes = self.quotes([99.0, 100.0, 101.0])
        increasing_parity = quotes["strike"] - 100.0

        quotes["c_bid"] = 20.0 + increasing_parity - 0.1
        quotes["c_ask"] = 20.0 + increasing_parity + 0.1

        with self.assertRaisesRegex(
            ValueError, "discount factor.*positive"
        ):
            ParityCarryEstimator(0.5).fit(
                self.eligible(quotes)
            )


if __name__ == "__main__":
    unittest.main()