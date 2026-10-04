"""Connect market inputs, carry estimation, IV extraction and spline fitting."""

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from black import BlackPricer
from calibration_data import CalibrationQuoteBuilder
from carry import ParityCarryEstimator, ParityQuotePolicy
from convex_spline import ConvexCallSpline, ConvexSplineCalibrator
from discount_curve import TreasuryYieldProxy
from implied_vol import ImpliedVolSolver
from market_data import QuoteSnapshot
from maturity import ExpiryConvention


@dataclass(frozen=True, eq=False)
class CalibratedExpiry:
    """Calibrated single-expiry model and its input diagnostics."""

    expiry_date: date
    spline: ConvexCallSpline
    parity_selection: pd.DataFrame
    parity_diagnostics: pd.DataFrame
    quote_selection: pd.DataFrame
    quotes: pd.DataFrame
    audit: dict


@dataclass(frozen=True)
class ExpirySliceCalibrator:
    """Run the calibration pipeline for one market snapshot."""

    discount_proxy: TreasuryYieldProxy
    relative_carry_window: float = 0.03
    min_log_moneyness: float = -0.10
    max_log_moneyness: float = 0.10
    n_intervals: int = 30
    smoothing_weight: float = 1e-3
    band_slack: float = 1.5
    curvature_floor: float = 1e-8

    def calibrate(
        self,
        snapshot: QuoteSnapshot,
        convention: ExpiryConvention,
    ) -> CalibratedExpiry:
        """Estimate carry, extract eligible quotes and fit a convex spline."""

        quote_date = pd.Timestamp(snapshot.quote_date).date()
        expiry_date = pd.Timestamp(snapshot.expiry_date).date()
        convention_expiry = pd.Timestamp(
            convention.expiry_date
        ).date()

        if self.discount_proxy.as_of != quote_date:
            raise ValueError(
                "Discount-reference date does not match quote date."
            )

        if convention_expiry != expiry_date:
            raise ValueError(
                "Settlement convention does not match snapshot expiry."
            )

        quote_timestamp = snapshot.quote_timestamp_utc
        maturity = convention.maturity(quote_timestamp)
        maturity_years = float(maturity.year_fraction)
        elapsed_days = float(maturity.elapsed_days)

        discount_factor = float(
            self.discount_proxy.discount_factor(maturity_years)
        )
        spot = float(snapshot.spot)

        settlement_timestamp = quote_timestamp + pd.Timedelta(
            days=elapsed_days
        )
        day_count = (
            "ACT/365F (fractional days from elapsed UTC time)"
        )

        source_quotes = snapshot.with_midpoints()
        source_quotes["model_settlement_timestamp_utc"] = (
            settlement_timestamp
        )
        source_quotes["maturity_years"] = maturity_years
        source_quotes["maturity_day_count"] = day_count
        source_quotes["settlement_kind"] = (
            convention.settlement_kind
        )
        source_quotes["settlement_verification_status"] = (
            convention.verification_status
        )

        parity_policy = ParityQuotePolicy(
            spot=spot,
            relative_window=self.relative_carry_window,
        )
        parity_selection = parity_policy.select(source_quotes)
        parity_pairs = parity_selection.loc[
            parity_selection["eligible"]
        ].copy()

        carry = ParityCarryEstimator(
            maturity=maturity_years
        ).fit(
            parity_pairs,
            discount_factor=discount_factor,
        )
        parity_diagnostics = carry.quote_diagnostics(
            parity_pairs
        )

        pricer = BlackPricer(
            forward=carry.forward,
            discount_factor=carry.discount_factor,
            maturity=maturity_years,
        )

        quote_builder = CalibrationQuoteBuilder(
            solver=ImpliedVolSolver(pricer),
            min_log_moneyness=self.min_log_moneyness,
            max_log_moneyness=self.max_log_moneyness,
        )
        quote_selection = quote_builder.build(source_quotes)

        quotes = (
            quote_selection.loc[quote_selection["eligible"]]
            .sort_values("strike")
            .reset_index(drop=True)
            .copy()
        )

        if len(quotes) < 4:
            raise ValueError(
                "At least four eligible quotes are required "
                "for spline calibration."
            )

        spline = ConvexSplineCalibrator(
            pricer=pricer,
            n_intervals=self.n_intervals,
            smoothing_weight=self.smoothing_weight,
            band_slack=self.band_slack,
            curvature_floor=self.curvature_floor,
        ).fit(
            quotes["strike"].to_numpy(dtype=float),
            quotes["call_mid"].to_numpy(dtype=float),
            quotes["call_half_width"].to_numpy(dtype=float),
        )

        native_error_columns = [
            "native_bid_repricing_error",
            "native_mid_repricing_error",
            "native_ask_repricing_error",
        ]
        call_error_columns = [
            "call_bid_repricing_error",
            "call_mid_repricing_error",
            "call_ask_repricing_error",
        ]

        maximum_native_repricing_error = float(
            np.max(
                np.abs(
                    quotes[native_error_columns].to_numpy(
                        dtype=float
                    )
                )
            )
        )
        maximum_call_repricing_error = float(
            np.max(
                np.abs(
                    quotes[call_error_columns].to_numpy(
                        dtype=float
                    )
                )
            )
        )

        parity_residuals = parity_diagnostics[
            "cp_residual"
        ].to_numpy(dtype=float)
        scaled_parity_residuals = parity_diagnostics[
            "residual_half_widths"
        ].to_numpy(dtype=float)

        selection_counts = {
            str(reason): int(count)
            for reason, count in quote_selection[
                "selection_reason"
            ].value_counts().items()
        }

        audit = {
            "quote_date": quote_date.isoformat(),
            "expiry_date": expiry_date.isoformat(),
            "quote_timestamp_utc": quote_timestamp.isoformat(),
            "model_settlement_timestamp_utc": (
                settlement_timestamp.isoformat()
            ),
            "settlement_kind": convention.settlement_kind,
            "settlement_verification_status": (
                convention.verification_status
            ),
            "settlement_evidence": convention.evidence,
            "elapsed_days": elapsed_days,
            "maturity_years": maturity_years,
            "spot": spot,
            "raw_rows": int(len(source_quotes)),
            "parity": {
                "mode": "fixed_discount",
                "pairs": int(len(parity_pairs)),
                "forward": float(carry.forward),
                "discount_factor": float(
                    carry.discount_factor
                ),
                "midpoint_rmse": float(
                    np.sqrt(np.mean(parity_residuals**2))
                ),
                "rms_half_widths": float(
                    np.sqrt(
                        np.mean(scaled_parity_residuals**2)
                    )
                ),
                "outside_bands": int(
                    parity_diagnostics["outside_band"].sum()
                ),
            },
            "calibration_quotes": int(len(quotes)),
            "quote_selection_counts": selection_counts,
            "maximum_native_repricing_error": (
                maximum_native_repricing_error
            ),
            "maximum_call_repricing_error": (
                maximum_call_repricing_error
            ),
            "spline": spline.metadata(),
            "shape_checks": spline.shape_checks(),
        }

        return CalibratedExpiry(
            expiry_date=expiry_date,
            spline=spline,
            parity_selection=parity_selection,
            parity_diagnostics=parity_diagnostics,
            quote_selection=quote_selection,
            quotes=quotes,
            audit=audit,
        )