"""Quote selection and weighted put-call parity carry estimation."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


BAND_TOLERANCE = 1e-8


@dataclass(frozen=True, slots=True)
class ParityQuotePolicy:
    """Eligibility policy for a single-expiry, single-time snapshot."""

    spot: float
    relative_window: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.spot) or self.spot <= 0.0:
            raise ValueError("Spot must be finite and positive.")

        if (
            not np.isfinite(self.relative_window)
            or self.relative_window <= 0.0
        ):
            raise ValueError(
                "The relative strike window must be finite and positive."
            )

    def select(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Return all source rows with eligibility and parity columns.

        Every member of a duplicated strike is excluded. A locked leg
        can qualify if the combined parity interval has positive width.
        Vendor IV is not used in the eligibility decision.
        """

        table = quotes.copy(deep=True)

        strike = table["strike"].to_numpy(dtype=float)
        bids = table[["c_bid", "p_bid"]].to_numpy(dtype=float)
        asks = table[["c_ask", "p_ask"]].to_numpy(dtype=float)

        table["cp_lower"] = table["c_bid"] - table["p_ask"]
        table["cp_upper"] = table["c_ask"] - table["p_bid"]
        table["cp_mid"] = (
            table["cp_lower"] + table["cp_upper"]
        ) / 2.0
        table["cp_half_width"] = (
            table["cp_upper"] - table["cp_lower"]
        ) / 2.0

        half_width = table["cp_half_width"].to_numpy(dtype=float)

        checks = [
            (
                "invalid_strike",
                ~np.isfinite(strike) | (strike <= 0.0),
            ),
            (
                "duplicate_strike",
                table["strike"].duplicated(
                    keep=False
                ).to_numpy(),
            ),
            (
                "nonfinite_quote",
                ~(
                    np.isfinite(bids).all(axis=1)
                    & np.isfinite(asks).all(axis=1)
                ),
            ),
            (
                "nonpositive_bid",
                (bids <= 0.0).any(axis=1),
            ),
            (
                "crossed_quote",
                (asks < bids).any(axis=1),
            ),
            (
                "nonpositive_combined_width",
                ~np.isfinite(half_width) | (half_width <= 0.0),
            ),
            (
                "outside_window",
                np.abs(strike - self.spot)
                > self.relative_window * self.spot,
            ),
        ]

        reason = np.full(len(table), "eligible", dtype=object)

        for label, failed in checks:
            assign = (reason == "eligible") & failed
            reason[assign] = label

        table["selection_reason"] = reason
        table["eligible"] = reason == "eligible"

        return table


@dataclass(frozen=True, slots=True)
class CarryEstimate:
    """Carry parameters and diagnostics for a fitted parity line."""

    forward: float
    discount_factor: float
    maturity: float
    mode: Literal["free_discount", "fixed_discount"]

    @property
    def implied_continuous_rate(self) -> float:
        return float(-np.log(self.discount_factor) / self.maturity)

    def quote_diagnostics(
        self,
        pairs: pd.DataFrame,
    ) -> pd.DataFrame:
        """Evaluate the fitted line against each supplied parity interval."""

        table = pairs.copy(deep=True)

        table["model_cp"] = self.discount_factor * (
            self.forward - table["strike"]
        )
        table["cp_residual"] = table["model_cp"] - table["cp_mid"]
        table["residual_half_widths"] = (
            table["cp_residual"] / table["cp_half_width"]
        )

        excess = np.maximum(
            table["cp_lower"].to_numpy(dtype=float)
            - table["model_cp"].to_numpy(dtype=float),
            table["model_cp"].to_numpy(dtype=float)
            - table["cp_upper"].to_numpy(dtype=float),
        )

        table["band_excess_points"] = np.maximum(excess, 0.0)
        table["outside_band"] = (
            table["band_excess_points"] > BAND_TOLERANCE
        )

        return table

    def summary(self, pairs: pd.DataFrame) -> dict:
        diagnostics = self.quote_diagnostics(pairs)

        residual = diagnostics["cp_residual"].to_numpy(dtype=float)
        normalized = diagnostics[
            "residual_half_widths"
        ].to_numpy(dtype=float)

        return {
            "mode": self.mode,
            "pairs": len(diagnostics),
            "discount_factor": self.discount_factor,
            "forward": self.forward,
            "implied_rate_pct": 100.0 * self.implied_continuous_rate,
            "midpoint_rmse": float(
                np.sqrt(np.mean(residual**2))
            ),
            "rms_half_widths": float(
                np.sqrt(np.mean(normalized**2))
            ),
            "outside_bands": int(
                diagnostics["outside_band"].sum()
            ),
            "max_band_excess_points": float(
                diagnostics["band_excess_points"].max()
            ),
        }


@dataclass(frozen=True, slots=True)
class ParityCarryEstimator:
    """Weighted least-squares carry estimator.

    Weights are inverse squared parity-interval half-widths.
    The freely estimated discount factor is not capped at one:
    discount factors above one can represent negative rates.
    """

    maturity: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.maturity) or self.maturity <= 0.0:
            raise ValueError("Maturity must be finite and positive.")

    def fit(
        self,
        pairs: pd.DataFrame,
        discount_factor: float | None = None,
    ) -> CarryEstimate:
        """Fit eligible pairs produced by ParityQuotePolicy."""

        if len(pairs) < 2:
            raise ValueError("At least two eligible pairs are required.")

        strike = pairs["strike"].to_numpy(dtype=float)
        midpoint = pairs["cp_mid"].to_numpy(dtype=float)
        half_width = pairs["cp_half_width"].to_numpy(dtype=float)

        if (
            not np.isfinite(strike).all()
            or not np.isfinite(midpoint).all()
            or not np.isfinite(half_width).all()
            or np.any(strike <= 0.0)
            or np.any(half_width <= 0.0)
        ):
            raise ValueError(
                "Fit inputs must be finite, with positive strikes "
                "and parity half-widths."
            )

        scale = float(np.ptp(strike))

        if scale <= 0.0:
            raise ValueError(
                "At least two distinct strikes are required."
            )

        # Rescaling all weights by the same constant preserves the fit.
        root_weight = float(half_width.min()) / half_width
        weight = root_weight**2

        if discount_factor is None:
            centre = float(np.average(strike, weights=weight))
            x = (strike - centre) / scale

            design = np.column_stack(
                [np.ones(len(strike)), x]
            )

            coefficients, _, rank, _ = np.linalg.lstsq(
                design * root_weight[:, None],
                midpoint * root_weight,
                rcond=None,
            )

            if rank != 2:
                raise ValueError(
                    "The weighted parity fit does not have full rank."
                )

            # midpoint = A - D * (strike - centre)
            discount = float(-coefficients[1] / scale)

            if not np.isfinite(discount) or discount <= 0.0:
                raise ValueError(
                    "The fitted discount factor must be finite "
                    "and positive."
                )

            forward = float(
                centre + coefficients[0] / discount
            )
            mode = "free_discount"

        else:
            discount = float(discount_factor)

            if not np.isfinite(discount) or discount <= 0.0:
                raise ValueError(
                    "The supplied discount factor must be finite "
                    "and positive."
                )

            forward = float(
                np.average(
                    strike + midpoint / discount,
                    weights=weight,
                )
            )
            mode = "fixed_discount"

        if not np.isfinite(forward) or forward <= 0.0:
            raise ValueError(
                "The fitted forward must be finite and positive."
            )

        return CarryEstimate(
            forward=forward,
            discount_factor=discount,
            maturity=self.maturity,
            mode=mode,
        )