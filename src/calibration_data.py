"""Build OTM calibration quotes and conditional call-price bands."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from black import OptionKind
from implied_vol import ImpliedVolSolver


@dataclass(frozen=True, slots=True)
class CalibrationQuoteBuilder:
    """Select OTM quotes, recover IVs and translate put-price bands.

    IVs are decimal annualised volatilities. Source quote columns are
    preserved. Only the selected option leg determines eligibility.
    """

    solver: ImpliedVolSolver
    min_log_moneyness: float = -0.10
    max_log_moneyness: float = 0.10

    def __post_init__(self) -> None:
        limits = [
            self.min_log_moneyness,
            self.max_log_moneyness,
        ]

        if (
            not np.isfinite(limits).all()
            or self.min_log_moneyness >= self.max_log_moneyness
        ):
            raise ValueError(
                "Log-moneyness limits must be finite and increasing."
            )

        if self.solver.pricer.maturity <= 0.0:
            raise ValueError(
                "IV calibration requires positive maturity."
            )

    def build(self, quotes: pd.DataFrame) -> pd.DataFrame:
        """Return all input rows with selection and pricing diagnostics."""

        table = quotes.copy(deep=True).reset_index(drop=True)
        pricer = self.solver.pricer

        strike = table["strike"].to_numpy(dtype=float)
        valid_strike = np.isfinite(strike) & (strike > 0.0)
        is_put = strike < pricer.forward

        log_moneyness = np.full(len(table), np.nan)
        log_moneyness[valid_strike] = (
            np.log(strike[valid_strike])
            - np.log(pricer.forward)
        )

        table["log_moneyness"] = log_moneyness
        table["option_kind"] = np.where(is_put, "put", "call")

        table["native_bid"] = np.where(
            is_put, table["p_bid"], table["c_bid"]
        )
        table["native_ask"] = np.where(
            is_put, table["p_ask"], table["c_ask"]
        )
        table["native_mid"] = (
            table["native_bid"] + table["native_ask"]
        ) / 2.0
        table["native_half_width"] = (
            table["native_ask"] - table["native_bid"]
        ) / 2.0

        bid = table["native_bid"].to_numpy(dtype=float)
        ask = table["native_ask"].to_numpy(dtype=float)

        # The selected OTM option has zero intrinsic value.
        # Its upper bound is D*K for a put and D*F for a call.
        upper_bound = pricer.discount_factor * np.where(
            is_put, strike, pricer.forward
        )

        checks = [
            ("invalid_strike", ~valid_strike),
            (
                "duplicate_strike",
                table["strike"].duplicated(
                    keep=False
                ).to_numpy(),
            ),
            (
                "outside_domain",
                (log_moneyness < self.min_log_moneyness)
                | (log_moneyness > self.max_log_moneyness),
            ),
            (
                "nonfinite_selected_quote",
                ~np.isfinite(bid) | ~np.isfinite(ask),
            ),
            ("nonpositive_selected_bid", bid <= 0.0),
            ("nonpositive_selected_spread", ask <= bid),
            (
                "outside_finite_iv_bounds",
                ask >= upper_bound,
            ),
        ]

        reason = np.full(len(table), "eligible", dtype=object)

        for label, failed in checks:
            assign = (reason == "eligible") & failed
            reason[assign] = label

        table["selection_reason"] = reason
        table["eligible"] = reason == "eligible"

        parity_shift = np.where(
            is_put,
            pricer.discount_factor * (pricer.forward - strike),
            0.0,
        )

        for level in ("bid", "mid", "ask"):
            table[f"call_{level}"] = (
                table[f"native_{level}"] + parity_shift
            )

        # Translation preserves the native quote interval's width.
        table["call_half_width"] = table["native_half_width"]

        table["forward"] = pricer.forward
        table["discount_factor"] = pricer.discount_factor
        table["maturity_years"] = pricer.maturity

        numeric_columns = [
            "iv_bid",
            "iv_mid",
            "iv_ask",
            "iv_width_pp",
            "iv_mid_vega",
            "iv_mid_iterations",
            "iv_mid_price_evaluations",
        ]

        for level in ("bid", "mid", "ask"):
            numeric_columns.extend(
                [
                    f"native_{level}_repricing_error",
                    f"call_{level}_repricing_error",
                ]
            )

        for column in numeric_columns:
            table[column] = np.nan

        table["iv_solver_message"] = ""

        candidates = table.loc[table["eligible"]]

        for row in candidates.itertuples():
            kind: OptionKind = (
                "put" if row.option_kind == "put" else "call"
            )
            results = {}

            try:
                for level in ("bid", "mid", "ask"):
                    results[level] = self.solver.solve(
                        price=float(
                            getattr(row, f"native_{level}")
                        ),
                        strike=float(row.strike),
                        kind=kind,
                    )
            except RuntimeError as error:
                table.at[row.Index, "eligible"] = False
                table.at[
                    row.Index, "selection_reason"
                ] = "iv_solver_failure"
                table.at[
                    row.Index, "iv_solver_message"
                ] = f"{level}: {error}"
                continue

            values = {}

            for level, result in results.items():
                values[f"iv_{level}"] = result.volatility
                values[
                    f"native_{level}_repricing_error"
                ] = result.price_error

            middle = results["mid"]

            values["iv_mid_vega"] = middle.vega
            values["iv_mid_iterations"] = middle.iterations
            values["iv_mid_price_evaluations"] = (
                middle.price_evaluations
            )

            table.loc[
                row.Index, list(values)
            ] = list(values.values())

        successful = table["eligible"]

        table["iv_width_pp"] = 100.0 * (
            table["iv_ask"] - table["iv_bid"]
        )

        if successful.any():
            retained_strikes = table.loc[
                successful, "strike"
            ].to_numpy(dtype=float)

            for level in ("bid", "mid", "ask"):
                volatility = table.loc[
                    successful, f"iv_{level}"
                ].to_numpy(dtype=float)

                repriced_call = pricer.price(
                    retained_strikes,
                    volatility,
                    kind="call",
                )

                table.loc[
                    successful, f"call_{level}_repricing_error"
                ] = (
                    repriced_call
                    - table.loc[
                        successful, f"call_{level}"
                    ].to_numpy(dtype=float)
                )

        return table