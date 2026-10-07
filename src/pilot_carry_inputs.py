"""Daily research carry assumptions and parity-equivalent calibration quotes."""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from pilot_carry import CarrySettings, PilotCarryEstimator


@dataclass(frozen=True)
class CarryInputSettings:
    primary_rate: float = 0.05
    rates: tuple = (0.03, 0.05, 0.07)
    parity_window: float = 0.03
    minimum_pairs: int = 6

    def __post_init__(self):
        if (
            not self.rates
            or not np.isfinite(self.rates).all()
            or len(set(self.rates)) != len(self.rates)
            or self.primary_rate not in self.rates
        ):
            raise ValueError(
                "Require distinct finite rates including the primary rate."
            )

        CarrySettings(
            windows=(self.parity_window,),
            primary_window=self.parity_window,
            minimum_pairs=self.minimum_pairs,
            rate_scenarios=self.rates,
        )

        if len({self.case_name(r) for r in self.rates}) != len(self.rates):
            raise ValueError("Rate scenario labels are not distinct.")

    @staticmethod
    def case_name(rate):
        number = (
            f"{100 * rate:.12g}"
            .replace("-", "m")
            .replace(".", "p")
        )
        return f"rate_{number}pct"


class PilotCarryInputs:
    GROUP = PilotCarryEstimator.GROUP
    KEY = PilotCarryEstimator.KEY

    def __init__(self, settings=None):
        self.settings = settings or CarryInputSettings()

    def prepare(self, quotes):
        s = self.settings

        estimator = PilotCarryEstimator(
            CarrySettings(
                windows=(s.parity_window,),
                primary_window=s.parity_window,
                minimum_pairs=s.minimum_pairs,
                rate_scenarios=s.rates,
            )
        )

        q = estimator._checked(quotes)

        for key, group in q.groupby(["quote_date", "root"]):
            if (
                group["underlying_last"].max()
                - group["underlying_last"].min()
                > 0.01
                or group["quote_timestamp_utc"].nunique() != 1
            ):
                raise ValueError(
                    f"Inconsistent daily snapshot for {key}."
                )

        estimates, upstream = estimator.run(q)

        free_columns = self.GROUP + [
            "spot",
            "maturity_years",
            "days",
            "pairs",
            "available_pair_count",
            "unpaired_quote_sides",
            "status",
            "forward",
            "discount_factor",
            "original_bands_feasible",
            "minimum_band_multiplier",
        ]

        base = (
            estimates["carry_estimates"]
            .reindex(columns=free_columns)
            .rename(
                columns={
                    "status": "free_fit_status",
                    "forward": "free_forward",
                    "discount_factor": "free_discount_factor",
                    "original_bands_feasible":
                        "free_discount_bands_feasible",
                }
            )
        )

        metadata = (
            q.groupby(self.GROUP)
            .agg(
                quote_timestamp_utc=("quote_timestamp_utc", "first"),
                assumed_fixing_utc=("assumed_fixing_utc", "first"),
                quote_sides=("kind", "size"),
                minimum_strike=("strike", "min"),
                maximum_strike=("strike", "max"),
            )
            .reset_index()
        )

        base = base.merge(
            metadata,
            on=self.GROUP,
            validate="one_to_one",
        )

        scenarios = estimates["fixed_rate_sensitivity"]

        redundant = [
            "spot",
            "maturity_years",
            "days",
            "available_pair_count",
            "unpaired_quote_sides",
            "annual_rate_pct",
        ]

        parts = []

        for rate in sorted(s.rates):
            fitted = (
                scenarios.loc[
                    scenarios["annual_rate_pct"].eq(100 * rate)
                ]
                .drop(columns=redundant)
                .rename(
                    columns={"status": "conditional_fit_status"}
                )
            )

            frame = base.merge(
                fitted,
                on=self.GROUP,
                how="left",
                validate="one_to_one",
            )

            frame["case"] = s.case_name(rate)
            frame["primary_case"] = rate == s.primary_rate
            frame["annual_rate"] = rate
            frame["annual_rate_pct"] = 100 * rate

            # The discount assumption exists even if a forward is unavailable.
            frame["discount_factor"] = np.exp(
                -rate * frame["maturity_years"]
            )

            frame["forward"] = pd.to_numeric(
                frame["forward"],
                errors="raise",
            ).astype(float)

            frame["conditional_fit_status"] = (
                frame["conditional_fit_status"]
                .fillna("unavailable")
            )

            frame["carry_ready"] = (
                frame["conditional_fit_status"].eq("conditional_fit")
                & np.isfinite(frame["forward"])
                & frame["forward"].gt(0)
                & np.isfinite(frame["discount_factor"])
                & frame["discount_factor"].gt(0)
            )

            frame["parity_bands_incompatible"] = (
                frame["original_bands_feasible"].eq(False)
            )

            frame["fitted_parity_outside_bands"] = (
                frame["outside_parity_bands"].gt(0)
            )

            frame["carry_status"] = np.select(
                [
                    ~frame["carry_ready"],
                    frame["parity_bands_incompatible"],
                ],
                [
                    "unavailable",
                    "ready_with_parity_incompatibility",
                ],
                default="ready",
            )

            for column in [
                "net_carry_rate",
                "implied_yield_proxy",
                "prepaid_forward",
            ]:
                frame[column] = np.nan

            ready = frame["carry_ready"]

            frame.loc[ready, "net_carry_rate"] = (
                np.log(
                    frame.loc[ready, "forward"]
                    / frame.loc[ready, "spot"]
                )
                / frame.loc[ready, "maturity_years"]
            )

            frame.loc[ready, "implied_yield_proxy"] = (
                rate - frame.loc[ready, "net_carry_rate"]
            )

            frame.loc[ready, "prepaid_forward"] = (
                frame.loc[ready, "discount_factor"]
                * frame.loc[ready, "forward"]
            )

            parts.append(frame)

        carry = pd.concat(parts, ignore_index=True)
        primary = carry.loc[carry["primary_case"]].copy()

        selected = self._calibration_quotes(q, primary)

        counts = (
            selected.groupby(self.GROUP)
            .agg(
                calibration_quotes=("strike", "size"),
                carry_resolved_quotes=("carry_ready", "sum"),
                preferred_side_missing=("preferred_side_missing", "sum"),
                bands_disjoint_from_call_bounds=(
                    "band_disjoint_from_call_bounds",
                    "sum",
                ),
                midpoints_outside_call_bounds=(
                    "midpoint_outside_call_bounds",
                    "sum",
                ),
            )
            .reset_index()
        )

        primary = primary.merge(
            counts,
            on=self.GROUP,
            validate="one_to_one",
        )

        daily = (
            primary.groupby("quote_date")
            .agg(
                expiry_groups=("root", "size"),
                ready_groups=("carry_ready", "sum"),
                parity_incompatible_groups=(
                    "parity_bands_incompatible",
                    "sum",
                ),
                fitted_parity_outside_groups=(
                    "fitted_parity_outside_bands",
                    "sum",
                ),
                calibration_quotes=("calibration_quotes", "sum"),
                carry_resolved_quotes=("carry_resolved_quotes", "sum"),
                preferred_side_missing=("preferred_side_missing", "sum"),
                bands_disjoint_from_call_bounds=(
                    "bands_disjoint_from_call_bounds",
                    "sum",
                ),
                midpoints_outside_call_bounds=(
                    "midpoints_outside_call_bounds",
                    "sum",
                ),
            )
            .reset_index()
        )

        audit = {
            "settings": asdict(s),
            "primary_case": s.case_name(s.primary_rate),
            "input_quote_sides": len(q),
            "matched_pairs": upstream["matched_pairs"],
            "quote_dates": int(q["quote_date"].nunique()),
            "expiry_groups": len(primary),
            "carry_rows": len(carry),
            "primary_calibration_quotes": len(selected),
            "discount_convention":
                "D(T)=exp(-r*T); r is an assumed constant annual rate, "
                "ACT/365F.",
            "forward_fit":
                "Same-date matched pairs in the fixed parity window; "
                "robust conditional fit separately for each rate.",
            "calibration_selection":
                "Primary forward: put below forward, call at or above; "
                "available opposite side is flagged as a fallback.",
            "put_conversion":
                "Equivalent call bid/ask = original put bid/ask + D*(F-K).",
            "parity_incompatibility_policy":
                "Retained and flagged for research calibration; "
                "no band expansion or automatic removal.",
            "call_bound_policy":
                "Report bands and midpoints outside individual call bounds; "
                "no clipping or automatic removal.",
            "carry_ready_scope":
                "A finite positive forward and discount are available; "
                "this is not a quote-fit or backtest approval.",
            "implied_yield_proxy":
                "r-log(F/spot)/T; pricing proxy, not observed dividends "
                "or hedge income.",
            "future_observations_used": False,
            "carry_interpolated": False,
            "funding_curve_verified": False,
            "contract_identity_verified": False,
            "snapshot_provenance_verified": False,
            "raw_quotes_modified": False,
            "prices_clipped": False,
            "diffusion_models_refitted": False,
            "candidate_promoted": False,
            "backtest_performed": False,
        }

        return {
            "carry_inputs": carry,
            "primary_carry": primary,
            "calibration_quotes": selected,
            "daily_summary": daily,
        }, audit

    def _calibration_quotes(self, quotes, primary):
        fields = self.GROUP + [
            "case",
            "forward",
            "discount_factor",
            "carry_ready",
            "carry_status",
            "parity_bands_incompatible",
            "fitted_parity_outside_bands",
        ]

        q = quotes.merge(
            primary[fields],
            on=self.GROUP,
            how="left",
            validate="many_to_one",
        )

        preferred = np.where(
            q["strike"] < q["forward"],
            "put",
            "call",
        )

        q["side_priority"] = q["kind"].ne(preferred).astype(int)

        # Duplicate input sides have already failed validation.
        # Here we select between call and put observations at one strike.
        q = (
            q.sort_values(
                self.KEY + ["side_priority"],
                kind="stable",
            )
            .drop_duplicates(self.KEY)
            .copy()
        )

        q["preferred_side_missing"] = (
            q["carry_ready"] & q["side_priority"].eq(1)
        )

        q = q.rename(
            columns={
                "kind": "source_kind",
                "bid": "source_bid",
                "ask": "source_ask",
            }
        )

        f = q["forward"].where(q["carry_ready"])
        d = q["discount_factor"]

        adjustment = np.where(
            q["source_kind"].eq("put"),
            d * (f - q["strike"]),
            0.0,
        )

        for side in ["bid", "ask"]:
            q[f"call_{side}"] = (
                q[f"source_{side}"] + adjustment
            ).where(q["carry_ready"])

        q["call_mid"] = (q["call_bid"] + q["call_ask"]) / 2

        q["call_half_width"] = (
            q["source_ask"] - q["source_bid"]
        ) / 2

        q["normalized_strike"] = q["strike"] / f
        q["forward_log_moneyness"] = np.log(q["normalized_strike"])
        q["normalized_call_mid"] = q["call_mid"] / (d * f)

        lower = d * np.maximum(f - q["strike"], 0.0)
        upper = d * f
        tolerance = 1e-6  # Index points, used for reporting only.

        q["band_disjoint_from_call_bounds"] = q["carry_ready"] & (
            (q["call_ask"] < lower - tolerance)
            | (q["call_bid"] > upper + tolerance)
        )

        q["midpoint_outside_call_bounds"] = q["carry_ready"] & (
            (q["call_mid"] < lower - tolerance)
            | (q["call_mid"] > upper + tolerance)
        )

        return q.drop(columns="side_priority").reset_index(drop=True)