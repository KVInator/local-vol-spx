"""Dated Treasury-yield proxies for option discounting."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


def _numeric_result(values: FloatArray) -> float | FloatArray:
    return float(values) if values.ndim == 0 else values


@dataclass(frozen=True, slots=True)
class TreasuryYieldProxy:
    """Interpolate nominal semiannual yields and derive discount factors.

    Maturity is expressed in ACT/365F years. Treasury CMT yields are
    treated as zero yields for this explicitly approximate conversion.

    Nominal tenor anchors are supplied in days. Below the first anchor,
    its yield is held constant. Evaluation beyond the final anchor is
    rejected.
    """

    as_of: date
    tenor_days: FloatArray = field(repr=False, compare=False)
    bey_rates: FloatArray = field(repr=False, compare=False)
    source_url: str

    def __post_init__(self) -> None:
        days = np.array(self.tenor_days, dtype=float, copy=True)
        rates = np.array(self.bey_rates, dtype=float, copy=True)

        if (
            days.ndim != 1
            or rates.shape != days.shape
            or len(days) < 2
        ):
            raise ValueError(
                "Provide matching one-dimensional tenor and yield "
                "arrays with at least two anchors."
            )

        if (
            np.any(~np.isfinite(days))
            or np.any(~np.isfinite(rates))
        ):
            raise ValueError("Tenors and yields must be finite.")

        if np.any(days <= 0.0) or np.any(np.diff(days) <= 0.0):
            raise ValueError(
                "Tenor days must be positive and strictly increasing."
            )

        if np.any(rates <= -2.0):
            raise ValueError(
                "Semiannual conversion requires yields greater than -2."
            )

        if not isinstance(self.as_of, date):
            raise ValueError("as_of must be a date.")

        if not self.source_url.strip():
            raise ValueError("A yield source reference is required.")

        days.setflags(write=False)
        rates.setflags(write=False)

        object.__setattr__(self, "tenor_days", days)
        object.__setattr__(self, "bey_rates", rates)

    @classmethod
    def from_json(cls, path: str | Path) -> TreasuryYieldProxy:
        definition = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        return cls(
            as_of=date.fromisoformat(definition["as_of"]),
            tenor_days=definition["tenor_days"],
            bey_rates=(
                np.asarray(
                    definition["bey_percent"],
                    dtype=float,
                )
                / 100.0
            ),
            source_url=definition["source_url"],
        )

    def yield_bey(
        self,
        maturity: ArrayLike,
    ) -> float | FloatArray:
        times = np.asarray(maturity, dtype=float)

        if np.any(~np.isfinite(times)) or np.any(times < 0.0):
            raise ValueError(
                "Maturities must be finite and nonnegative."
            )

        if np.any(times > self.tenor_days[-1] / 365.0):
            raise ValueError(
                "Maturity exceeds the final Treasury tenor."
            )

        interpolated = np.asarray(
            np.interp(
                365.0 * times,
                self.tenor_days,
                self.bey_rates,
            )
        )
        return _numeric_result(interpolated)

    def continuous_rate(
        self,
        maturity: ArrayLike,
    ) -> float | FloatArray:
        yields = np.asarray(self.yield_bey(maturity))
        rates = np.asarray(2.0 * np.log1p(yields / 2.0))
        return _numeric_result(rates)

    def discount_factor(
        self,
        maturity: ArrayLike,
    ) -> float | FloatArray:
        times = np.asarray(maturity, dtype=float)
        rates = np.asarray(self.continuous_rate(times))
        discounts = np.asarray(np.exp(-rates * times))
        return _numeric_result(discounts)

    def metadata(self) -> dict:
        return {
            "as_of": self.as_of.isoformat(),
            "source_url": self.source_url,
            "tenor_days": self.tenor_days.tolist(),
            "bey_rates_decimal": self.bey_rates.tolist(),
            "interpolation": "Linear in nominal yield versus tenor days.",
            "short_end": "First yield held constant down to zero maturity.",
            "long_end": "Evaluation beyond final tenor rejected.",
            "tenor_mapping": (
                "Nominal month = 30 days; one year = 365 days."
            ),
            "discounting": "r=2*log(1+y/2); D=exp(-r*T).",
            "limitation": (
                "Treasury CMT par yields treated as zero yields; "
                "discounting to the model fixing horizon."
            ),
        }