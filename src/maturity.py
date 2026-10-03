"""Explicit expiry conventions and fractional-year model maturities."""

from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Literal

import pandas as pd


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.0 * SECONDS_PER_DAY

DAY_COUNT = "ACT/365F (fractional days from elapsed UTC time)"

SettlementKind = Literal["AM", "PM"]
VerificationStatus = Literal["inferred", "verified"]


@dataclass(frozen=True, slots=True)
class MaturityResult:
    """A positive pricing horizon between two timezone-aware instants."""

    quote_timestamp_utc: pd.Timestamp
    settlement_timestamp_utc: pd.Timestamp

    @property
    def elapsed_seconds(self) -> float:
        return float(
            (
                self.settlement_timestamp_utc
                - self.quote_timestamp_utc
            ).total_seconds()
        )

    @property
    def elapsed_days(self) -> float:
        return self.elapsed_seconds / SECONDS_PER_DAY

    @property
    def year_fraction(self) -> float:
        return self.elapsed_seconds / SECONDS_PER_YEAR


@dataclass(frozen=True, slots=True)
class ExpiryConvention:
    """An explicitly supplied model horizon for one expiry date.

    No settlement family, holiday adjustment or expiry time is inferred
    automatically. The caller supplies the convention and its evidence.
    """

    expiry_date: str | pd.Timestamp
    settlement_time_ny: str
    settlement_kind: SettlementKind
    verification_status: VerificationStatus
    evidence: str

    settlement_timestamp_utc: pd.Timestamp = field(init=False)

    def __post_init__(self) -> None:
        expiry = pd.Timestamp(self.expiry_date)

        if (
            pd.isna(expiry)
            or expiry.tzinfo is not None
            or expiry != expiry.normalize()
        ):
            raise ValueError(
                "expiry_date must be a timezone-naive calendar date."
            )

        clock = time.fromisoformat(self.settlement_time_ny)

        if clock.tzinfo is not None:
            raise ValueError(
                "settlement_time_ny must be a local time without an offset."
            )

        if self.settlement_kind not in ("AM", "PM"):
            raise ValueError("settlement_kind must be 'AM' or 'PM'.")

        if self.verification_status not in ("inferred", "verified"):
            raise ValueError(
                "verification_status must be 'inferred' or 'verified'."
            )

        evidence = self.evidence.strip()

        if not evidence:
            raise ValueError("Settlement evidence must be supplied.")

        local_timestamp = pd.Timestamp(
            datetime.combine(expiry.date(), clock)
        ).tz_localize(
            "America/New_York",
            ambiguous="raise",
            nonexistent="raise",
        )

        object.__setattr__(self, "expiry_date", expiry)
        object.__setattr__(
            self, "settlement_time_ny", clock.isoformat()
        )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self,
            "settlement_timestamp_utc",
            local_timestamp.tz_convert("UTC"),
        )

    @property
    def settlement_timestamp_ny(self) -> pd.Timestamp:
        return self.settlement_timestamp_utc.tz_convert(
            "America/New_York"
        )

    def maturity(
        self,
        quote_timestamp_utc: str | pd.Timestamp,
    ) -> MaturityResult:
        """Calculate the positive model maturity from an explicit instant."""

        quote = pd.Timestamp(quote_timestamp_utc)

        if pd.isna(quote) or quote.tzinfo is None:
            raise ValueError(
                "The quote timestamp must be valid and timezone-aware."
            )

        quote = quote.tz_convert("UTC")

        if self.settlement_timestamp_utc <= quote:
            raise ValueError(
                "The settlement horizon must be after the quote timestamp."
            )

        return MaturityResult(
            quote_timestamp_utc=quote,
            settlement_timestamp_utc=self.settlement_timestamp_utc,
        )