"""Read SPX quote files and select auditable market snapshots."""

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = (
    "quote_unixtime",
    "quote_readtime",
    "quote_date",
    "underlying_last",
    "expire_date",
    "expire_unix",
    "dte",
    "strike",
    "c_bid",
    "c_ask",
    "c_iv",
    "p_bid",
    "p_ask",
    "p_iv",
)

TEXT_COLUMNS = {
    "quote_readtime",
    "quote_date",
    "expire_date",
}


def _column_name(value: str) -> str:
    return value.strip().strip("[]").lower()


def _date(value: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if (
        pd.isna(timestamp)
        or timestamp.tzinfo is not None
        or timestamp != timestamp.normalize()
    ):
        raise ValueError("Supply a calendar date without a time or timezone.")

    return timestamp


@dataclass(frozen=True, slots=True, eq=False)
class QuoteSnapshot:
    """One observation date, expiry date and quote timestamp.

    Created by SPXQuoteFile.snapshot. Accessing quotes returns a copy.
    Vendor expiry timestamps are retained without assuming settlement
    conventions. No maturity is assigned by this class.
    """

    source_path: Path
    quote_date: pd.Timestamp
    expiry_date: pd.Timestamp
    quote_timestamp_utc: pd.Timestamp
    _quotes: pd.DataFrame = field(repr=False)

    def __post_init__(self) -> None:
        quotes = (
            self._quotes
            .sort_values("strike", kind="stable")
            .reset_index(drop=True)
            .copy(deep=True)
        )
        object.__setattr__(self, "_quotes", quotes)

    @property
    def quotes(self) -> pd.DataFrame:
        return self._quotes.copy(deep=True)

    @property
    def spot(self) -> float:
        """Return the unique positive underlying value in this snapshot."""
        values = self._quotes["underlying_last"].to_numpy()

        if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("The snapshot contains invalid underlying values.")

        unique = np.unique(values)
        if len(unique) != 1:
            raise ValueError(
                "The snapshot has multiple underlying values; "
                "an explicit policy is required before pricing."
            )

        return float(unique[0])

    def with_midpoints(self) -> pd.DataFrame:
        """Return quotes with derived midpoints and spreads.

        Original bids and asks remain unchanged, including problem values.
        """
        quotes = self.quotes

        for side in ("c", "p"):
            quotes[f"{side}_mid"] = (
                quotes[f"{side}_bid"] + quotes[f"{side}_ask"]
            ) / 2.0
            quotes[f"{side}_spread"] = (
                quotes[f"{side}_ask"] - quotes[f"{side}_bid"]
            )

        return quotes

    def row_diagnostics(self) -> pd.Series:
        """Describe identity, underlying and timing conditions."""
        quotes = self._quotes
        strikes = quotes["strike"].to_numpy()
        underlying = quotes["underlying_last"].to_numpy()
        dte = quotes["dte"].to_numpy()

        ny_dates = (
            quotes["quote_timestamp_utc"]
            .dt.tz_convert("America/New_York")
            .dt.tz_localize(None)
            .dt.normalize()
        )

        counts = {
            "rows": len(quotes),
            "unique_strikes": quotes["strike"].nunique(),
            "duplicate_key_rows": quotes.duplicated(
                ["quote_unixtime", "expire_date", "strike"],
                keep=False,
            ).sum(),
            "invalid_strike_rows": (
                ~np.isfinite(strikes) | (strikes <= 0.0)
            ).sum(),
            "invalid_underlying_rows": (
                ~np.isfinite(underlying) | (underlying <= 0.0)
            ).sum(),
            "distinct_underlying_values": (
                quotes["underlying_last"].nunique()
            ),
            "nonfinite_dte_rows": (~np.isfinite(dte)).sum(),
            "nonpositive_dte_rows": (dte <= 0.0).sum(),
            "ny_quote_date_mismatch_rows": (
                ny_dates != self.quote_date
            ).sum(),
            "invalid_vendor_expiry_timestamp_rows": (
                quotes["vendor_expiry_timestamp_utc"].isna().sum()
            ),
        }

        return pd.Series(counts, name="count", dtype="int64")

    def quote_diagnostics(self) -> pd.DataFrame:
        """Count quote conditions separately for calls and puts.

        Counts can overlap. They are not a count of rejected observations.
        """
        reports = {}

        for prefix, name in (("c", "call"), ("p", "put")):
            bid = self._quotes[f"{prefix}_bid"].to_numpy()
            ask = self._quotes[f"{prefix}_ask"].to_numpy()
            vendor_iv = self._quotes[f"{prefix}_iv"].to_numpy()

            reports[name] = {
                "missing_bid_or_ask": (
                    np.isnan(bid) | np.isnan(ask)
                ).sum(),
                "nonfinite_bid_or_ask": (
                    ~np.isfinite(bid) | ~np.isfinite(ask)
                ).sum(),
                "negative_bid_or_ask": (
                    (bid < 0.0) | (ask < 0.0)
                ).sum(),
                "zero_bid": (bid == 0.0).sum(),
                "crossed_quote": (bid > ask).sum(),
                "locked_quote": (
                    np.isfinite(bid)
                    & np.isfinite(ask)
                    & (bid == ask)
                ).sum(),
                "missing_vendor_iv": np.isnan(vendor_iv).sum(),
            }

        return pd.DataFrame(reports, dtype="int64")


class SPXQuoteFile:
    """Load the core fields of one monthly vendor file once.

    Dates are parsed strictly. Missing numeric quote values are retained.
    source_record identifies the original one-based data-record position,
    excluding the header.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

        if not self.path.is_file():
            raise FileNotFoundError(self.path)

        with self.path.open(
            encoding="utf-8-sig",
            newline="",
        ) as handle:
            header = next(
                csv.reader(handle, skipinitialspace=True),
                [],
            )

        canonical = [_column_name(name) for name in header]

        if len(set(canonical)) != len(canonical):
            raise ValueError("The source header contains duplicate field names.")

        field_map = dict(zip(canonical, header))
        missing = sorted(set(CORE_COLUMNS) - set(field_map))

        if missing:
            raise ValueError(
                "Missing required fields: " + ", ".join(missing)
            )

        numeric_types = {
            field_map[name]: "float64"
            for name in CORE_COLUMNS
            if name not in TEXT_COLUMNS
        }

        quotes = pd.read_csv(
            self.path,
            usecols=[field_map[name] for name in CORE_COLUMNS],
            dtype=numeric_types,
            skipinitialspace=True,
            encoding="utf-8-sig",
            low_memory=False,
            on_bad_lines="error",
        )
        quotes = quotes.rename(
            columns={
                original: name
                for name, original in field_map.items()
            }
        )
        quotes = quotes.loc[:, list(CORE_COLUMNS)].copy()

        if quotes.empty:
            raise ValueError("The source file contains no data records.")

        for name in ("quote_date", "expire_date"):
            quotes[name] = pd.to_datetime(
                quotes[name],
                format="%Y-%m-%d",
                errors="raise",
            )
            if quotes[name].isna().any():
                raise ValueError(f"The source contains missing {name} values.")

        quotes["source_record"] = np.arange(
            1,
            len(quotes) + 1,
            dtype=np.int64,
        )
        quotes["quote_timestamp_utc"] = pd.to_datetime(
            quotes["quote_unixtime"],
            unit="s",
            utc=True,
            errors="coerce",
        )
        quotes["vendor_expiry_timestamp_utc"] = pd.to_datetime(
            quotes["expire_unix"],
            unit="s",
            utc=True,
            errors="coerce",
        )

        self._quotes = quotes

    @property
    def rows(self) -> int:
        return len(self._quotes)

    @property
    def observation_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            self._quotes["quote_date"]
            .drop_duplicates()
            .sort_values()
        )

    @property
    def invalid_quote_timestamp_rows(self) -> int:
        return int(self._quotes["quote_timestamp_utc"].isna().sum())

    def available_expiries(self, quote_date: str) -> pd.DatetimeIndex:
        date = _date(quote_date)
        selected = self._quotes.loc[
            self._quotes["quote_date"] == date,
            "expire_date",
        ]

        if selected.empty:
            raise ValueError(f"No observations for {date.date()}.")

        return pd.DatetimeIndex(
            selected.drop_duplicates().sort_values()
        )

    def snapshot(
        self,
        quote_date: str,
        expiry_date: str,
        quote_timestamp_utc: str | None = None,
    ) -> QuoteSnapshot:
        date = _date(quote_date)
        expiry = _date(expiry_date)

        selected = self._quotes.loc[
            (self._quotes["quote_date"] == date)
            & (self._quotes["expire_date"] == expiry)
        ]

        if selected.empty:
            raise ValueError(
                f"No quotes for {date.date()} / {expiry.date()}."
            )

        if selected["quote_timestamp_utc"].isna().any():
            raise ValueError(
                "The selected date/expiry contains invalid quote timestamps."
            )

        if quote_timestamp_utc is None:
            timestamps = (
                selected["quote_timestamp_utc"]
                .drop_duplicates()
                .sort_values()
            )
            if len(timestamps) != 1:
                raise ValueError(
                    "Multiple quote timestamps are available; "
                    "supply quote_timestamp_utc explicitly."
                )
            timestamp = timestamps.iloc[0]

        else:
            timestamp = pd.Timestamp(quote_timestamp_utc)
            if pd.isna(timestamp) or timestamp.tzinfo is None:
                raise ValueError(
                    "The quote timestamp must include an explicit timezone."
                )
            timestamp = timestamp.tz_convert("UTC")

        selected = selected.loc[
            selected["quote_timestamp_utc"] == timestamp
        ]

        if selected.empty:
            raise ValueError("No quotes match the supplied quote timestamp.")

        return QuoteSnapshot(
            source_path=self.path,
            quote_date=date,
            expiry_date=expiry,
            quote_timestamp_utc=timestamp,
            _quotes=selected,
        )