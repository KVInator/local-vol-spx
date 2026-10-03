"""Extract a quote snapshot and record its source and diagnostics."""

import argparse
import hashlib
import json
from pathlib import Path

from market_data import SPXQuoteFile


ROOT = Path(__file__).resolve().parents[1]


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--quote-date", required=True)
    parser.add_argument("--expiry-date", required=True)
    parser.add_argument("--quote-timestamp-utc")
    args = parser.parse_args()

    reader = SPXQuoteFile(args.file)
    snapshot = reader.snapshot(
        quote_date=args.quote_date,
        expiry_date=args.expiry_date,
        quote_timestamp_utc=args.quote_timestamp_utc,
    )

    quotes = snapshot.with_midpoints()
    row_report = snapshot.row_diagnostics()
    quote_report = snapshot.quote_diagnostics()
    dates = reader.observation_dates

    print(f"Source file: {reader.path.name}")
    print(f"Rows in source: {reader.rows:,}")
    print(f"Observation dates: {len(dates)}")
    print(
        "Source date range: "
        f"{dates[0].date()} to {dates[-1].date()}"
    )
    print(
        "Invalid quote timestamp rows in source: "
        f"{reader.invalid_quote_timestamp_rows}"
    )

    print(f"\nObservation date: {snapshot.quote_date.date()}")
    print(f"Expiry date:      {snapshot.expiry_date.date()}")
    print(f"Quote time UTC:   {snapshot.quote_timestamp_utc}")
    print(
        "Quote time NY:    "
        f"{snapshot.quote_timestamp_utc.tz_convert('America/New_York')}"
    )

    print("\nSnapshot diagnostics:")
    print(row_report.to_string())

    print("\nQuote diagnostics:")
    print(quote_report.to_string())

    spot = snapshot.spot
    print(f"\nUnderlying: {spot:.6f}")
    print(f"Reported DTE values: {quotes['dte'].unique().tolist()}")
    print(
        f"Strike range: {quotes['strike'].min():.2f} "
        f"to {quotes['strike'].max():.2f}"
    )

    nearest = (quotes["strike"] - spot).abs().nsmallest(11).index
    display = quotes.loc[nearest].sort_values("strike")

    print("\nEleven strikes nearest the underlying:")
    print(
        display[
            [
                "strike",
                "c_bid",
                "c_ask",
                "c_mid",
                "p_bid",
                "p_ask",
                "p_mid",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: f"{value:.2f}",
        )
    )

    clock_tag = (
        snapshot.quote_timestamp_utc.isoformat()
        .replace("+00:00", "Z")
        .replace("-", "")
        .replace(":", "")
        .replace(".", "")
    )
    stem = (
        f"spx_{snapshot.quote_date.date()}_"
        f"{snapshot.expiry_date.date()}_{clock_tag}"
    )

    data_directory = ROOT / "data" / "processed" / "snapshots"
    audit_directory = ROOT / "outputs" / "data_audits"
    data_directory.mkdir(parents=True, exist_ok=True)
    audit_directory.mkdir(parents=True, exist_ok=True)

    data_path = data_directory / f"{stem}.csv"
    audit_path = audit_directory / f"{stem}.json"

    manifest = {
        "source_file": reader.path.name,
        "source_sha256": file_digest(reader.path),
        "source_rows": reader.rows,
        "quote_date": str(snapshot.quote_date.date()),
        "expiry_date": str(snapshot.expiry_date.date()),
        "quote_timestamp_utc": snapshot.quote_timestamp_utc.isoformat(),
        "snapshot_rows": len(quotes),
        "settlement_convention": "unverified",
        "row_diagnostics": {
            name: int(value)
            for name, value in row_report.items()
        },
        "quote_diagnostics": {
            issue: {
                side: int(value)
                for side, value in values.items()
            }
            for issue, values in quote_report.to_dict(
                orient="index"
            ).items()
        },
    }

    quotes.to_csv(data_path, index=False)
    audit_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"\nSnapshot saved to: {data_path}")
    print(f"Audit saved to:    {audit_path}")
    print(
        "Vendor expiry timestamps are preserved; "
        "maturity has not been assigned."
    )


if __name__ == "__main__":
    main()