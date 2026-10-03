"""Assign an explicit model maturity to an SPX quote snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from market_data import SPXQuoteFile
from maturity import DAY_COUNT, ExpiryConvention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign an explicit maturity to an SPX snapshot."
    )
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--quote-date", required=True)
    parser.add_argument("--expiry-date", required=True)
    parser.add_argument("--quote-timestamp-utc")

    parser.add_argument("--settlement-time-ny", required=True)
    parser.add_argument(
        "--settlement-kind",
        choices=("AM", "PM"),
        required=True,
    )
    parser.add_argument(
        "--settlement-status",
        choices=("inferred", "verified"),
        required=True,
    )
    parser.add_argument("--settlement-evidence", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reader = SPXQuoteFile(args.file)
    snapshot = reader.snapshot(
        args.quote_date,
        args.expiry_date,
        quote_timestamp_utc=args.quote_timestamp_utc,
    )

    convention = ExpiryConvention(
        expiry_date=snapshot.expiry_date,
        settlement_time_ny=args.settlement_time_ny,
        settlement_kind=args.settlement_kind,
        verification_status=args.settlement_status,
        evidence=args.settlement_evidence,
    )
    maturity = convention.maturity(
        snapshot.quote_timestamp_utc
    )

    quotes = snapshot.with_midpoints()

    vendor_difference = (
        convention.settlement_timestamp_utc
        - quotes["vendor_expiry_timestamp_utc"]
    ).dt.total_seconds()

    valid_vendor_difference = vendor_difference.dropna()

    max_vendor_difference = (
        float(valid_vendor_difference.abs().max())
        if not valid_vendor_difference.empty
        else None
    )

    reported_dte = quotes["dte"].to_numpy(dtype=float)
    finite_dte = np.isfinite(reported_dte)

    max_dte_difference = (
        float(
            np.max(
                np.abs(
                    reported_dte[finite_dte]
                    - maturity.elapsed_days
                )
            )
        )
        if finite_dte.any()
        else None
    )

    quotes["model_settlement_timestamp_utc"] = (
        convention.settlement_timestamp_utc
    )
    quotes["maturity_years"] = maturity.year_fraction
    quotes["maturity_day_count"] = DAY_COUNT
    quotes["settlement_kind"] = convention.settlement_kind
    quotes["settlement_verification_status"] = (
        convention.verification_status
    )

    with snapshot.source_path.open("rb") as handle:
        source_sha256 = hashlib.file_digest(
            handle, "sha256"
        ).hexdigest()

    project_root = Path(__file__).resolve().parents[1]

    quote_tag = snapshot.quote_timestamp_utc.strftime(
        "%Y%m%dT%H%M%SZ"
    )
    settlement_tag = convention.settlement_time_ny.replace(
        ":", ""
    )
    stem = (
        f"spx_{snapshot.quote_date.date()}_"
        f"{snapshot.expiry_date.date()}_{quote_tag}_"
        f"{convention.settlement_kind}_{settlement_tag}"
    )

    snapshot_path = (
        project_root / "data" / "processed" / "maturities"
        / f"{stem}.csv"
    )
    audit_path = (
        project_root / "outputs" / "maturity_audits"
        / f"{stem}.json"
    )

    audit = {
        "source_file": snapshot.source_path.name,
        "source_sha256": source_sha256,
        "data_provider": "unconfirmed",
        "source_rows": reader.rows,
        "snapshot_rows": len(quotes),
        "quote_date": str(snapshot.quote_date.date()),
        "expiry_date": str(snapshot.expiry_date.date()),
        "quote_timestamp_utc": (
            maturity.quote_timestamp_utc.isoformat()
        ),
        "model_settlement_timestamp_utc": (
            maturity.settlement_timestamp_utc.isoformat()
        ),
        "model_settlement_timestamp_ny": (
            convention.settlement_timestamp_ny.isoformat()
        ),
        "settlement_kind": convention.settlement_kind,
        "settlement_verification_status": (
            convention.verification_status
        ),
        "settlement_evidence": convention.evidence,
        "day_count": DAY_COUNT,
        "elapsed_days": maturity.elapsed_days,
        "maturity_years": maturity.year_fraction,
        "vendor_comparison": {
            "missing_expiry_timestamp_rows": int(
                vendor_difference.isna().sum()
            ),
            "max_abs_clock_difference_seconds": (
                max_vendor_difference
            ),
            "finite_reported_dte_rows": int(finite_dte.sum()),
            "max_abs_reported_dte_difference_days": (
                max_dte_difference
            ),
        },
    }

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)

    quotes.to_csv(snapshot_path, index=False)
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"Snapshot rows:          {len(quotes)}")
    print(f"Quote time UTC:         {maturity.quote_timestamp_utc}")
    print(
        "Model expiry time UTC:  "
        f"{maturity.settlement_timestamp_utc}"
    )
    print(
        "Model expiry time NY:   "
        f"{convention.settlement_timestamp_ny}"
    )
    print(f"Settlement kind:        {convention.settlement_kind}")
    print(f"Verification status:    {convention.verification_status}")
    print(f"Elapsed days:           {maturity.elapsed_days:.8f}")
    print(f"Maturity, years:        {maturity.year_fraction:.10f}")
    print(f"Day count:              {DAY_COUNT}")
    print(
        "Max vendor clock difference, seconds: "
        f"{max_vendor_difference}"
    )
    print(
        "Max reported DTE difference, days:     "
        f"{max_dte_difference}"
    )
    print(f"\nSnapshot saved to: {snapshot_path}")
    print(f"Audit saved to:    {audit_path}")


if __name__ == "__main__":
    main()