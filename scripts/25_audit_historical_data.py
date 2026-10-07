"""Audit monthly raw files without combining their quote panels in memory."""

import argparse
from pathlib import Path

from historical_data_audit import (
    HistoricalAuditSettings,
    HistoricalDataAudit,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-directory", type=Path, default=Path("data/raw")
    )
    parser.add_argument("--pattern", default="spx_eod_*.txt")
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--root-column")
    parser.add_argument(
        "--readtime-timezone", default="America/New_York"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/hedging_data_diagnostics/historical_2013_2023"
        ),
    )
    args = parser.parse_args()

    files = sorted(args.raw_directory.glob(args.pattern))
    if not files:
        parser.error("No matching monthly raw files.")

    settings = HistoricalAuditSettings(
        start=args.start,
        end=args.end,
        root_column=args.root_column,
        readtime_timezone=args.readtime_timezone,
    )
    tables, audit = HistoricalDataAudit(settings).run(
        files, args.output, script_path=__file__
    )

    print("\nFile status:")
    print(tables["file_summary"]["status"].value_counts().to_string())

    print("\nSession status:")
    print(tables["daily_summary"]["status"].value_counts().to_string())

    print("\nYear coverage:")
    print(tables["year_summary"].to_string(index=False))

    print("\nObserved snapshot clocks, New York time:")
    daily = tables["daily_summary"]
    if "clocks_ny" in daily:
        print(
            daily.loc[daily["raw_rows"].gt(0), "clocks_ny"]
            .value_counts()
            .to_string()
        )

    adjacent = tables["adjacent_summary"]
    print("\nAdjacent reference sessions:")
    print(adjacent.groupby("status")["contracts"].sum().to_string())

    print("\nCross-month continuity:")
    print(
        adjacent.loc[adjacent["cross_month"]]
        .groupby("status")["contracts"]
        .sum()
        .to_string()
    )

    print(f"\nDiagnostics: {args.output.resolve()}")
    print(
        "Calendar discrepancies and vendor clocks require "
        "review before universe selection."
    )
    print(
        "Contract identity, fixing conventions and daily "
        "carry remain unverified."
    )
    return int(
        any(k != "audited" for k in audit["files_by_status"])
    )


if __name__ == "__main__":
    raise SystemExit(main())