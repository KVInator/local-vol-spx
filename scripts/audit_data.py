"""Audit monthly files and produce the session policy used by a study."""

import argparse
from pathlib import Path

from data_audit import HistoricalAuditSettings, HistoricalDataAudit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-directory", type=Path, default=Path("data/raw"))
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/hedging_data_diagnostics/historical_2013_2023"),
    )
    args = parser.parse_args()
    files = sorted(args.raw_directory.glob("spx_eod_*.txt"))
    if not files:
        parser.error("No monthly quote files found.")
    tables, audit = HistoricalDataAudit(
        HistoricalAuditSettings(start=args.start, end=args.end)
    ).run(files, args.output, script_path=__file__)
    print(tables["year_summary"].to_string(index=False))
    print("Saved audit:", args.output.resolve())
    return int(any(status != "audited" for status in audit["files_by_status"]))


if __name__ == "__main__":
    raise SystemExit(main())
