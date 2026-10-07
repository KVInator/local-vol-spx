"""Audit raw EOD quotes before specifying a daily hedging experiment."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

from hedging_data_audit import AuditSettings, HedgingDataAudit


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024), b""
        ):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", default="data/raw/spx_eod_202309.txt"
    )
    parser.add_argument(
        "--output",
        default="outputs/hedging_data_diagnostics/september_2023",
    )
    parser.add_argument("--start", default="2023-09-01")
    parser.add_argument("--end", default="2023-09-30")
    parser.add_argument("--close-ny", default="16:00:00")
    parser.add_argument("--root-column", default=None)
    parser.add_argument("--sep", default=",")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    output = Path(args.output)

    settings = AuditSettings(
        start=args.start,
        end=args.end,
        close_ny=args.close_ny,
        root_column=args.root_column,
    )

    raw = pd.read_csv(
        raw_path,
        sep=args.sep,
        dtype=str,
        low_memory=False,
    )
    tables = HedgingDataAudit(settings).run(raw)

    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)

    rows = tables["row_audit"]
    counts = {
        c: int(rows[c].sum())
        for c in rows
        if rows[c].dtype == bool
    }

    flags = (
        "bad_timestamp",
        "timestamp_mismatch",
        "date_mismatch",
        "bad_spot",
        "bad_contract",
        "duplicate_close_key",
        "spot_inconsistent",
        "c_bad_quote",
        "p_bad_quote",
        "c_locked_quote",
        "p_locked_quote",
    )
    flagged = rows[list(flags)].any(axis=1)

    original = (
        raw.reset_index(drop=True)
        .loc[flagged]
        .add_prefix("raw_")
    )
    pd.concat(
        [
            rows.loc[flagged, ["source_row", *flags]],
            original,
        ],
        axis=1,
    ).to_csv(
        output / "flagged_source_records.csv",
        index=False,
    )

    module_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "hedging_data_audit.py"
    )

    audit = {
        "input_sha256": {
            str(raw_path): sha256(raw_path),
        },
        "source_sha256": {
            str(Path(__file__)): sha256(__file__),
            "hedging_data_audit.py": sha256(module_path),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
        "settings": asdict(settings),
        "raw_rows": len(raw),
        "row_flag_counts": counts,
        "row_flag_counts_scope": (
            "Whole raw file; date and coverage flags "
            "are reported separately."
        ),
        "observed_dates": (
            tables["daily_summary"]["quote_date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        ),
        "adjacent_status_counts": (
            tables["adjacent_observations"]["status"]
            .value_counts()
            .to_dict()
        ),
        "contract_identity_certified": False,
        "settlement_fixings_verified": False,
        "historical_exchange_calendar_verified": False,
        "daily_discount_curves_verified": False,
        "parity_method": (
            "Free-discount robust C-P regression on "
            "two-sided near-spot pairs; diagnostic only."
        ),
        "moneyness": (
            "log(K/observed spot), not log(K/forward)."
        ),
        "coverage_window": (
            "Descriptive subset only; quote_panel and "
            "expiry_coverage retain wider available close coverage."
        ),
        "day_count": (
            "Calendar-date difference; "
            "not a verified time to fixing."
        ),
        "mark_quality": (
            "Nonnegative ordered bid/ask with positive spread. "
            "Zero bids are separately flagged."
        ),
        "duplicates": (
            "All duplicate close keys quarantined; "
            "none chosen or averaged."
        ),
        "continuity": (
            "Consecutive quote_date values in range, including "
            "dates with unusable closes; "
            "exchange-session completeness unverified."
        ),
        "settlement": (
            "Vendor expiry timestamps are retained as metadata, "
            "not accepted as fixing evidence."
        ),
        "unchanged_quotes": (
            "Suspicion flag only; unchanged quotes "
            "do not prove staleness."
        ),
        "selection_scope": (
            "Retrospective descriptive audit; continuity "
            "must not screen future backtest entries."
        ),
        "input_modified": False,
        "models_refitted": False,
        "backtest_performed": False,
    }

    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print(
        f"Raw rows: {len(raw):,}; "
        f"dated observations: {len(tables['daily_summary'])}"
    )

    print("\nDaily coverage:")
    print(tables["daily_summary"].to_string(index=False))

    print("\nAdjacent observation status:")
    print(
        tables["adjacent_observations"]["status"]
        .value_counts()
        .to_string()
    )

    print("\nParity diagnostic status:")
    print(
        tables["parity_summary"]["status"]
        .value_counts()
        .to_string()
    )

    print(f"\nDiagnostics: {output.resolve()}")
    print(
        "Contract identity, fixing conventions and "
        "daily discount inputs remain unverified."
    )
    print(
        "This audit does not select a backtest universe "
        "or change calibrated models."
    )


if __name__ == "__main__":
    main()