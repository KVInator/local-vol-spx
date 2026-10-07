"""Prepare the September research panel under recorded, unverified conventions."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from hedging_pilot import HedgingPilot, PilotSettings


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw", type=Path,
        default=Path("data/raw/spx_eod_202309.txt"),
    )
    parser.add_argument(
        "--policy", type=Path,
        default=Path(
            "outputs/hedging_data_diagnostics/historical_2013_2023/"
            "provisional_session_policy.csv"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data/processed/hedging_pilot/september_2023"),
    )
    args = parser.parse_args()

    raw = pd.read_csv(args.raw, dtype="string", encoding="utf-8-sig")
    tables, audit = HedgingPilot(
        pd.read_csv(args.policy), PilotSettings()
    ).prepare(raw)

    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)

    import hedging_pilot
    import hedging_data_audit

    audit["input_sha256"] = {
        str(p): digest(p) for p in [args.raw, args.policy]
    }
    audit["source_sha256"] = {
        str(p): digest(p) for p in [
            Path(__file__),
            Path(hedging_pilot.__file__),
            Path(hedging_data_audit.__file__),
        ]
    }
    audit["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )

    print("Provisional research panel; no contract identity inferred.")
    print(
        tables["quote_selection_summary"]
        .groupby("quote_status")["quotes"].sum().to_string()
    )
    quotes = tables["pilot_quotes"]
    print(f"\nSelected quote dates: {quotes['quote_date'].nunique()}")
    print(f"Selected calibration quotes: {len(quotes):,}")
    print(
        "Current-day entry candidates: "
        f"{int(quotes['entry_research_candidate'].sum()):,}"
    )
    print(
        "Zero-bid bounds retained separately: "
        f"{len(tables['zero_bid_bounds']):,}"
    )
    print(f"\nSaved: {args.output.resolve()}")
    print("Daily carry, calibration and hedge accounting are the next stages.")


if __name__ == "__main__":
    main()