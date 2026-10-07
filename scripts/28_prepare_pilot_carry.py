"""Prepare assumed-rate daily carry and primary calibration quote inputs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import pilot_carry
import pilot_carry_inputs
from pilot_carry_inputs import CarryInputSettings, PilotCarryInputs


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--quotes",
        type=Path,
        default=Path(
            "data/processed/hedging_pilot/september_2023/pilot_quotes.csv"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/processed/hedging_pilot/september_2023/carry_inputs"
        ),
    )

    parser.add_argument("--primary-rate-pct", type=float, default=5.0)

    parser.add_argument(
        "--rates-pct",
        type=float,
        nargs="+",
        default=[3.0, 5.0, 7.0],
    )

    args = parser.parse_args()

    settings = CarryInputSettings(
        primary_rate=args.primary_rate_pct / 100,
        rates=tuple(r / 100 for r in args.rates_pct),
    )

    quotes = pd.read_csv(args.quotes)

    print(
        f"Loaded {len(quotes):,} current-date quote sides.",
        flush=True,
    )

    print(
        "Preparing conditional forwards at assumed annual rates:",
        args.rates_pct,
        flush=True,
    )

    tables, audit = PilotCarryInputs(settings).prepare(quotes)

    args.output.mkdir(parents=True, exist_ok=True)

    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)

    audit["input_sha256"] = {
        str(args.quotes): digest(args.quotes)
    }

    parent = args.quotes.parent / "audit.json"
    if parent.exists():
        audit["input_sha256"][str(parent)] = digest(parent)

    audit["source_sha256"] = {
        str(path): digest(path)
        for path in [
            Path(__file__),
            Path(pilot_carry.__file__),
            Path(pilot_carry_inputs.__file__),
        ]
    }

    audit["output_sha256"] = {
        f"{name}.csv": digest(args.output / f"{name}.csv")
        for name in tables
    }

    audit["versions"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }

    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    summary = tables["carry_inputs"].groupby(
        ["case", "annual_rate_pct"]
    ).agg(
        groups=("root", "size"),
        ready=("carry_ready", "sum"),
        parity_incompatible=("parity_bands_incompatible", "sum"),
        fitted_parity_outside=("fitted_parity_outside_bands", "sum"),
    )

    print("\nCarry scenarios:")
    print(summary.to_string())

    print("\nPrimary-case daily calibration inputs:")
    print(tables["daily_summary"].to_string(index=False))

    print(f"\nPrimary case: {audit['primary_case']}")
    print(
        f"One quote per strike: "
        f"{audit['primary_calibration_quotes']:,}"
    )

    print(f"Saved: {args.output.resolve()}")

    print(
        "Parity and call-bound flags remain in the tables. "
        "No diffusion calibrated or hedge backtest run."
    )


if __name__ == "__main__":
    main()