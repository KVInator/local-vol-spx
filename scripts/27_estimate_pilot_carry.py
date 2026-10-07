"""Estimate option-implied carry and identification diagnostics for the pilot."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

from pilot_carry import CarrySettings, PilotCarryEstimator


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
            "outputs/hedging_data_diagnostics/september_2023/carry"
        ),
    )
    args = parser.parse_args()
    settings = CarrySettings()
    q = pd.read_csv(args.quotes)
    print(
        f"Loaded {len(q):,} quote sides; "
        "matching within date, expiry, root and strike.",
        flush=True,
    )

    tables, audit = PilotCarryEstimator(settings).run(q)
    fits = tables["carry_estimates"]
    primary = fits.loc[fits["window"].eq(settings.primary_window)]
    daily = primary.groupby("quote_date").agg(
        expiry_groups=("status", "size"),
        fitted_groups=("status", lambda x: x.eq("fitted").sum()),
        median_parity_rms=("rms_parity_half_widths", "median"),
        median_implied_rate_pct=("implied_zero_rate_pct", "median"),
        median_rate_band_width_pp=("rate_band_width_pp", "median"),
        max_required_band_multiplier=("minimum_band_multiplier", "max"),
    ).reset_index()
    tables["daily_summary"] = daily

    args.output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output / f"{name}.csv", index=False)

    import pilot_carry

    audit["input_sha256"] = {str(args.quotes): digest(args.quotes)}
    parent_audit = args.quotes.parent / "audit.json"
    if parent_audit.exists():
        audit["input_sha256"][str(parent_audit)] = digest(parent_audit)
    audit["source_sha256"] = {
        str(p): digest(p)
        for p in [Path(__file__), Path(pilot_carry.__file__)]
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

    print(f"Matched pairs: {audit['matched_pairs']:,}")
    print(
        "Quote sides without a matching side: "
        f"{audit['unpaired_quote_sides']:,}"
    )
    print("\nEstimation status:")
    print(
        fits.groupby(["window", "status"])
        .size().rename("expiry_groups").to_string()
    )
    print("\nPrimary-window daily diagnostics:")
    print(
        daily.to_string(
            index=False, float_format=lambda x: f"{x:.6g}"
        )
    )
    print("\nChanges across fitted strike windows:")
    columns = [
        "forward_window_range_points", "rate_window_range_pp",
    ]
    print(
        tables["window_sensitivity"][columns]
        .agg(["median", "max"]).to_string()
    )

    good = primary.loc[primary["status"].eq("fitted")]
    if "original_bands_feasible" in good:
        print(
            "\nPrimary groups incompatible with original parity bands: "
            f"{int(good['original_bands_feasible'].eq(False).sum())}"
        )
    print(f"\nSaved: {args.output.resolve()}")
    print(
        "Option-implied estimates and sensitivity scenarios only; "
        "no funding curve or model promoted."
    )


if __name__ == "__main__":
    main()