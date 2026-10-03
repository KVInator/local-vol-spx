"""Estimate and audit carry from a maturity-assigned SPX snapshot."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from carry import ParityCarryEstimator, ParityQuotePolicy


WINDOWS = (0.005, 0.01, 0.03)


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate carry from audited SPX quotes."
    )
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--maturity-audit", type=Path, required=True)
    parser.add_argument("--discount-factor", type=float, required=True)
    parser.add_argument("--discount-evidence", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.discount_evidence.strip():
        raise ValueError("Discounting evidence must be supplied.")

    quotes = pd.read_csv(args.snapshot)
    maturity_audit = json.loads(
        args.maturity_audit.read_text(encoding="utf-8")
    )

    if len(quotes) != maturity_audit["snapshot_rows"]:
        raise ValueError("Snapshot row count differs from its audit.")

    expected_fields = {
        "quote_date": maturity_audit["quote_date"],
        "expire_date": maturity_audit["expiry_date"],
        "settlement_kind": maturity_audit["settlement_kind"],
        "settlement_verification_status": (
            maturity_audit["settlement_verification_status"]
        ),
    }

    for column, expected in expected_fields.items():
        if not quotes[column].eq(expected).all():
            raise ValueError(
                f"Snapshot field {column!r} differs from its audit."
            )

    expected_clocks = {
        "quote_timestamp_utc": maturity_audit["quote_timestamp_utc"],
        "model_settlement_timestamp_utc": (
            maturity_audit["model_settlement_timestamp_utc"]
        ),
    }

    for column, expected in expected_clocks.items():
        timestamps = pd.to_datetime(
            quotes[column], utc=True, errors="raise"
        )

        if not timestamps.eq(pd.Timestamp(expected)).all():
            raise ValueError(
                f"Snapshot clock {column!r} differs from its audit."
            )

    maturity = float(maturity_audit["maturity_years"])
    stored_maturity = quotes["maturity_years"].to_numpy(dtype=float)

    if (
        not np.isfinite(stored_maturity).all()
        or np.any(np.abs(stored_maturity - maturity) > 1e-12)
    ):
        raise ValueError("Snapshot maturities differ from their audit.")

    spots = quotes["underlying_last"].unique()

    if len(spots) != 1:
        raise ValueError("The snapshot must contain one underlying value.")

    spot = float(spots[0])
    estimator = ParityCarryEstimator(maturity)

    project_root = Path(__file__).resolve().parents[1]
    pair_directory = (
        project_root / "data" / "processed" / "carry"
    )
    output_directory = (
        project_root / "outputs" / "carry_diagnostics"
    )

    pair_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    stem = args.snapshot.stem
    summaries = []
    diagnostic_tables = []
    selection_counts = {}

    for window in WINDOWS:
        policy = ParityQuotePolicy(
            spot=spot,
            relative_window=window,
        )
        selection = policy.select(quotes)
        pairs = selection.loc[selection["eligible"]].copy()

        window_tag = f"{100.0 * window:g}pct"

        selection.to_csv(
            pair_directory / f"{stem}_{window_tag}_selection.csv",
            index=False,
        )

        counts = selection["selection_reason"].value_counts()
        selection_counts[window_tag] = {
            str(reason): int(count)
            for reason, count in counts.items()
        }

        for discount in (None, args.discount_factor):
            estimate = estimator.fit(
                pairs,
                discount_factor=discount,
            )

            summary = estimate.summary(pairs)
            summary["window_pct"] = 100.0 * window
            summaries.append(summary)

            diagnostics = estimate.quote_diagnostics(pairs)
            diagnostics["window_pct"] = 100.0 * window
            diagnostics["fit_mode"] = estimate.mode
            diagnostic_tables.append(diagnostics)

    comparison = pd.DataFrame(summaries)
    diagnostics = pd.concat(
        diagnostic_tables, ignore_index=True
    )

    summary_path = output_directory / f"{stem}_summary.csv"
    diagnostic_path = pair_directory / f"{stem}_fit_quotes.csv"
    audit_path = output_directory / f"{stem}_audit.json"

    comparison.to_csv(summary_path, index=False)
    diagnostics.to_csv(diagnostic_path, index=False)

    audit = {
        "snapshot_file": args.snapshot.name,
        "snapshot_sha256": file_sha256(args.snapshot),
        "maturity_audit_file": args.maturity_audit.name,
        "maturity_audit_sha256": file_sha256(args.maturity_audit),
        "maturity_audit": maturity_audit,
        "spot": spot,
        "relative_windows": list(WINDOWS),
        "weighting": "inverse squared parity-interval half-width",
        "fixed_discount_factor": args.discount_factor,
        "discount_evidence": args.discount_evidence.strip(),
        "selection_counts": selection_counts,
        "fits": summaries,
    }

    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    display_columns = [
        "mode",
        "window_pct",
        "pairs",
        "discount_factor",
        "forward",
        "implied_rate_pct",
        "midpoint_rmse",
        "rms_half_widths",
        "outside_bands",
    ]

    print(f"Underlying: {spot:.6f}")
    print(f"Maturity:   {maturity:.10f} years")
    print(f"Fixed D:    {args.discount_factor:.8f}")
    print("\nCarry comparison:")
    print(
        comparison[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.6f}",
        )
    )

    print("\nSelection counts:")
    for window_tag, counts in selection_counts.items():
        print(f"  {window_tag}: {counts}")

    print(f"\nSummary saved to:     {summary_path}")
    print(f"Diagnostics saved to: {diagnostic_path}")
    print(f"Audit saved to:       {audit_path}")


if __name__ == "__main__":
    main()