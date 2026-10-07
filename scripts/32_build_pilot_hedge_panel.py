"""Select current-date contracts, calculate deltas, and match one-session exits."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy

import ah_backward_pricer
import ah_local_vol
import andreasen_huge
import backward_pricer
import daily_ah_validation
import pilot_hedge_panel
from andreasen_huge import AndreasenHugeSurface
from pilot_hedge_panel import (
    DailyHedgeDeltas,
    HedgePanelSettings,
    PilotContractSelector,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path(
        "data/processed/hedging_pilot/september_2023"
    )

    parser.add_argument(
        "--calibration-run", type=Path, required=True
    )
    parser.add_argument(
        "--quotes", type=Path,
        default=base / "pilot_quotes.csv",
    )
    parser.add_argument(
        "--bounds", type=Path,
        default=base / "zero_bid_bounds.csv",
    )
    parser.add_argument(
        "--timeline", type=Path,
        default=base / "session_timeline.csv",
    )
    parser.add_argument(
        "--carry", type=Path,
        default=base / "carry_inputs/primary_carry.csv",
    )
    parser.add_argument(
        "--output", type=Path,
        default=base / "hedge_panel",
    )
    parser.add_argument(
        "--dates", nargs="+",
        help=(
            "Entry-date subset; exits still use the full timeline."
        ),
    )
    parser.add_argument(
        "--intervals", type=int, default=24000
    )
    parser.add_argument(
        "--steps-per-day", type=int, default=128
    )
    args = parser.parse_args()

    settings = HedgePanelSettings(
        intervals=args.intervals,
        steps_per_day=args.steps_per_day,
    )

    calibration = args.calibration_run.resolve()
    audit_file = calibration / "audit.json"
    manifest_file = calibration / "model_manifest.csv"

    inputs = [
        args.quotes,
        args.bounds,
        args.timeline,
        args.carry,
        audit_file,
        manifest_file,
    ]
    input_hashes = {
        str(path.resolve()): digest(path)
        for path in inputs
    }

    parent = json.loads(audit_file.read_text())

    if digest(manifest_file) != parent[
        "output_sha256"
    ].get("model_manifest.csv"):
        raise ValueError(
            "Calibration manifest checksum mismatch"
        )

    if digest(args.carry) not in parent[
        "input_sha256"
    ].values():
        raise ValueError(
            "Carry input differs from the pinned calibration run"
        )

    dtype = {
        "quote_date": str,
        "expire_date": str,
        "root": str,
    }
    quotes, bounds, timeline, carry = [
        pd.read_csv(path, dtype=dtype)
        for path in [
            args.quotes,
            args.bounds,
            args.timeline,
            args.carry,
        ]
    ]
    manifest = pd.read_csv(
        manifest_file, dtype=dtype
    )

    if manifest.duplicated(
        ["quote_date", "root"]
    ).any():
        raise ValueError("Duplicate model manifest key")

    if (
        carry["case"].nunique() != 1
        or carry["annual_rate"].nunique() != 1
    ):
        raise ValueError(
            "Use one carry case and assumed annual rate"
        )

    selector = PilotContractSelector(settings)
    entries, buckets = selector.select(
        quotes, timeline, args.dates
    )

    if entries.empty:
        raise ValueError(
            "No current-date entries were selected"
        )

    pairs = selector.match_next(
        entries, quotes, bounds, timeline
    )

    output = args.output / datetime.now(
        timezone.utc
    ).strftime("run_%Y%m%dT%H%M%S_%fZ")

    output.mkdir(parents=True, exist_ok=False)

    for name, table in [
        ("entries", entries),
        ("selection_buckets", buckets),
        ("endpoint_pairs", pairs),
    ]:
        table.to_csv(
            output / f"{name}.csv", index=False
        )

    results, statuses = [], []
    calculator = DailyHedgeDeltas(settings)

    print(
        f"Selected {len(entries):,} fixed-contract entries "
        "using current-date information.",
        flush=True,
    )
    print(
        "Black uses observed-price IV; AH uses the radius "
        "0.0005 diffusion. Exit marks stay observed.",
        flush=True,
    )

    for (date, root), group in entries.groupby(
        ["quote_date", "root"], sort=True
    ):
        model = None
        match = manifest.loc[
            manifest.quote_date.eq(date)
            & manifest.root.eq(root)
        ]
        model_status = "model_missing"

        if len(match) and match.status.iloc[0] == "fitted":
            row = match.iloc[0]

            if row["case"] != carry["case"].iloc[0]:
                raise ValueError(
                    "Manifest and carry cases disagree"
                )

            path = (
                calibration / row.model_file
            ).resolve()

            if (
                not path.is_relative_to(calibration)
                or digest(path) != row.model_sha256
            ):
                raise ValueError(
                    f"Model path or checksum mismatch: {date}"
                )

            model = AndreasenHugeSurface.load(path)
            input_hashes[str(path)] = row.model_sha256
            model_status = "loaded"

        elif len(match):
            model_status = "model_not_fitted"

        result = calculator.calculate(
            group,
            carry,
            model,
            progress=lambda text: print(text, flush=True),
        )
        result["model_status"] = model_status
        results.append(result)

        status = {
            "quote_date": date,
            "root": root,
            "entries": len(result),
            "model_status": model_status,
            "black_ready": int(
                result.black_status.eq("ready").sum()
            ),
            "ah_ready": int(
                result.ah_status.eq("ready").sum()
            ),
            "ah_shape_flags": int(
                result.ah_status.eq("shape_flag").sum()
            ),
        }
        statuses.append(status)

        day = (
            output
            / date
            / hashlib.sha256(root.encode()).hexdigest()[:12]
        )
        day.mkdir(parents=True)

        result.to_csv(
            day / "delta_panel.csv", index=False
        )
        pd.DataFrame(statuses).to_csv(
            output / "daily_status.csv", index=False
        )

    panel = pd.concat(results, ignore_index=True)
    panel = panel.merge(
        pairs.drop(columns=["contract_id", "quote_date"]),
        on="entry_id",
        validate="one_to_one",
    )

    panel["both_deltas_ready"] = (
        panel.black_status.eq("ready")
        & panel.ah_status.eq("ready")
    )
    panel["matched_comparison_available"] = (
        panel.both_deltas_ready
        & panel.end_status.eq("matched")
    )

    panel.to_csv(
        output / "delta_panel.csv", index=False
    )

    summary = panel.groupby("quote_date").agg(
        entries=("entry_id", "size"),
        both_deltas_ready=("both_deltas_ready", "sum"),
        matched_comparison_available=(
            "matched_comparison_available", "sum"
        ),
    ).reset_index()

    requested_dates = (
        sorted(set(args.dates))
        if args.dates
        else selector.checked_timeline(timeline).loc[
            lambda frame: frame.reference_session,
            "quote_date",
        ].tolist()
    )

    summary = summary.set_index(
        "quote_date"
    ).reindex(
        requested_dates, fill_value=0
    ).reset_index()

    summary.to_csv(
        output / "daily_summary.csv", index=False
    )

    modules = [
        pilot_hedge_panel,
        daily_ah_validation,
        ah_backward_pricer,
        ah_local_vol,
        andreasen_huge,
        backward_pricer,
    ]
    sources = [Path(__file__).resolve()] + [
        Path(module.__file__).resolve()
        for module in modules
    ]

    if any(
        digest(path) != expected
        for path, expected in input_hashes.items()
    ):
        raise ValueError(
            "Inputs changed during the panel run"
        )

    audit = {
        "status": "completed",
        "settings": asdict(settings),
        "entries": len(entries),
        "entry_dates": sorted(
            entries.quote_date.unique().tolist()
        ),
        "requested_entry_dates": requested_dates,
        "selection_status_counts": (
            buckets.status.value_counts().to_dict()
        ),
        "carry_case": carry["case"].iloc[0],
        "assumed_annual_rate": float(
            carry.annual_rate.iloc[0]
        ),
        "endpoint_status_counts": (
            panel.end_status.value_counts().to_dict()
        ),
        "black_status_counts": (
            panel.black_status.value_counts().to_dict()
        ),
        "ah_status_counts": (
            panel.ah_status.value_counts().to_dict()
        ),
        "matched_comparison_available": int(
            panel.matched_comparison_available.sum()
        ),
        "selection": (
            "Current-date positive-bid entry candidates; "
            "nearest calendar-day expiry then nearest "
            "log-spot strike; ties use earlier expiry "
            "and lower strike."
        ),
        "duplicates": (
            "Repeated basket buckets map to one "
            "date/root/expiry/strike/kind entry."
        ),
        "holding_period": (
            "Next scheduled reference session in the "
            "supplied pilot timeline; gaps are not bridged."
        ),
        "endpoint_marks": (
            "Original same-contract bid/ask midpoints; "
            "zero-bid bounds may supply exits, never entries."
        ),
        "black_convention": (
            "IV from the original option midpoint; spot "
            "delta holds strike, IV and forward/spot "
            "ratio fixed."
        ),
        "AH_convention": (
            "Current-day physical local variance and "
            "deterministic carry fixed; smoothing "
            "changes the diffusion."
        ),
        "put_greeks": (
            "Call backward solution minus exact "
            "deterministic-carry parity terms."
        ),
        "marks_replaced_by_model_prices": False,
        "entries_filtered_by_future_availability": False,
        "entry_calculations_use_future_quotes": False,
        "funding_curve_verified": False,
        "contract_identity_verified": False,
        "snapshot_provenance_verified": False,
        "zero_time_coefficient_requested": False,
        "models_refitted": False,
        "values_clipped": False,
        "hedge_backtest_performed": False,
        "all_entry_greeks_independently_validated": False,
        "input_sha256": input_hashes,
        "source_sha256": {
            str(path): digest(path) for path in sources
        },
        "output_sha256": {
            path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob("*.csv"))
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }

    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    print(
        "\nDaily panel:\n"
        + summary.to_string(index=False)
    )
    print(
        "\nEndpoint statuses:\n"
        + panel.end_status.value_counts().to_string()
    )
    print(
        "\nBlack statuses:\n"
        + panel.black_status.value_counts().to_string()
    )
    print(
        "\nAH statuses:\n"
        + panel.ah_status.value_counts().to_string()
    )
    print(f"\nPanel: {output.resolve()}")
    print(
        "Entries and failures retained. "
        "Hedge-ledger strategy comparison is next."
    )


if __name__ == "__main__":
    main()