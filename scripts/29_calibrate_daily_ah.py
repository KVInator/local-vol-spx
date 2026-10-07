"""Calibrate independent daily research AH models; no hedging or promotion."""

import argparse
import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

import andreasen_huge
import daily_ah
from daily_ah import DailyAHCalibrator, DailyAHSettings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path("data/processed/hedging_pilot/september_2023")
    parser.add_argument(
        "--quotes",
        type=Path,
        default=base / "carry_inputs/calibration_quotes.csv",
    )
    parser.add_argument(
        "--carry",
        type=Path,
        default=base / "carry_inputs/primary_carry.csv",
    )
    parser.add_argument(
        "--output", type=Path, default=base / "daily_ah"
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        help="Optional current-date subset; otherwise all dates.",
    )
    parser.add_argument("--intervals", type=int, default=8000)
    parser.add_argument("--width", type=float, default=1.0)
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument("--control-points", type=int, default=31)
    parser.add_argument("--max-evaluations", type=int, default=300)
    args = parser.parse_args()

    dtype = {
        "quote_date": str,
        "root": str,
        "expire_date": str,
    }
    q = pd.read_csv(args.quotes, dtype=dtype)
    c = pd.read_csv(args.carry, dtype=dtype)

    if (
        q.empty
        or c.empty
        or c["case"].nunique() != 1
        or c["annual_rate"].nunique() != 1
        or set(q["case"]) != set(c["case"])
    ):
        raise ValueError(
            "Need nonempty inputs containing one matching carry scenario."
        )

    dates = sorted(set(c["quote_date"]) | set(q["quote_date"]))
    if args.dates:
        if not set(args.dates).issubset(dates):
            raise ValueError(
                "Requested date is absent from the prepared inputs."
            )
        dates = sorted(set(args.dates))

    q = q.loc[q["quote_date"].isin(dates)]
    c = c.loc[c["quote_date"].isin(dates)]
    keys = sorted(
        set(map(tuple, q[["quote_date", "root"]].values))
        | set(map(tuple, c[["quote_date", "root"]].values))
    )

    settings = DailyAHSettings(
        args.intervals,
        args.width,
        args.smoothing,
        args.control_points,
        max_evaluations=args.max_evaluations,
    )
    runner = DailyAHCalibrator(settings)

    stamp = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    output = args.output / stamp
    output.mkdir(parents=True, exist_ok=False)
    records, tables = [], {}

    print(
        f"{len(keys)} independent daily fits; "
        f"{len(q):,} selected calibration targets.",
        flush=True,
    )
    print(
        "Original targets and carry flags retained; "
        "proxy bounds do not cap Dupire volatility.",
        flush=True,
    )

    for date, root in keys:
        quotes = q.loc[
            q["quote_date"].eq(date) & q["root"].eq(root)
        ]
        carry = c.loc[
            c["quote_date"].eq(date) & c["root"].eq(root)
        ]
        record = {
            "quote_date": date,
            "root": root,
            "case": c["case"].iloc[0],
            "input_quotes": len(quotes),
            "input_expiries": len(carry),
            "status": "failed",
            "model_file": "",
            "message": "",
        }

        def progress(index, time):
            print(
                f"{date} / {root}: "
                f"expiry {index + 1}/{len(carry)}, "
                f"{365 * time:.4g} days...",
                flush=True,
            )

        try:
            model, result = runner.calibrate(
                quotes, carry, progress=progress
            )
        except (
            ValueError,
            RuntimeError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            record["message"] = (
                f"{type(error).__name__}: {error}"
            )
            records.append(record)
            print(
                f"  FAILED: {record['message']}", flush=True
            )
            continue

        root_id = hashlib.sha256(
            root.encode("utf-8")
        ).hexdigest()[:12]
        relative = (
            Path("models") / date / root_id / "ah_surface.json"
        )
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path)

        for name, frame in result.items():
            tables.setdefault(name, []).append(frame)

        flags = (
            result["shape_checks"].iloc[:, 3:]
            .to_numpy(int).sum()
        )
        record.update(
            status="fitted",
            model_file=relative.as_posix(),
            model_sha256=digest(path),
            **runner.fit_metrics(result["quote_residuals"]),
            active_proxy_bounds=int(
                result["expiry_summary"]["active_proxy_bounds"].sum()
            ),
            sampled_shape_violations=int(flags),
        )
        records.append(record)
        print(
            f"  RMS {record['rms_half_spreads']:.6g} half-spreads; "
            f"outside {record['outside_original_bands']}; "
            f"shape flags {flags}.",
            flush=True,
        )

    manifest = pd.DataFrame(records)
    manifest.to_csv(
        output / "model_manifest.csv", index=False
    )
    for name, frames in tables.items():
        pd.concat(frames, ignore_index=True).to_csv(
            output / f"{name}.csv", index=False
        )

    sources = [
        Path(__file__),
        Path(daily_ah.__file__),
        Path(andreasen_huge.__file__),
    ]
    audit = {
        "settings": asdict(settings),
        "carry_case": c["case"].iloc[0],
        "assumed_annual_rate": float(c["annual_rate"].iloc[0]),
        "dates": dates,
        "successful_daily_models": int(
            manifest["status"].eq("fitted").sum()
        ),
        "failed_daily_models": int(
            manifest["status"].eq("failed").sum()
        ),
        "calibration": (
            "Independent same-date fits; "
            "sequential expiries within each date."
        ),
        "objective": (
            "Original equivalent-call midpoint errors divided by "
            "original half-spreads; log-proxy regularization."
        ),
        "flag_policy": (
            "Parity-incompatible and call-bound-disjoint targets "
            "retained without clipping or band expansion."
        ),
        "shape_scope": (
            "Native finite grid at expiry pillars and interval "
            "midpoints; normalized-strike calendar order."
        ),
        "shape_tolerances": {
            "normalized_price": 1e-10,
            "slope": 1e-8,
            "slope_change": 1e-8,
        },
        "conditioning_scope": (
            "Native nodes inside absolute forward log-moneyness "
            "0.10; positive pillar one-sided values."
        ),
        "quote_band_tolerance_points": 1e-6,
        "proxy_volatility_bounds": [
            settings.min_proxy_vol,
            settings.max_proxy_vol,
        ],
        "actual_local_volatility_capped": False,
        "short_end_smoothing_applied": False,
        "future_observations_used": False,
        "cross_date_parameter_warm_start": False,
        "prices_clipped": False,
        "failed_models_exported": False,
        "funding_curve_verified": False,
        "contract_identity_verified": False,
        "snapshot_provenance_verified": False,
        "day_zero_pde_validated": False,
        "backward_greeks_validated": False,
        "candidate_promoted": False,
        "backtest_performed": False,
        "input_sha256": {
            str(p): digest(p)
            for p in [args.quotes, args.carry]
        },
        "source_sha256": {
            str(p): digest(p) for p in sources
        },
        "output_sha256": {
            p.relative_to(output).as_posix(): digest(p)
            for p in sorted(output.rglob("*"))
            if p.is_file()
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = [
        "quote_date",
        "root",
        "status",
        "input_expiries",
        "input_quotes",
        "rms_half_spreads",
        "outside_original_bands",
        "active_proxy_bounds",
        "sampled_shape_violations",
    ]
    print(
        "\nDaily calibration:\n"
        + manifest.reindex(columns=columns).to_string(index=False)
    )
    print(f"\nModels and diagnostics: {output.resolve()}")
    print(
        "Calibration results only. Daily PDE/Greek validation "
        "and hedge accounting remain."
    )
    if audit["failed_daily_models"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()