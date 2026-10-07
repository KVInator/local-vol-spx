"""Two backward checks against the cached study-22 wider-domain baseline."""

import argparse
import importlib.util
import inspect
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy


KEYS = ["days", "log_strike", "spot_log_change"]
FIELDS = ["price", "delta", "gamma"]
FLAGS = [
    "negative_gamma",
    "delta_bound_violation",
    "price_bound_violation",
]


def load_study():
    path = Path(__file__).with_name("22_validate_smoothed_ah.py")
    spec = importlib.util.spec_from_file_location(
        "smoothed_ah_study22", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path


def check_sources(study, path, audit):
    sources = [path] + [
        Path(inspect.getfile(getattr(study, name)))
        for name in (
            "AHBackwardPricer",
            "AHLocalVariance",
            "AndreasenHugeSurface",
            "BackwardPriceGrid",
            "ForwardPDESolver",
        )
    ]
    recorded = {
        Path(p).name: h
        for p, h in audit["source_sha256"].items()
    }
    for source in sources:
        if study.digest(source) != recorded.get(source.name):
            raise ValueError(
                "Source changed since the cached baseline: "
                f"{source.name}"
            )
    return sources


def select_baseline(frame, days, strikes_y, spot, windows):
    parts = []
    for day in days:
        for y in strikes_y:
            part = frame[
                np.isclose(frame.days, day, rtol=0, atol=1e-10)
                & np.isclose(frame.log_strike, y, rtol=0, atol=1e-12)
            ].copy()
            if len(part) != 1801 or part.duplicated(KEYS).any():
                raise ValueError(
                    "Missing/duplicate cached samples: "
                    f"{day:g} days, {y:+g}"
                )
            part = part.sort_values("spot_log_change")
            expected = np.linspace(
                -max(windows), max(windows), 1801
            )
            if (
                not np.all(
                    np.isfinite(part[KEYS + ["spot"] + FIELDS])
                )
                or not np.allclose(
                    part.spot_log_change,
                    expected,
                    rtol=0,
                    atol=1e-12,
                )
                or not np.allclose(
                    part.spot,
                    spot * np.exp(expected),
                    rtol=0,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    "Cached physical spots or curve values "
                    "are inconsistent."
                )
            parts.append(part)
    return pd.concat(parts, ignore_index=True)


def compare(check, candidate, baseline, windows):
    merged = candidate.merge(
        baseline,
        on=KEYS,
        suffixes=("_new", "_old"),
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if (
        not merged._merge.eq("both").all()
        or not np.array_equal(merged.spot_new, merged.spot_old)
    ):
        raise ValueError(
            "Comparison requires identical contract keys "
            "and physical spots."
        )

    rows, original = [], []
    for (day, y), frame in merged.groupby(
        ["days", "log_strike"], sort=True
    ):
        for window in windows:
            scoped = frame[
                abs(frame.spot_log_change) <= window + 1e-12
            ]
            row = {
                "check": check,
                "days": day,
                "log_strike": y,
                "spot_log_half_width": window,
            }
            for field in FIELDS:
                difference = (
                    scoped[field + "_new"]
                    - scoped[field + "_old"]
                )
                i = difference.abs().idxmax()
                row.update({
                    f"max_{field}_change":
                        float(abs(difference.loc[i])),
                    f"signed_{field}_change":
                        float(difference.loc[i]),
                    f"{field}_change_spot_y":
                        float(scoped.loc[i, "spot_log_change"]),
                    f"{field}_change_spot":
                        float(scoped.loc[i, "spot_new"]),
                })
            for side in ("old", "new"):
                peak = scoped.loc[
                    scoped["gamma_" + side].idxmax()
                ]
                row[f"{side}_gamma_peak"] = float(
                    peak["gamma_" + side]
                )
                row[f"{side}_gamma_peak_spot"] = float(
                    peak["spot_" + side]
                )
                for flag in FLAGS:
                    row[f"{side}_{flag}_samples"] = int(
                        scoped[flag + "_" + side].sum()
                    )
            row["gamma_peak_spot_shift"] = (
                row["new_gamma_peak_spot"]
                - row["old_gamma_peak_spot"]
            )
            rows.append(row)

        at_zero = frame[
            abs(frame.spot_log_change) <= 1e-14
        ]
        if len(at_zero) != 1:
            raise ValueError(
                "Need exactly one original-spot sample "
                "per contract."
            )
        point = at_zero.iloc[0]
        row = {
            "check": check,
            "days": day,
            "log_strike": y,
        }
        for field in FIELDS:
            row[f"old_{field}"] = float(point[field + "_old"])
            row[f"new_{field}"] = float(point[field + "_new"])
            row[field + "_change"] = (
                row[f"new_{field}"] - row[f"old_{field}"]
            )
        row["gamma_change_pct"] = (
            100 * row["gamma_change"] / row["old_gamma"]
            if abs(row["old_gamma"]) > 1e-12
            else None
        )
        original.append(row)

    return pd.DataFrame(rows), pd.DataFrame(original)


def summarize(comparisons):
    rows = []
    for (check, window), frame in comparisons.groupby(
        ["check", "spot_log_half_width"], sort=False
    ):
        row = {
            "check": check,
            "spot_log_half_width": window,
        }
        for field in FIELDS:
            worst = frame.loc[
                frame[f"max_{field}_change"].idxmax()
            ]
            row[f"max_{field}_change"] = float(
                worst[f"max_{field}_change"]
            )
            row[f"{field}_days"] = float(worst.days)
            row[f"{field}_log_strike"] = float(worst.log_strike)
            row[f"{field}_spot_y"] = float(
                worst[f"{field}_change_spot_y"]
            )
        for flag in FLAGS:
            row[flag + "_samples"] = int(
                frame[f"new_{flag}_samples"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument(
        "--native-intervals", type=int, default=32000
    )
    parser.add_argument("--width", type=float, default=0.90)
    parser.add_argument(
        "--days", type=float, nargs="+", default=[4, 35, 60]
    )
    parser.add_argument(
        "--log-strikes",
        type=float,
        nargs="+",
        default=[-0.02, 0, 0.02],
    )
    parser.add_argument(
        "--run-name", default="ah_grid_domain_check"
    )
    args = parser.parse_args()

    study, study_path = load_study()
    audit_path = args.study / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("status") != "completed":
        raise ValueError("The cached study must be completed.")

    sources = check_sources(study, study_path, audit)
    inputs = [Path(p) for p in audit["input_sha256"]]
    for path, expected in audit["input_sha256"].items():
        if study.digest(path) != expected:
            raise ValueError(
                f"Cached study input changed: {path}"
            )

    model_paths = [p for p in inputs if p.suffix == ".json"]
    if len(model_paths) != 1:
        raise ValueError(
            "Expected one original AH input model "
            "in the study audit."
        )
    original = study.AndreasenHugeSurface.load(model_paths[0])

    quote_paths = [p for p in inputs if p.name == "quotes.csv"]
    if (
        not quote_paths
        or len({
            p.parent.parent.resolve() for p in quote_paths
        }) != 1
    ):
        raise ValueError(
            "Expected retained quotes under one common root."
        )
    quotes = study.load_quotes(
        quote_paths[0].parent.parent, original
    )
    if (
        {p.resolve() for p, _ in quotes}
        != {p.resolve() for p in quote_paths}
    ):
        raise ValueError(
            "Quote files differ from the completed "
            "baseline inputs."
        )

    settings = [
        c for c in audit["cases"]
        if c["case"] == "domain_wide"
    ]
    if len(settings) != 1:
        raise ValueError(
            "Expected one cached domain_wide case."
        )
    baseline_settings = settings[0]
    radius = float(audit["radius_in_log_state"])
    windows = sorted(audit["spot_log_windows"])
    days = sorted(set(args.days))
    strikes_y = sorted(set(args.log_strikes))
    base_width = float(baseline_settings["width"])
    spacing = (
        2 * base_width / baseline_settings["pde_intervals"]
    )
    count = (
        int(round(2 * args.width / spacing))
        if np.isfinite(args.width)
        else 0
    )
    if (
        not np.all(np.isfinite([
            args.width, radius, *days, *strikes_y, *windows
        ]))
        or radius <= 0
        or not windows
        or min(windows) <= 0
        or not days
        or min(days) <= 0
        or not strikes_y
        or args.native_intervals
            <= baseline_settings["native_intervals"]
        or args.native_intervals % 2
        or args.width <= base_width
        or count < 4
        or count % 2
        or not np.isclose(
            2 * args.width / count,
            spacing,
            rtol=0,
            atol=1e-14,
        )
        or max(windows) >= base_width
        or max(abs(np.array(strikes_y))) >= base_width
        or baseline_settings["state_shift"] != 0
        or baseline_settings["wing_vol_factor"] != 1
        or args.run_name in ("", ".", "..")
        or Path(args.run_name).name != args.run_name
    ):
        raise ValueError(
            "Invalid refinement, domain, "
            "or cached-baseline settings."
        )

    times = [original._time(day / 365) for day in days]
    days = [365 * time for time in times]
    curve_path = args.study / "curves_domain_wide.csv"
    contract_path = args.study / "contracts.csv"
    cached = select_baseline(
        pd.read_csv(curve_path),
        days,
        strikes_y,
        original.spot,
        windows,
    )
    contracts_old = pd.read_csv(contract_path).query(
        "case == 'domain_wide'"
    )
    for (day, y), frame in cached.groupby(
        ["days", "log_strike"]
    ):
        contract = contracts_old[
            np.isclose(
                contracts_old.days, day, rtol=0, atol=1e-10
            )
            & np.isclose(
                contracts_old.log_strike,
                y,
                rtol=0,
                atol=1e-12,
            )
        ]
        point = frame.loc[
            abs(frame.spot_log_change).idxmin()
        ]
        if len(contract) != 1 or not np.allclose(
            point[FIELDS].to_numpy(float),
            contract[FIELDS].iloc[0].to_numpy(float),
            rtol=1e-10,
            atol=1e-10,
        ):
            raise ValueError(
                "Cached curves do not match the "
                "baseline contract results."
            )

    model = study.AndreasenHugeSurface(
        study.AHGrid(
            original.grid.width, args.native_intervals
        ),
        original.spot,
        original.maturities,
        original.forwards,
        original.discounts,
        original.controls,
        original.parameters,
    )
    bounds = np.array([
        [
            np.log(data[:, 0] / f).min(),
            np.log(data[:, 0] / f).max(),
        ]
        for (_, data), f in zip(quotes, original.forwards)
    ])
    variance = study.StudyVariance(model, radius, bounds)
    if args.width + 4 * radius >= variance.max_log_moneyness:
        raise ValueError(
            "The wider PDE domain needs Gaussian-kernel "
            "room inside the AH domain."
        )

    cases = [
        (
            "native_refinement",
            base_width,
            baseline_settings["pde_intervals"],
        ),
        ("domain_extension", args.width, count),
    ]
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
    }
    output = (
        Path("outputs/backward_pde_diagnostics")
        / args.run_name
    )
    if output.exists():
        raise FileExistsError(
            "Use a fresh --run-name to preserve "
            "previous results."
        )
    output.mkdir(parents=True)

    provenance = [
        audit_path, curve_path, contract_path, *inputs
    ]
    report = {
        "input_sha256": {
            str(p): study.digest(p) for p in provenance
        },
        "source_sha256": {
            str(p): study.digest(p)
            for p in [Path(__file__), *sources]
        },
        "versions": versions,
        "baseline_versions": audit["versions"],
        "versions_match_baseline":
            versions == audit["versions"],
        "cached_baseline": baseline_settings,
        "radius_in_log_state": radius,
        "native_intervals": args.native_intervals,
        "log_spacing": spacing,
        "steps_per_day": baseline_settings["steps_per_day"],
        "early_time_power": 3,
        "rannacher_steps": 2,
        "spot_log_windows": windows,
        "generated_days": days,
        "generated_forward_log_strikes": strikes_y,
        "cases": [
            {
                "case": name,
                "width": w,
                "pde_intervals": n,
            }
            for name, w, n in cases
        ],
        "coefficient": (
            "Study-22 Gaussian variance average; unchanged "
            "physical radius and proxy parameters."
        ),
        "boundaries": (
            "Martingale payoff asymptotes "
            "on each finite domain."
        ),
        "greek_convention": (
            "Physical coefficient and original carry "
            "fixed separately for each case."
        ),
        "comparison_scope": (
            "Backward prices and Greeks at identical "
            "cached physical spots."
        ),
        "price_minus_original_AH_scope": (
            "Includes the diffusion change "
            "and discretization."
        ),
        "reference_is_exact_greek_truth": False,
        "parameters_refitted": False,
        "input_models_modified": False,
        "prices_clipped": False,
        "volatility_cap_added": False,
        "global_wing_robustness_certified": False,
        "new_forward_validation_performed": False,
        "wing_stress_repeated": False,
        "candidate_promoted": False,
        "status": "running",
    }

    def save_audit():
        (output / "audit.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    save_audit()
    print(
        f"Radius {radius:g}; native grid "
        f"{args.native_intervals:,}; "
        f"PDE spacing {spacing:g}.",
        flush=True,
    )
    print(
        "Two new backward cases; cached wider-domain "
        "baseline; no optimizer.",
        flush=True,
    )
    if not report["versions_match_baseline"]:
        print(
            "Dependency versions differ from the baseline; "
            "both versions are recorded.",
            flush=True,
        )

    curves = {"cached_domain_wide": cached}
    contracts = []
    try:
        for name, width, intervals in cases:
            solver = study.AHBackwardPricer(
                variance,
                domain_width=width,
                space_intervals=intervals,
                steps_per_day=
                    baseline_settings["steps_per_day"],
                early_time_power=3,
                state_shift=0,
                rannacher_steps=2,
            )
            parts = []
            for day, time in zip(days, times):
                subset = cached[
                    np.isclose(
                        cached.days, day, rtol=0, atol=1e-10
                    )
                ]
                labels = sorted(
                    subset.log_strike.unique()
                )
                strikes = (
                    model.forward(time) * np.exp(labels)
                )
                print(
                    f"{name}: width {width:g}, "
                    f"N={intervals:,}, {day:g} days, "
                    f"{len(labels)} calls...",
                    flush=True,
                )
                results = solver.solve_many(strikes, time)
                diagnostics = solver.last_diagnostics
                if (
                    diagnostics[
                        "zero_time_coefficient_requested"
                    ]
                    or diagnostics[
                        "minimum_coefficient_time_years"
                    ] <= 0
                ):
                    raise ArithmeticError(
                        "A coefficient was requested "
                        "at a nonpositive time."
                    )

                for y, strike, result in zip(
                    labels, strikes, results
                ):
                    frame = subset[
                        subset.log_strike == y
                    ][KEYS + ["spot"]].copy()
                    values = result.greeks(
                        frame.spot.to_numpy()
                    )
                    if not np.all(np.isfinite([
                        values.price,
                        values.delta,
                        values.gamma,
                    ])):
                        raise ArithmeticError(
                            "Nonfinite backward prices "
                            "or Greeks."
                        )
                    for field in FIELDS:
                        frame[field] = getattr(values, field)

                    scale = (
                        model.forward(time)
                        * model.discount_factor(time)
                    )
                    lower = scale * np.maximum(
                        frame.spot / original.spot
                        - strike / model.forward(time),
                        0,
                    )
                    frame[FLAGS[0]] = frame.gamma < -1e-8
                    frame[FLAGS[1]] = (
                        (frame.delta < -1e-6)
                        | (
                            frame.delta
                            > scale / original.spot + 1e-6
                        )
                    )
                    frame[FLAGS[2]] = (
                        (frame.price < lower - 1e-6)
                        | (
                            frame.price
                            > scale * frame.spot
                            / original.spot + 1e-6
                        )
                    )
                    parts.append(frame)

                    at_spot = result.greeks(original.spot)
                    node_spot, node_gamma = study.node_gamma(
                        result, original.spot
                    )
                    contracts.append({
                        "case": name,
                        "days": day,
                        "log_strike": y,
                        **{
                            field: float(
                                getattr(at_spot, field)
                            )
                            for field in FIELDS
                        },
                        "node_spot": node_spot,
                        "node_gamma": node_gamma,
                        "price_minus_original_AH": (
                            float(at_spot.price)
                            - float(
                                original.call_price(
                                    strike, time
                                )
                            )
                        ),
                        **solver.last_diagnostics,
                    })

            curves[name] = pd.concat(
                parts, ignore_index=True
            )
            curves[name].to_csv(
                output / f"curves_{name}.csv",
                index=False,
            )
            pd.DataFrame(contracts).to_csv(
                output / "contracts.csv", index=False
            )

        checks = [
            (
                "native_grid",
                "native_refinement",
                "cached_domain_wide",
            ),
            (
                "PDE_domain",
                "domain_extension",
                "native_refinement",
            ),
        ]
        comparisons, at_spot = zip(*[
            compare(
                label, curves[new], curves[old], windows
            )
            for label, new, old in checks
        ])
        comparisons = pd.concat(comparisons)
        at_spot = pd.concat(at_spot)
        summary = summarize(comparisons)

        for name, frame in (
            ("contract_window_sensitivity", comparisons),
            ("original_spot_sensitivity", at_spot),
            ("summary", summary),
        ):
            frame.to_csv(
                output / f"{name}.csv", index=False
            )

        figure, axes = plt.subplots(
            len(days),
            len(strikes_y),
            figsize=(
                5 * len(strikes_y),
                3.5 * len(days),
            ),
            squeeze=False,
        )
        for i, day in enumerate(days):
            for j, y in enumerate(strikes_y):
                for name, frame in curves.items():
                    mask = (
                        np.isclose(
                            frame.days,
                            day,
                            rtol=0,
                            atol=1e-10,
                        )
                        & np.isclose(
                            frame.log_strike,
                            y,
                            rtol=0,
                            atol=1e-12,
                        )
                        & (
                            abs(frame.spot_log_change)
                            <= windows[0] + 1e-12
                        )
                    )
                    axes[i, j].plot(
                        frame.loc[mask, "spot"],
                        frame.loc[mask, "gamma"],
                        label=name,
                    )
                axes[i, j].set(
                    title=(
                        f"{day:g} days; "
                        f"forward log-strike {y:+g}"
                    ),
                    xlabel="Physical spot",
                    ylabel="Gamma per index point",
                )
        axes[0, -1].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(
            output / "gamma_comparison.png", dpi=150
        )
        plt.close(figure)

        report.update(
            status="completed",
            comparisons=[
                {"check": c, "new": n, "old": o}
                for c, n, o in checks
            ],
            zero_time_coefficient_requested=False,
            shape_tolerances={
                "conditional_gamma": 1e-8,
                "conditional_delta": 1e-6,
                "conditional_price_points": 1e-6,
            },
        )
        save_audit()

    except Exception as error:
        report.update(status="failed", error=str(error))
        save_audit()
        raise

    print(
        "\nConditional sensitivity across "
        "the selected calls:"
    )
    columns = (
        ["check", "spot_log_half_width"]
        + [f"max_{f}_change" for f in FIELDS]
        + [f + "_samples" for f in FLAGS]
    )
    print(summary[columns].to_string(index=False))

    print("\nOriginal-spot changes:")
    print(at_spot[[
        "check",
        "days",
        "log_strike",
        "price_change",
        "delta_change",
        "gamma_change",
        "gamma_change_pct",
    ]].to_string(index=False))

    print(f"\nDiagnostics: {output.resolve()}")
    print(
        "Sensitivity estimates only; residual wing "
        "dependence remains documented in study 22."
    )


if __name__ == "__main__":
    main()