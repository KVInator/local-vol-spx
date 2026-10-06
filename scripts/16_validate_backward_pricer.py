"""Validate backward vanilla prices and fixed-model spot Greeks."""

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from backward_pricer import BackwardVanillaPricer
from tailed_surface import TailedShortEndSurface


@dataclass(frozen=True)
class BenchmarkSurface:
    min_log_moneyness: float = -1.0
    max_log_moneyness: float = 1.0
    spot: float = 100.0
    sigma: float = 0.2
    rate: float = 0.05
    carry: float = 0.03

    @property
    def maturities(self):
        return np.array([0.0, 0.25, 0.5])

    def forward(self, time):
        return self.spot * np.exp(self.carry * time)

    def discount_factor(self, time):
        return np.exp(-self.rate * time)

    def normalized_variance(self, states, time, side="right"):
        return np.full_like(states, self.sigma**2)


def benchmark_values(surface, spots, strike, time):
    forward = spots * np.exp(surface.carry * time)
    discount = surface.discount_factor(time)
    root_w = surface.sigma * np.sqrt(time)
    d1 = np.log(forward / strike) / root_w + 0.5 * root_w
    d2 = d1 - root_w
    factor = np.exp((surface.carry - surface.rate) * time)
    phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    return (
        discount * (forward * ndtr(d1) - strike * ndtr(d2)),
        factor * ndtr(d1),
        factor * phi / (spots * root_w),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--surface", type=Path)
    parser.add_argument("--space-intervals", nargs="+", type=int, default=[400, 800, 1600])
    parser.add_argument("--steps-per-day", type=int, default=32)
    parser.add_argument("--domain-width", type=float, default=1.0)
    parser.add_argument("--days", nargs="+", type=float, default=[4.0, 35.0, 60.0])
    parser.add_argument("--run-name")
    args = parser.parse_args()
    counts = sorted(set(args.space_intervals))
    if (
        len(counts) < 2 or min(counts) < 4 or any(count % 2 for count in counts)
        or args.steps_per_day <= 0 or not np.isfinite(args.domain_width)
        or args.domain_width <= 0.0
        or not np.all(np.isfinite(args.days)) or min(args.days) <= 0.0
    ):
        raise ValueError("Use increasing positive settings and even spatial counts.")
    market = args.surface is not None
    surface = TailedShortEndSurface.load(args.surface) if market else BenchmarkSurface()
    name = args.run_name or (args.surface.parent.name if market else "black_benchmark")
    if Path(name).name != name or name in (".", ".."):
        raise ValueError("Run name must be a single directory name.")
    output = Path("outputs/backward_pde_diagnostics") / name
    output.mkdir(parents=True, exist_ok=True)
    if market:
        contracts = [(day / 365.0, y) for day in args.days for y in (-0.02, 0.0, 0.02)]
        spots = np.array([surface.spot])
    else:
        contracts = [(0.5, 0.0)]
        spots = np.array([90.0, 100.0, 110.0])
    records, bumps = [], []
    for count in counts:
        pricer = BackwardVanillaPricer(
            surface, -args.domain_width, args.domain_width, count,
            max_time_step=1.0 / (365.0 * args.steps_per_day),
        )
        for time, y in contracts:
            strike = surface.forward(time) * np.exp(y)
            print(f"Solving N={count}, days={365.0*time:g}, forward log-strike={y:+.3f}...", flush=True)
            result = pricer.solve(strike, time)
            value = result.greeks(spots)
            if market:
                target_price = np.asarray([surface.call_price(strike, time)])
                target_delta = target_gamma = np.full(spots.shape, np.nan)
            else:
                target_price, target_delta, target_gamma = benchmark_values(surface, spots, strike, time)
            for index, spot in enumerate(spots):
                records.append({
                    "space_intervals": count, "log_spacing": 2.0 * args.domain_width / count,
                    "days": 365.0 * time, "log_strike": y, "strike": strike, "spot": spot,
                    "price": float(value.price[index]), "delta": float(value.delta[index]),
                    "gamma": float(value.gamma[index]), "target_price": float(target_price[index]),
                    "price_error_points": float(value.price[index] - target_price[index]),
                    "delta_error": float(value.delta[index] - target_delta[index]),
                    "gamma_error": float(value.gamma[index] - target_gamma[index]),
                    "actual_time_steps": result.time_steps,
                })
                if count == counts[-1]:
                    for fraction in (0.005, 0.0025, 0.001):
                        bump = fraction * spot
                        lower, centre, upper = result.price([spot - bump, spot, spot + bump])
                        fd_delta = (upper - lower) / (2.0 * bump)
                        fd_gamma = (upper - 2.0 * centre + lower) / bump**2
                        bumps.append({
                            "days": 365.0*time, "log_strike": y, "spot": spot, "bump_points": bump,
                            "delta": float(value.delta[index]), "fd_delta": float(fd_delta),
                            "delta_difference": float(fd_delta - value.delta[index]),
                            "gamma": float(value.gamma[index]), "fd_gamma": float(fd_gamma),
                            "gamma_difference": float(fd_gamma - value.gamma[index]),
                        })
    table = pd.DataFrame(records)
    finest = table[table["space_intervals"] == counts[-1]]
    keys = ["days", "log_strike", "spot"]
    reference = finest[keys + ["price", "delta", "gamma"]].rename(
        columns={key: f"reference_{key}" for key in ("price", "delta", "gamma")}
    )
    table = table.merge(reference, on=keys, validate="many_to_one")
    for key in ("price", "delta", "gamma"):
        table[f"{key}_change_vs_finest"] = table[key] - table[f"reference_{key}"]
    table.to_csv(output / "price_and_greeks.csv", index=False)
    pd.DataFrame(bumps).to_csv(output / "spot_bump_checks.csv", index=False)
    summary = table.groupby("space_intervals").agg(
        max_price_error_points=("price_error_points", lambda x: x.abs().max()),
        max_delta_error=("delta_error", lambda x: x.abs().max()),
        max_gamma_error=("gamma_error", lambda x: x.abs().max()),
        max_price_change_vs_finest=("price_change_vs_finest", lambda x: x.abs().max()),
        max_delta_change_vs_finest=("delta_change_vs_finest", lambda x: x.abs().max()),
        max_gamma_change_vs_finest=("gamma_change_vs_finest", lambda x: x.abs().max()),
    ).reset_index()
    summary.to_csv(output / "convergence_summary.csv", index=False)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, metric in zip(axes, ("price", "delta", "gamma")):
        column = f"max_{metric}_change_vs_finest" if market else (
            "max_price_error_points" if metric == "price" else f"max_{metric}_error"
        )
        plotted = summary[summary[column] > 0.0]
        axis.loglog(plotted["space_intervals"], plotted[column], "o-")
        axis.set(xlabel="Spatial intervals", ylabel="Absolute difference", title=metric.capitalize())
    figure.savefig(output / "backward_convergence.png", dpi=160)
    plt.close(figure)
    audit = {
        "mode": "constructed SPX candidate" if market else "analytical Black benchmark",
        "domain_width": args.domain_width, "space_intervals": counts,
        "steps_per_day": args.steps_per_day, "contracts": len(contracts),
        "greek_convention": "Original physical local-volatility function and carry held fixed under spot bumps.",
        "boundary_policy": "Martingale payoff asymptotes, not fitted surface prices.",
        "greek_interpolation": "Cubic interpolation of conditional values in log state; no shape certification implied.",
        "spot_bump_scope": "Same solved conditional value curve; independent Black Greek tests are in the unit suite.",
        "market_scope": "Nine generated vanilla contracts, not repricing of every retained quote." if market else None,
        "code_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).resolve().parents[1] / "src/backward_pricer.py")
        },
    }
    if market:
        model_specification = json.loads(args.surface.read_text())
        tails_path = args.surface.parent / model_specification["tail_slices_file"]
        tail_specification = json.loads(tails_path.read_text())
        inputs = [args.surface, tails_path, *[
            tails_path.parent / row["core_file"] for row in tail_specification["slices"]
        ]]
        audit["input_sha256"] = {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs
        }
    (output / "backward_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("\nConvergence summary:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print("\nFinest-grid values:")
    print(finest[["days", "log_strike", "spot", "price", "delta", "gamma", "price_error_points"]].to_string(index=False))
    print(f"\nResults: {output.resolve()}")
    print("Spot-bump checks: spot_bump_checks.csv")


if __name__ == "__main__":
    main()