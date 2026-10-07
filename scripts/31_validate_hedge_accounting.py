"""Controlled accounting checks, followed by a synthetic Black hedge path."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.special import ndtr

import hedge_ledger
from hedge_ledger import HedgeLedgerSettings, SelfFinancingHedgeLedger


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manual_path():
    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2023-09-01", periods=3, tz="UTC"
        ),
        "spot": [100.0, 110.0, 105.0],
        "option_mark": [5.0, 8.0, 6.0],
        "delta": [0.5, 0.25, 0.9],
    })


def synthetic_black_path(seed):
    """Fixed strike; synthetic calendar days, not exchange sessions."""
    rate, sigma, strike = 0.05, 0.2, 100.0
    rng = np.random.default_rng(seed)

    shocks = (
        (rate - 0.5 * sigma**2) / 365
        + sigma / np.sqrt(365) * rng.standard_normal(30)
    )
    spots = 100 * np.exp(np.r_[0.0, np.cumsum(shocks)])
    remaining = (30 - np.arange(31)) / 365
    marks, deltas = [], []

    for spot, maturity in zip(spots, remaining):
        if maturity == 0:
            marks.append(max(spot - strike, 0.0))
            deltas.append(float(spot > strike))
        else:
            root_variance = sigma * np.sqrt(maturity)
            d1 = (
                np.log(spot / strike)
                + (rate + 0.5 * sigma**2) * maturity
            ) / root_variance

            marks.append(
                spot * ndtr(d1)
                - strike
                * np.exp(-rate * maturity)
                * ndtr(d1 - root_variance)
            )
            deltas.append(ndtr(d1))

    return pd.DataFrame({
        "timestamp": pd.date_range(
            "2023-09-01", periods=31, tz="UTC"
        ),
        "spot": spots,
        "option_mark": marks,
        "delta": deltas,
        "strike": strike,
    })


def controls(seed):
    zero_rates = HedgeLedgerSettings(
        lend_rate=0, borrow_rate=0
    )

    linear = manual_path()
    linear["option_mark"] = linear["spot"]
    linear["delta"] = 1.0

    cash_claim = manual_path()
    cash_claim["option_mark"] = (
        10 * np.exp(0.05 * np.arange(3) / 365)
    )
    cash_claim["delta"] = 0.0

    days = np.array([0, 7, 30])
    deterministic = pd.DataFrame({
        "timestamp": (
            pd.Timestamp("2023-09-01", tz="UTC")
            + pd.to_timedelta(days, unit="D")
        ),
        "spot": 100 * np.exp(0.05 * days / 365),
        "option_mark": (
            100 * np.exp(0.05 * days / 365)
            - 90 * np.exp(-0.05 * (30 - days) / 365)
        ),
        "delta": 1.0,
        "strike": 90.0,
    })

    quoted = manual_path()
    quoted["delta"] = 0.5
    quoted["option_bid"] = [4.8, 7.8, 5.9]
    quoted["option_ask"] = [5.2, 8.2, 6.1]
    quoted["hedge_bid"] = quoted["spot"] - 0.1
    quoted["hedge_ask"] = quoted["spot"] + 0.1

    black = synthetic_black_path(seed)

    return {
        "linear_claim": (
            linear, HedgeLedgerSettings(), 0.0
        ),
        "cash_claim": (
            cash_claim, HedgeLedgerSettings(), 0.0
        ),
        "deterministic_itm": (
            deterministic, HedgeLedgerSettings(), 0.0
        ),
        "manual_rebalance": (
            manual_path(), zero_rates, 2.75
        ),
        "manual_fees": (
            manual_path(),
            HedgeLedgerSettings(
                lend_rate=0,
                borrow_rate=0,
                hedge_fee_bps=10,
                hedge_fee_per_trade=0.01,
                option_fee_per_contract=0.01,
            ),
            2.59625,
        ),
        "bid_ask": (
            quoted,
            HedgeLedgerSettings(
                lend_rate=0,
                borrow_rate=0,
                execution="bid_ask",
            ),
            1.1,
        ),
        "black_no_costs": (
            black, HedgeLedgerSettings(), None
        ),
        "black_with_costs": (
            black,
            HedgeLedgerSettings(
                hedge_fee_bps=2,
                option_fee_per_contract=0.01,
            ),
            None,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/hedging_accounting_diagnostics/"
            "self_financing_controls"
        ),
    )
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()

    run_id = datetime.now(timezone.utc).strftime(
        "run_%Y%m%dT%H%M%S_%fZ"
    )
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)

    cases = controls(args.seed)
    results, summaries = {}, []

    for name, (data, settings, expected) in cases.items():
        data = data.assign(contract_id=name)
        result = SelfFinancingHedgeLedger(settings).run(data)
        results[name] = result

        if expected is not None:
            if abs(result.summary["net_pnl"] - expected) > 1e-9:
                raise AssertionError(
                    f"Analytical accounting control failed: {name}"
                )

        result.ledger.to_csv(
            output / f"{name}_ledger.csv", index=False
        )
        summaries.append({
            "case": name,
            **result.summary,
            "expected_net_pnl": expected,
        })

    black = cases["black_no_costs"][0]

    # Independent identity: equal funding rates and identical hedge targets.
    held = np.r_[black["delta"].to_numpy()[:-1], 0.0]
    trades = np.diff(np.r_[0.0, held])

    fees = (
        np.abs(trades)
        * black["spot"].to_numpy()
        * 2
        / 10000
    )
    fees[[0, -1]] += 0.01

    future_years = (30 - np.arange(31)) / 365
    expected_cost_effect = -float(
        np.sum(fees * np.exp(0.05 * future_years))
    )
    actual_cost_effect = (
        results["black_with_costs"].summary["net_pnl"]
        - results["black_no_costs"].summary["net_pnl"]
    )
    cost_residual = actual_cost_effect - expected_cost_effect

    if abs(cost_residual) > 1e-9:
        raise AssertionError("Cost-funding identity failed")

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "summary.csv", index=False)

    sources = (
        Path(__file__).resolve(),
        Path(hedge_ledger.__file__).resolve(),
    )

    audit = {
        "status": "completed",
        "seed": args.seed,
        "cases": {
            name: asdict(values[1])
            for name, values in cases.items()
        },
        "time_basis": (
            "Timezone-aware elapsed seconds, ACT/365F."
        ),
        "cash_flows": (
            "Accrue opening cash, pay endpoint dividends, "
            "then rebalance."
        ),
        "entry_exit": (
            "Option opened at the first supplied mark/quote "
            "and liquidated at the last."
        ),
        "gross_pnl": (
            "Before direct execution costs on the actual "
            "funded cash path."
        ),
        "synthetic_schedule": (
            "30 calendar-day intervals; "
            "not an exchange-session schedule."
        ),
        "black_control": (
            "Analytical fixed-strike prices and deltas; "
            "discrete hedging P&L need not be zero."
        ),
        "cost_funding_expected_effect": expected_cost_effect,
        "cost_funding_actual_effect": actual_cost_effect,
        "cost_funding_residual": cost_residual,
        "max_reconciliation": float(
            summary["max_reconciliation"].max()
        ),
        "historical_backtest_performed": False,
        "contract_identity_verified": False,
        "funding_curve_verified": False,
        "hedge_instrument": (
            "Synthetic index position; "
            "not an executable SPX position."
        ),
        "dividend_income_inferred": False,
        "models_modified": False,
        "source_sha256": {
            str(path): sha256(path) for path in sources
        },
        "output_sha256": {
            path.name: sha256(path)
            for path in sorted(output.glob("*.csv"))
        },
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }

    (output / "audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )

    columns = [
        "case", "net_pnl", "expected_net_pnl",
        "direct_costs", "total_interest",
        "max_reconciliation",
    ]
    print(summary[columns].to_string(index=False))
    print(
        f"\nCost-funding identity residual: {cost_residual:.3g}"
    )
    print(f"Diagnostics: {output.resolve()}")
    print(
        "Controlled accounting checks passed. "
        "Historical marks and daily deltas are the next stage."
    )


if __name__ == "__main__":
    main()