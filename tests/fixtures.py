import json
from paths import METHODS

"""Small deterministic inputs shared by the hedge and chronology checks."""

import numpy as np
import pandas as pd
from hedging import black_greeks, PastOnlyEmpiricalMV


def empirical_fixture(days=15):
    """Exact declared quadratic response, not historical marks or a BS path."""
    rows = []
    beta = np.array([-0.12, 0.07, -0.035])
    for i, stamp in enumerate(pd.date_range("2023-08-01T20:00:00Z", periods=days)):
        fixing = pd.Timestamp("2023-11-01T20:00:00Z")
        t = (fixing - stamp).total_seconds() / (365 * 86400)
        f, d = (100 * np.exp(0.01 * t), np.exp(-0.05 * t))
        for kind in ("call", "put"):
            for k in (93.0, 100.0, 107.0):
                b = black_greeks(100, k, t, f, d, 0.25, kind)
                row = dict(
                    entry_id=f"{i}-{kind}-{k}",
                    contract_id=f"{kind}-{k}",
                    quote_date=stamp.date().isoformat(),
                    quote_timestamp_utc=stamp,
                    assumed_fixing_utc=fixing,
                    expire_date="2023-11-01",
                    root="UNKNOWN",
                    kind=kind,
                    strike=k,
                    underlying_last=100.0,
                    forward=f,
                    discount_factor=d,
                    assumed_maturity_years=t,
                    black_status="ready",
                    black_iv=0.25,
                    black_delta=b["delta"],
                    ah_status="ready",
                    ah_delta=b["delta"],
                    mid=b["price"],
                    bid=b["price"] - 0.01,
                    ask=b["price"] + 0.01,
                    end_status="matched",
                    end_timestamp=stamp + pd.Timedelta(days=1),
                    end_date=(stamp + pd.Timedelta(days=1)).date().isoformat(),
                    end_spot=100 + (0.4 if i % 2 else -0.3),
                    calendar_days=t * 365,
                    entry_spot_y=np.log(k / 100),
                )
                actual_beta = (
                    beta if kind == "call" else beta * np.array([0.8, 1.2, 0.7])
                )
                delta = b["delta"] + PastOnlyEmpiricalMV.feature(row) @ actual_beta
                row["end_mid"] = row["mid"] + delta * (row["end_spot"] - 100)
                row["end_bid"], row["end_ask"] = (
                    row["end_mid"] - 0.01,
                    row["end_mid"] + 0.01,
                )
                rows.append(row)
    return (pd.DataFrame(rows), beta)


from surface import AHGrid, AndreasenHugeSurface
from pricing import black_time_value


def surface_fixture():
    times = np.array([5 / 365, 12 / 365, 24 / 365])
    forwards = 100 * np.exp(0.01 * times)
    discounts = np.exp(-0.05 * times)
    model = AndreasenHugeSurface(
        AHGrid(1.0, 4000),
        100.0,
        times,
        forwards,
        discounts,
        [np.array([-0.2, 0.2]) for _ in times],
        [np.log(np.array([0.2, 0.2])) for _ in times],
    )
    carry, quotes = ([], [])
    date, timestamp = ("2023-10-02", pd.Timestamp("2023-10-02 20:00:00", tz="UTC"))
    for t, f, d in zip(times, forwards, discounts):
        fixing = timestamp + pd.Timedelta(days=round(t * 365))
        expiry = fixing.strftime("%Y-%m-%d")
        group = dict(
            quote_date=date,
            root="UNKNOWN",
            expire_date=expiry,
            case="rate_5pct",
            forward=f,
            discount_factor=d,
            carry_ready=True,
            quote_timestamp_utc=str(timestamp),
            assumed_fixing_utc=str(fixing),
        )
        carry.append(
            dict(
                **group,
                spot=100.0,
                maturity_years=t,
                annual_rate=0.05,
                parity_bands_incompatible=False,
                fitted_parity_outside_bands=False,
            )
        )
        for y in np.linspace(-0.1, 0.1, 31):
            strike = f * np.exp(y)
            otm = float(d * f * black_time_value(y, 0.04 * t))
            width = min(0.001, otm / 4)
            kind = "put" if y < 0 else "call"
            offset = d * (f - strike) if kind == "put" else 0
            quotes.append(
                dict(
                    **group,
                    strike=strike,
                    underlying_last=100.0,
                    source_kind=kind,
                    source_bid=otm - width,
                    source_ask=otm + width,
                    call_bid=otm - width + offset,
                    call_ask=otm + width + offset,
                    call_mid=otm + offset,
                    call_half_width=width,
                    assumed_maturity_years=t,
                    preferred_side_missing=False,
                    band_disjoint_from_call_bounds=False,
                    midpoint_outside_call_bounds=False,
                )
            )
    return (model, pd.DataFrame(quotes), pd.DataFrame(carry))


from paths import HistoricalSettings
from inference import matched_pairs, stationary_weights


def gain_report(trials, reference_dates, settings=None, common_strategies=False):
    """Pooled SSE ratios and pointwise date-block sensitivity intervals.

    Missing paths are not imputed; inference is conditional on compared paths.
    No model/empirical coefficients are refitted inside a bootstrap replicate.
    """
    settings = settings or HistoricalSettings()
    dates = list(reference_dates)
    if len(set(dates)) != len(dates) or dates != sorted(dates):
        raise ValueError("Reference dates must be unique and chronological.")
    paired = matched_pairs(trials, common_strategies)
    if paired.empty:
        return (pd.DataFrame(), paired)
    if not set(paired.quote_date) <= set(dates):
        raise ValueError("Result dates escape the declared reference calendar.")
    rng = np.random.default_rng(settings.seed)
    weights = {
        block: stationary_weights(len(dates), settings.bootstrap_replicates, block, rng)
        for block in settings.block_lengths
    }
    date_index = {d: i for i, d in enumerate(dates)}
    records = []
    by = ["holding_sessions", "rebalance_sessions", "scenario", "strategy"]
    for values, candidate in paired.groupby(by, sort=True):
        groups = [("overall", "all", candidate)]
        for dimension in (
            "phase",
            "year",
            "kind",
            "maturity_bucket",
            "moneyness_bucket",
            "delta_bucket",
        ):
            groups += [
                (dimension, str(name), group)
                for name, group in candidate.groupby(dimension, sort=True)
            ]
        for dimension, name, group in groups:
            a = group.net_pnl.to_numpy(float)
            b = group.net_pnl_black.to_numpy(float)
            aa, bb = (float(a @ a), float(b @ b))
            ar, br = (
                group.raw_mark_error.to_numpy(float),
                group.raw_mark_error_black.to_numpy(float),
            )
            raw_b = float(br @ br)
            out = dict(zip(by, values))
            out.update(
                dimension=dimension,
                group=name,
                comparisons=len(group),
                entry_dates=group.quote_date.nunique(),
                black_sse=bb,
                strategy_sse=aa,
                gain=1 - aa / bb if bb > 1e-20 else np.nan,
                gain_status="ready" if bb > 1e-20 else "zero_black_sse",
                raw_gain=1 - float(ar @ ar) / raw_b if raw_b > 1e-20 else np.nan,
                mae_improvement=float(np.mean(abs(b) - abs(a))),
                mse_improvement=float(np.mean(b * b - a * a)),
                strategy_rms=float(np.sqrt(aa / len(group))),
                black_rms=float(np.sqrt(bb / len(group))),
                strategy_mean=float(a.mean()),
                black_mean=float(b.mean()),
                fraction_lower_abs_error=float(np.mean(abs(a) < abs(b))),
                sample_scope=(
                    "all_method_intersection"
                    if common_strategies
                    else "candidate_Black_pairs"
                ),
            )
            sums = np.zeros((len(dates), 3))
            for date, daily in group.groupby("quote_date"):
                sums[date_index[date]] = [
                    np.sum(daily.net_pnl**2),
                    np.sum(daily.net_pnl_black**2),
                    len(daily),
                ]
            for block, w in weights.items():
                rec = dict(
                    out,
                    mean_block_sessions=block,
                    bootstrap_replicates=settings.bootstrap_replicates,
                    ci_status="insufficient_dates",
                    gain_ci_low=np.nan,
                    gain_ci_high=np.nan,
                    mae_ci_low=np.nan,
                    mae_ci_high=np.nan,
                    resolved_gain_sign=0,
                )
                if out["entry_dates"] >= max(20, 2 * block) and bb > 1e-20:
                    totals = w @ sums
                    valid = totals[:, 1] > 1e-20
                    gain = 1 - totals[valid, 0] / totals[valid, 1]
                    absolute = np.zeros((len(dates), 2))
                    for date, daily in group.groupby("quote_date"):
                        absolute[date_index[date]] = [
                            np.sum(abs(daily.net_pnl_black) - abs(daily.net_pnl)),
                            len(daily),
                        ]
                    total_abs = w @ absolute
                    good_abs = total_abs[:, 1] > 0
                    mae = total_abs[good_abs, 0] / total_abs[good_abs, 1]
                    rec["valid_bootstrap_replicates"] = int(valid.sum())
                    if valid.sum() >= 0.95 * settings.bootstrap_replicates:
                        lo, hi = np.quantile(gain, [0.025, 0.975])
                        rec.update(
                            ci_status="ready",
                            gain_ci_low=float(lo),
                            gain_ci_high=float(hi),
                            mae_ci_low=float(np.quantile(mae, 0.025)),
                            mae_ci_high=float(np.quantile(mae, 0.975)),
                            resolved_gain_sign=(
                                1 if lo > 1e-10 else -1 if hi < -1e-10 else 0
                            ),
                        )
                    else:
                        rec["ci_status"] = "unstable_denominator"
                records.append(rec)
    return (pd.DataFrame(records), paired)


from paths import HedgePathEvaluator


def run_path(
    meta, common, failure_date, path, settings=None, representative_limit=None
):
    return HedgePathEvaluator(settings).run(
        meta, common, failure_date, path, representative_limit
    )


from pathlib import Path
from dataclasses import asdict
from pricing import BlackFixedIVBenchmark


def study_research_repository(parent, missing_date=None):
    """Small artificial quote history, explicitly not historical SPX evidence."""
    repo = Path(parent)
    (repo / "scripts").mkdir(parents=True, exist_ok=True)
    (repo / "scripts/run_history.py").write_text("# frozen script fixture\n")
    (repo / "data/raw").mkdir(parents=True, exist_ok=True)
    days = pd.bdate_range("2013-01-02", "2013-01-14").strftime("%Y-%m-%d").tolist()
    policy_dates = pd.date_range("2012-12-01", "2013-04-30")
    pd.DataFrame(
        dict(
            quote_date=policy_dates.strftime("%Y-%m-%d"),
            reference_session=policy_dates.weekday < 5,
            early_cash_close=False,
            session_policy=np.where(
                policy_dates.weekday < 5,
                "provisional_standard_session",
                "nonreference_date",
            ),
        )
    ).to_csv(repo / "policy.csv", index=False)
    raw = []
    for i, date in enumerate(days):
        if date == missing_date:
            continue
        stamp = pd.Timestamp(date + " 16:00", tz="America/New_York")
        spot = 100.0 + 0.6 * np.sin(i / 2)
        for expiry in ("2013-01-25", "2013-02-06", "2013-02-12"):
            fixing = pd.Timestamp(expiry + " 16:00", tz="America/New_York")
            t = (fixing - stamp).total_seconds() / (365 * 86400)
            forward, discount = (spot * np.exp(0.01 * t), np.exp(-0.05 * t))
            for strike in np.linspace(90, 110, 41):
                c = BlackFixedIVBenchmark.price(
                    forward, discount, strike, 0.22 * np.sqrt(t), "call"
                )
                p = BlackFixedIVBenchmark.price(
                    forward, discount, strike, 0.22 * np.sqrt(t), "put"
                )
                raw.append(
                    dict(
                        quote_date=date,
                        quote_unixtime=stamp.timestamp(),
                        quote_readtime=stamp.strftime("%Y-%m-%d %H:%M:%S"),
                        expire_date=expiry,
                        underlying_last=spot,
                        strike=strike,
                        c_bid=max(0.0, c - 0.025),
                        c_ask=c + 0.025,
                        p_bid=max(0.0, p - 0.025),
                        p_ask=p + 0.025,
                    )
                )
    pd.DataFrame(raw).to_csv(repo / "data/raw/spx_eod_201301.txt", index=False)
    settings = HistoricalSettings(
        start=days[0],
        end=days[-1],
        schedules=((1, 1), (5, 1), (5, 5)),
        block_lengths=(5, 10),
        bootstrap_replicates=100,
        representative_ledgers=2,
        calibration=dict(
            intervals=600,
            width=1.0,
            smoothing=1.0,
            control_points=9,
            min_proxy_vol=0.005,
            max_proxy_vol=3.0,
            max_evaluations=100,
        ),
        panel=dict(
            target_days=(21, 35, 45),
            target_spot_y=(-0.02, 0.0, 0.02),
            kinds=("call", "put"),
            radius=0.005,
            width=0.75,
            intervals=400,
            steps_per_day=2,
        ),
        empirical=dict(
            window_dates=60,
            minimum_dates=3,
            minimum_rows=6,
            condition_limit=100000000.0,
        ),
    )
    config = dict(
        raw_directory="data/raw",
        session_policy="policy.csv",
        output="outputs/history",
        settings=asdict(settings),
    )
    config_path = repo / "config.json"
    config_path.write_text(json.dumps(config))
    return (repo, config_path, days)


def study_path_fixture(count=12):
    dates = pd.bdate_range("2023-09-01", periods=count).strftime("%Y-%m-%d").tolist()
    timeline = pd.DataFrame(
        dict(
            quote_date=dates,
            reference_session=True,
            session_policy="provisional_standard_session",
        )
    )
    lookup, decisions = ({}, {})
    for i, date in enumerate(dates):
        row = dict(
            quote_date=date,
            root="UNKNOWN",
            strike=100.0,
            kind="call",
            expire_date="2023-11-30",
            assumed_fixing_utc=pd.Timestamp("2023-11-30T21:00:00Z"),
            quote_timestamp_utc=pd.Timestamp(date + "T20:00:00Z"),
            underlying_last=100.0 + i,
            mid=5.0 + 0.5 * i,
            bid=4.9 + 0.5 * i,
            ask=5.1 + 0.5 * i,
        )
        key = (date, "UNKNOWN", "2023-11-30", 100.0, "call")
        lookup[key] = row
        decision = dict(
            black_status="ready",
            black_delta=0.5,
            ah_status="ready",
            ah_delta=0.6,
            smile_status="ready",
            surface_sticky_strike_delta=0.55,
            sticky_delta_status="ready",
            surface_sticky_delta_delta=0.7,
            lv_smile_status="ready",
            lv_smile_delta=0.4,
            empirical_mv_status="ready",
            empirical_mv_delta=0.5,
        )
        decisions[key] = decision
    entry = dict(
        lookup[next(iter(lookup))],
        entry_id="entry",
        contract_id="fixed",
        calendar_days=90,
        entry_spot_y=0.0,
    )
    return (timeline, entry, lookup, decisions)


def study_trial_fixture(days=100):
    dates = pd.bdate_range("2020-01-02", periods=days).strftime("%Y-%m-%d").tolist()
    rows = []
    for i, date in enumerate(dates):
        for kind in ("call", "put"):
            for strategy in METHODS:
                error = (1.0 + 0.2 * np.sin(i / 4)) * (1 if kind == "call" else -1)
                ratio = 1.0 if strategy == "black" else 0.5
                rows.append(
                    dict(
                        entry_id=f"{date}-{kind}",
                        quote_date=date,
                        strategy=strategy,
                        holding_sessions=5,
                        rebalance_sessions=1,
                        scenario="mid",
                        net_pnl=ratio * error,
                        raw_mark_error=ratio * error,
                        black_entry_delta=0.55,
                        calendar_days=35,
                        entry_spot_y=0.0,
                        kind=kind,
                        phase="retrospective_history",
                    )
                )
    return (pd.DataFrame(rows), dates)
