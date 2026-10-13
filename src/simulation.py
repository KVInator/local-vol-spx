"""Controlled replication under Black-Scholes and square-root CEV."""

from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq
from scipy.special import ndtr, ive
from scipy.stats import ncx2, t as student_t
from pde import ForwardPDESolver
from ledger import HedgeLedgerSettings, SelfFinancingHedgeLedger


@dataclass(frozen=True)
class KnownModelSettings:
    paths: int = 20000
    seed: int = 20261008
    spot: float = 100.0
    rate: float = 0.05
    sigma: float = 0.2
    maturity_days: int = 60
    strike_ratios: tuple = (0.95, 1.0, 1.05)
    frequencies: tuple = (4, 8, 16, 32, 64, 128, 256)
    pde_intervals: int = 1600
    pde_half_width: float = 1.0
    pde_steps_per_day: int = 32
    hedge_fee_bps: float = 1.0
    misspecified_iv_factor: float = 1.25
    bootstrap_replicates: int = 200
    representative_paths: int = 3

    def __post_init__(self):
        integers = (
            "paths",
            "seed",
            "maturity_days",
            "pde_intervals",
            "pde_steps_per_day",
            "bootstrap_replicates",
            "representative_paths",
        )
        for name in integers:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer.")
        if self.paths < 8 or self.seed < 0 or self.maturity_days < 1:
            raise ValueError(
                "Require paths >= 8, nonnegative seed and positive maturity."
            )
        if (
            self.pde_intervals < 16
            or self.pde_intervals % 2
            or self.pde_steps_per_day < 1
        ):
            raise ValueError(
                "Require an even PDE interval count >= 16 and positive time resolution."
            )
        if (
            self.bootstrap_replicates < 20
            or not 1 <= self.representative_paths <= self.paths
        ):
            raise ValueError("Invalid bootstrap or representative-path count.")
        numeric = [
            self.spot,
            self.rate,
            self.sigma,
            self.pde_half_width,
            self.hedge_fee_bps,
            self.misspecified_iv_factor,
            *self.strike_ratios,
        ]
        if (
            not np.isfinite(numeric).all()
            or min(self.spot, self.sigma, self.pde_half_width) <= 0
        ):
            raise ValueError(
                "Require finite parameters and positive spot, sigma and domain width."
            )
        if self.hedge_fee_bps < 0 or self.misspecified_iv_factor <= 0:
            raise ValueError("Invalid fee or misspecified-volatility factor.")
        if (
            not self.strike_ratios
            or min(self.strike_ratios) <= 0
            or len(set(self.strike_ratios)) != len(self.strike_ratios)
        ):
            raise ValueError("Strike ratios must be distinct and positive.")
        f = self.frequencies
        if len(f) < 3 or any(
            (
                isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1
                for n in f
            )
        ):
            raise ValueError("Require at least three positive integer frequencies.")
        if tuple(sorted(set(f))) != tuple(f) or any(
            (b % a for a, b in zip(f[:-1], f[1:]))
        ):
            raise ValueError("Frequencies must increase and form nested schedules.")
        if (
            max(abs(np.log(np.asarray(self.strike_ratios)) - self.rate * self.horizon))
            >= self.pde_half_width
        ):
            raise ValueError("Strikes must lie inside the normalized PDE domain.")

    @property
    def horizon(self):
        return self.maturity_days / 365.0

    @classmethod
    def from_json(cls, path):
        values = json.loads(Path(path).read_text())
        for name in ("strike_ratios", "frequencies"):
            if name in values:
                values[name] = tuple(values[name])
        return cls(**values)

    def plan(self):
        return {
            "settings": asdict(self),
            "models": ["black_scholes", "sqrt_cev"],
            "strategies": list(STRATEGIES),
            "scenarios": ["zero_cost", "hedge_fee"],
            "sampling": "IID exact Markov transitions; nested schedules on each model's common paths.",
            "summary_rows": 2
            * len(self.strike_ratios)
            * 2
            * len(STRATEGIES)
            * 2
            * len(self.frequencies),
            "estimated_main_path_array_bytes": self.paths
            * (self.frequencies[-1] + 1)
            * 8,
            "historical_inputs_used": False,
            "historical_models_modified": False,
        }


def _state(spot, strike, tau, kind):
    s = np.asarray(spot, dtype=float)
    if (
        not np.isfinite(s).all()
        or np.any(s < 0)
        or (not np.isfinite([strike, tau]).all())
        or (strike <= 0)
        or (tau < 0)
    ):
        raise ValueError("Invalid nonnegative spot, positive strike or remaining time.")
    if kind not in ("call", "put"):
        raise ValueError("kind must be call or put.")
    return s


@dataclass(frozen=True)
class BlackScholesModel:
    spot: float
    rate: float
    sigma: float
    name: str = "black_scholes"

    def __post_init__(self):
        if (
            not np.isfinite([self.spot, self.rate, self.sigma]).all()
            or min(self.spot, self.sigma) <= 0
        ):
            raise ValueError("Invalid known Black model.")

    def greeks(self, spot, strike, tau, kind="call"):
        s = _state(spot, strike, tau, kind)
        if tau == 0:
            return (
                np.maximum(s - strike if kind == "call" else strike - s, 0),
                np.full_like(s, np.nan),
                np.full_like(s, np.nan),
            )
        std = self.sigma * np.sqrt(tau)
        d1 = (
            np.log(np.maximum(s, np.finfo(float).tiny) / strike)
            + (self.rate + 0.5 * self.sigma**2) * tau
        ) / std
        call = s * ndtr(d1) - strike * np.exp(-self.rate * tau) * ndtr(d1 - std)
        delta = ndtr(d1)
        gamma = np.divide(
            np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi),
            s * std,
            out=np.zeros_like(s),
            where=s > 0,
        )
        if kind == "put":
            call = strike * np.exp(-self.rate * tau) * ndtr(std - d1) - s * ndtr(-d1)
            delta = -ndtr(-d1)
        return (call, delta, gamma)

    def delta_gamma(self, spot, strike, tau):
        _, delta, gamma = self.greeks(spot, strike, tau)
        return (delta, gamma)

    def diffusion_squared(self, spot):
        return self.sigma**2 * np.asarray(spot) ** 2

    def step(self, spot, dt, rng):
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("Transition interval must be positive.")
        return np.asarray(spot) * np.exp(
            (self.rate - 0.5 * self.sigma**2) * dt
            + self.sigma * np.sqrt(dt) * rng.standard_normal(np.shape(spot))
        )

    def increment_moments(self, spot, time, dt):
        """Variance of dM and of (dM)^2, M_t = exp(-rt) S_t."""
        m = np.asarray(spot) * np.exp(-self.rate * time)
        v = self.sigma**2 * dt
        variance = m * m * np.expm1(v)
        squared_variance = m**4 * (
            np.expm1(6 * v) - 4 * np.expm1(3 * v) + 6 * np.expm1(v) - np.expm1(v) ** 2
        )
        return (variance, squared_variance)


@dataclass(frozen=True)
class SquareRootCEVModel:
    spot: float
    rate: float
    sigma: float
    name: str = "sqrt_cev"

    def __post_init__(self):
        if (
            not np.isfinite([self.spot, self.rate, self.sigma]).all()
            or min(self.spot, self.sigma) <= 0
        ):
            raise ValueError("Invalid known CEV model.")

    @property
    def k(self):
        return self.sigma * np.sqrt(self.spot)

    def scale(self, tau):
        if tau <= 0:
            raise ValueError("CEV transition interval must be positive.")
        return (
            self.k**2 * tau / 4
            if self.rate == 0
            else self.k**2 * np.expm1(self.rate * tau) / (4 * self.rate)
        )

    def delta_gamma(self, spot, strike, tau):
        s = _state(spot, strike, tau, "call")
        if tau == 0:
            return (np.full_like(s, np.nan), np.full_like(s, np.nan))
        c = self.scale(tau)
        lam, z = (s * np.exp(self.rate * tau) / c, strike / c)
        delta = ncx2.sf(z, 2, lam)
        root = np.sqrt(lam * z)
        ratio = np.sqrt(np.divide(z, lam, out=np.zeros_like(lam), where=lam > 0))
        difference = (
            np.exp(-0.5 * (np.sqrt(lam) - np.sqrt(z)) ** 2) * ratio * ive(1, root)
        )
        difference = np.where(lam == 0, 0.5 * z * np.exp(-0.5 * z), difference)
        return (delta, np.exp(self.rate * tau) * difference / (2 * c))

    def greeks(self, spot, strike, tau, kind="call"):
        s = _state(spot, strike, tau, kind)
        if tau == 0:
            return (
                np.maximum(s - strike if kind == "call" else strike - s, 0),
                np.full_like(s, np.nan),
                np.full_like(s, np.nan),
            )
        c = self.scale(tau)
        lam, z = (s * np.exp(self.rate * tau) / c, strike / c)
        call = s * ncx2.sf(z, 4, lam) - strike * np.exp(-self.rate * tau) * ncx2.cdf(
            lam, 2, z
        )
        delta, gamma = self.delta_gamma(s, strike, tau)
        if kind == "put":
            call = strike * np.exp(-self.rate * tau) * ncx2.sf(
                lam, 2, z
            ) - s * ncx2.cdf(z, 4, lam)
            delta = delta - 1
        return (call, delta, gamma)

    def diffusion_squared(self, spot):
        return self.k**2 * np.asarray(spot)

    def step(self, spot, dt, rng):
        s = np.asarray(spot, dtype=float)
        if (
            not np.isfinite(s).all()
            or np.any(s < 0)
            or dt <= 0
            or (not np.isfinite(dt))
        ):
            raise ValueError("Invalid CEV transition state or interval.")
        c = self.scale(dt)
        lam = s * np.exp(self.rate * dt) / c
        counts = rng.poisson(lam / 2)
        result = np.zeros_like(s)
        positive = counts > 0
        result[positive] = c * rng.gamma(counts[positive], 2.0)
        return result

    def increment_moments(self, spot, time, dt):
        c = self.scale(dt)
        lam = np.asarray(spot) * np.exp(self.rate * dt) / c
        variance = 4 * lam * c * c
        return (
            variance * np.exp(-2 * self.rate * (time + dt)),
            (192 * lam * c**4 + 2 * variance**2) * np.exp(-4 * self.rate * (time + dt)),
        )


def simulate_paths(model, settings, rng):
    times = np.linspace(0, settings.horizon, settings.frequencies[-1] + 1)
    spots = np.empty((settings.paths, len(times)))
    spots[:, 0] = model.spot
    for i, dt in enumerate(np.diff(times)):
        spots[:, i + 1] = model.step(spots[:, i], float(dt), rng)
    if not np.isfinite(spots).all() or np.any(spots < 0):
        raise ArithmeticError("Invalid exact simulated path.")
    return (times, spots)


class KnownModelPDE:
    """Existing PDE generator, with explicit reverse time and fixed carry."""

    def __init__(
        self, model, strike, settings, *, intervals=None, width=None, steps_per_day=None
    ):
        self.model, self.strike, self.settings = (model, strike, settings)
        width = settings.pde_half_width if width is None else width
        intervals = settings.pde_intervals if intervals is None else intervals
        steps_per_day = (
            settings.pde_steps_per_day if steps_per_day is None else steps_per_day
        )
        self.solver = ForwardPDESolver(
            -width, width, intervals, 1 / (365 * steps_per_day)
        )
        horizon = settings.horizon
        self.times = np.linspace(0, horizon, settings.frequencies[-1] + 1)
        forward = model.spot * np.exp(model.rate * horizon)
        k = strike / forward
        if (
            not self.solver.normalized_strikes[0]
            < k
            < self.solver.normalized_strikes[-1]
        ):
            raise ValueError("Strike outside PDE domain.")
        payoff = np.maximum(self.solver.normalized_strikes - k, 0)

        def variance(states, tau, side):
            s = states * model.spot * np.exp(model.rate * (horizon - tau))
            return model.diffusion_squared(s) / (s * s)

        self.result = self.solver.solve(
            self.times, payoff, variance, lambda tau: tuple(payoff[[0, -1]])
        )
        self.curves = [
            CubicSpline(self.result.log_moneyness, v, extrapolate=False)
            for v in self.result.normalized_calls[::-1]
        ]

    def greeks(self, index, spot, kind="call"):
        s = np.asarray(spot, dtype=float)
        time = self.times[index]
        tau = self.settings.horizon - time
        if tau == 0:
            return (
                np.maximum(s - self.strike if kind == "call" else self.strike - s, 0),
                np.full_like(s, np.nan),
                np.full_like(s, np.nan),
            )
        y = np.log(
            np.maximum(s, np.finfo(float).tiny)
            / (self.model.spot * np.exp(self.model.rate * time))
        )
        curve = self.curves[index]
        scale = self.model.spot * np.exp(self.model.rate * time)
        price = scale * curve(y)
        delta = scale * curve(y, 1) / np.maximum(s, np.finfo(float).tiny)
        gamma = (
            scale
            * (curve(y, 2) - curve(y, 1))
            / np.maximum(s * s, np.finfo(float).tiny)
        )
        price, delta, gamma = [np.where(s == 0, 0.0, v) for v in (price, delta, gamma)]
        if kind == "put":
            price = price - s + self.strike * np.exp(-self.model.rate * tau)
            delta = delta - 1
        return (price, delta, gamma)


STRATEGIES = (
    "analytical_delta",
    "pde_delta",
    "entry_iv_black",
    "misspecified_black",
    "unhedged",
)


def entry_implied_volatility(model, strike, horizon):
    price = float(model.greeks(model.spot, strike, horizon)[0])
    objective = (
        lambda sigma: float(
            BlackScholesModel(model.spot, model.rate, sigma).greeks(
                model.spot, strike, horizon
            )[0]
        )
        - price
    )
    return (brentq(objective, 1e-05, 5.0, xtol=1e-12), price)


def funded_hedge(spots, times, deltas, premium, strike, kind, rate, fee_bps=0.0):
    """Vectorized one-short-option cash ledger plus independent discounted sum.

    deltas contains only preterminal decisions: shape (paths, intervals).
    Missing PDE decisions propagate to NaN; no fallback or retrospective fill.
    All costs are assumed underlying notional fees; option execution is fair mid.
    """
    spots, times, deltas = map(np.asarray, (spots, times, deltas))
    if (
        spots.ndim != 2
        or deltas.shape != (spots.shape[0], spots.shape[1] - 1)
        or times.shape != (spots.shape[1],)
    ):
        raise ValueError("Inconsistent path, time or preterminal-delta shape.")
    if (
        not np.isfinite(spots).all()
        or np.any(spots < 0)
        or (not np.isfinite(times).all())
        or np.any(np.diff(times) <= 0)
        or (times[0] != 0)
    ):
        raise ValueError("Invalid simulated path or time order.")
    if (
        kind not in ("call", "put")
        or not np.isfinite([premium, strike, rate, fee_bps]).all()
        or premium < 0
        or (strike <= 0)
        or (fee_bps < 0)
    ):
        raise ValueError("Invalid claim or fee.")
    position = deltas[:, 0].copy()
    entry_fee = np.abs(position) * spots[:, 0] * fee_bps / 10000
    cash = premium - position * spots[:, 0] - entry_fee
    direct, turnover = (entry_fee.copy(), np.abs(position) * spots[:, 0])
    discounted_fee = entry_fee.copy()
    discounted_gain = np.zeros(len(spots))
    for i, dt in enumerate(np.diff(times)):
        discounted_gain += position * (
            spots[:, i + 1] * np.exp(-rate * times[i + 1])
            - spots[:, i] * np.exp(-rate * times[i])
        )
        cash *= np.exp(rate * dt)
        target = np.zeros(len(spots)) if i == deltas.shape[1] - 1 else deltas[:, i + 1]
        trade = target - position
        notional = np.abs(trade) * spots[:, i + 1]
        fee = notional * fee_bps / 10000
        cash -= trade * spots[:, i + 1] + fee
        direct += fee
        turnover += notional
        discounted_fee += fee * np.exp(-rate * times[i + 1])
        position = target
    payoff = np.maximum(
        spots[:, -1] - strike if kind == "call" else strike - spots[:, -1], 0.0
    )
    net = cash - payoff
    independent = (
        np.exp(rate * times[-1]) * (premium + discounted_gain - discounted_fee) - payoff
    )
    residual = net - independent
    valid = np.isfinite(deltas).all(axis=1)
    if np.any(valid & (np.abs(residual) > 1e-09 + 1e-11 * np.maximum(1, np.abs(net)))):
        raise ArithmeticError("Vectorized funded-account reconciliation failed.")
    return {
        "net_pnl": net,
        "direct_costs": direct,
        "funded_costs": discounted_fee * np.exp(rate * times[-1]),
        "turnover": turnover,
        "reconciliation": residual,
        "valid": valid,
    }


def estimate(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {
            "samples": len(values),
            "mean": np.nan,
            "se": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }
    mean, se = (float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values))))
    half = float(student_t.ppf(0.975, len(values) - 1)) * se
    return dict(
        samples=len(values), mean=mean, se=se, ci_low=mean - half, ci_high=mean + half
    )


def quadratic_error_proxy(model, spots, times, gamma):
    """Centred quadratic martingale proxy, plus its predictable variance.

    Discounted option gamma in the M coordinate is exp(rt)*Gamma.
    The proxy's variance is exact for this proxy, not for full hedge P&L.
    """
    proxy = np.zeros(len(spots))
    qv = np.zeros(len(spots))
    for i, dt in enumerate(np.diff(times)):
        dm = spots[:, i + 1] * np.exp(-model.rate * times[i + 1]) - spots[
            :, i
        ] * np.exp(-model.rate * times[i])
        variance, squared_variance = model.increment_moments(spots[:, i], times[i], dt)
        coefficient = -0.5 * np.exp(model.rate * (times[-1] + times[i])) * gamma[:, i]
        proxy += coefficient * (dm * dm - variance)
        qv += coefficient * coefficient * squared_variance
    return (proxy, qv)


def _prefix(values, prefix):
    return {f"{prefix}_{key}": value for key, value in estimate(values).items()}


def _summary(result):
    pnl = result["net_pnl"]
    valid = result["valid"]
    mean, squared = (estimate(pnl), estimate(pnl * pnl))
    return {
        "paths": len(pnl),
        "ready_paths": int(valid.sum()),
        "failed_paths": int((~valid).sum()),
        "status": "completed" if valid.all() else "partial",
        **_prefix(pnl, "pnl"),
        **_prefix(pnl * pnl, "mse"),
        "rms": np.sqrt(squared["mean"]),
        "rms_ci_low": np.sqrt(max(0.0, squared["ci_low"])),
        "rms_ci_high": np.sqrt(max(0.0, squared["ci_high"])),
        "mean_direct_costs": float(np.nanmean(result["direct_costs"])),
        "mean_funded_costs": float(np.nanmean(result["funded_costs"])),
        "mean_turnover": float(np.nanmean(result["turnover"])),
        "max_reconciliation": (
            float(np.nanmax(np.abs(result["reconciliation"])))
            if valid.any()
            else np.nan
        ),
    }


def analytical_checks(model, strikes, settings):
    rows = []
    for strike in strikes:
        for tau in (
            settings.horizon,
            settings.horizon / 2,
            settings.horizon / settings.frequencies[-1],
        ):
            for spot in np.unique(
                np.r_[settings.spot * np.array([0.9, 1.0, 1.1]), strike]
            ):
                price, delta, gamma = [
                    float(v) for v in model.greeks(spot, strike, tau)
                ]
                put = float(model.greeks(spot, strike, tau, "put")[0])
                h = settings.spot * 0.0001
                plus, minus = (
                    model.greeks(spot + h, strike, tau),
                    model.greeks(spot - h, strike, tau),
                )
                delta_fd = float((plus[0] - minus[0]) / (2 * h))
                gamma_fd = float((plus[1] - minus[1]) / (2 * h))
                ht = tau * 0.0001
                calendar_derivative = float(
                    (
                        model.greeks(spot, strike, tau - ht)[0]
                        - model.greeks(spot, strike, tau + ht)[0]
                    )
                    / (2 * ht)
                )
                residual = (
                    calendar_derivative
                    + model.rate * spot * delta
                    + 0.5 * model.diffusion_squared(spot) * gamma
                    - model.rate * price
                )
                rows.append(
                    dict(
                        model=model.name,
                        strike=strike,
                        spot=spot,
                        remaining_years=tau,
                        price=price,
                        delta=delta,
                        gamma=gamma,
                        parity_residual=price
                        - put
                        - spot
                        + strike * np.exp(-model.rate * tau),
                        delta_minus_fd=delta - delta_fd,
                        gamma_minus_delta_fd=gamma - gamma_fd,
                        analytical_pde_residual=float(residual),
                    )
                )
    return rows


def pde_checks(model, strike, settings, base):
    cases = [("base", base)]
    variants = [
        (
            "space_coarse",
            settings.pde_intervals // 2,
            settings.pde_half_width,
            settings.pde_steps_per_day,
        ),
        (
            "space_fine",
            settings.pde_intervals * 2,
            settings.pde_half_width,
            settings.pde_steps_per_day,
        ),
        (
            "time_fine",
            settings.pde_intervals,
            settings.pde_half_width,
            settings.pde_steps_per_day * 2,
        ),
        (
            "domain_wide",
            settings.pde_intervals * 3 // 2,
            settings.pde_half_width * 1.5,
            settings.pde_steps_per_day,
        ),
    ]
    for name, n, width, steps in variants:
        cases.append(
            (
                name,
                KnownModelPDE(
                    model,
                    strike,
                    settings,
                    intervals=n,
                    width=width,
                    steps_per_day=steps,
                ),
            )
        )
    rows = []
    sample = np.unique(np.r_[settings.spot * np.array([0.9, 1.0, 1.1]), strike])
    for name, grid in cases:
        for index in (0, settings.frequencies[-1] // 2, settings.frequencies[-1] - 1):
            remaining = settings.horizon - grid.times[index]
            exact = model.greeks(sample, strike, remaining)
            numerical, reference = (
                grid.greeks(index, sample),
                base.greeks(index, sample),
            )
            row = dict(
                model=model.name,
                strike=strike,
                case=name,
                calendar_time=grid.times[index],
                remaining_years=remaining,
                time_steps=grid.result.time_steps,
                intervals=grid.solver.n_space_intervals,
                half_width=grid.solver.max_log_moneyness,
                steps_per_day=1 / (365 * grid.solver.max_time_step),
                control_spots=";".join(map(str, sample)),
            )
            for key, num, ana, ref in zip(
                ("price", "delta", "gamma"), numerical, exact, reference
            ):
                row[f"max_{key}_error"] = float(np.max(np.abs(num - ana)))
                row[f"max_{key}_change_vs_base"] = float(np.max(np.abs(num - ref)))
            row["status"] = (
                "finite" if np.isfinite(numerical).all() else "outside_domain"
            )
            rows.append(row)
    return rows


def slope_bootstrap(errors, frequencies, replicates, rng):
    """Resample entire independent paths, preserving frequency dependence."""
    errors = np.asarray(errors, dtype=float)
    mask = np.isfinite(errors).all(axis=1)
    errors = errors[mask]
    if len(errors) < 8:
        return dict(
            status="insufficient_common_paths",
            paths=len(errors),
            slope=np.nan,
            slope_ci_low=np.nan,
            slope_ci_high=np.nan,
        )
    x = np.log(np.asarray(frequencies, dtype=float))
    x = x - x.mean()
    squared = errors**2
    coefficient = lambda mse: (
        float(x @ (0.5 * np.log(mse)) / (x @ x)) if np.all(mse > 0) else np.nan
    )
    slope = coefficient(squared.mean(axis=0))
    boot = np.array(
        [
            coefficient(squared[rng.integers(0, len(errors), len(errors))].mean(axis=0))
            for _ in range(replicates)
        ]
    )
    finite = boot[np.isfinite(boot)]
    low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else (np.nan, np.nan)
    return dict(
        status="estimated" if len(finite) == replicates else "degenerate_replicates",
        paths=len(errors),
        slope=slope,
        slope_ci_low=float(low),
        slope_ci_high=float(high),
    )


def representative_ledger(
    model, strike, kind, spots, times, deltas, premium, rate, fee, expected, path_id
):
    if np.any(spots <= 0) or not np.isfinite(deltas).all():
        return (None, dict(status="absorbed_or_unresolved", path_id=path_id))
    marks = np.array(
        [
            float(model.greeks(s, strike, max(0.0, times[-1] - t), kind)[0])
            for s, t in zip(spots, times)
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2023-01-01", tz="UTC")
            + pd.to_timedelta(times * 365 * 86400, unit="s"),
            "spot": spots,
            "option_mark": marks,
            "delta": np.r_[deltas, 0.0],
            "kind": kind,
            "strike": strike,
            "contract_id": f"{model.name}:{strike}:{kind}",
        }
    )
    result = SelfFinancingHedgeLedger(
        HedgeLedgerSettings(lend_rate=rate, borrow_rate=rate, hedge_fee_bps=fee)
    ).run(data)
    difference = result.summary["net_pnl"] - expected
    if abs(difference) > 1e-08:
        raise ArithmeticError("Existing scalar ledger and batch hedge disagree.")
    return (
        result.ledger,
        dict(
            status="verified",
            path_id=path_id,
            scalar_net_pnl=result.summary["net_pnl"],
            batch_net_pnl=float(expected),
            difference=difference,
            scalar_max_reconciliation=result.summary["max_reconciliation"],
        ),
    )


class KnownModelExperiment:
    """Run the same replication controls under Black-Scholes and CEV."""

    def __init__(self, settings=None):
        self.settings = settings or KnownModelSettings()

    def run(self, output, progress=print):
        """Write reproducible diagnostics; random disagreement is reported, not hidden."""
        settings = self.settings
        output = Path(output)
        (output / "errors").mkdir(parents=True, exist_ok=True)
        (output / "representative_ledgers").mkdir(exist_ok=True)
        streams = np.random.SeedSequence(settings.seed).spawn(3)
        boot_rng = np.random.default_rng(streams[2])
        summaries, price_rows, simulation_rows, numerical_rows, control_rows = (
            [],
            [],
            [],
            [],
            [],
        )
        theory_rows, frequency_rows, slope_rows, ledger_rows, cost_rows, parity_rows = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        numerical_hedge_rows = []
        surface_rows, elapsed_rows = ([], [])
        from time import perf_counter

        models = (
            BlackScholesModel(settings.spot, settings.rate, settings.sigma),
            SquareRootCEVModel(settings.spot, settings.rate, settings.sigma),
        )
        for stream, model in zip(streams[:2], models):
            started = perf_counter()
            progress(
                f"{model.name}: {settings.paths:,} exact paths, {settings.frequencies[-1]} intervals..."
            )
            times, spots = simulate_paths(
                model, settings, np.random.default_rng(stream)
            )
            end = spots[:, -1] * np.exp(-model.rate * times[-1])
            var, _ = model.increment_moments(
                np.full(settings.paths, model.spot), 0.0, settings.horizon
            )
            observed_var = (end - model.spot) ** 2
            simulation_rows.append(
                dict(
                    model=model.name,
                    **_prefix(end - model.spot, "martingale_error"),
                    theoretical_discounted_terminal_variance=float(var[0]),
                    **_prefix(observed_var - var, "variance_error"),
                    absorbed_paths=int((spots[:, -1] == 0).sum()),
                    min_observed_spot=float(spots.min()),
                    max_observed_spot=float(spots.max()),
                )
            )
            pd.DataFrame(
                spots[: settings.representative_paths].T, index=times
            ).rename_axis("time_years").to_csv(
                output / f"{model.name}_sample_paths.csv"
            )
            strikes = settings.spot * np.asarray(settings.strike_ratios)
            control_rows.extend(analytical_checks(model, strikes, settings))
            for strike in strikes:
                contract_started = perf_counter()
                progress(
                    f"  strike {strike:g}: analytical/PDE hedges and numerical controls..."
                )
                iv, premium = entry_implied_volatility(model, strike, settings.horizon)
                grid = KnownModelPDE(model, strike, settings)
                numerical_rows.extend(pde_checks(model, strike, settings, grid))
                n = settings.frequencies[-1]
                true_delta, gamma, pde_delta = [
                    np.empty((settings.paths, n)) for _ in range(3)
                ]
                entry_delta, wrong_delta = [np.empty_like(true_delta) for _ in range(2)]
                iv_model = BlackScholesModel(model.spot, model.rate, iv)
                wrong = BlackScholesModel(
                    model.spot, model.rate, iv * settings.misspecified_iv_factor
                )
                for index, time in enumerate(times[:-1]):
                    tau = settings.horizon - time
                    true_delta[:, index], gamma[:, index] = model.delta_gamma(
                        spots[:, index], strike, tau
                    )
                    pde_delta[:, index] = grid.greeks(index, spots[:, index])[1]
                    entry_delta[:, index] = iv_model.delta_gamma(
                        spots[:, index], strike, tau
                    )[0]
                    wrong_delta[:, index] = wrong.delta_gamma(
                        spots[:, index], strike, tau
                    )[0]
                errors = np.empty(
                    (2, len(settings.frequencies), len(STRATEGIES), 2, settings.paths)
                )
                proxies = np.empty((len(settings.frequencies), settings.paths))
                all_deltas = (true_delta, pde_delta, entry_delta, wrong_delta, None)
                for fi, frequency in enumerate(settings.frequencies):
                    indices = np.arange(0, n + 1, n // frequency)
                    observed, schedule = (spots[:, indices], times[indices])
                    selected_gamma = gamma[:, indices[:-1]]
                    proxy, qv = quadratic_error_proxy(
                        model, observed, schedule, selected_gamma
                    )
                    proxies[fi] = proxy
                    for ki, kind in enumerate(("call", "put")):
                        initial_price = (
                            premium
                            if kind == "call"
                            else premium
                            - model.spot
                            + strike * np.exp(-model.rate * settings.horizon)
                        )
                        terminal_payoff = np.maximum(
                            (
                                spots[:, -1] - strike
                                if kind == "call"
                                else strike - spots[:, -1]
                            ),
                            0.0,
                        )
                        if fi == 0:
                            price_rows.append(
                                dict(
                                    model=model.name,
                                    strike=strike,
                                    kind=kind,
                                    analytical_price=initial_price,
                                    entry_black_iv=iv,
                                    **_prefix(
                                        terminal_payoff
                                        * np.exp(-model.rate * settings.horizon),
                                        "mc_price",
                                    ),
                                )
                            )
                        for si, (strategy, source) in enumerate(
                            zip(STRATEGIES, all_deltas)
                        ):
                            deltas = (
                                np.zeros((settings.paths, frequency))
                                if source is None
                                else source[:, indices[:-1]]
                                - (ki if strategy != "unhedged" else 0)
                            )
                            no_cost = funded_hedge(
                                observed,
                                schedule,
                                deltas,
                                initial_price,
                                strike,
                                kind,
                                model.rate,
                            )
                            with_fee = funded_hedge(
                                observed,
                                schedule,
                                deltas,
                                initial_price,
                                strike,
                                kind,
                                model.rate,
                                settings.hedge_fee_bps,
                            )
                            for ci, (scenario, result) in enumerate(
                                (("zero_cost", no_cost), ("hedge_fee", with_fee))
                            ):
                                errors[ki, fi, si, ci] = result["net_pnl"]
                                summaries.append(
                                    dict(
                                        model=model.name,
                                        strike=strike,
                                        kind=kind,
                                        strategy=strategy,
                                        scenario=scenario,
                                        intervals=frequency,
                                        entry_iv=iv,
                                        **_summary(result),
                                    )
                                )
                            cost_residual = (
                                with_fee["net_pnl"]
                                - no_cost["net_pnl"]
                                + with_fee["funded_costs"]
                            )
                            maximum = (
                                float(np.nanmax(np.abs(cost_residual)))
                                if no_cost["valid"].any()
                                else np.nan
                            )
                            if np.isfinite(maximum) and maximum > 1e-08:
                                raise ArithmeticError("Cost-funding identity failed.")
                            cost_rows.append(
                                dict(
                                    model=model.name,
                                    strike=strike,
                                    kind=kind,
                                    strategy=strategy,
                                    intervals=frequency,
                                    max_cost_funding_residual=maximum,
                                )
                            )
                            if ki == 0 and strategy == "analytical_delta":
                                actual = no_cost["net_pnl"]
                                actual_mse, predictable = (
                                    estimate(actual * actual),
                                    estimate(qv),
                                )
                                theory_rows.append(
                                    dict(
                                        model=model.name,
                                        strike=strike,
                                        intervals=frequency,
                                        **_prefix(actual - proxy, "proxy_residual"),
                                        **_prefix(
                                            proxy * proxy - qv,
                                            "proxy_variance_identity",
                                        ),
                                        actual_mse=actual_mse["mean"],
                                        proxy_predictable_variance=predictable["mean"],
                                        **_prefix(
                                            actual * actual - qv,
                                            "actual_mse_minus_proxy_variance",
                                        ),
                                        actual_mse_to_proxy_variance=actual_mse["mean"]
                                        / predictable["mean"],
                                        proxy_rms=float(
                                            np.sqrt(np.mean(proxy * proxy))
                                        ),
                                        proxy_residual_rms=float(
                                            np.sqrt(np.mean((actual - proxy) ** 2))
                                        ),
                                    )
                                )
                            if frequency == settings.frequencies[
                                min(2, len(settings.frequencies) - 1)
                            ] and strategy in ("analytical_delta", "pde_delta"):
                                for ci, fee in enumerate((0.0, settings.hedge_fee_bps)):
                                    for path_id in range(settings.representative_paths):
                                        ledger, check = representative_ledger(
                                            model,
                                            strike,
                                            kind,
                                            observed[path_id],
                                            schedule,
                                            deltas[path_id],
                                            initial_price,
                                            model.rate,
                                            fee,
                                            errors[ki, fi, si, ci, path_id],
                                            path_id,
                                        )
                                        identity = dict(
                                            model=model.name,
                                            strike=strike,
                                            kind=kind,
                                            strategy=strategy,
                                            scenario=(
                                                "zero_cost" if ci == 0 else "hedge_fee"
                                            ),
                                            intervals=frequency,
                                        )
                                        if ledger is not None:
                                            file = f"{model.name}_{strike:g}_{kind}_{strategy}_{ci}_{path_id}.csv"
                                            ledger.to_csv(
                                                output
                                                / "representative_ledgers"
                                                / file,
                                                index=False,
                                            )
                                            check["ledger_file"] = (
                                                "representative_ledgers/" + file
                                            )
                                        ledger_rows.append({**identity, **check})
                    for si, strategy in enumerate(STRATEGIES[:-1]):
                        diff = errors[0, fi, si, 0] - errors[1, fi, si, 0]
                        parity_rows.append(
                            dict(
                                model=model.name,
                                strike=strike,
                                strategy=strategy,
                                intervals=frequency,
                                max_call_put_pnl_difference=float(
                                    np.nanmax(np.abs(diff))
                                ),
                            )
                        )
                    numerical_gap = errors[0, fi, 1, 0] - errors[0, fi, 0, 0]
                    common = np.isfinite(numerical_gap)
                    numerical_hedge_rows.append(
                        dict(
                            model=model.name,
                            strike=strike,
                            kind="call",
                            intervals=frequency,
                            failed_common_paths=int((~common).sum()),
                            **_prefix(numerical_gap, "pde_minus_analytical"),
                            rms_pde_minus_analytical=(
                                float(np.sqrt(np.mean(numerical_gap[common] ** 2)))
                                if common.any()
                                else np.nan
                            ),
                            maximum_absolute_pde_minus_analytical=(
                                float(np.max(np.abs(numerical_gap[common])))
                                if common.any()
                                else np.nan
                            ),
                        )
                    )
                np.savez_compressed(
                    output / "errors" / f"{model.name}_{strike:g}.npz",
                    net_pnl=errors,
                    quadratic_proxy=proxies,
                    frequencies=np.asarray(settings.frequencies),
                    strategies=np.asarray(STRATEGIES),
                    kinds=np.array(["call", "put"]),
                    scenarios=np.array(["zero_cost", "hedge_fee"]),
                )
                for si, strategy in enumerate(STRATEGIES):
                    for ci, scenario in enumerate(("zero_cost", "hedge_fee")):
                        for fi in range(len(settings.frequencies) - 1):
                            diff = (
                                errors[0, fi, si, ci] ** 2
                                - errors[0, fi + 1, si, ci] ** 2
                            )
                            change = estimate(diff)
                            frequency_rows.append(
                                dict(
                                    model=model.name,
                                    strike=strike,
                                    kind="call",
                                    strategy=strategy,
                                    scenario=scenario,
                                    coarse_intervals=settings.frequencies[fi],
                                    fine_intervals=settings.frequencies[fi + 1],
                                    **change,
                                    improvement_resolved=bool(change["ci_low"] > 0),
                                )
                            )
                    tail = min(4, len(settings.frequencies))
                    slope_rows.append(
                        dict(
                            model=model.name,
                            strike=strike,
                            kind="call",
                            strategy=strategy,
                            frequencies=";".join(
                                map(str, settings.frequencies[-tail:])
                            ),
                            asymptotic_reference=(
                                -0.5
                                if strategy in ("analytical_delta", "pde_delta")
                                else np.nan
                            ),
                            **slope_bootstrap(
                                errors[0, -tail:, si, 0].T,
                                settings.frequencies[-tail:],
                                settings.bootstrap_replicates,
                                boot_rng,
                            ),
                        )
                    )
                state_grid = model.spot * np.exp(np.linspace(-0.15, 0.15, 121))
                for state in state_grid:
                    price, delta, gm = [
                        float(v) for v in model.greeks(state, strike, settings.horizon)
                    ]
                    surface_rows.append(
                        dict(
                            model=model.name,
                            strike=strike,
                            spot=state,
                            price=price,
                            delta=delta,
                            gamma=gm,
                            local_volatility=np.sqrt(model.diffusion_squared(state))
                            / state,
                        )
                    )
                elapsed_rows.append(
                    dict(
                        model=model.name,
                        strike=strike,
                        elapsed_wall_seconds=perf_counter() - contract_started,
                    )
                )
            elapsed_rows.append(
                dict(
                    model=model.name,
                    strike=np.nan,
                    elapsed_wall_seconds=perf_counter() - started,
                )
            )
        tables = dict(
            frequency_summary=summaries,
            price_checks=price_rows,
            simulation_checks=simulation_rows,
            pde_checks=numerical_rows,
            analytical_checks=control_rows,
            quadratic_proxy_checks=theory_rows,
            paired_frequency_changes=frequency_rows,
            convergence_slopes=slope_rows,
            scalar_ledger_checks=ledger_rows,
            cost_funding_checks=cost_rows,
            call_put_hedge_checks=parity_rows,
            known_surface_samples=surface_rows,
            stage_times=elapsed_rows,
        )
        tables["pde_hedge_difference"] = numerical_hedge_rows
        for name, rows in tables.items():
            pd.DataFrame(rows).to_csv(output / f"{name}.csv", index=False)
        return pd.DataFrame(summaries)
