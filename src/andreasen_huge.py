"""Spread-weighted Andreasen-Huge interpolation on a finite strike grid.

One resolvent step per quoted expiry: (I - dt * L) c = c_previous.
Calibration proxy volatility is distinct from recovered Dupire volatility.
Prices interpolate linearly in normalized strike. Curvature and local
variance are discrete node diagnostics, not derivatives of that interpolant.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import solve_banded
from scipy.optimize import least_squares

from black import BlackPricer
from implied_vol import ImpliedVolSolver


def linear_weights(points, nodes):
    """Linear interpolation weights, with constant endpoint continuation."""
    points, nodes = np.asarray(points), np.asarray(nodes)
    index = np.clip(np.searchsorted(nodes, points) - 1, 0, len(nodes) - 2)
    fraction = np.clip(
        (points - nodes[index]) / (nodes[index + 1] - nodes[index]), 0.0, 1.0
    )
    weights = np.zeros((len(points), len(nodes)))
    weights[np.arange(len(points)), index] = 1.0 - fraction
    weights[np.arange(len(points)), index + 1] = fraction
    return weights


@dataclass(frozen=True)
class AHGrid:
    width: float = 1.0
    intervals: int = 2000

    def __post_init__(self):
        if (
            not np.isfinite(self.width) or not 0.1 <= self.width <= 3.0
            or not isinstance(self.intervals, (int, np.integer))
            or self.intervals < 100 or self.intervals % 2
        ):
            raise ValueError("Use width in [0.1, 3] and an even count >= 100.")
        z = np.exp(np.linspace(-self.width, self.width, self.intervals + 1))
        left, right = np.diff(z)[:-1], np.diff(z)[1:]
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "lower", 2.0 / (left * (left + right)))
        object.__setattr__(self, "diagonal", -2.0 / (left * right))
        object.__setattr__(self, "upper", 2.0 / (right * (left + right)))

    def curvature(self, calls):
        # Difference of slopes is independent of the density transport solve.
        slopes = np.diff(calls) / np.diff(self.z)
        return 2.0 * np.diff(slopes) / (self.z[2:] - self.z[:-2])

    def matrix(self, q, elapsed, density=False):
        """I-dt*diag(q)*D2, or I-dt*D2*diag(q) for density."""
        bands = np.zeros((3, len(q)))
        bands[1] = 1.0 - elapsed * q * self.diagonal
        if density:
            bands[0, 1:] = -elapsed * self.upper[:-1] * q[1:]
            bands[2, :-1] = -elapsed * self.lower[1:] * q[:-1]
        else:
            bands[0, 1:] = -elapsed * q[:-1] * self.upper[:-1]
            bands[2, :-1] = -elapsed * q[1:] * self.lower[1:]
        return bands

    def step(self, previous, previous_density, variance, elapsed):
        """Positive density transport avoids subtracting intrinsic prices."""
        q = 0.5 * variance * self.z[1:-1] ** 2
        density = solve_banded(
            (1, 1), self.matrix(q, elapsed, density=True), previous_density
        )
        calls = previous.copy()
        calls[1:-1] += elapsed * q * density
        return calls, density, q


class AndreasenHugeSurface:
    def __init__(self, grid, spot, maturities, forwards, discounts, controls, parameters):
        self.grid, self.spot = grid, float(spot)
        self.maturities = np.asarray(maturities, float).copy()
        self.forwards = np.asarray(forwards, float).copy()
        self.discounts = np.asarray(discounts, float).copy()
        self.controls = tuple(np.asarray(row, float).copy() for row in controls)
        self.parameters = tuple(np.asarray(row, float).copy() for row in parameters)
        n = len(self.maturities)
        if (
            n < 1 or self.maturities.ndim != 1
            or not np.isfinite(self.spot) or self.spot <= 0.0
            or np.any(self.maturities <= 0.0)
            or np.any(np.diff(self.maturities) <= 0.0)
            or len(self.forwards) != n or len(self.discounts) != n
            or len(self.controls) != n or len(self.parameters) != n
            or not all(np.all(np.isfinite(a)) for a in (
                self.maturities, self.forwards, self.discounts
            ))
            or np.any(self.forwards <= 0.0) or np.any(self.discounts <= 0.0)
        ):
            raise ValueError("Invalid carry or maturity arrays.")

        self._calls = [np.maximum(1.0 - grid.z, 0.0)]
        self._time_values = [np.zeros_like(grid.z)]
        self._densities = [grid.curvature(self._calls[0])]
        self._variances = []
        previous_time = 0.0

        for time, nodes, log_vol in zip(self.maturities, self.controls, self.parameters):
            if (
                nodes.ndim != 1 or len(nodes) < 2 or nodes.shape != log_vol.shape
                or np.any(np.diff(nodes) <= 0.0)
                or not np.all(np.isfinite(nodes))
                or not np.all(np.isfinite(log_vol))
            ):
                raise ValueError("Invalid proxy control nodes.")

            variance = np.exp(
                2.0 * linear_weights(np.log(grid.z[1:-1]), nodes) @ log_vol
            )
            if not np.all(np.isfinite(variance)) or np.any(variance <= 0.0):
                raise ValueError("Proxy variance must be finite and positive.")

            calls, density, q = grid.step(
                self._calls[-1], self._densities[-1], variance, time - previous_time
            )
            self._calls.append(calls)
            time_value = self._time_values[-1].copy()
            time_value[1:-1] += (time - previous_time) * q * density
            self._time_values.append(time_value)
            self._densities.append(density)
            self._variances.append(variance)
            previous_time = time

    def _time(self, time):
        values = np.asarray(time, float)
        if values.ndim != 0:
            raise ValueError("Evaluate one scalar maturity at a time.")
        value = float(values)
        pillars = np.r_[0.0, self.maturities]
        nearest = int(np.argmin(abs(pillars - value)))
        if abs(pillars[nearest] - value) <= 1e-14:
            value = float(pillars[nearest])
        if not np.isfinite(value) or not 0.0 <= value <= pillars[-1]:
            raise ValueError("Maturity is outside the supported range.")
        return value

    def forward(self, time):
        if self._time(time) == 0.0:
            return self.spot
        return float(np.exp(np.interp(
            self._time(time), np.r_[0.0, self.maturities],
            np.log(np.r_[self.spot, self.forwards])
        )))

    def discount_factor(self, time):
        if self._time(time) == 0.0:
            return 1.0
        return float(np.exp(np.interp(
            self._time(time), np.r_[0.0, self.maturities],
            np.log(np.r_[1.0, self.discounts])
        )))

    def node_state(self, time, side="right"):
        time = self._time(time)
        if side not in ("left", "right"):
            raise ValueError("Side must be left or right.")

        index = min(
            int(np.searchsorted(self.maturities, time, side=side)),
            len(self.maturities) - 1
        )
        start = 0.0 if index == 0 else self.maturities[index - 1]

        calls, density, q = self.grid.step(
            self._calls[index], self._densities[index],
            self._variances[index], time - start
        )
        time_value = self._time_values[index].copy()
        time_value[1:-1] += (time - start) * q * density

        # dc/dT = (I-dt*L)^(-1) Lc; it is not simply Lc.
        time_derivative = solve_banded(
            (1, 1), self.grid.matrix(q, time - start), q * density
        )
        local_variance = np.full(density.shape, np.nan)
        valid = (
            (density > 0.0)
            & np.isfinite(density)
            & (time_derivative >= 0.0)
        )
        np.divide(
            2.0 * time_derivative,
            self.grid.z[1:-1] ** 2 * density,
            out=local_variance,
            where=valid
        )
        if time == 0.0:
            # Initial payoff is not a smooth density.
            local_variance[:] = np.nan

        return {
            "calls": calls,
            "time_value": time_value,
            "curvature": density,
            "time_derivative": time_derivative,
            "proxy_variance": self._variances[index].copy(),
            "local_variance": local_variance,
        }

    def normalized_call(self, normalized_strikes, time):
        z = np.asarray(normalized_strikes, float)
        if (
            not np.all(np.isfinite(z)) or np.any(z < self.grid.z[0])
            or np.any(z > self.grid.z[-1])
        ):
            raise ValueError("Strikes are outside the finite AH grid.")
        if self._time(time) == 0.0:
            return np.maximum(1.0 - z, 0.0)
        return np.interp(z, self.grid.z, self.node_state(time)["calls"])

    def call_price(self, strikes, time):
        if self._time(time) == 0.0:
            return np.maximum(self.spot - np.asarray(strikes, float), 0.0)
        f, d = self.forward(time), self.discount_factor(time)
        return d * f * self.normalized_call(
            np.asarray(strikes, float) / f, time
        )

    def implied_volatility(self, strikes, time):
        time = self._time(time)
        if time <= 0.0:
            raise ValueError("Implied volatility is undefined at zero maturity.")

        strikes = np.asarray(strikes, float)
        f, d = self.forward(time), self.discount_factor(time)
        z = strikes / f
        self.normalized_call(z, time)  # Domain validation.
        time_value = np.interp(
            z, self.grid.z, self.node_state(time)["time_value"]
        )
        prices = d * f * time_value
        solver = ImpliedVolSolver(BlackPricer(f, d, time))

        return np.array([
            solver.solve(
                price=float(p),
                strike=float(k),
                kind="put" if k < f else "call"
            ).volatility
            for k, p in zip(strikes.ravel(), np.asarray(prices).ravel())
        ]).reshape(strikes.shape)

    def save(self, path):
        specification = {
            "format_version": 1,
            "model_type": "discrete_andreasen_huge",
            "spot": self.spot,
            "width": self.grid.width,
            "intervals": self.grid.intervals,
            "maturities": self.maturities.tolist(),
            "forwards": self.forwards.tolist(),
            "discounts": self.discounts.tolist(),
            "controls": [row.tolist() for row in self.controls],
            "log_proxy_volatility": [
                row.tolist() for row in self.parameters
            ],
            "boundaries": "Intrinsic normalized call at both finite endpoints.",
            "strike_interpolation": "Piecewise linear in normalized strike.",
            "derivative_scope": (
                "Discrete curvature and Dupire variance at interior nodes."
            ),
            "zero_time_local_variance": "Not defined by the nodal payoff.",
            "global_tail_extension": False,
        }
        Path(path).write_text(
            json.dumps(specification, indent=2) + "\n",
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path):
        row = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            row.get("format_version") != 1
            or row.get("model_type") != "discrete_andreasen_huge"
        ):
            raise ValueError("Unsupported AH specification.")
        return cls(
            AHGrid(row["width"], row["intervals"]),
            row["spot"],
            row["maturities"],
            row["forwards"],
            row["discounts"],
            row["controls"],
            row["log_proxy_volatility"]
        )


@dataclass(frozen=True)
class AndreasenHugeCalibrator:
    grid: AHGrid
    control_points: int = 31
    smoothing: float = 0.1
    min_proxy_vol: float = 0.005
    max_proxy_vol: float = 3.0
    max_evaluations: int = 300

    def calibrate(
        self, spot, maturities, forwards, discounts, quotes, *, progress=None
    ):
        if (
            self.control_points < 3 or self.max_evaluations < 1
            or not np.isfinite(self.smoothing) or self.smoothing < 0.0
            or not 0.0 < self.min_proxy_vol < self.max_proxy_vol < np.inf
        ):
            raise ValueError("Invalid calibration settings.")

        # Validate carry before starting an optimization.
        count = len(maturities)
        if len(quotes) != count:
            raise ValueError("Need one quote array per maturity.")
        AndreasenHugeSurface(
            self.grid, spot, maturities, forwards, discounts,
            [np.array([-0.1, 0.1])] * count,
            [np.log([0.2, 0.2])] * count
        )

        calls = np.maximum(1.0 - self.grid.z, 0.0)
        density = self.grid.curvature(calls)
        controls, parameters, reports = [], [], []
        previous_time = 0.0

        for index, (time, f, d, data) in enumerate(
            zip(maturities, forwards, discounts, quotes)
        ):
            if progress is not None:
                progress(index, float(time))
            data = np.asarray(data, float)
            if (
                data.ndim != 2 or data.shape[1] != 3 or len(data) < 3
                or not np.all(np.isfinite(data))
                or np.any(data[:, 0] <= 0.0)
                or np.any(data[:, 2] <= 0.0)
                or len(np.unique(data[:, 0])) != len(data)
            ):
                raise ValueError(
                    "Quotes must be unique [strike, call_mid, half_width] rows."
                )

            z, scale = data[:, 0] / f, f * d
            if (
                np.any(z <= self.grid.z[1])
                or np.any(z >= self.grid.z[-2])
            ):
                raise ValueError(
                    "Calibration quotes must be inside the AH grid interior."
                )

            y = np.log(z)
            nodes = np.linspace(
                y.min(), y.max(), min(self.control_points, len(data))
            )
            weights = linear_weights(np.log(self.grid.z[1:-1]), nodes)
            quote_weights = linear_weights(z, self.grid.z)
            elapsed = float(time - previous_time)
            smooth = np.diff(np.eye(len(nodes)), axis=0)
            penalty = np.sqrt(self.smoothing) * smooth
            initial = (
                np.full(len(nodes), np.log(0.15))
                if index == 0
                else np.interp(nodes, controls[-1], parameters[-1])
            )
            cache = {}

            def evaluate(p):
                if "p" in cache and np.array_equal(p, cache["p"]):
                    return cache["residual"], cache["jacobian"]

                variance = np.exp(2.0 * (weights @ p))
                fitted, gamma, q = self.grid.step(
                    calls, density, variance, elapsed
                )
                residual = (
                    scale * (quote_weights @ fitted) - data[:, 1]
                ) / data[:, 2]

                derivatives = solve_banded(
                    (1, 1),
                    self.grid.matrix(q, elapsed),
                    (2.0 * elapsed * q * gamma)[:, None] * weights
                )
                jacobian = (
                    scale * (quote_weights[:, 1:-1] @ derivatives)
                ) / data[:, 2, None]

                residual = np.r_[residual, penalty @ p]
                jacobian = np.vstack([jacobian, penalty])
                cache.update(
                    p=p.copy(),
                    residual=residual,
                    jacobian=jacobian
                )
                return residual, jacobian

            fit = least_squares(
                lambda p: evaluate(p)[0],
                initial,
                jac=lambda p: evaluate(p)[1],
                bounds=(
                    np.log(self.min_proxy_vol),
                    np.log(self.max_proxy_vol)
                ),
                max_nfev=self.max_evaluations,
                ftol=1e-9,
                xtol=1e-9,
                gtol=1e-7
            )
            if not fit.success:
                raise RuntimeError(
                    f"Expiry {index + 1} optimizer stopped: {fit.message}"
                )

            variance = np.exp(2.0 * (weights @ fit.x))
            calls, density, _ = self.grid.step(
                calls, density, variance, elapsed
            )
            controls.append(nodes)
            parameters.append(fit.x.copy())
            reports.append({
                "optimizer_evaluations": int(fit.nfev),
                "optimizer_message": str(fit.message),
                "active_proxy_bounds": int(
                    np.count_nonzero(fit.active_mask)
                ),
                "min_proxy_vol_pct": float(
                    100.0 * np.exp(fit.x).min()
                ),
                "max_proxy_vol_pct": float(
                    100.0 * np.exp(fit.x).max()
                ),
            })
            previous_time = time

        return AndreasenHugeSurface(
            self.grid, spot, maturities, forwards, discounts,
            controls, parameters
        ), reports