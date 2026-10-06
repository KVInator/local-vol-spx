"""Positive-time AH local variance without underflow in wing densities."""

from functools import lru_cache

import numpy as np
from scipy.linalg.lapack import dpttrf

from andreasen_huge import AndreasenHugeSurface


def log_positive_solve(bands, log_rhs):
    """Solve an irreducible tridiagonal M-matrix using a logarithmic RHS.

    The AH matrices have positive diagonal and negative off-diagonals.
    Their diagonal similarity to a symmetric positive-definite matrix
    supplies the same elimination pivots, without forming tiny densities.
    Negative infinity represents an exact zero in the nonnegative RHS.
    """
    bands = np.asarray(bands, float)
    rhs = np.asarray(log_rhs, float)
    if (
        rhs.ndim != 1 or len(rhs) < 2 or bands.shape != (3, len(rhs))
        or not np.all(np.isfinite(bands))
        or np.any(np.isnan(rhs)) or np.any(np.isposinf(rhs))
        or not np.any(np.isfinite(rhs)) or np.any(bands[1] <= 0)
    ):
        raise ValueError("Invalid positive tridiagonal system.")

    lower, upper = bands[2, :-1], bands[0, 1:]
    if np.all(lower == 0) and np.all(upper == 0):
        return rhs - np.log(bands[1])
    if np.any(lower >= 0) or np.any(upper >= 0):
        raise ValueError("Require strictly negative off-diagonals.")

    symmetric_off = -np.sqrt(-lower) * np.sqrt(-upper)
    pivots, _, info = dpttrf(bands[1].copy(), symmetric_off)
    if info != 0 or not np.all(np.isfinite(pivots)) or np.any(pivots <= 0):
        raise ArithmeticError("Positive elimination failed.")

    # Forward substitution.
    cumulative = np.r_[
        0.0,
        np.cumsum(np.log(-lower) - np.log(pivots[:-1])),
    ]
    forward = cumulative + np.logaddexp.accumulate(rhs - cumulative)

    # Back substitution as a positive recurrence in reverse.
    base = forward - np.log(pivots)
    reverse = np.r_[
        0.0,
        np.cumsum(
            (np.log(-upper) - np.log(pivots[:-1]))[::-1]
        ),
    ]
    solution = (
        reverse + np.logaddexp.accumulate(base[::-1] - reverse)
    )[::-1]

    if not np.all(np.isfinite(solution)):
        raise ArithmeticError("Logarithmic solve produced nonfinite values.")
    return solution


class AHLocalVariance:
    """Continue discrete AH Dupire variance linearly in log-strike.

    This is a numerical coefficient adapter, not a derivative of the
    piecewise-linear price interpolant. Zero-time variance and extrapolation
    outside the interior AH nodes are deliberately unsupported.
    """

    def __init__(self, model):
        self.model = model
        self.y = np.log(model.grid.z[1:-1])
        self._q = tuple(
            0.5 * v * model.grid.z[1:-1] ** 2
            for v in model._variances
        )

        # One unit slope jump at z=1; flat wings have exactly zero density.
        # This avoids cancellation noise from differencing intrinsic prices.
        center = int(np.argmin(abs(model.grid.z - 1)))
        if (
            model.grid.z[center] != 1
            or center in (0, len(model.grid.z) - 1)
        ):
            raise ValueError("The AH grid must contain an interior z=1 node.")

        log_density = np.full(len(self.y), -np.inf)
        log_density[center - 1] = np.log(
            2 / (model.grid.z[center + 1] - model.grid.z[center - 1])
        )
        self._log_density = [log_density]

        start = 0.0
        for time, q in zip(model.maturities, self._q):
            log_density = log_positive_solve(
                model.grid.matrix(q, time - start, density=True),
                log_density,
            )
            self._log_density.append(log_density)
            start = float(time)

        # Per-instance cache avoids repeated endpoint calculations.
        self._state = lru_cache(maxsize=4)(self._node_variance)
        self.evaluations = 0

    @classmethod
    def load(cls, path):
        return cls(AndreasenHugeSurface.load(path))

    def _node_variance(self, time, index):
        self.evaluations += 1
        start = 0.0 if index == 0 else self.model.maturities[index - 1]
        elapsed = time - start
        q = self._q[index]

        density = log_positive_solve(
            self.model.grid.matrix(q, elapsed, density=True),
            self._log_density[index],
        )
        derivative = log_positive_solve(
            self.model.grid.matrix(q, elapsed),
            np.log(q) + density,
        )
        log_variance = np.log(2.0) + derivative - 2 * self.y - density

        with np.errstate(over="ignore", under="ignore"):
            variance = np.exp(log_variance)

        if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
            raise ArithmeticError(
                "AH local variance cannot be represented as a float."
            )

        variance.setflags(write=False)
        return variance

    def normalized_variance(self, normalized_strikes, time, side="right"):
        time = self.model._time(time)
        if time <= 0:
            raise ValueError("AH local variance requires positive maturity.")
        if side not in ("left", "right"):
            raise ValueError("Side must be left or right.")

        z = np.asarray(normalized_strikes, float)
        if z.size == 0 or not np.all(np.isfinite(z)) or np.any(z <= 0):
            raise ValueError(
                "Normalized strikes must be finite, positive and nonempty."
            )

        y = np.log(z)
        if np.any(y < self.y[0]) or np.any(y > self.y[-1]):
            raise ValueError("AH variance extrapolation is unsupported.")

        index = min(
            int(np.searchsorted(self.model.maturities, time, side=side)),
            len(self.model.maturities) - 1,
        )
        variance = self._state(time, index)
        return np.interp(y.ravel(), self.y, variance).reshape(z.shape)

    def __call__(self, normalized_strikes, time, side="right"):
        return self.normalized_variance(normalized_strikes, time, side)