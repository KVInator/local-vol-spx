"""Forward call-price PDE on a uniform log-moneyness grid."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded


LocalVariance = Callable[
    [NDArray[np.float64], float, str],
    ArrayLike,
]

BoundaryValues = Callable[
    [float],
    tuple[float, float],
]


@dataclass(frozen=True, eq=False)
class ForwardPDEResult:
    """Normalized call prices at the requested maturities."""

    log_moneyness: NDArray[np.float64]
    maturities: NDArray[np.float64]
    normalized_calls: NDArray[np.float64]
    time_steps: int

    def __post_init__(self) -> None:
        for name in (
            "log_moneyness",
            "maturities",
            "normalized_calls",
        ):
            values = np.array(
                getattr(self, name),
                dtype=float,
                copy=True,
            )
            values.setflags(write=False)
            object.__setattr__(self, name, values)


@dataclass(frozen=True)
class ForwardPDESolver:
    """
    Solve

        u_T = 0.5 * v(exp(y), T) * (u_yy - u_y),

    where u(y, T) is the normalized call price.

    The local-variance callback receives:
        normalized strikes, maturity, derivative side.

    The boundary callback supplies the left and right normalized
    call prices at a maturity.

    Time breaks align the grid with coefficient discontinuities.
    Initial Rannacher steps replace each nominal step with two
    implicit Euler half-steps.
    """

    min_log_moneyness: float
    max_log_moneyness: float
    n_space_intervals: int = 400
    max_time_step: float = 1.0 / 365.0
    theta: float = 0.5
    rannacher_steps: int = 2

    _log_grid: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _strike_grid: NDArray[np.float64] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _spacing: float = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        bounds = (
            self.min_log_moneyness,
            self.max_log_moneyness,
        )

        if not np.all(np.isfinite(bounds)):
            raise ValueError("Log-moneyness bounds must be finite.")

        if self.min_log_moneyness >= self.max_log_moneyness:
            raise ValueError(
                "Minimum log-moneyness must be below the maximum."
            )

        if (
            not isinstance(self.n_space_intervals, (int, np.integer))
            or self.n_space_intervals < 4
        ):
            raise ValueError(
                "At least four spatial intervals are required."
            )

        if (
            not np.isfinite(self.max_time_step)
            or self.max_time_step <= 0.0
        ):
            raise ValueError("Maximum time step must be positive.")

        if not 0.5 <= self.theta <= 1.0:
            raise ValueError("Theta must lie between 0.5 and 1.")

        if (
            not isinstance(self.rannacher_steps, (int, np.integer))
            or self.rannacher_steps < 0
        ):
            raise ValueError(
                "Rannacher steps must be a nonnegative integer."
            )

        spacing = (
            self.max_log_moneyness - self.min_log_moneyness
        ) / self.n_space_intervals

        if spacing >= 2.0:
            raise ValueError(
                "Spatial spacing must be below 2 for this stencil."
            )

        log_grid = np.linspace(
            self.min_log_moneyness,
            self.max_log_moneyness,
            self.n_space_intervals + 1,
        )
        strike_grid = np.exp(log_grid)

        if not np.all(np.isfinite(strike_grid)):
            raise ValueError("Normalized strikes must be finite.")

        log_grid.setflags(write=False)
        strike_grid.setflags(write=False)

        object.__setattr__(self, "_log_grid", log_grid)
        object.__setattr__(self, "_strike_grid", strike_grid)
        object.__setattr__(self, "_spacing", float(spacing))

    @property
    def log_moneyness(self) -> NDArray[np.float64]:
        return self._log_grid.copy()

    @property
    def normalized_strikes(self) -> NDArray[np.float64]:
        return self._strike_grid.copy()

    @staticmethod
    def _boundaries(
        boundary_values: BoundaryValues,
        maturity: float,
    ) -> NDArray[np.float64]:
        values = np.asarray(
            boundary_values(maturity),
            dtype=float,
        )

        if values.shape != (2,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "Boundary callback must return two finite prices."
            )

        return values

    def _operator(
        self,
        local_variance: LocalVariance,
        maturity: float,
        side: str,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        interior_strikes = self._strike_grid[1:-1]

        values = np.asarray(
            local_variance(interior_strikes, maturity, side),
            dtype=float,
        )

        try:
            variance = np.broadcast_to(
                values,
                interior_strikes.shape,
            )
        except ValueError as error:
            raise ValueError(
                "Local variance must be scalar or match the "
                "interior strike grid."
            ) from error

        if (
            not np.all(np.isfinite(variance))
            or np.any(variance < 0.0)
        ):
            raise ValueError(
                "Local variance must be finite and nonnegative."
            )

        h = self._spacing
        diffusion = 0.5 * variance

        lower = diffusion * (1.0 / h**2 + 1.0 / (2.0 * h))
        diagonal = -2.0 * diffusion / h**2
        upper = diffusion * (1.0 / h**2 - 1.0 / (2.0 * h))

        return lower, diagonal, upper

    def _advance(
        self,
        prices: NDArray[np.float64],
        old_time: float,
        new_time: float,
        theta: float,
        local_variance: LocalVariance,
        boundary_values: BoundaryValues,
    ) -> NDArray[np.float64]:
        dt = new_time - old_time

        if theta == 1.0:
            rhs = prices[1:-1].copy()
        else:
            lower_old, diagonal_old, upper_old = self._operator(
                local_variance,
                old_time,
                "right",
            )

            rhs = (
                (
                    1.0
                    + (1.0 - theta) * dt * diagonal_old
                )
                * prices[1:-1]
                + (1.0 - theta)
                * dt
                * (
                    lower_old * prices[:-2]
                    + upper_old * prices[2:]
                )
            )

        lower_new, diagonal_new, upper_new = self._operator(
            local_variance,
            new_time,
            "left",
        )
        boundaries = self._boundaries(
            boundary_values,
            new_time,
        )

        rhs[0] += theta * dt * lower_new[0] * boundaries[0]
        rhs[-1] += theta * dt * upper_new[-1] * boundaries[1]

        interior_count = len(rhs)
        matrix = np.zeros((3, interior_count), dtype=float)

        matrix[0, 1:] = -theta * dt * upper_new[:-1]
        matrix[1, :] = 1.0 - theta * dt * diagonal_new
        matrix[2, :-1] = -theta * dt * lower_new[1:]

        updated = np.empty_like(prices)
        updated[0] = boundaries[0]
        updated[-1] = boundaries[1]
        updated[1:-1] = solve_banded(
            (1, 1),
            matrix,
            rhs,
            check_finite=False,
        )

        if not np.all(np.isfinite(updated)):
            raise RuntimeError(
                "PDE solve produced nonfinite prices."
            )

        return updated

    def solve(
        self,
        maturities: ArrayLike,
        initial_prices: ArrayLike,
        local_variance: LocalVariance,
        boundary_values: BoundaryValues,
        *,
        time_breaks: ArrayLike = (),
    ) -> ForwardPDEResult:
        """
        Start from initial_prices at maturities[0].

        Return prices at every requested maturity. Internal time
        steps respect max_time_step and the supplied time breaks.
        """
        times = np.asarray(maturities, dtype=float)

        if (
            times.ndim != 1
            or len(times) < 2
            or not np.all(np.isfinite(times))
            or times[0] < 0.0
            or np.any(np.diff(times) <= 0.0)
        ):
            raise ValueError(
                "Maturities must be finite, nonnegative and "
                "strictly increasing."
            )

        prices = np.array(
            initial_prices,
            dtype=float,
            copy=True,
        )

        if (
            prices.shape != self._strike_grid.shape
            or not np.all(np.isfinite(prices))
        ):
            raise ValueError(
                "Initial prices must be finite and match the "
                "spatial grid."
            )

        initial_boundaries = self._boundaries(
            boundary_values,
            float(times[0]),
        )

        if not np.allclose(
            prices[[0, -1]],
            initial_boundaries,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "Initial prices and boundary prices disagree."
            )

        breaks = np.asarray(time_breaks, dtype=float)

        if (
            breaks.ndim != 1
            or not np.all(np.isfinite(breaks))
            or np.any(breaks < times[0])
            or np.any(breaks > times[-1])
        ):
            raise ValueError(
                "Time breaks must lie inside the maturity range."
            )

        milestones = np.unique(
            np.concatenate((times, breaks))
        )

        saved = np.empty(
            (len(times), len(self._strike_grid)),
            dtype=float,
        )
        saved[0] = prices

        checkpoint = 1
        nominal_steps = 0
        actual_steps = 0

        for start, stop in zip(
            milestones[:-1],
            milestones[1:],
        ):
            step_count = max(
                1,
                int(np.ceil(
                    (stop - start) / self.max_time_step
                )),
            )

            edges = np.linspace(
                start,
                stop,
                step_count + 1,
            )

            for old_time, new_time in zip(
                edges[:-1],
                edges[1:],
            ):
                use_startup = (
                    nominal_steps < self.rannacher_steps
                    and self.theta < 1.0
                )

                if use_startup:
                    middle = 0.5 * (old_time + new_time)

                    prices = self._advance(
                        prices,
                        float(old_time),
                        float(middle),
                        1.0,
                        local_variance,
                        boundary_values,
                    )
                    prices = self._advance(
                        prices,
                        float(middle),
                        float(new_time),
                        1.0,
                        local_variance,
                        boundary_values,
                    )
                    actual_steps += 2
                else:
                    prices = self._advance(
                        prices,
                        float(old_time),
                        float(new_time),
                        self.theta,
                        local_variance,
                        boundary_values,
                    )
                    actual_steps += 1

                nominal_steps += 1

            if checkpoint < len(times) and stop == times[checkpoint]:
                saved[checkpoint] = prices
                checkpoint += 1

        return ForwardPDEResult(
            log_moneyness=self._log_grid,
            maturities=times,
            normalized_calls=saved,
            time_steps=actual_steps,
        )