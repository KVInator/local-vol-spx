from __future__ import annotations

import numpy as np


def _validate_grid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("grid must be one-dimensional.")
    if x.size < 3:
        raise ValueError("grid must have at least 3 points.")
    if not np.all(np.isfinite(x)):
        raise ValueError("grid must contain only finite points.")
    if not np.all(np.diff(x) > 0):
        raise ValueError("grid must be strictly increasing.")
    return x


def _validate_values(f: np.ndarray, n: int) -> np.ndarray:
    f = np.asarray(f, dtype=float)
    if f.ndim != 1:
        raise ValueError("values must be one-dimensional.")
    if f.size != n:
        raise ValueError("values size must match grid size.")
    return f


def first_derivative_1d(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Differentiate the local quadratic, including one-sided endpoints.

    The three-point stencil is second order on uniform and nonuniform grids
    with bounded spacing ratios. NaNs propagate through each local stencil.
    """
    x = _validate_grid(x)
    f = _validate_values(f, x.size)

    out = np.empty_like(f)

    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    out[0] = (
        -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * f[0]
        + (h1 + h2) / (h1 * h2) * f[1]
        - h1 / (h2 * (h1 + h2)) * f[2]
    )

    for i in range(1, x.size - 1):
        h_minus = x[i] - x[i - 1]
        h_plus = x[i + 1] - x[i]
        out[i] = (
            h_plus * (f[i] - f[i - 1]) / h_minus
            + h_minus * (f[i + 1] - f[i]) / h_plus
        ) / (h_minus + h_plus)

    h1 = x[-1] - x[-2]
    h2 = x[-2] - x[-3]
    out[-1] = (
        (2.0 * h1 + h2) / (h1 * (h1 + h2)) * f[-1]
        - (h1 + h2) / (h1 * h2) * f[-2]
        + h1 / (h2 * (h1 + h2)) * f[-3]
    )

    return out


def _endpoint_second_derivative(x: np.ndarray, f: np.ndarray, endpoint: int) -> float:
    """Four-point cubic stencil, scaled before solving for its weights."""
    offsets = x - x[endpoint]
    scale = np.max(np.abs(offsets))
    z = offsets / scale
    moments = np.vstack([z**power for power in range(4)])
    weights = np.linalg.solve(moments, np.array([0.0, 0.0, 2.0, 0.0]))
    return float(np.dot(weights, f - f[endpoint]) / scale**2)


def second_derivative_1d(x: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Three-point interior and four-point one-sided endpoint curvature.

    Interior accuracy is second order on uniform/smoothly graded grids, but
    only first order on arbitrary nonuniform grids. Endpoints are second
    order with at least four points and bounded spacing ratios. For three
    points, the quadratic's constant curvature is returned everywhere;
    endpoint accuracy is then generally first order. NaNs are not filled.
    """
    x = _validate_grid(x)
    f = _validate_values(f, x.size)

    out = np.empty_like(f)

    for i in range(1, x.size - 1):
        h_plus = x[i + 1] - x[i]
        h_minus = x[i] - x[i - 1]
        out[i] = 2.0 * (
            (f[i + 1] - f[i]) / h_plus - (f[i] - f[i - 1]) / h_minus
        ) / (h_plus + h_minus)

    if x.size < 4:
        out[0] = out[1]
        out[-1] = out[-2]
        return out

    out[0] = _endpoint_second_derivative(x[:4], f[:4], 0)
    out[-1] = _endpoint_second_derivative(x[-4:], f[-4:], -1)

    return out


def apply_1d_operator_along_axis(
    arr: np.ndarray,
    grid: np.ndarray,
    axis: int,
    operator,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    grid = _validate_grid(grid)

    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        raise ValueError("invalid axis.")

    if arr.shape[axis] != grid.size:
        raise ValueError("grid length must match array length on the selected axis.")

    moved = np.moveaxis(arr, axis, 0)
    out = np.empty_like(moved)

    for idx in np.ndindex(moved.shape[1:]):
        out[(slice(None),) + idx] = operator(grid, moved[(slice(None),) + idx])

    return np.moveaxis(out, 0, axis)


def first_derivative(arr: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
    return apply_1d_operator_along_axis(arr=arr, grid=grid, axis=axis, operator=first_derivative_1d)


def second_derivative(arr: np.ndarray, grid: np.ndarray, axis: int) -> np.ndarray:
    return apply_1d_operator_along_axis(arr=arr, grid=grid, axis=axis, operator=second_derivative_1d)
