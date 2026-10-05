"""Power-law strike tails attached to convex fitted call curves."""

from dataclasses import dataclass, field

import numpy as np

from convex_spline import ConvexCallSpline


@dataclass(frozen=True)
class TailAnchor:
    normalized_strike: float
    time_value: float
    power: float
    core_curvature: float


@dataclass(frozen=True)
class PowerTailSlice:
    """Preserve a fitted core and extend it over positive strikes.

    Left tail:
        c(z) = 1 - z + P_left * (z / z_left)**p_left

    Right tail:
        c(z) = C_right * (z / z_right)**(-p_right)

    The powers match the fitted slopes. Prices and first derivatives
    are continuous at the joins. Second derivatives can jump.
    """

    core: ConvexCallSpline
    left_log_moneyness: float
    right_log_moneyness: float
    left: TailAnchor = field(init=False)
    right: TailAnchor = field(init=False)

    def __post_init__(self):
        lower = float(self.left_log_moneyness)
        upper = float(self.right_log_moneyness)

        if not np.isfinite(lower) or not np.isfinite(upper):
            raise ValueError("Tail joins must be finite.")

        if not lower < 0.0 < upper:
            raise ValueError("Tail joins must lie on either side of forward.")

        forward = self.core.pricer.forward
        discount = self.core.pricer.discount_factor
        z = np.exp([lower, upper])
        strikes = forward * z

        if (
            strikes[0] < self.core.strike_origin
            or strikes[1] > self.core.strike_max
        ):
            raise ValueError("Tail joins must lie inside the fitted domain.")

        calls = np.asarray(self.core.price(strikes)) / (
            discount * forward
        )
        slopes = np.asarray(self.core.strike_slope(strikes)) / discount
        curvature = (
            np.asarray(self.core.strike_curvature(strikes))
            * forward
            / discount
        )

        left_put = float(calls[0] + z[0] - 1.0)
        right_call = float(calls[1])

        if left_put <= 0.0 or right_call <= 0.0:
            raise ValueError("Tail anchors need positive option time values.")

        if not np.all((-1.0 < slopes) & (slopes < 0.0)):
            raise ValueError(
                "Anchor call slopes must be strictly between -1 and 0."
            )

        left_power = float(z[0] * (1.0 + slopes[0]) / left_put)
        right_power = float(-z[1] * slopes[1] / right_call)

        if not np.isfinite(left_power) or left_power <= 1.0:
            raise ValueError(
                "The left anchor does not admit a power tail with power > 1."
            )

        if not np.isfinite(right_power) or right_power <= 0.0:
            raise ValueError(
                "The right anchor does not admit a positive power tail."
            )

        if not np.all(np.isfinite(curvature)) or np.any(curvature <= 0.0):
            raise ValueError("Anchor core curvatures must be positive.")

        object.__setattr__(
            self,
            "left",
            TailAnchor(
                float(z[0]),
                left_put,
                left_power,
                float(curvature[0]),
            ),
        )
        object.__setattr__(
            self,
            "right",
            TailAnchor(
                float(z[1]),
                right_call,
                right_power,
                float(curvature[1]),
            ),
        )

    @property
    def maturity(self):
        return float(self.core.pricer.maturity)

    @staticmethod
    def _checked_strikes(z):
        z = np.asarray(z, dtype=float)

        if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
            raise ValueError("Normalized strikes must be finite and positive.")

        return z

    def log_tail_time_value(self, z, side):
        z = self._checked_strikes(z)

        if side == "left":
            anchor = self.left
            exponent = anchor.power
        elif side == "right":
            anchor = self.right
            exponent = -anchor.power
        else:
            raise ValueError("Tail side must be 'left' or 'right'.")

        return (
            np.log(anchor.time_value)
            + exponent
            * (np.log(z) - np.log(anchor.normalized_strike))
        )

    def _tail_value(self, z, side, derivative):
        log_value = self.log_tail_time_value(z, side)
        exponent = (
            self.left.power if side == "left" else -self.right.power
        )

        if derivative == 0:
            return np.exp(log_value)

        if derivative == 1:
            return exponent * np.exp(log_value - np.log(z))

        return (
            exponent
            * (exponent - 1.0)
            * np.exp(log_value - 2.0 * np.log(z))
        )

    def normalized_call(self, z, derivative=0):
        z = self._checked_strikes(z)

        if derivative not in (0, 1, 2):
            raise ValueError("Supported derivative orders are 0, 1 and 2.")

        left = z < self.left.normalized_strike
        right = z > self.right.normalized_strike
        body = ~(left | right)
        values = np.empty_like(z)

        if np.any(left):
            tail = self._tail_value(z[left], "left", derivative)

            if derivative == 0:
                tail = 1.0 - z[left] + tail
            elif derivative == 1:
                tail = -1.0 + tail

            values[left] = tail

        if np.any(right):
            values[right] = self._tail_value(
                z[right], "right", derivative
            )

        if np.any(body):
            forward = self.core.pricer.forward
            discount = self.core.pricer.discount_factor
            strikes = forward * z[body]

            if derivative == 0:
                values[body] = self.core.price(strikes) / (
                    discount * forward
                )
            elif derivative == 1:
                values[body] = (
                    self.core.strike_slope(strikes) / discount
                )
            else:
                values[body] = (
                    self.core.strike_curvature(strikes)
                    * forward
                    / discount
                )

        return values

    def normalized_put(self, z):
        """Evaluate scalar or array puts with stable left-tail prices."""
        z = self._checked_strikes(z)

        # Scalar arithmetic can return np.float64. Keep a writable array,
        # including a zero-dimensional array for a scalar input.
        values = np.array(
            self.normalized_call(z) + z - 1.0,
            dtype=float,
            copy=True,
        )
        left = z < self.left.normalized_strike

        if np.any(left):
            values[left] = self._tail_value(z[left], "left", 0)

        return values

    def metadata(self):
        a = self.left.normalized_strike
        b = self.right.normalized_strike
        put = self.left.time_value
        call = self.right.time_value
        p = self.left.power
        q = self.right.power

        left_call = 1.0 - a + put
        left_slope = -1.0 + p * put / a
        right_slope = -q * call / b

        left_mass = p * put / a
        right_mass = q * call / b
        core_mass = right_slope - left_slope

        core_mean = (
            b * right_slope
            - call
            - a * left_slope
            + left_call
        )
        total_mean = (
            (p - 1.0) * put
            + core_mean
            + (q + 1.0) * call
        )

        return {
            "maturity_years": self.maturity,
            "left_log_moneyness": float(self.left_log_moneyness),
            "right_log_moneyness": float(self.right_log_moneyness),
            "left_time_value": put,
            "right_time_value": call,
            "left_power": p,
            "right_power": q,
            "left_curvature_ratio": (
                p * (p - 1.0) * put / a**2
                / self.left.core_curvature
            ),
            "right_curvature_ratio": (
                q * (q + 1.0) * call / b**2
                / self.right.core_curvature
            ),
            "left_probability_mass": left_mass,
            "core_probability_mass": core_mass,
            "right_probability_mass": right_mass,
            "total_probability_mass": left_mass + core_mass + right_mass,
            "normalized_mean": total_mean,
            "join_continuity": "Price and slope; curvature may jump.",
        }


def check_tail_calendar(
    earlier,
    later,
    price_tolerance=1e-12,
    log_ratio_tolerance=1e-12,
    max_depth=24,
    max_intervals=50000,
):
    """Check calendar order over the whole positive strike range.

    The two unbounded regions use analytical power-tail comparisons.

    The remaining bounded region uses adaptive lower bounds derived
    from the monotone slopes of the two convex call curves.
    Unresolved intervals produce a failed check.
    """
    if earlier.maturity >= later.maturity:
        raise ValueError("Calendar checks require increasing maturities.")

    left_limit = min(
        earlier.left.normalized_strike,
        later.left.normalized_strike,
    )
    right_limit = max(
        earlier.right.normalized_strike,
        later.right.normalized_strike,
    )

    left_log_ratio = float(
        later.log_tail_time_value(left_limit, "left")
        - earlier.log_tail_time_value(left_limit, "left")
    )
    right_log_ratio = float(
        later.log_tail_time_value(right_limit, "right")
        - earlier.log_tail_time_value(right_limit, "right")
    )

    left_ordered = bool(
        later.left.power <= earlier.left.power
        and left_log_ratio >= -log_ratio_tolerance
    )
    right_ordered = bool(
        later.right.power <= earlier.right.power
        and right_log_ratio >= -log_ratio_tolerance
    )

    cache = {}

    def evaluate(z):
        if z not in cache:
            if z < 1.0:
                gap = (
                    later.normalized_put(z)
                    - earlier.normalized_put(z)
                )
            else:
                gap = (
                    later.normalized_call(z)
                    - earlier.normalized_call(z)
                )

            cache[z] = (
                float(gap),
                float(earlier.normalized_call(z, derivative=1)),
                float(later.normalized_call(z, derivative=1)),
            )

        return cache[z]

    stack = [(left_limit, right_limit, 0)]
    checked = 0
    accepted_lower_bound = np.inf
    minimum_gap = np.inf
    minimum_location = None
    middle_status = "passed"

    while stack:
        left, right, depth = stack.pop()
        checked += 1

        if checked > max_intervals:
            middle_status = "unresolved"
            break

        gap_left, early_left, late_left = evaluate(left)
        gap_right, early_right, late_right = evaluate(right)

        for point, gap in ((left, gap_left), (right, gap_right)):
            if gap < minimum_gap:
                minimum_gap = gap
                minimum_location = point

        if min(gap_left, gap_right) < -price_tolerance:
            middle_status = "violated"
            break

        # Each individual call slope increases with strike.
        derivative_lower = late_left - early_right
        derivative_upper = late_right - early_left
        width = right - left

        if derivative_lower >= 0.0:
            lower_bound = gap_left
        elif derivative_upper <= 0.0:
            lower_bound = gap_right
        else:
            lower_bound = (
                min(gap_left, gap_right)
                - 0.5
                * width
                * max(abs(derivative_lower), abs(derivative_upper))
            )

        lower_bound -= 64.0 * np.finfo(float).eps

        if lower_bound >= -price_tolerance:
            accepted_lower_bound = min(
                accepted_lower_bound, lower_bound
            )
            continue

        if depth >= max_depth:
            middle_status = "unresolved"
            break

        midpoint = 0.5 * (left + right)
        stack.append((midpoint, right, depth + 1))
        stack.append((left, midpoint, depth + 1))

    return {
        "earlier_days": 365.0 * earlier.maturity,
        "later_days": 365.0 * later.maturity,
        "left_tail_ordered": left_ordered,
        "right_tail_ordered": right_ordered,
        "left_log_ratio_at_common_join": left_log_ratio,
        "right_log_ratio_at_common_join": right_log_ratio,
        "left_power_change": later.left.power - earlier.left.power,
        "right_power_change": later.right.power - earlier.right.power,
        "middle_status": middle_status,
        "middle_intervals_checked": checked,
        "middle_accepted_lower_bound": (
            float(accepted_lower_bound)
            if np.isfinite(accepted_lower_bound)
            else None
        ),
        "minimum_evaluated_middle_gap": (
            float(minimum_gap) if np.isfinite(minimum_gap) else None
        ),
        "log_moneyness_at_minimum": (
            float(np.log(minimum_location))
            if minimum_location is not None
            else None
        ),
        "passed": bool(
            left_ordered
            and right_ordered
            and middle_status == "passed"
        ),
    }