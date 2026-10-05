"""Calendar-ordered strike tails with shared asymptotic powers."""

from dataclasses import dataclass, field

import numpy as np

from convex_spline import ConvexCallSpline


@dataclass(frozen=True)
class TailAnchor:
    normalized_strike: float
    time_value: float
    power: float
    shape: float
    core_curvature: float


@dataclass(frozen=True)
class CoupledTailSlice:
    core: ConvexCallSpline
    left_log_moneyness: float
    right_log_moneyness: float
    left_power: float = 3.0
    right_power: float = 1.5

    left: TailAnchor = field(init=False)
    right: TailAnchor = field(init=False)

    def __post_init__(self):
        settings = np.array(
            [
                self.left_log_moneyness,
                self.right_log_moneyness,
                self.left_power,
                self.right_power,
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(settings)):
            raise ValueError("Tail settings must be finite.")

        if not self.left_log_moneyness < 0.0 < self.right_log_moneyness:
            raise ValueError("Tail joins must lie on opposite sides of zero.")

        if self.left_power <= 1.0 or self.right_power <= 1.0:
            raise ValueError(
                "Both powers must exceed one. The right-tail restriction "
                "ensures a finite second moment."
            )

        for side, y, power in (
            ("left", self.left_log_moneyness, self.left_power),
            ("right", self.right_log_moneyness, self.right_power),
        ):
            z = float(np.exp(y))
            strike = self.core.pricer.forward * z

            if not self.core.strike_origin <= strike <= self.core.strike_max:
                raise ValueError("A tail join lies outside the fitted curve.")

            forward = self.core.pricer.forward
            discount = self.core.pricer.discount_factor

            call = float(self.core.price(strike) / (discount * forward))
            slope = float(self.core.strike_slope(strike) / discount)
            curvature = float(
                self.core.strike_curvature(strike) * forward / discount
            )

            time_value = call + z - 1.0 if side == "left" else call
            numerator = (
                z * (1.0 + slope)
                if side == "left"
                else -z * slope
            )
            elasticity = numerator / time_value if time_value > 0.0 else np.nan
            shape = (
                elasticity - power
                if side == "left"
                else elasticity / power
            )

            if not (
                np.isfinite(time_value)
                and time_value > 0.0
                and np.isfinite(curvature)
                and curvature > 0.0
                and -1.0 < slope < 0.0
                and np.isfinite(shape)
            ):
                raise ValueError(f"Invalid {side} tail anchor.")

            if side == "left" and shape < 0.0:
                raise ValueError(
                    "The left anchor elasticity is below the shared power."
                )
            if side == "right" and shape <= 0.0:
                raise ValueError("The right tail needs a positive shape.")

            object.__setattr__(
                self,
                side,
                TailAnchor(z, time_value, power, shape, curvature),
            )

    @property
    def maturity(self):
        return float(self.core.pricer.maturity)

    @staticmethod
    def _checked_strikes(z):
        z = np.asarray(z, dtype=float)
        if not np.all(np.isfinite(z)) or np.any(z <= 0.0):
            raise ValueError("Normalized strikes must be positive and finite.")
        return z

    def _tail_components(self, z, side):
        anchor = self.left if side == "left" else self.right
        log_z = np.log(z)
        a = anchor.normalized_strike
        b = anchor.shape
        p = anchor.power

        if side == "left":
            denominator = 1.0 + b * (1.0 - z / a)
            ratio = b * (z / a) / denominator
            log_value = (
                np.log(anchor.time_value)
                + p * (log_z - np.log(a))
                - np.log(denominator)
            )
        else:
            inverse_ratio = a / z
            scaled_denominator = b + (1.0 - b) * inverse_ratio
            log_denominator = (
                log_z - np.log(a) + np.log(scaled_denominator)
            )
            ratio = b / scaled_denominator
            log_value = (
                np.log(anchor.time_value) - p * log_denominator
            )

        return anchor, log_value, ratio

    def log_time_value(self, z, side):
        if side not in ("left", "right"):
            raise ValueError("Side must be 'left' or 'right'.")
        z = self._checked_strikes(z)
        return self._tail_components(z, side)[1]

    def _tail_value(self, z, side, derivative):
        anchor, log_value, ratio = self._tail_components(z, side)
        p = anchor.power

        if derivative == 0:
            return np.exp(log_value)

        if side == "left":
            factor = (
                p + ratio
                if derivative == 1
                else p * (p - 1.0) + 2.0 * p * ratio + 2.0 * ratio**2
            )
        else:
            factor = (
                p * ratio
                if derivative == 1
                else p * (p + 1.0) * ratio**2
            )

        values = np.exp(
            log_value + np.log(factor) - derivative * np.log(z)
        )
        return -values if side == "right" and derivative == 1 else values

    def normalized_call(self, z, derivative=0):
        if derivative not in (0, 1, 2):
            raise ValueError("Derivative must be zero, one or two.")

        z = self._checked_strikes(z)
        values = np.empty_like(z)

        left = z < self.left.normalized_strike
        right = z > self.right.normalized_strike
        body = ~(left | right)

        if np.any(left):
            tail = self._tail_value(z[left], "left", derivative)
            if derivative == 0:
                tail = tail + 1.0 - z[left]
            elif derivative == 1:
                tail = tail - 1.0
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
                values[body] = self.core.strike_slope(strikes) / discount
            else:
                values[body] = (
                    self.core.strike_curvature(strikes)
                    * forward
                    / discount
                )

        return values

    def normalized_put(self, z):
        z = self._checked_strikes(z)
        values = np.array(
            self.normalized_call(z) + z - 1.0,
            dtype=float,
            copy=True,
        )
        left = z < self.left.normalized_strike
        if np.any(left):
            values[left] = self._tail_value(z[left], "left", 0)
        return values

    def log_asymptotic_coefficient(self, side):
        if side == "left":
            anchor = self.left
            return float(
                np.log(anchor.time_value)
                - anchor.power * np.log(anchor.normalized_strike)
                - np.log1p(anchor.shape)
            )
        if side == "right":
            anchor = self.right
            return float(
                np.log(anchor.time_value)
                + anchor.power
                * (
                    np.log(anchor.normalized_strike)
                    - np.log(anchor.shape)
                )
            )
        raise ValueError("Side must be 'left' or 'right'.")

    def metadata(self):
        a = self.left.normalized_strike
        b = self.right.normalized_strike
        left_call = float(self.normalized_call(a))
        right_call = float(self.normalized_call(b))
        left_slope = float(self.normalized_call(a, 1))
        right_slope = float(self.normalized_call(b, 1))

        left_mass = 1.0 + left_slope
        right_mass = -right_slope
        core_mass = right_slope - left_slope

        left_mean = a * left_mass - self.left.time_value
        right_mean = right_call - b * right_slope
        core_mean = (
            b * right_slope
            - right_call
            - a * left_slope
            + left_call
        )

        return {
            "maturity_years": self.maturity,
            "left_log_moneyness": self.left_log_moneyness,
            "right_log_moneyness": self.right_log_moneyness,
            "left_power": self.left.power,
            "right_power": self.right.power,
            "left_shape": self.left.shape,
            "right_shape": self.right.shape,
            "left_time_value": self.left.time_value,
            "right_time_value": self.right.time_value,
            "left_log_coefficient": self.log_asymptotic_coefficient("left"),
            "right_log_coefficient": self.log_asymptotic_coefficient("right"),
            "left_curvature_ratio": float(
                self._tail_value(np.asarray(a), "left", 2)
                / self.left.core_curvature
            ),
            "right_curvature_ratio": float(
                self._tail_value(np.asarray(b), "right", 2)
                / self.right.core_curvature
            ),
            "left_probability_mass": left_mass,
            "core_probability_mass": core_mass,
            "right_probability_mass": right_mass,
            "total_probability_mass": left_mass + core_mass + right_mass,
            "normalized_mean": left_mean + core_mean + right_mean,
            "right_conditional_mean_over_forward": float(
                b * (1.0 + 1.0 / (self.right.power * self.right.shape))
            ),
            "join_continuity": "Price and slope; curvature may jump.",
            "finite_second_moment": True,
        }


def _check_middle(earlier, later, lower, upper, tolerance):
    cache = {}
    minimum_gap = np.inf
    minimum_bound = np.inf
    checked = 0
    status = "passed"
    stack = [(lower, upper, 0)]

    def evaluate(z):
        nonlocal minimum_gap
        if z not in cache:
            if z < 1.0:
                gap = float(
                    later.normalized_put(z) - earlier.normalized_put(z)
                )
            else:
                gap = float(
                    later.normalized_call(z) - earlier.normalized_call(z)
                )
            cache[z] = (
                gap,
                float(earlier.normalized_call(z, 1)),
                float(later.normalized_call(z, 1)),
            )
            minimum_gap = min(minimum_gap, gap)
        return cache[z]

    while stack:
        left, right, depth = stack.pop()
        checked += 1

        gap_left, early_left, late_left = evaluate(left)
        gap_right, early_right, late_right = evaluate(right)

        if min(gap_left, gap_right) < -tolerance:
            status = "violated"
            break

        width = right - left
        slope_lower = late_left - early_right
        slope_upper = late_right - early_left

        if slope_lower >= 0.0:
            bound = gap_left
        elif slope_upper <= 0.0:
            bound = gap_right
        else:
            bound = (
                min(gap_left, gap_right)
                - 0.5 * width * max(abs(slope_lower), abs(slope_upper))
            )

        bound -= 64.0 * np.finfo(float).eps * max(
            1.0,
            abs(gap_left),
            abs(gap_right),
            width * abs(slope_lower),
            width * abs(slope_upper),
        )

        if bound >= -tolerance:
            minimum_bound = min(minimum_bound, bound)
            continue

        if depth >= 24 or checked >= 50_000:
            status = "unresolved"
            break

        middle = 0.5 * (left + right)
        stack.append((middle, right, depth + 1))
        stack.append((left, middle, depth + 1))

    return {
        "middle_status": status,
        "middle_intervals_checked": checked,
        "minimum_evaluated_middle_gap": float(minimum_gap),
        "minimum_accepted_middle_bound": (
            float(minimum_bound) if np.isfinite(minimum_bound) else None
        ),
    }


def check_coupled_calendar(
    earlier,
    later,
    region="all",
    price_tolerance=1e-12,
    log_ratio_tolerance=1e-12,
):
    """Check infinite tails analytically and the bounded middle continuously."""
    if earlier.maturity >= later.maturity:
        raise ValueError("Calendar checks require increasing maturities.")
    if region not in ("all", "left", "right"):
        raise ValueError("Region must be 'all', 'left' or 'right'.")
    if (
        earlier.left_power != later.left_power
        or earlier.right_power != later.right_power
    ):
        raise ValueError("Calendar checks require shared tail powers.")

    result = {
        "earlier_days": 365.0 * earlier.maturity,
        "later_days": 365.0 * later.maturity,
        "region": region,
    }

    sides = ("left", "right") if region == "all" else (region,)
    tails_pass = True

    for side in sides:
        early_anchor = earlier.left if side == "left" else earlier.right
        late_anchor = later.left if side == "left" else later.right
        common_join = (
            min(
                early_anchor.normalized_strike,
                late_anchor.normalized_strike,
            )
            if side == "left"
            else max(
                early_anchor.normalized_strike,
                late_anchor.normalized_strike,
            )
        )

        coefficient_change = (
            later.log_asymptotic_coefficient(side)
            - earlier.log_asymptotic_coefficient(side)
        )
        join_log_ratio = float(
            later.log_time_value(common_join, side)
            - earlier.log_time_value(common_join, side)
        )
        ordered = bool(
            coefficient_change >= -log_ratio_tolerance
            and join_log_ratio >= -log_ratio_tolerance
        )

        result[f"{side}_tail_ordered"] = ordered
        result[f"{side}_log_coefficient_change"] = coefficient_change
        result[f"{side}_log_ratio_at_common_join"] = join_log_ratio
        tails_pass = tails_pass and ordered

    if not tails_pass:
        result.update(
            middle_status="not_checked",
            middle_intervals_checked=0,
            minimum_evaluated_middle_gap=None,
            minimum_accepted_middle_bound=None,
            passed=False,
        )
        return result

    lower = (
        min(
            earlier.left.normalized_strike,
            later.left.normalized_strike,
        )
        if region in ("all", "left")
        else 1.0
    )
    upper = (
        max(
            earlier.right.normalized_strike,
            later.right.normalized_strike,
        )
        if region in ("all", "right")
        else 1.0
    )

    middle = _check_middle(
        earlier, later, lower, upper, price_tolerance
    )
    result.update(middle)
    result["passed"] = middle["middle_status"] == "passed"
    return result


@dataclass(frozen=True)
class CoupledTailBuilder:
    left_power: float = 3.0
    right_power: float = 1.5
    join_width: float = 0.09
    inner_join_width: float = 0.01
    join_points: int = 401
    preserve_from_days: float = 21.0

    def construct(self, splines):
        splines = tuple(splines)
        maturities = np.array(
            [model.pricer.maturity for model in splines],
            dtype=float,
        )
        if len(splines) < 2 or np.any(np.diff(maturities) <= 0.0):
            raise ValueError("Provide at least two increasing expiry slices.")
        if not 0.0 < self.inner_join_width <= self.join_width:
            raise ValueError("Invalid inner or outer join width.")
        if self.join_points < 3:
            raise ValueError("At least three candidate joins are required.")

        selected = [None] * len(splines)

        for index in range(len(splines) - 1, -1, -1):
            core = splines[index]
            forward = core.pricer.forward

            left_start = max(
                -self.join_width,
                np.log(core.strike_origin / forward) + 1e-8,
            )
            right_start = min(
                self.join_width,
                np.log(core.strike_max / forward) - 1e-8,
            )
            if not left_start < 0.0 < right_start:
                raise ValueError("The fitted curve must contain the forward.")

            preserve = (
                365.0 * core.pricer.maturity
                >= self.preserve_from_days - 1e-10
            )
            left_candidates = (
                np.array([left_start])
                if preserve
                else np.linspace(
                    left_start,
                    max(left_start, -self.inner_join_width),
                    self.join_points,
                )
            )
            right_candidates = (
                np.array([right_start])
                if preserve
                else np.linspace(
                    right_start,
                    min(right_start, self.inner_join_width),
                    self.join_points,
                )
            )

            later = (
                selected[index + 1]
                if index + 1 < len(splines)
                else None
            )

            left_choice = None
            for left_y in left_candidates:
                try:
                    candidate = CoupledTailSlice(
                        core,
                        float(left_y),
                        float(right_start),
                        self.left_power,
                        self.right_power,
                    )
                except ValueError:
                    continue

                if later is None or check_coupled_calendar(
                    candidate, later, region="left"
                )["passed"]:
                    left_choice = float(left_y)
                    break

            if left_choice is None:
                raise ValueError(
                    f"No acceptable left join at "
                    f"{365.0 * core.pricer.maturity:.8g} days."
                )

            for right_y in right_candidates:
                try:
                    candidate = CoupledTailSlice(
                        core,
                        left_choice,
                        float(right_y),
                        self.left_power,
                        self.right_power,
                    )
                except ValueError:
                    continue

                if later is None or check_coupled_calendar(
                    candidate, later, region="right"
                )["passed"]:
                    selected[index] = candidate
                    break

            if selected[index] is None:
                raise ValueError(
                    f"No acceptable right join at "
                    f"{365.0 * core.pricer.maturity:.8g} days."
                )

        checks = tuple(
            check_coupled_calendar(earlier, later)
            for earlier, later in zip(selected[:-1], selected[1:])
        )
        if not all(check["passed"] for check in checks):
            raise ValueError("The complete calendar check did not pass.")

        return tuple(selected), checks