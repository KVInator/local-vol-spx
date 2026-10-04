"""Extend a fitted call surface from zero to its first maturity."""

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.special import ndtr

from black import BlackPricer
from local_vol import DupireLocalVolatility
from vol_surface import NormalizedCallSurface


@dataclass(frozen=True)
class ShortEndSurface:
    """Scale the first smile's total variance proportionally with time."""

    surface: NormalizedCallSurface
    spot: float

    _cached_anchor: Callable = field(
        init=False, repr=False, compare=False
    )
    _market_local_vol: DupireLocalVolatility = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self):
        spot = float(self.spot)

        if not np.isfinite(spot) or spot <= 0.0:
            raise ValueError("Spot must be finite and positive.")

        if self.first_maturity <= 0.0:
            raise ValueError("The first fitted maturity must be positive.")

        object.__setattr__(self, "spot", spot)
        object.__setattr__(
            self,
            "_cached_anchor",
            lru_cache(maxsize=8)(self._calculate_anchor),
        )
        object.__setattr__(
            self,
            "_market_local_vol",
            DupireLocalVolatility(self.surface),
        )

    @property
    def first_maturity(self):
        return float(self.surface.maturities[0])

    @property
    def maturities(self):
        return np.concatenate(
            [np.array([0.0]), self.surface.maturities]
        )

    @property
    def min_log_moneyness(self):
        return self.surface.min_log_moneyness

    @property
    def max_log_moneyness(self):
        return self.surface.max_log_moneyness

    def _checked_time(self, time):
        time = float(time)

        if (
            not np.isfinite(time)
            or time < 0.0
            or time > float(self.surface.maturities[-1])
        ):
            raise ValueError("Maturity is outside the supported range.")

        return time

    def _checked_strikes(self, normalized_strikes):
        z = np.asarray(normalized_strikes, dtype=float)

        if np.any(~np.isfinite(z)) or np.any(z <= 0.0):
            raise ValueError(
                "Normalized strikes must be finite and positive."
            )

        y = np.log(z)
        tolerance = 1e-12

        if (
            np.any(y < self.min_log_moneyness - tolerance)
            or np.any(y > self.max_log_moneyness + tolerance)
        ):
            raise ValueError(
                "Normalized strikes are outside the supported domain."
            )

        return z

    @staticmethod
    def _checked_side(side):
        if side not in ("left", "right"):
            raise ValueError("Derivative side must be 'left' or 'right'.")

    def forward(self, time):
        time = self._checked_time(time)

        if time >= self.first_maturity:
            return self.surface.forward(time)

        first_forward = self.surface.forward(self.first_maturity)
        fraction = time / self.first_maturity

        return float(
            self.spot
            * np.exp(fraction * np.log(first_forward / self.spot))
        )

    def discount_factor(self, time):
        time = self._checked_time(time)

        if time >= self.first_maturity:
            return self.surface.discount_factor(time)

        first_discount = self.surface.discount_factor(
            self.first_maturity
        )

        return float(
            np.exp(
                time / self.first_maturity * np.log(first_discount)
            )
        )

    def _calculate_anchor(self, strike_key):
        """Recover total variance and its log-strike derivatives."""
        z = np.asarray(strike_key, dtype=float)
        y = np.log(z)
        time = self.first_maturity
        forward = self.surface.forward(time)

        sigma = np.asarray(
            self.surface.implied_volatility(forward * z, time),
            dtype=float,
        )
        w = sigma**2 * time

        if np.any(~np.isfinite(w)) or np.any(w <= 0.0):
            raise ValueError(
                "The anchor smile requires positive finite variance."
            )

        root_w = np.sqrt(w)
        d1 = -y / root_w + 0.5 * root_w
        d2 = d1 - root_w
        phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

        black_w = phi / (2.0 * root_w)

        if np.any(black_w <= 0.0):
            raise ValueError(
                "Anchor price sensitivity is too small for IV derivatives."
            )

        black_y = -z * ndtr(d2)
        black_yy = black_y + phi / root_w

        d1_w = y / (2.0 * w * root_w) + 1.0 / (4.0 * root_w)
        d2_w = y / (2.0 * w * root_w) - 1.0 / (4.0 * root_w)

        black_yw = -phi * d2_w
        black_ww = black_w * (
            -d1 * d1_w - 1.0 / (2.0 * w)
        )

        call_z = self.surface.normalized_call(
            z, time, derivative=1
        )
        call_zz = self.surface.normalized_call(
            z, time, derivative=2
        )

        call_y = z * call_z
        call_yy = z**2 * call_zz + call_y

        w_y = (call_y - black_y) / black_w
        w_yy = (
            call_yy
            - black_yy
            - 2.0 * black_yw * w_y
            - black_ww * w_y**2
        ) / black_w

        values = (w, w_y, w_yy)

        if any(np.any(~np.isfinite(value)) for value in values):
            raise ValueError("Anchor smile derivatives are not finite.")

        for value in values:
            value.setflags(write=False)

        return values

    def _anchor(self, z):
        values = self._cached_anchor(tuple(z.ravel()))
        return tuple(value.reshape(z.shape) for value in values)

    def _short_geometry(self, z, time):
        w_first, w_y_first, w_yy_first = self._anchor(z)
        y = np.log(z)
        fraction = time / self.first_maturity

        density_factor = (
            (1.0 - y * w_y_first / (2.0 * w_first)) ** 2
            + fraction
            * (
                0.5 * w_yy_first
                - w_y_first**2 / (4.0 * w_first)
            )
            - fraction**2 * w_y_first**2 / 16.0
        )

        if (
            np.any(~np.isfinite(density_factor))
            or np.any(density_factor <= 0.0)
        ):
            raise ValueError(
                "Short-end density factor must be finite and positive."
            )

        return (
            w_first,
            fraction * w_y_first,
            fraction * w_first,
            density_factor,
        )

    def normalized_call(self, normalized_strikes, time, derivative=0):
        time = self._checked_time(time)
        z = self._checked_strikes(normalized_strikes)

        if derivative not in (0, 1, 2):
            raise ValueError("Supported strike derivative orders are 0–2.")

        if time == 0.0:
            if derivative != 0:
                raise ValueError(
                    "The zero-maturity payoff has a strike kink."
                )
            return np.maximum(1.0 - z, 0.0)

        if time >= self.first_maturity:
            return self.surface.normalized_call(
                z, time, derivative=derivative
            )

        _, w_y, w, density_factor = self._short_geometry(z, time)
        root_w = np.sqrt(w)
        d1 = -np.log(z) / root_w + 0.5 * root_w
        d2 = d1 - root_w
        phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

        if derivative == 0:
            sigma = np.sqrt(w / time)
            return BlackPricer(1.0, 1.0, time).price(
                z, sigma, "call"
            )

        if derivative == 1:
            return -ndtr(d2) + phi * w_y / (2.0 * z * root_w)

        return phi * density_factor / (z**2 * root_w)

    def normalized_time_derivative(
        self, normalized_strikes, time, side="right"
    ):
        time = self._checked_time(time)
        z = self._checked_strikes(normalized_strikes)
        self._checked_side(side)

        if time == 0.0:
            raise ValueError(
                "The payoff's maturity derivative is undefined at zero."
            )

        use_short_end = (
            time < self.first_maturity
            or (time == self.first_maturity and side == "left")
        )

        if not use_short_end:
            return self.surface.normalized_time_derivative(
                z, time, side=side
            )

        w_first, _, w, _ = self._short_geometry(z, time)
        root_w = np.sqrt(w)
        d1 = -np.log(z) / root_w + 0.5 * root_w
        phi = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

        return phi * w_first / (
            2.0 * root_w * self.first_maturity
        )

    def normalized_variance(
        self, normalized_strikes, time, side="right"
    ):
        """Return local variance, including its right limit at time zero."""
        time = self._checked_time(time)
        z = self._checked_strikes(normalized_strikes)
        self._checked_side(side)

        if time == 0.0 and side == "left":
            raise ValueError("There is no left-hand side at time zero.")

        use_short_end = (
            time < self.first_maturity
            or (time == self.first_maturity and side == "left")
        )

        if not use_short_end:
            return self._market_local_vol.normalized_variance(
                z, time, side=side
            )

        w_first, _, _, density_factor = self._short_geometry(z, time)

        return (
            w_first / self.first_maturity / density_factor
        )

    def call_price(self, strikes, time):
        time = self._checked_time(time)
        strikes = np.asarray(strikes, dtype=float)
        forward = self.forward(time)
        z = self._checked_strikes(strikes / forward)

        if time == 0.0:
            return np.maximum(self.spot - strikes, 0.0)

        discount = self.discount_factor(time)

        return (
            discount
            * forward
            * self.normalized_call(z, time)
        )

    def implied_volatility(self, strikes, time):
        time = self._checked_time(time)
        z = self._checked_strikes(
            np.asarray(strikes, dtype=float) / self.forward(time)
        )

        if time == 0.0:
            raise ValueError(
                "Implied volatility is not identified at zero maturity."
            )

        if time >= self.first_maturity:
            return self.surface.implied_volatility(strikes, time)

        w_first, _, _ = self._anchor(z)
        return np.sqrt(w_first / self.first_maturity)

    def metadata(self):
        return {
            "format_version": 1,
            "model_type": "first_smile_short_end_extension",
            "spot": self.spot,
            "first_maturity_years": self.first_maturity,
            "min_log_moneyness": self.min_log_moneyness,
            "max_log_moneyness": self.max_log_moneyness,
            "short_end_method": "w(y,T)=(T/T1)*w(y,T1)",
            "short_end_iv": "Constant in time at fixed forward moneyness.",
            "short_end_carry": (
                "Log-linear forward from spot to the first forward; "
                "log-linear discount factor from 1 to the first discount."
            ),
            "zero_maturity_prices": "Intrinsic payoff.",
            "zero_maturity_local_variance": "Right-hand limit.",
            "global_tail_extension": False,
        }

    @classmethod
    def load(cls, path):
        path = Path(path)
        specification = json.loads(path.read_text(encoding="utf-8"))

        if (
            specification.get("format_version") != 1
            or specification.get("model_type")
            != "first_smile_short_end_extension"
        ):
            raise ValueError("Unsupported short-end model specification.")

        base_path = path.parent / specification["base_surface_file"]

        return cls(
            surface=NormalizedCallSurface.load(base_path),
            spot=specification["spot"],
        )