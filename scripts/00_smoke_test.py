from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from lv_project.black_scholes import call_price, implied_vol_call
from lv_project.finite_diff import first_derivative_1d, second_derivative_1d
from lv_project.plotting import synthetic_demo_surface


def run_black_scholes_smoke_test() -> None:
    spot = 100.0
    strike = 100.0
    maturity = 0.5
    rate = 0.03
    dividend_yield = 0.01
    vol = 0.20

    price = call_price(
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
        vol=vol,
    )
    iv = implied_vol_call(
        price=price,
        spot=spot,
        strike=strike,
        maturity=maturity,
        rate=rate,
        dividend_yield=dividend_yield,
    )

    print("BS smoke test")
    print(f"call price      : {price:.8f}")
    print(f"recovered iv    : {iv:.8f}")
    print(f"abs error       : {abs(iv - vol):.12f}")
    print()


def run_fd_smoke_test() -> None:
    x = np.linspace(-1.0, 1.0, 11)
    f = x**3 + 2.0 * x**2 - x + 1.0

    f1_true = 3.0 * x**2 + 4.0 * x - 1.0
    f2_true = 6.0 * x + 4.0

    f1_est = first_derivative_1d(x, f)
    f2_est = second_derivative_1d(x, f)

    print("Finite-difference smoke test")
    print(f"max abs error f'  : {np.max(np.abs(f1_est - f1_true)):.8f}")
    print(f"max abs error f'' : {np.max(np.abs(f2_est - f2_true)):.8f}")
    print()


def run_plotting_smoke_test() -> None:
    y, T, sigma = synthetic_demo_surface()
    print("Plotting smoke test")
    print(f"surface shape    : {sigma.shape}")
    print(f"y range          : [{y.min():.3f}, {y.max():.3f}]")
    print(f"T range          : [{T.min():.3f}, {T.max():.3f}]")
    print()


if __name__ == "__main__":
    run_black_scholes_smoke_test()
    run_fd_smoke_test()
    run_plotting_smoke_test()