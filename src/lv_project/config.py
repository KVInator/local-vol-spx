from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
OUTPUT_TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUTPUT_DIAGNOSTICS_DIR = PROJECT_ROOT / "outputs" / "diagnostics"
OUTPUT_INTERACTIVE_DIR = PROJECT_ROOT / "outputs" / "interactive"


def default_target_maturities_years() -> np.ndarray:
    days = np.array(
        [7, 14, 21, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180, 210, 240, 270, 300, 330, 360],
        dtype=float,
    )
    return days / 365.0


@dataclass(frozen=True)
class SurfaceGridConfig:
    y_min: float = -0.20
    y_max: float = 0.20
    n_y: int = 50
    maturities_years: np.ndarray = field(default_factory=default_target_maturities_years)

    @property
    def y_grid(self) -> np.ndarray:
        return np.linspace(self.y_min, self.y_max, self.n_y)


@dataclass(frozen=True)
class FilterConfig:
    min_days_to_expiry: int = 7
    max_days_to_expiry: int = 365
    min_bid: float = 0.05
    max_bid_ask_spread_ratio: float = 0.35
    min_log_moneyness: float = -0.35
    max_log_moneyness: float = 0.35


@dataclass(frozen=True)
class WidgetDefaults:
    quote_date: str = "2025-08-27"
    y_min: float = -0.20
    y_max: float = 0.20
    n_y: int = 50
    clip_floor: float = 0.05
    clip_cap: float = 1.20
    apply_calendar_repair: bool = True
    apply_butterfly_checks: bool = True