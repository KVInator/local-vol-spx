from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def make_3d_surface(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
    z_label: str,
) -> go.Figure:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
                hovertemplate=
                f"{x_label}: %{{x:.4f}}<br>"
                f"{y_label}: %{{y:.4f}}<br>"
                f"{z_label}: %{{z:.4f}}<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label,
            aspectmode="manual",
            aspectratio=dict(x=1.15, y=1.0, z=0.75),
            camera=dict(
                eye=dict(x=1.65, y=1.45, z=0.85)
            ),
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        height=720,
    )
    return fig


def make_smile_slice_figure(
    x: np.ndarray,
    curves: dict[str, np.ndarray],
    title: str,
    x_label: str,
    y_label: str,
) -> go.Figure:
    fig = go.Figure()
    x = np.asarray(x, dtype=float)

    for name, values in curves.items():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=np.asarray(values, dtype=float),
                mode="lines",
                name=name,
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=560,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="v"),
    )
    return fig


def save_interactive_html(fig: go.Figure, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path), include_plotlyjs="cdn")


def synthetic_demo_surface(
    y_min: float = -0.20,
    y_max: float = 0.20,
    n_y: int = 50,
    maturities: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maturities is None:
        maturities = np.array(
            [7, 14, 21, 30, 45, 60, 75, 90, 120, 150, 180, 240, 300, 360],
            dtype=float,
        ) / 365.0

    y = np.linspace(y_min, y_max, n_y)
    T = np.asarray(maturities, dtype=float)
    Y, TT = np.meshgrid(y, T)

    base = 0.16 + 0.06 * np.exp(-2.8 * TT)
    skew = -0.14 * Y * np.exp(-1.3 * TT)
    wing = 0.45 * (Y ** 2) * (0.7 + 0.3 * np.exp(-TT))
    short_dated_bump = 0.028 * np.exp(-((Y + 0.09) / 0.045) ** 2) * np.exp(-((TT - 0.18) / 0.12) ** 2)
    term_lift = 0.008 * (1.0 - np.exp(-2.0 * TT))

    sigma = base + skew + wing + short_dated_bump + term_lift
    return y, T, sigma