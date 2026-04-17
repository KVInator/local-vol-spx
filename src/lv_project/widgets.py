from __future__ import annotations

import ipywidgets as widgets
from IPython.display import display

from lv_project.plotting import make_3d_surface, make_smile_slice_figure, synthetic_demo_surface


def build_surface_controls() -> dict[str, widgets.Widget]:
    controls = {
        "y_min": widgets.FloatSlider(
            value=-0.20, min=-0.50, max=0.00, step=0.01, description="y min"
        ),
        "y_max": widgets.FloatSlider(
            value=0.20, min=0.00, max=0.50, step=0.01, description="y max"
        ),
        "n_y": widgets.IntSlider(
            value=50, min=20, max=120, step=5, description="grid pts"
        ),
        "show_slices": widgets.IntRangeSlider(
            value=[0, 5], min=0, max=13, step=1, description="slice idx"
        ),
    }
    return controls


def launch_demo_surface_widget() -> None:
    controls = build_surface_controls()

    output = widgets.Output()

    def refresh(*_) -> None:
        with output:
            output.clear_output(wait=True)

            y, T, sigma = synthetic_demo_surface(
                y_min=controls["y_min"].value,
                y_max=controls["y_max"].value,
                n_y=controls["n_y"].value,
            )

            fig_surface = make_3d_surface(
                x=y,
                y=T,
                z=sigma,
                title="Interactive Demo Implied Volatility Surface",
                x_label="Log-Moneyness",
                y_label="Time to Maturity",
                z_label="Implied Volatility",
            )
            fig_surface.show()

            start_idx, end_idx = controls["show_slices"].value
            end_idx = min(end_idx, len(T) - 1)

            curves = {
                f"T={T[i]:.3f}": sigma[i, :]
                for i in range(start_idx, end_idx + 1)
            }

            fig_slices = make_smile_slice_figure(
                x=y,
                curves=curves,
                title="Smile Slices by Maturity",
                x_label="Log-Moneyness",
                y_label="Implied Volatility",
            )
            fig_slices.show()

    for control in controls.values():
        control.observe(refresh, names="value")

    panel = widgets.VBox(
        [
            widgets.HTML("<h3>Surface Playground Controls</h3>"),
            controls["y_min"],
            controls["y_max"],
            controls["n_y"],
            controls["show_slices"],
        ]
    )

    refresh()
    display(widgets.HBox([panel]))
    display(output)