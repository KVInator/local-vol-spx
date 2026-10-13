"""Curated publication artifacts from saved results; no study or calibration run."""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import subprocess

import pandas as pd

from artifacts import atomic_json, sha256
from explorer import SurfaceExplorer
from history_analysis import HistoricalAnalysis
from results import StudyResults


class PublicationArtifacts:
    """Export small pinned tables and figures through the existing plotting objects."""

    def __init__(self, saved: StudyResults, repository: Path):
        self.saved = saved
        self.repository = Path(repository).resolve()
        if saved.index.get("status") != "execution_complete":
            raise ValueError("Publication requires the completed historical study.")

    def export(self, output: Path, date: str = "2015-06-19", half_width: float = 0.09):
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator

        output = self.saved.safe_output(output)
        figures, tables = output / "figures", output / "tables"
        figures.mkdir(parents=True, exist_ok=True)
        tables.mkdir(parents=True, exist_ok=True)
        pairs = self.saved.table("gain_summary")
        common = self.saved.table("common_strategy_gain")
        coverage = self.saved.table("coverage_summary")
        keys = [
            "holding_sessions",
            "rebalance_sessions",
            "scenario",
            "strategy",
            "mean_block_sessions",
        ]
        fields = keys + [
            "comparisons",
            "entry_dates",
            "strategy_sse",
            "black_sse",
            "gain",
            "gain_ci_low",
            "gain_ci_high",
            "ci_status",
        ]
        pair = pairs.loc[pairs.dimension.eq("overall"), fields]
        shared = common.loc[common.dimension.eq("overall"), fields]
        full = pair.merge(
            shared, on=keys, suffixes=("_pairwise", "_common"), validate="one_to_one"
        )
        full.to_csv(tables / "full_history_gain.csv", index=False)
        coverage.to_csv(tables / "full_history_coverage.csv", index=False)
        for scenario in sorted(full.scenario.unique()):
            data = full.loc[
                full.scenario.eq(scenario)
                & full.mean_block_sessions.eq(20)
                & full.strategy.ne("unhedged")
            ].sort_values(keys[:2] + ["strategy"])
            lines = [
                "| Schedule | Method | Pair N | Pair Gain (%) | Common Gain (%) | Common 95% CI (%) |",
                "| --- | --- | ---: | ---: | ---: | ---: |",
            ]
            names = dict(
                ah_pde="AH PDE",
                empirical_mv="Empirical MV",
                lv_smile="LV smile",
                surface_sticky_delta="Sticky delta",
                surface_sticky_strike="Sticky strike",
            )
            for row in data.itertuples():
                lines.append(
                    f"| {row.holding_sessions}/{row.rebalance_sessions} | {names[row.strategy]} | "
                    f"{row.comparisons_pairwise:,} | {100*row.gain_pairwise:+.2f} | {100*row.gain_common:+.2f} | "
                    f"[{100*row.gain_ci_low_common:+.2f}, {100*row.gain_ci_high_common:+.2f}] |"
                )
            (tables / f"{scenario}_gain.md").write_text("\n".join(lines) + "\n")

        explorer = SurfaceExplorer(self.saved)
        view = explorer.select(date, half_width=half_width)
        if view.status != "ready":
            raise ValueError(f"No publication surface for {date}: {view.message}")
        expiry = min(
            view.tables["carry"].itertuples(),
            key=lambda r: abs(365 * r.maturity_years - 35),
        ).expire_date
        names = dict(
            black="Black",
            ah_pde="AH PDE",
            empirical_mv="Empirical MV",
            lv_smile="LV smile",
            surface_sticky_delta="Sticky delta",
            surface_sticky_strike="Sticky strike",
            unhedged="Unhedged",
        )
        with plt.rc_context(
            {
                "font.size": 16,
                "axes.labelsize": 15,
                "axes.titlesize": 17,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
            }
        ):
            surface = explorer.surface_figure(view)
            surface.set_layout_engine(None)
            surface.set_size_inches(14, 11)
            surface.subplots_adjust(
                left=0.10, right=0.94, bottom=0.06, top=0.90,
                wspace=0.28, hspace=0.32,
            )
            for ax in surface.axes:
                ax.set_title(ax.get_title().replace(": ", ":\n"), fontsize=14)
                for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                    axis.set_major_locator(MaxNLocator(nbins=3))
                    axis.label.set_fontsize(14)
                    axis.set_tick_params(labelsize=11)
            # A tight bounding box omits projected 3D z-axis labels.
            surface.savefig(figures / "daily_surface.png", dpi=220)
            plt.close(surface)
            slices = explorer.slice_figure(view, expiry)
            # Curate four existing panels; calculations remain in SurfaceExplorer.
            axes = list(slices.axes)
            axes[1].remove()
            axes[5].remove()
            slices.set_layout_engine(None)
            slices.set_size_inches(10, 7)
            for ax, box in zip(
                [axes[i] for i in (0, 2, 3, 4)],
                [
                    (0.09, 0.57, 0.38, 0.33),
                    (0.60, 0.57, 0.38, 0.33),
                    (0.09, 0.10, 0.38, 0.33),
                    (0.60, 0.10, 0.38, 0.33),
                ],
            ):
                ax.set_position(box)
                legend = ax.get_legend()
                if legend is not None:
                    for text in legend.get_texts():
                        text.set_fontsize(11)
            slices.savefig(figures / "daily_slice.png", dpi=220, bbox_inches="tight")
            plt.close(slices)
            analysis = HistoricalAnalysis(self.saved)
            analysis.build_widget()
            analysis.sample.value = "common"
            gain = analysis.figures[0]
            gain.axes[0].set_title("Common-entry squared funded errors: hedged methods")
            gain.axes[0].set_yticklabels(
                [
                    names.get(str(text.get_text()).split(" / ")[0], text.get_text())
                    for text in gain.axes[0].get_yticklabels()
                ]
            )
            gain.savefig(figures / "history_gain.png", dpi=220, bbox_inches="tight")
            cov = self.saved.plot_coverage()
            cov.set_size_inches(10, 6)
            ax = cov.axes[0]
            ax.set_xticklabels(
                [
                    names.get(text.get_text(), text.get_text())
                    for text in ax.get_xticklabels()
                ],
                rotation=20,
            )
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(
                handles,
                [label.replace("_", " ") for label in labels],
                loc="upper center",
                bbox_to_anchor=(0.5, -0.25),
                ncol=3,
                fontsize=11,
            )
            cov.savefig(figures / "history_coverage.png", dpi=220, bbox_inches="tight")
            plt.close(cov)
        inputs = dict(self.saved.inputs)
        for _, pins in explorer.cache.values():
            inputs.update(pins)
        for filename, digest in inputs.items():
            if sha256(filename) != digest:
                raise ValueError(f"Publication input changed: {filename}")
        artifacts = [
            figures / name
            for name in (
                "daily_surface.png",
                "daily_slice.png",
                "history_gain.png",
                "history_coverage.png",
            )
        ]
        artifacts += [
            tables / "full_history_gain.csv",
            tables / "full_history_coverage.csv",
        ]
        artifacts += list(tables.glob("*_gain.md"))
        atomic_json(
            tables / "publication_provenance.json",
            dict(
                frozen_fingerprint=self.saved.freeze["fingerprint"],
                figure_scope=dict(
                    date=date,
                    root="UNKNOWN",
                    log_strike_domain=[-half_width, half_width],
                    slice_expiry=expiry,
                    volatility_units="annualized percent",
                    price_units="index points",
                    holding_sessions=5,
                    rebalance_sessions=1,
                    scenario="mid",
                    gain_sample="common",
                    mean_block_sessions=20,
                    coverage_sample="all selected entries per method",
                ),
                input_sha256={
                    str(Path(path).relative_to(self.repository)): digest
                    for path, digest in inputs.items()
                },
                source_sha256={
                    str(path.relative_to(self.repository)): sha256(path)
                    for path in sorted((self.repository / "src").glob("*.py"))
                },
                output_sha256={
                    str(path.relative_to(self.repository)): sha256(path)
                    for path in artifacts
                },
            ),
        )
        return full


class ReportRenderer:
    """Typeset the authoritative Markdown through Pandoc and Tectonic in scratch."""

    def __init__(
        self, repository: Path, pandoc: str = "pandoc", tectonic: str = "tectonic"
    ):
        self.repository = Path(repository).resolve()
        self.pandoc = shutil.which(pandoc)
        self.tectonic = shutil.which(tectonic)
        if self.pandoc is None or self.tectonic is None:
            raise FileNotFoundError(
                "Install native Pandoc and Tectonic to render the report."
            )

    def render(self) -> Path:
        root = self.repository
        inputs = [root / "docs" / name for name in ("report.md", "report.yaml", "report.lua")]
        inputs += sorted((root / "docs/figures").glob("*.png"))
        pins = {str(path.relative_to(root)): sha256(path) for path in inputs}
        build = root / "scratch/build/report"
        build.mkdir(parents=True, exist_ok=True)
        # Tectonic resolves image references beside its input TeX file.
        shutil.copytree(root / "docs/figures", build / "figures", dirs_exist_ok=True)
        cache = root / "scratch/cache/tectonic"
        cache.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, TECTONIC_CACHE_DIR=str(cache))
        commands = [
            (
                [
                    self.pandoc,
                    str(root / "docs/report.md"),
                    "--from=markdown+tex_math_dollars",
                    "--standalone",
                    "--toc",
                    "--number-sections",
                    f"--metadata-file={root / 'docs/report.yaml'}",
                    f"--lua-filter={root / 'docs/report.lua'}",
                    f"--resource-path={root / 'docs'}",
                    f"--output={build / 'report.tex'}",
                ],
                root,
            ),
            (
                [
                    self.tectonic,
                    "--keep-logs",
                    "--keep-intermediates",
                    "--outdir",
                    str(build),
                    str(build / "report.tex"),
                ],
                root / "docs",
            ),
        ]
        logs = []
        for command, working in commands:
            result = subprocess.run(
                command, cwd=working, env=environment, text=True, capture_output=True
            )
            logs.append(result.stdout + result.stderr)
            (build / "render.log").write_text("\n".join(logs))
            if result.returncode:
                raise RuntimeError(
                    f"Report rendering failed; inspect {build / 'render.log'}"
                )
        output = root / "docs/report.pdf"
        for filename, digest in pins.items():
            if sha256(root / filename) != digest:
                raise ValueError(f"Report input changed: {filename}")
        shutil.copy2(build / "report.pdf", output)
        atomic_json(
            root / "docs/tables/report_provenance.json",
            dict(
                input_sha256=pins,
                source_sha256={
                    name: sha256(root / name)
                    for name in ("src/publication.py", "scripts/build_publication.py", "requirements.txt")
                },
                output_sha256={"docs/report.pdf": sha256(output)},
                tools={
                    name: subprocess.check_output([executable, "--version"], text=True).splitlines()[0]
                    for name, executable in (("pandoc", self.pandoc), ("tectonic", self.tectonic))
                },
            ),
        )
        return output
