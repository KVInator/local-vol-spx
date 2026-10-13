"""Date controls, display failures, and independent aggregate-statistic checks."""
from datetime import date
from pathlib import Path
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artifacts import ArtifactCache, atomic_json, sha256, signature
from explorer import SurfaceExplorer
from fixtures import study_research_repository
from history_analysis import HistoricalAnalysis
from results import StudyResults, SavedValidationResults
from study import HistoricalStudy


class AggregateChecks(unittest.TestCase):
    def test_zero_black_sse_has_no_defined_gain(self):
        daily = pd.DataFrame(
            dict(
                strategy=["ah_pde"],
                dimension=["overall"],
                group=["all"],
                quote_date=["2020-03-02"],
                rows=[1],
                strategy_sse=[4],
                black_sse=[0],
                strategy_sum=[2],
                black_sum=[0],
                strategy_absolute=[2],
                black_absolute=[0],
            )
        )
        result = HistoricalAnalysis.aggregate_daily(daily).iloc[0]
        self.assertTrue(np.isnan(result.gain))
        self.assertEqual(result.gain_status, "zero_black_sse")

    def test_gain_uses_summed_sse_and_counts(self):
        daily = pd.DataFrame(
            dict(
                strategy=["ah_pde"] * 2,
                dimension=["overall"] * 2,
                group=["all"] * 2,
                quote_date=["2020-03-02", "2020-03-03"],
                rows=[1, 3],
                strategy_sse=[4, 100],
                black_sse=[1, 100],
                strategy_sum=[2, 10],
                black_sum=[1, 10],
                strategy_absolute=[2, 10],
                black_absolute=[1, 10],
            )
        )
        result = HistoricalAnalysis.aggregate_daily(daily).iloc[0]
        self.assertAlmostEqual(result.gain, 1 - 104 / 101)
        self.assertEqual(result.comparisons, 4)
        self.assertEqual(result.entry_dates, 2)
        self.assertAlmostEqual(result.strategy_rms, np.sqrt(26))
        self.assertNotIn("gain_ci_low", result.index)

    def test_cache_rejects_changed_identity_and_escaping_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            identity = {"date": "2020-03-02"}
            atomic_json(
                folder / "cache.json",
                dict(
                    identity=identity, signature=signature(identity), output_sha256={}
                ),
            )
            ArtifactCache.inspect(folder)
            audit = json.loads((folder / "cache.json").read_text())
            audit["identity"]["date"] = "2020-03-03"
            atomic_json(folder / "cache.json", audit)
            with self.assertRaisesRegex(ValueError, "identity"):
                ArtifactCache.inspect(folder)
            audit["signature"] = signature(audit["identity"])
            audit["output_sha256"] = {"../unknown.csv": "bad"}
            atomic_json(folder / "cache.json", audit)
            with self.assertRaisesRegex(ValueError, "escapes"):
                ArtifactCache.inspect(folder)

    def test_validation_scope_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pd.DataFrame(
                dict(quote_date=["2015-06-19"], root=["UNKNOWN"], delta=[0.5])
            ).to_csv(folder / "panel.csv", index=False)
            atomic_json(
                folder / "audit.json",
                dict(
                    date="2015-06-19",
                    root="UNKNOWN",
                    output_sha256={"panel.csv": sha256(folder / "panel.csv")},
                ),
            )
            reader = SavedValidationResults(folder)
            self.assertTrue(reader.table("panel", "2020-03-16").empty)
            self.assertEqual(len(reader.table("panel", "2015-06-19")), 1)
            with (folder / "panel.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                reader.table("panel", "2015-06-19")


class NotebookControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        repo, config, cls.dates = study_research_repository(
            Path(cls.temp.name) / "fixture"
        )
        cls.study = HistoricalStudy(config, repo)
        cls.study.prepare_calendar()
        cls.study.run(progress=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def tearDown(self):
        plt.close("all")

    def test_unavailable_request_survives_display_changes_and_recovers_explicitly(self):
        explorer = SurfaceExplorer(StudyResults(self.study.output))
        explorer.build_widget(self.dates[0])
        with patch.object(explorer, "render"):
            explorer._refresh()
            explorer.picker.value = date(1900, 1, 1)
            for change in (
                lambda: setattr(explorer.domain, "value", 0.04),
                lambda: setattr(explorer.scale, "value", "log"),
                lambda: explorer._refresh(),
                lambda: explorer._render_changed({"new": None}),
            ):
                change()
                self.assertEqual(explorer.requested_date, "1900-01-01")
                self.assertEqual(explorer.slider.value, "1900-01-01")
                self.assertEqual(explorer.picker.value, date(1900, 1, 1))
                self.assertIsNone(explorer.current)
                self.assertFalse(explorer.output.outputs)
            explorer._updating = True
            explorer.root.options = ["UNKNOWN", "OTHER"]
            explorer._updating = False
            explorer.root.value = "OTHER"
            self.assertIsNone(explorer.current)
            self.assertIn("1900-01-01", explorer.status.value)
            explorer.picker.value = date.fromisoformat(self.dates[0])
            self.assertEqual(explorer.current.date, self.dates[0])
            explorer.picker.value = date(1900, 1, 1)
            explorer.slider.value = self.dates[1]
            self.assertEqual(explorer.current.date, self.dates[1])
            self.assertEqual(explorer.picker.value.isoformat(), self.dates[1])
            explorer.picker.value = date(1900, 1, 1)
            explorer.next.click()
            self.assertEqual(explorer.current.date, self.dates[0])

    def test_missing_optional_folders_do_not_block_required_study(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing_greek_panel"
            explorer = SurfaceExplorer(StudyResults(self.study.output), [missing])
            view = explorer.select(self.dates[0])
            self.assertEqual(view.status, "ready")
            self.assertIsNone(view.ssvi)
            self.assertEqual(
                view.tables["evidence_availability"].status.tolist(),
                ["unavailable_missing_folder"],
            )

    def test_present_optional_producers_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            saved = StudyResults(self.study.output)
            with self.assertRaises(FileNotFoundError):
                SurfaceExplorer(saved, [folder])
            (folder / "audit.json").write_text("{malformed")
            with self.assertRaises(json.JSONDecodeError):
                SurfaceExplorer(saved, [folder])
            atomic_json(
                folder / "audit.json", dict(input_sha256={"freeze.json": "wrong"})
            )
            with self.assertRaisesRegex(ValueError, "does not pin"):
                SurfaceExplorer(saved, [folder])
            panel = folder / "greek_refinement.csv"
            pd.DataFrame(
                dict(quote_date=[self.dates[0]], root=["UNKNOWN"], delta_change=[0.0])
            ).to_csv(panel, index=False)
            atomic_json(
                folder / "audit.json",
                dict(
                    date=self.dates[0],
                    root="UNKNOWN",
                    input_sha256={"freeze.json": sha256(saved.folder / "freeze.json")},
                    output_sha256={panel.name: sha256(panel)},
                ),
            )
            explorer = SurfaceExplorer(saved, [folder])
            with panel.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                explorer.select(self.dates[0])

    def test_widget_date_changes_are_synchronized_without_numerical_jobs(self):
        explorer = SurfaceExplorer(StudyResults(self.study.output), cache_size=2)
        explorer.build_widget(self.dates[0])
        with (
            patch(
                "calibration.DailyAHCalibrator.calibrate",
                side_effect=AssertionError("browse calibrated"),
            ),
            patch(
                "pde.AHBackwardPricer.solve_many",
                side_effect=AssertionError("browse solved PDE"),
            ),
            patch.object(
                HistoricalStudy, "run", side_effect=AssertionError("browse ran study")
            ),
            patch.object(explorer, "render") as render,
        ):
            explorer._refresh()
            first = explorer.current
            explorer.next.click()
            self.assertEqual(explorer.current.date, self.dates[1])
            self.assertEqual(explorer.picker.value.isoformat(), self.dates[1])
            explorer.picker.value = date.fromisoformat(self.dates[2])
            self.assertEqual(explorer.current.date, self.dates[2])
            self.assertEqual(explorer.slider.value, self.dates[2])
            self.assertLessEqual(len(explorer.cache), 2)
            self.assertNotIn((first.date, "UNKNOWN", 0.09), explorer.cache)
            for frame in explorer.current.tables.values():
                if "quote_date" in frame:
                    self.assertTrue(frame.quote_date.eq(self.dates[2]).all())
            self.assertEqual(render.call_count, 3)
            explorer.picker.value = date(1900, 1, 1)
            self.assertIsNone(explorer.current)
            self.assertIn("no saved decisions", explorer.status.value)
            explorer.picker.value = date.fromisoformat(self.dates[2])
            self.assertEqual(explorer.current.date, self.dates[2])

    def test_rendered_figures_stay_inside_the_clearable_widget_output(self):
        explorer = SurfaceExplorer(StudyResults(self.study.output))
        explorer.build_widget(self.dates[0])
        with patch("IPython.display.display"):
            explorer._refresh()
            self.assertEqual(len(explorer.figures), 2)
            self.assertEqual(plt.get_fignums(), [])
            self.assertGreater(len(explorer.output.outputs), 0)
            explorer.next.click()
            self.assertEqual(explorer.current.date, self.dates[1])
            self.assertEqual(plt.get_fignums(), [])
            rendered = str(explorer.output.outputs)
            self.assertIn(self.dates[1], rendered)
            self.assertNotIn(self.dates[0], rendered)
        analysis = HistoricalAnalysis(StudyResults(self.study.output))
        analysis.build_widget()
        with patch("IPython.display.display"):
            analysis._refresh()
            self.assertEqual(len(analysis.figures), 1)
            self.assertEqual(plt.get_fignums(), [])

    def test_failed_model_preserves_date_and_carry_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "relocated"
            shutil.copytree(self.study.output, copied)
            folder = copied / "cache/dates" / self.dates[0] / "decisions"
            manifest = pd.read_csv(folder / "model_manifest.csv")
            manifest["status"] = "calibration_failed"
            manifest["message"] = "Unavailable carry; insufficient_pairs"
            manifest.to_csv(folder / "model_manifest.csv", index=False)
            audit = json.loads((folder / "cache.json").read_text())
            audit["output_sha256"]["model_manifest.csv"] = sha256(
                folder / "model_manifest.csv"
            )
            atomic_json(folder / "cache.json", audit)
            explorer = SurfaceExplorer(StudyResults(copied))
            view = explorer.select(self.dates[0])
            self.assertEqual(view.status, "model_unavailable")
            self.assertEqual(view.date, self.dates[0])
            self.assertIn("insufficient_pairs", view.message)
            self.assertTrue(view.samples.empty)
            self.assertIsNone(explorer.surface_figure(view))

    def test_cached_selection_rejects_tampering_and_clears_state(self):
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "relocated"
            shutil.copytree(self.study.output, copied)
            explorer = SurfaceExplorer(StudyResults(copied))
            explorer.select(self.dates[0])
            path = explorer.saved.dated_folder(self.dates[0]) / "decisions.csv"
            with path.open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "changed"):
                explorer.select(self.dates[0])
            self.assertIsNone(explorer.current)

    def test_range_rejects_common_sample_and_unsupported_schedule(self):
        analysis = HistoricalAnalysis(StudyResults(self.study.output))
        with self.assertRaisesRegex(ValueError, "recorded scope"):
            analysis.select(
                block=5, sample="common", start=self.dates[0], end=self.dates[1]
            )
        with self.assertRaisesRegex(ValueError, "schedule"):
            analysis.select(holding=20, rebalance=20, block=5)
        range_result = analysis.select(block=5, start=self.dates[0], end=self.dates[1])
        self.assertTrue(range_result["coverage"].empty)
        self.assertNotIn("gain_ci_low", range_result["results"])

    def test_retained_validation_workflows_reproduce_saved_production_delta(self):
        from research_validation import (
            SavedDecisionRefinement,
            GreekPanelWorkflow,
            GreekPanelSettings,
        )

        saved = StudyResults(self.study.output)
        values, changes = SavedDecisionRefinement().run(
            saved, self.dates[0], progress=lambda _: None
        )
        self.assertLess(
            values.loc[values.case.eq("production"), "delta_minus_saved"].abs().max(),
            1e-10,
        )
        self.assertEqual(set(changes.check), {"double_space", "half_cell_shift"})
        with tempfile.TemporaryDirectory() as directory:
            settings = GreekPanelSettings(
                radii=(0.0, 0.005),
                log_strikes=(0.0,),
                spot_changes=(0.0,),
                coarse_intervals=800,
                fine_intervals=1600,
                steps_per_day=4,
            )
            result = GreekPanelWorkflow(settings, workers=2).run(
                saved, self.dates[0], "UNKNOWN", directory
            )
            self.assertEqual(set(result["greek_panel"].radius), {0.0, 0.005})
            self.assertTrue(result["validation_failures"].empty)
            self.assertEqual(
                len(result["panel_contracts"]),
                len(result["panel_contracts"].drop_duplicates()),
            )
            self.assertTrue(
                (Path(directory) / "checkpoints/radius_0/audit.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
