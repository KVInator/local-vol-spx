"""Saved output integrity, relocation and analysis of a small artificial study."""

from pathlib import Path
import shutil
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fixtures import study_research_repository
from results import StudyResults
from study import HistoricalStudy


class ResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.repo, config, cls.dates = study_research_repository(cls.temporary.name)
        cls.study = HistoricalStudy(config, cls.repo)
        cls.study.prepare_calendar()
        cls.study.run(progress=lambda _: None)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_original_saved_format_can_be_read(self):
        saved = StudyResults(self.study.output)
        self.assertEqual(saved.dates, tuple(self.dates))
        self.assertTrue(saved.summary_ready)
        self.assertGreater(len(saved.table("strategy_results")), 0)
        model = saved.model(self.dates[0])
        self.assertTrue(
            np.isfinite(
                model.implied_volatility(np.array([100.0]), model.maturities[0])
            ).all()
        )

    def test_changed_output_is_rejected(self):
        path = self.study.output / "summary" / "coverage_summary.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "Checksum"):
                StudyResults(self.study.output).table("coverage_summary")
        finally:
            path.write_bytes(original)

    def test_relocated_study_uses_its_own_cache(self):
        with tempfile.TemporaryDirectory() as parent:
            copied = Path(parent) / "study"
            shutil.copytree(self.study.output, copied)
            alias = Path(parent) / "study_alias"
            alias.symlink_to(copied, target_is_directory=True)
            for folder in (copied, alias):
                with self.subTest(folder=folder):
                    saved = StudyResults(folder)
                    self.assertEqual(saved.folder, copied.resolve())
                    self.assertTrue(
                        saved.dated_folder(self.dates[0]).is_relative_to(copied.resolve())
                    )
                    self.assertEqual(saved.model(self.dates[0]).spot, 100.0)

    def test_analysis_reads_models_and_exports_outside_the_study(self):
        saved = StudyResults(self.study.output)
        evidence = saved.surface_evidence(self.dates[0])
        self.assertIn("surface_samples", evidence)
        figure = saved.plot_surface(self.dates[0])
        self.assertEqual(len(figure.axes), 6)
        plt.close(figure)
        with tempfile.TemporaryDirectory() as target:
            saved.export(target)
            self.assertTrue((Path(target) / "gain.png").exists())
            self.assertTrue((Path(target) / "results.md").exists())
        with self.assertRaisesRegex(ValueError, "separate"):
            saved.export(saved.folder / "report")


if __name__ == "__main__":
    unittest.main()
