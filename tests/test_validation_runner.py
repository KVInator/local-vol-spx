"""Independent support/provenance examples for the market audit runner."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    'validate_surface', Path(__file__).resolve().parents[1] / 'scripts/validate_surface.py')
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class SupportProvenanceTests(unittest.TestCase):
    def test_inherited_edge_fill_remains_visible_after_calendar_repair(self):
        result = SimpleNamespace(
            y_grid=np.array([-.1, 0., .1]),
            pillar_maturities=np.array([.1, .2, .3]),
            target_maturities=np.array([.05, .1, .15, .2, .4]),
            expiry_coverage=pd.DataFrame({'y_min_native': [0., -.1, -.1],
                                          'y_max_native': [.1, .1, .1]}),
            pillar_total_variance=np.array([[.03, .02, .01], [.01, .03, .02], [.04, .04, .04]]),
        )
        native, repaired_native, time_ok, target_native, target_repaired, brackets = validation.support_masks(result)
        self.assertTrue(native[1, 0])
        self.assertFalse(repaired_native[1, 0])
        np.testing.assert_array_equal(time_ok, [False, True, True, True, False])
        self.assertTrue(target_native[3, 0])
        self.assertFalse(target_repaired[3, 0])
        self.assertFalse(target_repaired[[0, -1]].any())
        self.assertEqual(brackets[2], (0, 1))

    def test_native_repair_source_can_replace_an_edge_fill(self):
        result = SimpleNamespace(
            y_grid=np.array([-.1, 0., .1]), pillar_maturities=np.array([.1, .2]),
            target_maturities=np.array([.2]),
            expiry_coverage=pd.DataFrame({'y_min_native': [-.1, 0.], 'y_max_native': [.1, .1]}),
            pillar_total_variance=np.array([[.03, .02, .01], [.01, .03, .02]]),
        )
        native, repaired, time_ok, target_native, target_repaired, brackets = validation.support_masks(result)
        self.assertFalse(target_native[0, 0])
        self.assertTrue(target_repaired[0, 0])
        self.assertFalse((target_native & target_repaired)[0, 0])


class PreservationSnapshotTests(unittest.TestCase):
    def test_fresh_snapshot_excludes_local_notes_and_all_validation_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clean = root / 'data/processed/clean.csv'
            baseline = root / 'outputs/diagnostics/baseline.json'
            validation_root = root / 'outputs/diagnostics/validation'
            out = validation_root / 'checkpoint'
            out.mkdir(parents=True)
            for path, content in [(clean, 'original quotes'), (baseline, 'baseline diagnostics'),
                                  (validation_root / 'prior_run.csv', 'old validation output'),
                                  (root / '.local/notes/context.md', 'local context')]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            with patch.object(validation, 'ROOT', root):
                before = validation.preservation_snapshot(out)
                self.assertEqual({r['path'] for r in before},
                                 {'data/processed/clean.csv', 'outputs/diagnostics/baseline.json'})
                self.assertEqual(validation.verify_preservation(before, out)['changed_files'], [])
                clean.write_text('updated quotes')
                self.assertEqual(validation.verify_preservation(before, out)['changed_files'],
                                 ['data/processed/clean.csv'])
                current = validation.preservation_snapshot(out)
                self.assertNotEqual(before[0]['sha256'], current[0]['sha256'])
                self.assertEqual(validation.verify_preservation(current, out)['changed_files'], [])


if __name__ == '__main__':
    unittest.main()
