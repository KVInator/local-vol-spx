"""Saved-summary compatibility and integrity without running a study producer."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from artifacts import sha256
from results import StudyResults


class CompressedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name).resolve()
        (self.folder / "run_index.json").write_text(
            json.dumps(dict(frozen_fingerprint="fixture", dates={}))
        )
        (self.folder / "freeze.json").write_text(
            json.dumps(dict(fingerprint="fixture"))
        )
        summary = self.folder / "summary"
        summary.mkdir()
        self.csv = summary / "strategy_results.csv"
        self.payload = b"quote_date,entry_id,net_pnl\n2020-03-02,001,1.25\n"
        self.csv.write_bytes(self.payload)
        self.audit = summary / "audit.json"
        self.audit.write_text(
            json.dumps(dict(output_sha256={self.csv.name: sha256(self.csv)}))
        )
        self.compressed = self.csv.with_suffix(".csv.gz")

    def compress(self, payload=None):
        self.compressed.write_bytes(
            gzip.compress(self.payload if payload is None else payload, mtime=0)
        )
        self.csv.unlink(missing_ok=True)

    def test_original_and_compressed_studies_have_identical_tables_and_pins(self):
        original_audit = self.audit.read_bytes()
        plain = StudyResults(self.folder).table("strategy_results")
        self.compress()
        saved = StudyResults(self.folder)
        pd.testing.assert_frame_equal(plain, saved.table("strategy_results"))
        self.assertEqual(plain.entry_id.tolist(), ["001"])
        self.assertEqual(self.audit.read_bytes(), original_audit)
        self.assertEqual(saved.inputs[str(self.compressed)], sha256(self.compressed))
        saved.verify_unchanged()

    def test_original_takes_precedence_and_cannot_hide_tampering(self):
        self.compressed.write_bytes(gzip.compress(self.payload))
        saved = StudyResults(self.folder)
        saved.table("strategy_results")
        self.assertIn(str(self.csv), saved.inputs)
        self.csv.write_bytes(self.payload.replace(b"1.25", b"9.25"))
        with self.assertRaisesRegex(ValueError, "Checksum"):
            saved.table("strategy_results")

    def test_valid_gzip_with_changed_csv_is_rejected(self):
        self.compress(self.payload.replace(b"1.25", b"9.25"))
        with self.assertRaisesRegex(ValueError, "Checksum"):
            StudyResults(self.folder).table("strategy_results")

    def test_missing_original_output_pin_is_rejected(self):
        self.compress()
        self.audit.write_text(json.dumps(dict(output_sha256={})))
        with self.assertRaisesRegex(ValueError, "missing output pin"):
            StudyResults(self.folder).table("strategy_results")

    def test_corrupt_and_truncated_gzip_are_rejected(self):
        self.compress()
        content = self.compressed.read_bytes()
        for damaged in (b"invalid gzip", content[:-4], content[:-8] + b"\0" * 8):
            with self.subTest(damaged=damaged):
                self.compressed.write_bytes(damaged)
                with self.assertRaises((OSError, EOFError)):
                    StudyResults(self.folder).table("strategy_results")

    def test_reuse_checks_detect_changed_compressed_container(self):
        self.compress()
        saved = StudyResults(self.folder)
        saved.table("strategy_results")
        self.compressed.write_bytes(gzip.compress(self.payload, mtime=1))
        with self.assertRaisesRegex(ValueError, "Input changed"):
            saved.verify_unchanged()

    def test_compressed_symlink_cannot_escape_summary(self):
        self.compress()
        outside = self.folder / "outside.csv.gz"
        self.compressed.rename(outside)
        self.compressed.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "outside"):
            StudyResults(self.folder).table("strategy_results")


if __name__ == "__main__":
    unittest.main()
