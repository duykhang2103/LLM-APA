"""Loader rejects escaped paths and ambiguous/missing input sources."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_grading.data.loader import load_samples


class DataLoaderTests(unittest.TestCase):
    def test_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "samples.json"
            path.write_text(
                json.dumps(
                    [{"sample_id": "a", "code": "x"}, {"sample_id": "a", "code": "y"}]
                )
            )
            with self.assertRaises(ValueError):
                load_samples(path)

    def test_relative_code_and_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code.cpp").write_text("int main() {}")
            path = root / "samples.json"
            path.write_text(json.dumps({"sample_id": "s", "code_file": "code.cpp"}))
            self.assertIn("main", load_samples(path)[0]["code"])
            path.write_text(json.dumps({"sample_id": "s", "code_file": "../other.cpp"}))
            with self.assertRaises(ValueError):
                load_samples(path)
