"""CPU-free checks for Submission 04's per-ID prediction recovery plumbing."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import nbformat


NOTEBOOK = Path(__file__).resolve().parents[4] / "CS781_Phase1_Submission04_Infrastructure_FIXED.ipynb"


def _load_generation_function(namespace):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    tree = ast.parse(notebook.cells[35].source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "generate_official_predictions")
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, str(NOTEBOOK), "exec"), namespace)
    return namespace["generate_official_predictions"]


def _load_packaging_functions(namespace):
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    tree = ast.parse(notebook.cells[37].source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"validate_predictions", "create_submission_zip"}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(NOTEBOOK), "exec"), namespace)
    return namespace["create_submission_zip"]


class InferenceRecoveryTest(unittest.TestCase):
    def test_package_requires_all_ids_and_root_level_prediction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prediction_dir = root / "predictions"
            prediction_dir.mkdir()
            (prediction_dir / "inference_provenance.json").write_text(
                json.dumps({"checkpoint_id": "fixture"}), encoding="utf-8"
            )

            def atomic_write_json(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")

            def validate_zip(path):
                with zipfile.ZipFile(path) as archive:
                    return {"members": archive.namelist()}

            namespace = {
                "Path": Path, "json": json, "zipfile": zipfile, "tempfile": tempfile,
                "os": os, "Any": object, "Sequence": list, "Mapping": dict,
                "CFG": SimpleNamespace(decoding={"target_words": 128}),
                "DIRS": {"predictions": prediction_dir},
                "words": lambda value: value.split(),
                "atomic_write_json": atomic_write_json,
                "validate_zip": validate_zip,
                "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "generation_quality_report": lambda value: {"word_count": len(value.split())},
            }
            package = _load_packaging_functions(namespace)
            prediction = {"text": " ".join(f"word{i}" for i in range(128))}
            with self.assertRaisesRegex(ValueError, "missing IDs"):
                package({"a": prediction}, ["a", "b"], root / "submission")
            self.assertFalse((root / "submission" / "submission.zip").exists())
            zip_path, report = package({"a": prediction, "b": prediction},
                                       ["a", "b"], root / "submission")
            self.assertTrue(report["passed"])
            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(archive.namelist(), ["prediction.json"])
                self.assertEqual(set(json.loads(archive.read("prediction.json"))), {"a", "b"})

    def test_failed_ids_resume_with_new_rounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoints" / "selected_model.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"verified-checkpoint-fixture")
            predictions_dir = root / "predictions"
            predictions_dir.mkdir()
            records = [{"id": item, "prompt": item} for item in ("a", "b", "c")]
            attempts = {}
            text = " ".join(f"word{i}" for i in range(128))

            def atomic_write_json(path, value):
                Path(path).write_text(json.dumps(value), encoding="utf-8")

            def append_event(path, event):
                with Path(path).open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(event) + "\n")

            def continuation(_model, _tokenizer, prompt, _device, **_kwargs):
                attempts[prompt] = attempts.get(prompt, 0) + 1
                if prompt == "b" and attempts[prompt] <= 3:
                    raise RuntimeError("transient b")
                if prompt == "c" and attempts[prompt] <= 9:
                    raise RuntimeError("transient c")
                return text, {"word_count": 128}

            cfg = SimpleNamespace(decoding={
                "target_words": 128,
                "max_new_tokens": 512,
                "recovery_rounds": 3,
                "retry_seed_stride": 100_003,
                "policies": [
                    {"name": "first", "seed": 781},
                    {"name": "second", "seed": 1781},
                    {"name": "length_only", "seed": 1781},
                ],
            })
            namespace = {
                "CFG": cfg,
                "DIRS": {"checkpoints": checkpoint.parent, "predictions": predictions_dir},
                "Path": Path, "json": json, "hashlib": hashlib, "re": re, "time": time,
                "tqdm": lambda items, **_kwargs: items,
                "sha256_file": lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "test_records_fingerprint": lambda rows: hashlib.sha256(
                    json.dumps(rows, sort_keys=True).encode()
                ).hexdigest(),
                "MODEL_SOURCE_SHA256": "model-source-fixture",
                "atomic_write_json": atomic_write_json,
                "append_event": append_event,
                "efficient_continuation": continuation,
                "words": lambda value: value.split(),
                "utc_now": lambda: "2026-10-03T00:00:00Z",
            }
            generate = _load_generation_function(namespace)
            arguments = (None, None, records, "cpu")
            keywords = {"checkpoint_id": "selected-fixture",
                        "tokenizer_meta": {"fingerprint": "tokenizer-fixture"}}
            with self.assertRaisesRegex(RuntimeError, "1 official IDs remain invalid"):
                generate(*arguments, **keywords)
            partial = next(predictions_dir.glob("*.partial.json"))
            self.assertEqual(set(json.loads(partial.read_text(encoding="utf-8"))), {"a", "b"})
            before = dict(attempts)
            completed, failures = generate(*arguments, **keywords)
            self.assertEqual(set(completed), {"a", "b", "c"})
            self.assertEqual(failures, [])
            self.assertEqual(attempts["a"], before["a"])
            self.assertEqual(attempts["b"], before["b"])
            events = [json.loads(line) for line in next(
                predictions_dir.glob("generation_attempts.*.jsonl")
            ).read_text(encoding="utf-8").splitlines()]
            self.assertGreater(max(event["round"] for event in events if event["id"] == "c"), 2)


if __name__ == "__main__":
    unittest.main()
