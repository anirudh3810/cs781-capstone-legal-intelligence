"""Torch-free checks of Phase-2 offset, archive, and packaging safeguards."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


SOURCE = Path(__file__).resolve().parents[1] / 'phase2_infrastructure.py'


def functions(names, namespace):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f'Missing source functions: {missing}')
    module = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                              *selected], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), namespace)


class Phase2InfrastructureTest(unittest.TestCase):
    def setUp(self):
        self.namespace = {'Path': Path, 'hashlib': hashlib, 'json': json,
                          'tempfile': tempfile, 'zipfile': zipfile, 'os': os,
                          'shutil': shutil, 'sys': sys}
        functions({'normalize_span', 'validate_prediction_map', 'sha256_file',
                   'stable_json', 'atomic_json', 'make_submission_zip',
                   '_safe_archive_member', 'extract_member_verified',
                   'export_recovery', 'verify_recovery_zip', 'recovery_name',
                   'discover_exact', 'stage_official_phase2_files'}, self.namespace)

    def test_gated_hugging_face_download_uses_secret_and_one_revision(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            input_root = root / 'input'
            input_root.mkdir()
            names = ('fold1-text.json', 'fold1-gold.json', 'fold2-text.json',
                     'fold2-gold.json', 'labels.txt', 'fold3-text.json')
            self.namespace.update(INPUT_ROOT=input_root, WORK_ROOT=root / 'work',
                                  FILE_OVERRIDES={}, DATA_NAMES=names, RUN_MODE='auto',
                                  AUTO_DOWNLOAD_PHASE2=True,
                                  HF_DATASET_REPO='Exploration-Lab/CS-781-Capstone',
                                  HF_DATASET_FILES={name: 'Phase_2_LNER/' + name for name in names})
            fake_secrets = types.ModuleType('kaggle_secrets')
            fake_secrets.UserSecretsClient = lambda: types.SimpleNamespace(
                get_secret=lambda key: 'test-token' if key == 'HF_TOKEN' else None)
            fake_hub = types.ModuleType('huggingface_hub')

            class FakeApi:
                def repo_info(self, repo_id, repo_type, token):
                    assert token == 'test-token' and repo_type == 'dataset'
                    return types.SimpleNamespace(sha='a' * 40)

            def fake_download(**kwargs):
                self.assertEqual(kwargs['token'], 'test-token')
                self.assertEqual(kwargs['revision'], 'a' * 40)
                output = root / 'fake_cache' / Path(kwargs['filename']).name
                output.parent.mkdir(exist_ok=True)
                output.write_text('official-' + output.name, encoding='utf-8')
                return str(output)

            fake_hub.HfApi = FakeApi
            fake_hub.hf_hub_download = fake_download
            with patch.dict(sys.modules, {'kaggle_secrets': fake_secrets,
                                          'huggingface_hub': fake_hub}):
                self.namespace['stage_official_phase2_files']()
            self.assertEqual(set(self.namespace['FILE_OVERRIDES']), set(names))
            source = json.loads((root / 'work' / 'official_phase2_source.json').read_text())
            self.assertEqual(source['revision'], 'a' * 40)
            self.assertEqual(set(source['files']), set(names))
            for name in names:
                self.assertEqual(self.namespace['discover_exact'](name).read_text(),
                                 'official-' + name)

    def test_fully_attached_official_files_need_no_secret(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            names = ('fold1-text.json', 'fold1-gold.json', 'fold2-text.json',
                     'fold2-gold.json', 'labels.txt', 'fold3-text.json')
            for name in names:
                (root / name).write_text('fixture', encoding='utf-8')
            self.namespace.update(INPUT_ROOT=root, WORK_ROOT=root / 'work',
                                  FILE_OVERRIDES={}, DATA_NAMES=names,
                                  RUN_MODE='auto', AUTO_DOWNLOAD_PHASE2=True)
            self.namespace['stage_official_phase2_files']()
            self.assertFalse((root / 'work').exists())

    def test_unicode_character_spans_and_exact_ids(self):
        text = 'अधिकार Court'
        start = text.index('Court')
        end = start + len('Court')
        correct = {'doc1': [[start, end, 'COURT']]}
        self.assertTrue(self.namespace['validate_prediction_map'](
            correct, {'doc1': text}, ['COURT'])['passed'])
        with self.assertRaises(ValueError):
            self.namespace['validate_prediction_map'](
                {'doc1': [[-1, end, 'COURT']]}, {'doc1': text}, ['COURT'])
        with self.assertRaises(ValueError):
            self.namespace['validate_prediction_map'](
                correct, {'doc1': text, 'doc2': 'Other'}, ['COURT'])
        with self.assertRaises(ValueError):
            self.namespace['normalize_span']([start, end, 'COURT', 'Wrong'],
                                               document_id='doc1', text=text, allowed={'COURT'})

    def test_archive_manifest_is_checked_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            archive_path = root / 'source.zip'
            payload = b'checkpoint-fixture'
            with zipfile.ZipFile(archive_path, 'w') as archive:
                archive.writestr('checkpoints/selected_model.pt', payload)
            target = root / 'staged' / 'selected_model.pt'
            valid = {'checkpoints/selected_model.pt': {
                'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(),
            }}
            with zipfile.ZipFile(archive_path) as archive:
                row = self.namespace['extract_member_verified'](
                    archive, 'checkpoints/selected_model.pt', target, valid)
            self.assertTrue(row['manifest_verified'])
            self.assertEqual(target.read_bytes(), payload)
            self.assertFalse(self.namespace['_safe_archive_member']('../escape.pt'))
            with zipfile.ZipFile(archive_path) as archive:
                with self.assertRaises(ValueError):
                    self.namespace['extract_member_verified'](
                        archive, 'checkpoints/selected_model.pt', root / 'wrong.pt',
                        {'checkpoints/selected_model.pt': {'bytes': 1, 'sha256': 'bad'}})

    def test_incomplete_package_is_blocked_and_valid_zip_has_root_json(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self.namespace['experiment_directory'] = lambda _slot: root
            data = {'test_text': {'a': 'Court', 'b': 'Judge'}, 'labels': ['COURT', 'JUDGE']}
            with self.assertRaises(ValueError):
                self.namespace['make_submission_zip'](
                    1, {'a': [[0, 5, 'COURT']]}, data, {'checkpoint': 'fixture'})
            self.assertFalse((root / 'submission.zip').exists())
            report = self.namespace['make_submission_zip'](
                1, {'a': [[0, 5, 'COURT']], 'b': [[0, 5, 'JUDGE']]},
                data, {'checkpoint': 'fixture'})
            self.assertTrue(report['passed'])
            with zipfile.ZipFile(root / 'submission.zip') as archive:
                self.assertEqual(archive.namelist(), ['prediction.json'])
                self.assertEqual(set(json.loads(archive.read('prediction.json'))), {'a', 'b'})

    def test_recovery_contains_completion_result_but_not_redundant_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_name:
            root = Path(temporary_name)
            self.namespace['experiment_directory'] = lambda _slot: root
            (root / 'checkpoint.latest.pt').write_bytes(b'checkpoint')
            (root / 'checkpoint.latest.previous_known_good.pt').write_bytes(b'old')
            (root / 'selected_model.pt').write_bytes(b'selected')
            (root / 'training_result.json').write_text('{"status":"complete"}', encoding='utf-8')
            archive_path = self.namespace['export_recovery'](1)
            self.namespace['verify_recovery_zip'](archive_path, 1)
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
            self.assertIn('training_result.json', names)
            self.assertIn('checkpoint.latest.pt', names)
            self.assertNotIn('checkpoint.latest.previous_known_good.pt', names)


if __name__ == '__main__':
    unittest.main()
