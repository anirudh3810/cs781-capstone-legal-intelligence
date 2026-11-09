"""CPU-only checks for the self-contained improvement notebook builder."""
from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from typing import Any, Mapping

from tokenizers import Tokenizer


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / 'phase2' / 'notebook' / 'improvement_runall.py'
BUNDLE = ROOT / 'CS781_Phase2_COMPLETE_ARTIFACTS.zip'


class ImprovementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = RUNNER.read_text(encoding='utf-8')
        source = source.split('try:\n    IMPROVEMENT_RESULTS = run_improvement_all()', 1)[0]
        namespace = {
            'Path': Path, 'Any': Any, 'Mapping': Mapping, 'random': random,
            'math': math, 'hashlib': hashlib, 'stable_json': lambda x: json.dumps(x, sort_keys=True),
            'WORK_ROOT': Path(tempfile.mkdtemp()), 'SEED': 781,
            'TRAINING_CONFIGS': {i: {'learning_rate': 2e-5} for i in (1, 3, 5)},
            'RESTORED_RECOVERY_SLOTS': {},
        }
        exec(compile(source, str(RUNNER), 'exec'), namespace)
        cls.ns = namespace

    def test_notebook_cells_compile(self):
        notebook = json.loads((ROOT / 'CS781_Phase2_Improved_Three_RunAll.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(len(notebook['cells']), 11)
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), f'cell{index}', 'exec')

    def test_window_training_covers_gold_without_test_data(self):
        with zipfile.ZipFile(BUNDLE) as archive:
            text = json.loads(archive.read('official_phase2/fold1-text.json'))
            gold = json.loads(archive.read('official_phase2/fold1-gold.json'))
            tokenizer = types.SimpleNamespace(backend=Tokenizer.from_str(
                archive.read('phase1_bases/01/tokenizer/tokenizer.json').decode('utf-8')))
        identifier = next(iter(text))
        original = {'train_text': {identifier: text[identifier]},
                    'train_gold': {identifier: [(a, b, label) for a, b, label, *_ in gold[identifier]]},
                    'test_text': {}, 'labels': [], 'paths': {}, 'fingerprint': 'synthetic'}
        config = {'max_tokens': 384, 'window_stride': 288}
        staged, windows = self.ns['_windowed_training_data'](
            original, [identifier], tokenizer, config, 'unit')
        self.assertGreater(len(windows), 1)
        self.assertTrue(all('__' in name for name in windows))
        # Every official gold span must appear in at least one extracted window
        # with the same text slice, label and relative character boundaries.
        for a, b, label in original['train_gold'][identifier]:
            self.assertTrue(any(staged['train_text'][name][start:end] == text[identifier][a:b]
                                and name_label == label
                                for name in windows
                                for start, end, name_label in staged['train_gold'][name]))

    def test_step_budget_is_based_on_full_window_epochs(self):
        config = self.ns['_config_for'](1, 0, 160)
        self.assertEqual(config['batch_size'], 16)
        self.assertEqual(config['max_steps'], 160)
        self.assertEqual(config['training_window_count'], 160)


if __name__ == '__main__':
    unittest.main()
