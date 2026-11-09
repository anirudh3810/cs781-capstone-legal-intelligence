"""CPU contract tests against small instances of the four real Phase-1 sources."""
from __future__ import annotations

import importlib.util
import copy
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

try:
    import torch
    from tokenizers import Tokenizer
except ImportError:
    torch = None


ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / 'phase1_submission01_recovery.zip').is_file()),
            Path(__file__).resolve().parents[3])
METHOD_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = ('decoder_lm.py', 'encoder_decoder_lm.py',
                'prefix_hybrid_lm.py', 'efficient_causal_lm.py')


def _load_source(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipIf(torch is None, 'PyTorch and tokenizers are needed for model tests')
class Phase2MethodContractTest(unittest.TestCase):
    def test_character_offsets_can_be_inside_a_bpe_token(self):
        sys.path.insert(0, str(METHOD_ROOT))
        from phase2_method_common import _boundary_targets
        self.assertEqual(_boundary_targets([(0, 5)], [(1, 4, 'COURT')], 16),
                         ([1], [1]))

    def test_all_six_forward_backward_and_prediction(self):
        torch.set_num_threads(2)
        sys.path.insert(0, str(METHOD_ROOT))
        from phase2_method_common import collate

        labels = ['APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT',
                  'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO']
        text = 'Alice appeared before the Supreme Court in Delhi.'
        start = text.index('Court')
        rows = [{'id': 'fixture', 'text': text, 'spans': [(start, start + 5, 'COURT')]}]
        config = {'max_tokens': 32, 'window_stride': 16, 'unfrozen_layers': 1,
                  'rank': 4, 'adapter_alpha': 8., 'adapter_bottleneck': 8,
                  'teacher_max_tokens': 48, 'distill_temperature': 2.,
                  'distill_weight': .35, 'inference_window_batch_size': 2}
        with tempfile.TemporaryDirectory() as directory:
            models, tokenizers = {}, {}
            for index, filename in enumerate(SOURCE_NAMES, 1):
                archive_path = ROOT / f'phase1_submission{index:02d}_recovery.zip'
                if not archive_path.exists():
                    self.skipTest('Local Phase-1 recovery archives are unavailable')
                with zipfile.ZipFile(archive_path) as archive:
                    source_path = Path(directory) / filename
                    source_path.write_bytes(archive.read(f'model/{filename}'))
                    module = _load_source(source_path, f'phase2_fixture_base_{index}')
                    backend = Tokenizer.from_str(archive.read('tokenizer/tokenizer.json').decode())
                tokenizer = type('BPE', (), {})()
                tokenizer.backend = backend
                tokenizer.pad_token_id = backend.token_to_id('<pad>')
                tokenizer.unk_token_id = backend.token_to_id('<unk>')
                tokenizers[index] = tokenizer
                vocab = backend.get_vocab_size()
                if index == 1:
                    model = module.build_model(module.ModelConfig(
                        vocab_size=vocab, context_length=64, hidden_size=64,
                        num_layers=2, num_heads=2, ffn_hidden_size=128))
                else:
                    settings = dict(module.make_model_config(vocab))
                    if index == 2:
                        settings.update(encoder_layers=2, decoder_layers=1,
                                        hidden_size=64, attention_heads=2, ffn_size=128,
                                        max_encoder_length=64, max_decoder_length=64)
                    elif index == 3:
                        settings.update(num_layers=2, hidden_size=64, num_heads=2,
                                        head_dim=32, ffn_size=128, max_sequence_length=64)
                    else:
                        settings.update(num_layers=2, hidden_size=64, num_query_heads=2,
                                        num_kv_heads=1, intermediate_size=128,
                                        max_sequence_length=64, local_window=32, global_stride=16)
                    model = module.build_model(settings)
                models[index] = model
            teacher_task = None
            for slot, base_id in enumerate((1, 2, 3, 4, 1, 3), 1):
                filename = f'phase2_method_{slot:02d}.py'
                method = _load_source(METHOD_ROOT / filename, f'phase2_fixture_method_{slot}')
                teacher = teacher_task if slot == 6 else None
                task = method.build_task_model(copy.deepcopy(models[base_id]), tokenizers[base_id],
                                               labels, config, teacher_model=teacher,
                                               teacher_tokenizer=tokenizers[1] if teacher else None)
                trainable = {name for name, parameter in task.named_parameters()
                             if parameter.requires_grad}
                self.assertIn('head.weight', trainable)
                if slot == 1:
                    self.assertIn('base.token_embedding.weight', trainable)
                elif slot == 2:
                    self.assertNotIn('base.token_embedding.weight', trainable)
                    self.assertTrue(any('encoder_blocks.1' in name for name in trainable))
                elif slot == 3:
                    self.assertTrue(any('.attention.query.a.weight' in name for name in trainable))
                elif slot == 4:
                    self.assertTrue(any('.attention.q_proj.a.weight' in name for name in trainable))
                elif slot == 5:
                    self.assertTrue(any('.up.weight' in name for name in trainable))
                else:
                    self.assertEqual(trainable, {
                        'head.weight', 'head.bias',
                        'start_boundary_head.weight', 'start_boundary_head.bias',
                        'end_boundary_head.weight', 'end_boundary_head.bias'})
                batch = collate(rows, tokenizers[base_id], labels, config,
                                teacher_tokenizer=tokenizers[1] if teacher else None)
                loss = task(**batch)['loss']
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                predicted = method.predict_spans(task, text, tokenizers[base_id], labels, config)
                self.assertIsInstance(predicted, list)
                if slot == 1:
                    teacher_task = task
            self.assertTrue(all(not p.requires_grad for p in teacher_task.parameters()))


if __name__ == '__main__':
    unittest.main()
