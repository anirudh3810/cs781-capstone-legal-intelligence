"""Local model-interface smoke test; no official data or weights are loaded."""
import importlib.util
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

import runner

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[1]
with zipfile.ZipFile(root / 'CS781_Phase2_COMPLETE_ARTIFACTS.zip') as z:
    configs = {}
    for slot in range(1, 5):
        path = f'experiments/submission_{slot:02d}/experiment_identity.json'
        spec = json.loads(z.read(path))
        configs[spec['spec']['base_id']] = spec['base_meta']['model_config']

with tempfile.TemporaryDirectory() as tmp:
    for spec in runner.PLAN:
        bid = spec['base']
        with zipfile.ZipFile(root / runner.BASE_FILES[bid]) as z:
            source = Path(tmp) / f'{bid}_{runner.SOURCES[bid]}'
            source.write_bytes(z.read(f'model/{runner.SOURCES[bid]}'))
        module_spec = importlib.util.spec_from_file_location(f'smoke_base_{bid}_{spec["slot"]}', source)
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[module_spec.name] = module
        module_spec.loader.exec_module(module)
        base = module.build_model(configs[bid])
        model = runner.RRModel(base, spec, 13)
        model.train()
        ids = torch.randint(4, configs[bid]['vocab_size'], (2, 12))
        mask = torch.ones_like(ids)
        pos = torch.zeros(2, dtype=torch.long)
        domain = torch.zeros(2, dtype=torch.long)
        logits = model(ids, mask, pos, domain)
        assert tuple(logits.shape) == (2, 13), spec
        logits.float().sum().backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad), spec
        print(f'slot {spec["slot"]}: {spec["method"]} on base {bid} passed')
        del model, base
