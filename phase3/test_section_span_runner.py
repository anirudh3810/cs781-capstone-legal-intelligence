"""CPU shape and gradient smoke check for the Kaggle section runner."""
import importlib.util
import inspect
import json
import sys
import zipfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
for name, file in (
    ('cs781_phase3_hierarchical_runner', 'hierarchical_runner.py'),
    ('cs781_phase3_context_joint_runner', 'context_joint_runner.py'),
    ('cs781_phase3_section_span_runner', 'section_span_runner.py'),
):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

import cs781_phase3_section_span_runner as s

spec = importlib.util.spec_from_file_location('small_prefix_model', HERE.parent / 'prefix_hybrid_lm.py')
prefix = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prefix)
cfg = dict(prefix.make_model_config(100))
cfg.update(hidden_size=64, num_layers=2, num_heads=4, head_dim=16, ffn_size=128)
local = s.LocalEncoder(prefix.build_model(cfg))
ids = torch.randint(1, 100, (2, 12))
mask = torch.ones_like(ids, dtype=torch.bool)
target = mask.clone()
target[:, :3] = False
h, logits = local(ids, mask, target, torch.tensor([0, 3]), torch.tensor([0, 1]))
assert h.shape == (2, 128) and logits.shape == (2, 13)
logits.square().mean().backward()
assert local.classifier[-1].weight.grad is not None

doc = {'id': 'synthetic', 'domain': 'CL', 'text': ['counsel for respondent', 'we held', 'therefore'],
       'labels': ['ArgumentRespondent', 'RatioOfTheDecision', 'RatioOfTheDecision']}
assert s.lexical_flags(doc).shape == (3, len(s.FLAG_NAMES))
rows = [(doc['id'], i) for i in range(3)]
tokens = {doc['id']: [([1, 2, 3, 4], 1, 3) for _ in rows]}
vocab = ['ArgumentRespondent', 'RatioOfTheDecision']
batch = s.collate_sentences(rows, {doc['id']: doc}, tokens, 0, vocab, torch.device('cpu'))
assert batch[-1].tolist() == [0, 1, 1]

tracker = s.SectionTracker()
x = torch.randn(42, 768+13+len(s.FLAG_NAMES))
labels = (['Fact'] * 15 + ['ArgumentRespondent'] * 20 + ['RatioOfTheDecision'] * 7)
all_roles = ['Fact', 'Issue', 'ArgumentPetitioner', 'ArgumentRespondent', 'Statute',
             'Dissent', 'PrecedentReliedUpon', 'PrecedentNotReliedUpon',
             'PrecedentOverruled', 'RulingByLowerCourt', 'RatioOfTheDecision',
             'RulingByPresentCourt', 'None']
train_doc = {'labels': labels}
segments = s.gold_segments(labels, all_roles)
starts = torch.tensor([a for a, _, _ in segments])
ends = torch.tensor([b for _, b, _ in segments])
out = tracker(x, starts, ends)
assert out[1].shape == (42, 13) and out[5].shape == (3, 13)
loss = s.global_loss(out, train_doc, all_roles, torch.ones(13), torch.device('cpu'))
loss.backward()
assert tracker.span[-1].weight.grad is not None
assert tracker.argument.weight.grad is not None
assert tracker.precedent.weight.grad is not None
tracker.zero_grad(set_to_none=True)
no_pair = {'labels': ['Fact'] * 8}
out_no_pair = tracker(x[:8], torch.tensor([0]), torch.tensor([8]))
s.global_loss(out_no_pair, no_pair, all_roles, torch.ones(13), torch.device('cpu')).backward()
assert tracker.argument.weight.grad is not None
assert tracker.precedent.weight.grad is not None
tracker.eval()
with torch.no_grad():
    out = tracker(x)
    pred = s.decode_sections(tracker, out[0], out[1], out[2],
                             s.transition_prior([train_doc], all_roles))
assert pred.shape == (42,) and pred.min() >= 0 and pred.max() < 13

with zipfile.ZipFile(HERE.parent / 'CS781_Phase3_All_Artifacts.zip') as z:
    training = []
    for domain in ('CL','IT'):
        for item in json.loads(z.read(f'official/{domain}_train.json')):
            training.append({'id':item['id'],'domain':domain,'text':item['text'],
                             'labels':item['labels']})
conf = {'folds': s.u.fold_map(training)}
assert len(conf['folds']) == 80
for fold in range(5):
    fitting, held = s.local_stage_docs(f'outer_{fold}', training, conf)
    assert (len(fitting),len(held)) == (64,16)
    inner_held = []
    for side in (0,1):
        fit, unseen = s.local_stage_docs(f'inner_{fold}_{side}', training, conf)
        assert (len(fit),len(unseen)) == (32,32)
        inner_held.extend(d['id'] for d in unseen)
    assert set(inner_held) == {d['id'] for d in fitting}
for distributed_stage in (s.train_local,s.extract_local,s.train_global,s.infer_test):
    assert 'u.snapshot()' not in inspect.getsource(distributed_stage)
print('Local encoder, batching, full-document model, losses, and decoder passed.')
