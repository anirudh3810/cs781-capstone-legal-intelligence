"""Check the new encoder against the actual user Phase 1 checkpoint 03."""
import importlib.util
import json
import sys
import tempfile
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

import cs781_phase3_hierarchical_runner as u
import cs781_phase3_context_joint_runner as c
import cs781_phase3_section_span_runner as s

archive = HERE.parent / 'CS781_Phase3_All_Artifacts.zip'
needed = ('runner.py', 'bases/03/model/prefix_hybrid_lm.py',
          'bases/03/model/course_bpe.py', 'bases/03/checkpoints/selected_model.pt',
          'bases/03/tokenizer/course_bpe-merges.txt',
          'bases/03/tokenizer/course_bpe-vocab.json',
          'bases/03/tokenizer/tokenizer.json',
          'bases/03/tokenizer/tokenizer_config.json',
          'bases/03/tokenizer/tokenizer_metadata.json')
torch.set_num_threads(4)
with tempfile.TemporaryDirectory(prefix='cs781_real_base_') as directory:
    root = Path(directory)
    with zipfile.ZipFile(archive) as z:
        for name in needed:
            dest = root / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, dest.open('wb') as out:
                for block in iter(lambda: src.read(4 << 20), b''):
                    out.write(block)
    u.PRIOR = root
    old = u.legacy()
    old.ROOT = root
    model, tokenizer = s.local_model(old, torch.device('cpu'), ['Fact']*13)
    doc = {'id':'real-base-smoke', 'domain':'CL',
           'text':['The respondent submitted that the precedent was inapplicable.',
                   'For these reasons we hold that the petition fails.'],
           'labels':['ArgumentRespondent','RulingByPresentCourt']}
    tokens = c.make_tokens([doc], tokenizer)
    batch = s.collate_sentences([(doc['id'],0),(doc['id'],1)], {doc['id']:doc},
                                tokens, int(tokenizer.pad_token_id),
                                ['ArgumentRespondent','RulingByPresentCourt'],
                                torch.device('cpu'))
    model.eval()
    with torch.no_grad():
        embeddings, logits = model(*batch[:5])
    assert embeddings.shape == (2,768) and logits.shape == (2,13)
    print('Real Phase 1 base 03 plus target-aware section encoder passed.')
