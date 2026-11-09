"""Exercise archive restore, official data loading, and token/flag preparation."""
import importlib.util
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch
import nbformat

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
import cs781_phase3_section_span_runner as s

original = HERE.parent / 'CS781_Phase3_All_Artifacts.zip'
context = HERE.parent / 'CS781_Phase3_Context_Joint_All_Artifacts.zip'
with tempfile.TemporaryDirectory(prefix='cs781_prepare_') as folder:
    working = Path(folder)
    s.WORK = working
    s.ROOT = working / 'CS781_Capstone/phase3_sections'
    s.ARTIFACT = working / 'CS781_Phase3_Sections_All_Artifacts.zip'
    s.SUBMISSION = working / 'CS781_Phase3_Sections_Submission.zip'
    s.START_TIME_FILE = s.ROOT / 'run_start.json'
    u.WORK = working
    u.INPUT = working / 'input'
    u.INPUT.mkdir()
    u.PRIOR = working / 'CS781_Capstone/phase3'
    u.ROOT = s.ROOT
    u.ARTIFACT = s.ARTIFACT
    u.OLD_ARCHIVES = ('CS781_Phase3_All_Artifacts.zip',
                      'CS781_Phase3_Context_Joint_All_Artifacts.zip')
    locate = u.locate
    u.locate = lambda name: original if name == original.name else (
        context if name == context.name else locate(name))
    original_legacy = u.legacy
    def patched_legacy():
        module = original_legacy()
        module.ROOT = u.PRIOR
        return module
    u.legacy = patched_legacy
    original_count = torch.cuda.device_count
    torch.cuda.device_count = lambda: 2
    try:
        s.prepare()
    finally:
        torch.cuda.device_count = original_count
    old, train, dev, test, labels = u.sets()
    assert (len(train), len(dev), len(test), len(labels)) == (80, 10, 10, 13)
    assert (s.ROOT / 'tokens.pt').is_file()
    assert (s.ROOT / 'lexical_flags.pt').is_file()
    assert s.ARTIFACT.is_file()
    flags = torch.load(s.ROOT / 'lexical_flags.pt', weights_only=False)
    assert all(flags[d['id']].shape == (len(d['text']), len(s.FLAG_NAMES))
               for d in train+dev+test)
    # The failed Kaggle run's artifact must resume with the repaired source.
    old_nb = nbformat.read(HERE / 'CS781_Phase3_Sections_Failed_RunAll.ipynb', 4)
    old_source = old_nb.cells[5].source.split('\n',1)[1]
    assert hashlib.sha256(old_source.encode()).hexdigest() == s.FAILED_RUN_SOURCE_SHA256
    (s.ROOT / 'section_span_runner.py').write_bytes(old_source.encode('utf8'))
    assert u.digest(s.ROOT / 'section_span_runner.py') == s.FAILED_RUN_SOURCE_SHA256
    old_config = u.read_json(s.ROOT / 'config.json')
    old_config['source_sha256'] = s.FAILED_RUN_SOURCE_SHA256
    u.atomic_json(s.ROOT / 'config.json',old_config)
    torch.cuda.device_count = lambda: 2
    try:
        s.prepare()
    finally:
        torch.cuda.device_count = original_count
    assert (s.ROOT / 'migration.json').is_file()
    assert u.read_json(s.ROOT / 'config.json')['source_sha256'] == u.digest(HERE / 'section_span_runner.py')
    second_nb = nbformat.read(HERE / 'CS781_Phase3_Sections_SnapshotHang_RunAll.ipynb', 4)
    second_source = second_nb.cells[5].source.split('\n',1)[1]
    assert hashlib.sha256(second_source.encode()).hexdigest() == s.SNAPSHOT_HANG_SOURCE_SHA256
    (s.ROOT / 'section_span_runner.py').write_bytes(second_source.encode('utf8'))
    second_config = u.read_json(s.ROOT / 'config.json')
    second_config['source_sha256'] = s.SNAPSHOT_HANG_SOURCE_SHA256
    u.atomic_json(s.ROOT / 'config.json',second_config)
    complete = s.ROOT / 'local/done_test'
    incomplete = s.ROOT / 'local/live_test'
    complete.mkdir(parents=True)
    incomplete.mkdir(parents=True)
    u.atomic_json(complete/'done.json',{'synthetic_test_only':True})
    (complete/'latest.pt').write_bytes(b'not needed after completion')
    (incomplete/'latest.pt').write_bytes(b'resume this stage')
    torch.cuda.device_count = lambda: 2
    try:
        s.prepare()
    finally:
        torch.cuda.device_count = original_count
    assert not (complete/'latest.pt').exists()
    assert (incomplete/'latest.pt').is_file()
    assert u.read_json(s.ROOT / 'config.json')['source_sha256'] == u.digest(HERE / 'section_span_runner.py')
    # Exercise submission and recovery packaging with clearly synthetic scores.
    (s.ROOT / 'global/final').mkdir(parents=True)
    u.atomic_torch(s.ROOT / 'global/final/best.pt', {'synthetic_test_only':True})
    fake = {}
    for doc in test:
        probs = torch.zeros(len(doc['text']),len(labels))
        probs[:,0] = 1
        fake[doc['id']] = {'local':probs,'global':probs,'blend':probs,
                           'section':torch.zeros(len(doc['text']),dtype=torch.long),
                           'blend_section':torch.zeros(len(doc['text']),dtype=torch.long)}
    u.atomic_torch(s.ROOT / 'test_results.pt',fake)
    u.atomic_json(s.ROOT / 'selection.json',{'selected':'local','basis':'synthetic test',
        'completed_outer_folds':[],'development':{'local':{'macro_supported':0.}},
        'oof':None})
    u.atomic_json(s.ROOT / 'pilot_gate.json',{'continue_outer_folds':False,
        'best_fold0_supported_macro_f1':0.,'synthetic_test_only':True})
    s.package()
    assert u.read_json(s.ROOT / 'report.json')['status']=='WEAK_HELDOUT_PILOT_REVIEW_BEFORE_UPLOAD'
    with zipfile.ZipFile(s.SUBMISSION) as z:
        assert z.namelist() == ['prediction.json']
        pred = json.loads(z.read('prediction.json'))
    assert set(pred) == {d['id'] for d in test}
    assert sum(map(len,pred.values())) == 2197
    shutil.rmtree(s.ROOT)
    u.restore()
    assert (s.ROOT / 'config.json').is_file()
    assert (s.ROOT / 'tokens.pt').is_file()
    assert (s.ROOT / 'lexical_flags.pt').is_file()
    assert (s.ROOT / 'report.json').is_file()
    print('Archive restore, official-data preparation, submission ZIP, and recovery passed.')
