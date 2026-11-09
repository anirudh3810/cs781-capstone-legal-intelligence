"""Synthetic Phase 3 schema and self-contained archive round-trip."""
import json
import shutil
import tempfile
from pathlib import Path

import runner

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    runner.ROOT = tmp / 'work'
    runner.INPUT = tmp / 'input'
    runner.ARTIFACT = tmp / 'CS781_Phase3_All_Artifacts.zip'
    runner.ROOT.mkdir()
    runner.INPUT.mkdir()
    official = runner.ROOT / 'official'
    official.mkdir()
    (official / 'label_vocab.json').write_text(json.dumps(sorted(runner.EXPECTED_LABELS)))
    for domain in ('CL', 'IT'):
        for split in ('train', 'dev', 'test'):
            row = {'id': f'{domain}_{split}', 'text': ['First sentence.', 'Second sentence.']}
            if split != 'test': row['labels'] = ['Fact', 'Issue']
            (official / f'{domain}_{split}.json').write_text(json.dumps([row]))
    train, dev, test = runner.all_data()
    assert len(train) == len(dev) == len(test) == 2
    assert len(runner.labels()) == 13
    (runner.ROOT / 'run_config.json').write_text('{}')
    runner.snapshot()
    shutil.copyfile(runner.ARTIFACT, runner.INPUT / runner.ARTIFACT.name)
    shutil.rmtree(runner.ROOT)
    runner.ROOT.mkdir()
    runner.restore_artifact()
    assert runner.all_data() == (train, dev, test)
    print('Phase 3 schema and recovery ZIP round-trip passed')
