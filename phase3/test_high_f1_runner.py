"""Small CPU integration smoke test with the real Phase 1 recovery files."""
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

import runner as u
sys.modules['cs781_phase3_source_utils'] = u
import high_f1_runner as h


def extract_members(archive, destination, prefixes):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not any(name.startswith(prefix) for prefix in prefixes) or name.endswith('/'):
                continue
            target = destination/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(name))


def main():
    torch.set_num_threads(3)
    project = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='cs781_high_f1_') as name:
        temp = Path(name)
        u.ROOT = h.ROOT = temp/'run'
        u.INPUT = temp/'input'
        u.INPUT.mkdir()
        extract_members(project/'CS781_Phase3_All_Artifacts.zip',u.ROOT,['official/'])
        train,dev,test=u.all_data()
        assert (len(train),len(dev),len(test)) == (80,10,10)
        labels=u.labels()
        assert len(labels)==13
        assert len(h.balance_folds(train))==80
        for branch in ('03','01'):
            dataset=u.INPUT/f'phase1-submission{branch}-recovery'
            extract_members(project/f'phase1_submission{branch}_recovery.zip',dataset,
                            ['model/','checkpoints/selected_model.pt','tokenizer/'])
            h.stage_base_robust(branch)
            base,tok=u.load_base(branch)
            model=h.RoleTagger(base,branch,len(labels)).eval()
            assert any(isinstance(module,u.LoRA) for module in model.modules())
            one=dev[0]
            raw=[tok.backend.encode(sentence,add_special_tokens=False).ids for sentence in one['text'][:2]]
            saved_limit=h.MAX_TOKENS
            h.MAX_TOKENS=96
            encoded=h.encode_tokens([{**one,'text':one['text'][:2], 'labels':one['labels'][:2]}],tok,branch)
            h.MAX_TOKENS=saved_limit
            assert len(encoded[one['id']])==2
            item=(one['id'],0,2)
            tiny={**one,'text':one['text'][:2],'labels':one['labels'][:2]}
            inputs=h.collate(item,{one['id']:tiny},encoded,int(tok.pad_token_id),labels,torch.device('cpu'))
            with torch.no_grad():
                fine,pairs,trans=model(*inputs[:5])
                loss=h.loss_for((fine,pairs,trans),inputs[-1],labels,torch.ones(len(labels)))
            assert fine.shape==(2,13) and len(pairs)==3 and torch.isfinite(loss)
            state=h.trainable_state(model)
            h.load_state(model,state)
            print('PASS',branch,'tokens',len(raw[0]),'loss',round(float(loss),4),
                  'trainable tensors',len(state),flush=True)
            del model,base


if __name__=='__main__':
    main()
