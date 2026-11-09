"""CPU integration checks on real Phase 1 models and synthetic retrieval pools."""
import importlib.util
from pathlib import Path
import shutil
import sys
import tempfile
import time
import zipfile

import torch
from torch import nn

import utils as u
sys.modules['cs781_phase7_utils']=u
spec=importlib.util.spec_from_file_location('cs781_phase7_evaluator',Path(__file__).parent/'reference/evaluate_phase7.py')
official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
sys.modules['cs781_phase7_evaluator']=official
import pcr_runner as r


def extract(archive,destination):
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not name.endswith('/') and (name.startswith(('model/','tokenizer/')) or name=='checkpoints/selected_model.pt'):
                target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
                with z.open(name) as src,target.open('wb') as out:shutil.copyfileobj(src,out)
    u.write_json(destination/'provenance.json',{'checkpoint_sha256':u.sha(destination/'checkpoints/selected_model.pt')})


def main():
    torch.set_num_threads(2)
    project=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='pcr_smoke_',dir=project/'phase7') as tmp:
        tmp=Path(tmp);r.ROOT=u.ROOT=tmp/'run';r.WORK=tmp;u.ARTIFACT=tmp/'CS781_Phase7_All_Artifacts.zip'
        original_root=r.ROOT
        for split in ('train','dev','test'):
            cases={f'{split}_c{i:03}':{'text':[f'The court reviewed statute section {i} and allowed the appeal. The question concerned liability.']} for i in range(10)}
            queries={f'{split}_q{i:03}':{'text':[f'The court considered liability under statute section {i%10}. The petition was allowed.']} for i in range(24 if split=='train' else 8)}
            if split!='test':
                for i,row in enumerate(queries.values()):row['relevant_candidates']=[f'{split}_c{i%10:03}',f'{split}_c{(i+1)%10:03}']
            u.write_json(r.ROOT/'official'/f'{split}.json',queries)
            u.write_json(r.ROOT/'official'/f'{split}_candidates.json',cases)
        fit=list(r.data('train')[0])[:20];tune=list(r.data('train')[0])[20:]
        u.write_json(r.ROOT/'run_config.json',{'split':{'fit':fit,'tune':tune,'audit':list(r.data('dev')[0]),'tune_pool':'train','audit_pool':'dev'}})
        for split in ('train','dev','test'):r.lexical(split)
        for b in ('01','02','03','04'):
            extract(project/f'phase1_submission{b}_recovery.zip',r.ROOT/'bases'/b)
            r.tokenize(b)
        for slot in r.SLOTS:
            model,tok=r.model_for(slot,torch.device('cpu'));cache=torch.load(r.ROOT/'tokens'/f"{r.SLOTS[slot]['base']}.pt",weights_only=False)
            items=r.examples()[:2];qrows,cids,qkeys,ckeys=r.training_batch(items,0,cache)
            scores=r.score_vectors(model(*r.pack(qkeys,cache,torch.device('cpu'))),model(*r.pack(ckeys,cache,torch.device('cpu'))))
            loss=r.gold_loss(scores,qrows,cids)
            if slot=='06':loss=loss+.15*r.kd_loss(scores,qrows,cids,{row['id']:{k:float(i) for i,k in enumerate(cids)} for row in qrows})
            loss.backward()
            missing=[n for n,p in model.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
            assert not missing,(slot,missing)
            if slot=='05':assert model.prompt.grad.abs().sum()>0
            state=r.trainable_state(model);r.load_state(model,state)
            print('PASS',slot,r.SLOTS[slot]['method'],'loss',round(float(loss.detach()),4),flush=True)
            if slot=='01':
                folder=r.ROOT/'runs/01';u.save_torch(folder/'best.pt',{'model':state,'progress':{}})
                u.write_json(folder/'selected.json',{'neural_weight':.5,'metric':{'tp':2},'epoch':1})
                u.write_json(folder/'status.json',{'trainable_parameters':1,'active_parameters':2})
                model.eval();path=folder/'best.pt'
                partial=r.encode_records(slot,path,'test','c',model,cache,torch.device('cpu'),time.monotonic()-1)
                assert partial is None
                complete=r.encode_records(slot,path,'test','c',model,cache,torch.device('cpu'),time.monotonic()+600)
                assert len(complete['ids'])==10
            del model,state,loss,scores
        # Multiplicative identity and matching initializations of KD/control.
        linear=nn.Linear(7,5);adapter=r.LoRMA(linear,2);x=torch.randn(2,7)
        assert torch.allclose(adapter(x),linear(x))
        r.train('01',max_minutes=.00001);first=u.read_json(r.ROOT/'runs/01/status.json')['steps']
        r.train('01',max_minutes=.00001);assert u.read_json(r.ROOT/'runs/01/status.json')['steps']>first
        print('PASS OPTIMIZER RESUME',flush=True)
        # Restore the synthetic selected metadata for the packaging interface.
        u.write_json(r.ROOT/'runs/01/status.json',{'trainable_parameters':1,'active_parameters':2})
        r.package('01',max_minutes=.001)
        r.package('01',max_minutes=10)
        assert (r.ROOT/'reports/submission_01.json').is_file()
        with zipfile.ZipFile(tmp/'CS781_Phase7_Submission_01.zip') as z:
            assert z.namelist()==['prediction.json']
            import json
            pred=json.loads(z.read('prediction.json'))
            assert not official.validate(pred,{k:[] for k in r.data('test')[0]},set(r.data('test')[1]))
        print('PASS FULL-POOL RANKING AND PACKAGE',flush=True)
        # Recovery interruption marker forces a second idempotent extraction.
        u.ROOT=tmp/'mini';u.write_json(u.ROOT/'run_config.json',{'example':1});u.save_torch(u.ROOT/'latest.pt',{'step':3})
        u.snapshot();mount=tmp/'mount';mount.mkdir();shutil.copy2(u.ARTIFACT,mount/u.ARTIFACT.name)
        target=u.ROOT.resolve();assert tmp.resolve() in target.parents;shutil.rmtree(target)
        u.INPUT=mount;u.restore_artifact();(u.ROOT/'.restore_in_progress').write_text('test')
        u.restore_artifact();assert not (u.ROOT/'.restore_in_progress').exists()
        assert u.read_json(u.ROOT/'run_config.json')['example']==1
        print('PASS RECOVERY INCLUDING INTERRUPTED RESTORE',flush=True)


if __name__=='__main__':main()
