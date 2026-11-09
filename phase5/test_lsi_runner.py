"""CPU smoke test with the user's real Phase 1 weights and synthetic Phase 5 rows."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

import utils as u
sys.modules['cs781_phase5_utils']=u
import lsi_runner as r


def extract_base(archive,destination):
    with zipfile.ZipFile(archive) as source:
        names=[name for name in source.namelist()
               if name.startswith(('model/','tokenizer/')) or name=='checkpoints/selected_model.pt']
        for name in names:
            if name.endswith('/'):continue
            target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
            with source.open(name) as reader,target.open('wb') as writer:
                shutil.copyfileobj(reader,writer,4<<20)
    u.write_json(destination/'provenance.json',{'checkpoint_sha256':u.sha(destination/'checkpoints/selected_model.pt')})


def main():
    torch.set_num_threads(2)
    project=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='cs781_phase5_smoke_',dir=project/'phase5') as name:
        root=Path(name);r.ROOT=u.ROOT=root/'run';r.WORK=root;r.ARTIFACT=u.ARTIFACT=root/'all.zip'
        (r.ROOT/'official').mkdir(parents=True)
        train=[{'id':f'case_{i//2:03d}','text':f'Case {i}. Section 7 of the Act and Article 21 are discussed. The judgment examined legal provisions.',
                'labels':[] if i==35 else [1,2] if i%3==0 else [2] if i%3==1 else [3]} for i in range(36)]
        dev=[{'id':f'dev_{i}','text':f'Development case {i}. Section 7 of the Act is discussed.','labels':[] if i==5 else [1] if i%2 else [2,3]}
             for i in range(6)]
        test=[{'id':f'test_{i}','text':f'Test case {i}. Article 21 and the Act are mentioned.'} for i in range(5)]
        for split,rows in [('train',train),('dev',dev),('test',test)]:
            u.write_json(r.ROOT/'official'/f'{split}.json',rows)
        parsed_train,parsed_dev,parsed_test=r.data()
        mapping,mode=r.split_groups(parsed_train)
        assert len(mapping)==len(train) and len(set(mapping.values()))==3
        for group in r.case_groups(parsed_train):
            assert len({mapping[row['id']] for row in group})==1
        u.write_json(r.ROOT/'run_config.json',{'labels':[1,2,3],'split':mapping,
            'official_sha256':{split:u.sha(r.ROOT/'official'/f'{split}.json') for split in ('train','dev','test')}})
        assert r.metric([[.9,.2,.1]]*len(parsed_dev),parsed_dev,[1,2,3],[.5]*3)['all_labels']==3
        threshold,_=r.fit_thresholds([[.9,.2,.1]]*len(parsed_dev),parsed_dev,[1,2,3],{1:10,2:10,3:10})
        assert len(threshold)==3
        assert r.chunk_starts(list(range(2000)),[100,700,1300,1800],128,4,False,0,'x')[0]==0
        for base in ('01','02','03','04'):
            extract_base(project/f'phase1_submission{base}_recovery.zip',r.ROOT/'bases'/base)
            r.tokenize_base(base)
        original_util_root=u.ROOT
        u.INPUT=r.ROOT/'bases';u.ROOT=root/'staged_input_test'
        assert u.stage_base('02').joinpath('checkpoints/selected_model.pt').is_file()
        u.ROOT=original_util_root
        print('PASS EXTRACTED INPUT DISCOVERY',flush=True)
        for slot in r.SLOTS:
            model,_=r.model_for(slot,torch.device('cpu'))
            cache=torch.load(r.ROOT/'hf_cache'/f"tokens_{r.SLOTS[slot]['base']}.pt",map_location='cpu',weights_only=False)
            ids,mask,pos,counts=r.document_batch(parsed_train[:2],cache,torch.device('cpu'),True,1)
            model.train()
            logits=model(ids,mask,pos,counts)
            assert logits.shape==(2,3) and torch.isfinite(logits).all()
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,r.targets(parsed_train[:2],{1:0,2:1,3:2},torch.device('cpu')))
            loss.backward()
            missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing,(slot,missing[:5])
            state=r.trainable_state(model);r.load_state(model,state)
            print('PASS MODEL',slot,r.SLOTS[slot]['method'],'chunks',ids.shape[0],'trainable tensors',len(state),flush=True)
            del model,cache,loss,logits,state
        for stage in ('holdout','final'):
            directory=r.ROOT/'runs'/'01'/stage;directory.mkdir(parents=True)
            status={'complete':True,'best_epoch':2,'thresholds':[.4,.4,.4],'trainable_parameters':123,
                    'outer_metric':{'macro_f1':.2} if stage=='holdout' else None}
            u.write_json(directory/'status.json',status)
        u.save_torch(r.ROOT/'runs/01/final/inner_logits.pt',{row['id']:[2.,-1.,-2.] for row in parsed_dev})
        u.save_torch(r.ROOT/'runs/01/final/test_logits.pt',{row['id']:[2.,-1.,-2.] for row in parsed_test})
        r.package_slot('01')
        with zipfile.ZipFile(root/'CS781_Phase5_Submission_01.zip') as package:
            assert package.namelist()==['prediction.json']
            rows=json.loads(package.read('prediction.json'))
            assert [row['id'] for row in rows]==[row['id'] for row in test]
            assert all(row['pred']==[1] for row in rows)
        print('PASS PACKAGE',flush=True)
        # Exercise the archive integrity/restore path on a tiny separate tree.
        mini=root/'mini';u.ROOT=mini;u.ARTIFACT=root/'CS781_Phase5_All_Artifacts.zip'
        u.write_json(mini/'run_config.json',{'check':'recover'})
        u.save_torch(mini/'runs/latest.pt',{'step':torch.tensor(3)})
        u.snapshot()
        assert mini.resolve().is_relative_to(root.resolve())
        shutil.rmtree(mini)
        mount=root/'input';mount.mkdir()
        shutil.copy2(u.ARTIFACT,mount/u.ARTIFACT.name)
        u.INPUT=mount
        u.restore_artifact()
        assert u.read_json(mini/'run_config.json')['check']=='recover'
        assert int(torch.load(mini/'runs/latest.pt',weights_only=True)['step'])==3
        print('PASS RECOVERY',flush=True)


if __name__=='__main__':main()
