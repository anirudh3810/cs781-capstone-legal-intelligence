"""Local CPU smoke test with genuine Phase-1 recovery weights and mock CJPE rows."""
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

import utils as u
sys.modules['cs781_phase4_utils']=u
import cjpe_runner as r


def extract(archive,destination):
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            if not (name.startswith('model/') or name.startswith('tokenizer/') or
                    name=='checkpoints/selected_model.pt') or name.endswith('/'):
                continue
            target=destination/name
            target.parent.mkdir(parents=True,exist_ok=True)
            with source.open(name) as reader,target.open('wb') as writer:
                shutil.copyfileobj(reader,writer,4<<20)


def main():
    torch.set_num_threads(3)
    project=Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory(prefix='cs781_phase4_test_') as name:
        temp=Path(name)
        u.ROOT=r.ROOT=temp/'run'
        (r.ROOT/'official').mkdir(parents=True)
        train=[{'id':f'case_{i//2:03d}','text':f'Case number {i}. The petitioner requests relief. The court examined the arguments and issued an order.','label':i%2} for i in range(18)]
        dev=[{'id':f'dev_{i//2:03d}','text':f'Development case {i}. The dispute was heard. Judgment follows.','label':i%2} for i in range(4)]
        test=[{'id':f'test_{i//2:03d}','text':f'Test case {i}. The appeal was heard.'} for i in range(3)]
        for split,rows in (('train',train),('dev',dev),('test',test)):
            u.write_json(r.ROOT/'official'/f'{split}.json',rows)
        actual=r.data()
        assert tuple(len(x) for x in actual)==(18,4,3)
        fold_assignment=r.folds_for(actual[0])
        assert len(fold_assignment)==18
        for group in r.case_groups(actual[0]):
            assert len({fold_assignment[row['id']] for row in group})==1
        fit,inner=r.inner_split(actual[0],r.SEED)
        assert not {row['source_id'] for row in fit}&{row['source_id'] for row in inner}
        for base in ('01','02','03','04'):
            source=project/f'phase1_submission{base}_recovery.zip'
            extract(source,r.ROOT/'bases'/base)
            r.tokenize_base(base)
        # Kaggle often mounts extracted datasets under a different slug or
        # preserves a recovery ZIP with a different filename.
        original_root=r.ROOT
        mounts=temp/'input-layout-test'
        extracted=mounts/'different-dataset-name'/'nested-recovery'
        extract(project/'phase1_submission02_recovery.zip',extracted)
        arbitrary_zip=mounts/'opaque-dataset'/'base-weights.zip'
        arbitrary_zip.parent.mkdir(parents=True)
        shutil.copy2(project/'phase1_submission04_recovery.zip',arbitrary_zip)
        u.INPUT=mounts
        u.ROOT=r.ROOT=temp/'staged-layout-test'
        assert r.stage_base('02').joinpath('checkpoints/selected_model.pt').is_file()
        assert r.stage_base('04').joinpath('checkpoints/selected_model.pt').is_file()
        for base in ('02','04'):
            model,_=u.load_base(base)
            del model
        try:r.stage_base('01')
        except RuntimeError as exc:
            assert 'Mounted inputs' in str(exc)
        else:raise AssertionError('Missing Phase-1 recovery was not reported')
        r.ROOT=u.ROOT=original_root
        print('PASS KAGGLE INPUT DISCOVERY',flush=True)
        for slot in r.SLOTS:
            model,_=r.model_for(slot,torch.device('cpu'))
            cache=torch.load(r.ROOT/'hf_cache'/f"tokens_{r.SLOTS[slot]['base']}.pt",map_location='cpu',weights_only=False)
            ids,mask,pos=r.document_batch(actual[0][0],cache,torch.device('cpu'),False)
            model.train()
            logit=model(ids,mask,pos)
            assert logit.ndim==0 and torch.isfinite(logit)
            loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,torch.tensor(0.))
            loss.backward()
            missing=[name for name,p in model.named_parameters() if p.requires_grad and p.grad is None]
            assert not missing,(slot,missing)
            state=r.trainable_state(model)
            r.load_state(model,state)
            print('PASS',slot,r.SLOTS[slot]['method'],'chunks',ids.shape[0],
                  'loss',round(float(loss.detach()),4),'trainable tensors',len(state),flush=True)
            del model
        assert 0<=r.best_threshold([.1,.2,.7,.8],[0,0,1,1])['macro_f1']<=1
        r.WORK=temp
        foldmap=r.folds_for(actual[0])
        u.write_json(r.ROOT/'run_config.json',{'folds':foldmap})
        final=r.ROOT/'runs/01/final'
        final.mkdir(parents=True)
        u.write_json(final/'status.json',{'complete':True,'threshold':.5})
        u.save_torch(final/'dev_logits.pt',{row['id']:2. if row['label'] else -2. for row in actual[1]})
        u.save_torch(final/'test_logits.pt',{row['id']:2. if i%2 else -2. for i,row in enumerate(actual[2])})
        for fold in range(r.FOLDS):
            path=r.ROOT/'runs/01'/f'fold_{fold}'
            path.mkdir(parents=True)
            u.write_json(path/'status.json',{'complete':True,'threshold':.5})
            u.save_torch(path/'outer_logits.pt',{
                row['id']:2. if row['label'] else -2.
                for row in actual[0] if foldmap[row['id']]==fold})
        r.package_slot('01')
        with zipfile.ZipFile(temp/'CS781_Phase4_Submission_01.zip') as package:
            assert package.namelist()==['prediction.json']
            import json
            assert [row['id'] for row in json.loads(package.read('prediction.json'))]==[row['source_id'] for row in actual[2]]
        assert u.read_json(r.ROOT/'reports/submission_01.json')['outer_macro_f1']==1.
        print('PASS PACKAGING',flush=True)
        # Check recovery integrity without archiving the large real Phase-1 checkpoints.
        recovery_root=temp/'recovery_source'
        recovery_root.mkdir()
        u.write_json(recovery_root/'report.json',{'score':1.0})
        u.save_torch(recovery_root/'runs'/'best.pt',{'weight':torch.ones(2)})
        u.ROOT=recovery_root
        u.ARTIFACT=temp/'CS781_Phase4_All_Artifacts.zip'
        u.snapshot()
        mounted=temp/'mounted_input'
        mounted.mkdir()
        shutil.copy2(u.ARTIFACT,mounted/u.ARTIFACT.name)
        u.ROOT=temp/'restored'
        u.INPUT=mounted
        u.restore_artifact()
        assert u.read_json(u.ROOT/'report.json')=={'score':1.0}
        assert torch.equal(torch.load(u.ROOT/'runs'/'best.pt',weights_only=True)['weight'],torch.ones(2))
        print('PASS RECOVERY',flush=True)


if __name__=='__main__':main()
