"""Fast end-to-end stage test; method mathematics are tested on real bases separately."""
import sys
from pathlib import Path
import tempfile
import torch
from torch import nn
from torch.nn import functional as F
import test_pcr_runner

r=test_pcr_runner.r
u=test_pcr_runner.u

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.base=nn.Identity()
        self.embedding=nn.Embedding(100,16)
        self.project=nn.Linear(16,r.DIM)
        self.use_checkpoint=False
    def forward(self,ids,mask,counts):
        x=self.embedding(ids)
        vec=F.normalize(self.project((x*mask[:,:,None]).sum(1)/mask.sum(1)[:,None]).float(),dim=-1)
        groups=torch.split(vec,counts)
        chunks=torch.stack([F.pad(g,(0,0,0,r.CHUNKS-len(g))) for g in groups])
        cmask=torch.arange(r.CHUNKS)[None]<torch.tensor(counts)[:,None]
        return F.normalize(chunks.sum(1),dim=-1),chunks,cmask

def main():
    torch.set_num_threads(2)
    def model_for(slot,device):
        r.seed_all(r.SEED+r.stable('06' if slot=='control' else slot)%10000)
        return Tiny().to(device),None
    r.model_for=model_for
    for spec in r.SLOTS.values():spec['epochs']=2
    with tempfile.TemporaryDirectory(prefix='pipeline_',dir=Path(__file__).parent) as tmp:
        tmp=Path(tmp);r.ROOT=u.ROOT=tmp/'run';r.WORK=tmp
        for cache in (r.data,r.lexical,r.examples,r.training_queries,r.mining_bank):cache.cache_clear()
        tokens={}
        for split in ('train','dev','test'):
            cases={f'{split}_c{i}':{'text':[f'Legal appeal concerns section {i} and a liability order.']} for i in range(10)}
            queries={f'{split}_q{i}':{'text':[f'Query {i} considers statute section {i%10} and the holding.']} for i in range(12)}
            if split!='test':
                for i,row in enumerate(queries.values()):row['relevant_candidates']=[f'{split}_c{i%10}',f'{split}_c{(i+1)%10}']
            if split=='train':queries['train_q1']['text']=queries['train_q0']['text']
            for kind,rows in [('c',cases),('q',queries)]:
                for i,key in enumerate(rows):tokens[f'{split}/{kind}/{key}']=[[1,10+i,3,4,5,2]]
            u.write_json(r.ROOT/'official'/f'{split}.json',queries)
            u.write_json(r.ROOT/'official'/f'{split}_candidates.json',cases)
        tokens['_pad']=0
        for base in ('01','02','03','04'):u.save_torch(r.ROOT/'tokens'/f'{base}.pt',tokens)
        u.write_json(r.ROOT/'run_config.json',{'split':{'fit':list(r.data('train')[0])[:8],
            'tune':list(r.data('train')[0])[8:],'audit':list(r.data('dev')[0]),
            'tune_pool':'train','audit_pool':'dev'}})
        assert len(r.training_queries()['train_q0']['gold'])==3
        u.write_json(r.ROOT/'performance/01.json',{'microbatch':2,'gradient_checkpointing':False,'probe_version':2})
        original_train=r.train
        def injected_oom(*args,**kwargs):raise torch.cuda.OutOfMemoryError('injected memory failure')
        r.train=injected_oom
        assert r.advance('01',max_minutes=2)=='retry'
        assert u.read_json(r.ROOT/'performance/01.json')['gradient_checkpointing']
        r.train=original_train
        def finish(slot):
            converted=False
            for iteration in range(30):
                r.advance(slot,max_minutes=2)
                path=r.ROOT/'runs'/slot/'status.json'
                if slot=='01' and path.is_file() and not converted:
                    state=u.read_json(path)
                    if state['epoch']==1 and not state['pending_eval']:
                        settings=u.read_json(r.ROOT/'performance/01.json')
                        settings['microbatch']=1
                        u.write_json(r.ROOT/'performance/01.json',settings)
                        converted=True
                if slot=='control' and path.is_file() and u.read_json(path)['complete']:break
                if (r.ROOT/'reports'/f'submission_{slot}.json').is_file():break
            else:raise AssertionError(f'Infinite stage loop for {slot}')
        for slot in ('01','02','03','04'):finish(slot)
        r.teacher(max_minutes=2)
        assert (r.ROOT/'teacher/scores.json').is_file()
        for slot in ('control','05','06'):finish(slot)
        for slot in ('01','02'):r.score_selected(slot,'tune',max_minutes=2)
        r.score_selected('01','audit',max_minutes=2);r.report()
        value=u.read_json(r.ROOT/'report.json')
        assert len(value['reports'])==6 and value['gold_control_comparison']
        assert len(list(tmp.glob('CS781_Phase7_Submission_*.zip')))==6
        for slot in ('01','02','03','04','05','06'):
            pred=u.read_json(r.ROOT/'submissions'/slot/'prediction.json')
            assert not r.official.validate(pred,{k:[] for k in r.data('test')[0]},set(r.data('test')[1]))
        print('PASS COMPLETE SIX-SLOT PIPELINE, MINING, TEACHER, CONTROL, DISTILLATION, SCORES AND PACKAGES',flush=True)

if __name__=='__main__':main()
