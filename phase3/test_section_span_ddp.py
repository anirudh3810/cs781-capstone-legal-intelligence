"""Reproduce the failed batch pattern with two CPU distributed workers."""
import importlib.util
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

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

vocab = ['Fact', 'Issue', 'ArgumentPetitioner', 'ArgumentRespondent', 'Statute',
         'Dissent', 'PrecedentReliedUpon', 'PrecedentNotReliedUpon',
         'PrecedentOverruled', 'RulingByLowerCourt', 'RatioOfTheDecision',
         'RulingByPresentCourt', 'None']

def worker(rank, store_path):
    torch.set_num_threads(2)
    store=dist.FileStore(store_path,2)
    dist.init_process_group('gloo',store=store,rank=rank,world_size=2)
    try:
        model = DDP(s.SectionTracker(), broadcast_buffers=False, find_unused_parameters=False)
        optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
        for step in range(3):
            labels = (['Fact']*12 if step != 1 else
                      ['Fact']*4 + ['ArgumentRespondent']*4 + ['PrecedentReliedUpon']*4)
            doc = {'labels':labels}
            segments = s.gold_segments(labels,vocab)
            starts = torch.tensor([a for a,_,_ in segments])
            ends = torch.tensor([b for _,b,_ in segments])
            features = torch.randn(12,768+13+len(s.FLAG_NAMES)) + rank*.01
            optimizer.zero_grad(set_to_none=True)
            output = model(features,starts,ends)
            loss = s.global_loss(output,doc,vocab,torch.ones(13),torch.device('cpu'))
            loss.backward()
            assert model.module.argument.weight.grad is not None
            assert model.module.precedent.weight.grad is not None
            optimizer.step()
        dist.barrier()
        if rank == 0:
            print('Two-worker DDP passed across no-pair and paired-label batches.')
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    with tempfile.TemporaryDirectory(prefix='cs781_ddp_') as directory:
        mp.spawn(worker,args=(str(Path(directory)/'shared.store'),),nprocs=2,join=True)
