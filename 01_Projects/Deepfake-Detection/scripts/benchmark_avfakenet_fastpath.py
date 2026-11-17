"""Small real-batch timing study, separate model copies; no live model changes."""

# Workspace relocation: portable paths, historical metadata stays immutable.
import sys as _fk_sys
from pathlib import Path as _FkBootstrapPath
_fk_workspace = next((p for p in _FkBootstrapPath(__file__).resolve().parents
                      if (p / 'workspace-layout.json').is_file()), None)
if _fk_workspace is None:
    _fk_hint = _FkBootstrapPath(__import__('os').environ.get('FK_WORKSPACE_ROOT', '/inputs'))
    if (_fk_hint / 'workspace-layout.json').is_file():
        _fk_workspace = _fk_hint
if _fk_workspace is None:
    raise RuntimeError('workspace-layout.json is missing')
if str(_fk_workspace / '06_Tools/workspace') not in _fk_sys.path:
    _fk_sys.path.insert(0, str(_fk_workspace / '06_Tools/workspace'))
import workspace_paths as _fk_paths

import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
import sys,time,copy
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import numpy as np
import torch
import run_avfakenet_one_epoch as pilot
from baselines.common.io import read,digest
from baselines.common.training import load_checkpoint
from baselines.avfakenet.model import classification_loss
from baselines.avfakenet.normalized_recovery import set_training_mode
from avfakenet_progress_io import write_json
from avfakenet_training_fastpath import enable,loss as fast_loss


def main():
    run=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__);out=_fk_paths.relocate((ROOT/'runs/avfakenet-training-fastpath-benchmark-001'), __file__)
    out.mkdir(exist_ok=False)
    identity=read(_fk_paths.relocate((run/'identity.json'), __file__));profile=identity['profile']
    records=read(_fk_paths.relocate((run/'prepared-index.json'), __file__));cache=_fk_paths.relocate((ROOT/'runs/avfakenet-machine-b-prepared-001'), __file__)
    refs=pilot.references(records,'visual','train',cache);order=pilot.balanced_epoch(refs[:,2],8)
    loader=pilot.Inputs(cache,records,'visual');rows=[order[i:i+8] for i in range(400,800,8)]
    started=time.perf_counter();batches=[loader.batch(refs,ids) for ids in rows]
    read_seconds=time.perf_counter()-started;loader.close()
    state=load_checkpoint(_fk_paths.relocate((run/'visual/last.pt'), __file__))['model_state']
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    results={'live_runner_untouched':True,'cached_batches':len(batches),'read_seconds':read_seconds,
        'checkpoint_sha256':digest(_fk_paths.relocate((run/'visual/last.pt'), __file__)),'arms':{}}
    reference=None;reference_state=None
    for fast in (False,True):
        threads=2
        torch.set_num_threads(threads);torch.manual_seed(7);torch.cuda.manual_seed_all(7)
        model=pilot.make_model(profile,'visual',state,'cuda');set_training_mode(model,'off')
        if fast:enable(model)
        loss_fn=fast_loss if fast else classification_loss
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-8,weight_decay=profile['training']['weight_decay'])
        # The optimizer is diagnostic-only. All arms have exactly the same state,
        # dropout seed, batches and warmup sequence, independent of the live run.
        logits_list=[];started=time.perf_counter()
        for index,(x,y) in enumerate(batches):
            if index==5:torch.cuda.synchronize();started=time.perf_counter()
            optimizer.zero_grad(set_to_none=True);logits=model(x.cuda());loss=loss_fn(logits,y.cuda(),.1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite benchmark loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
            logits_list.append(logits.detach().cpu())
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        observed=torch.cat(logits_list)
        if reference is None:
            reference=observed;reference_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        delta=float((reference-observed).abs().max())
        name='fast' if fast else 'baseline'
        results['arms'][name]={'threads':threads,'timed_batches':45,'seconds':elapsed,
            'images_per_second':360/elapsed,'max_logit_difference_vs_two_threads':delta,
            'bitwise_logits_identical':bool(torch.equal(reference,observed)),
            'all_final_weights_bitwise_identical':all(torch.equal(v.detach().cpu(),reference_state[k]) for k,v in model.state_dict().items())}
        write_json(_fk_paths.relocate((out/'report.json'), __file__),results);print(results['arms'][name],flush=True)
        del model,optimizer;torch.cuda.empty_cache()
    results['note']='Timings include competition with the live worker; use as an indicative bottleneck test.'
    write_json(_fk_paths.relocate((out/'report.json'), __file__),results)


if __name__=='__main__':main()
