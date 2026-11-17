"""Validation-only execution benchmark; select batch size, never a threshold."""

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
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import numpy as np
import torch
from baselines.common.io import read,digest,fingerprint
from baselines.common.training import load_checkpoint
from run_avfakenet_one_epoch import make_model
from avfakenet_progress_io import write_json


def main():
    out=_fk_paths.relocate((ROOT/'runs/avfakenet-inference-batch-benchmark-001'), __file__);out.mkdir(exist_ok=False)
    training=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__);identity=read(_fk_paths.relocate((training/'identity.json'), __file__));cache=_fk_paths.relocate((ROOT/'runs/avfakenet-machine-b-prepared-001'), __file__)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    model=make_model(identity['profile'],'visual',load_checkpoint(_fk_paths.relocate((training/'visual/best.pt'), __file__))['model_state'],'cuda').eval().requires_grad_(False)
    selected=[];arrays=[]
    for row in read(_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__)):
        if row['split']!='validation':continue
        directory=_fk_paths.relocate((cache/fingerprint(row)), __file__);record=read(_fk_paths.relocate((directory/'record.json'), __file__))
        if 'visual' in record['reasons']:continue
        a=np.load(_fk_paths.relocate_value((_fk_paths.relocate((directory/'visual.npy'), __file__)), __file__),allow_pickle=False);arrays.append(a[:min(len(a),192)]);selected.append(row['video_id'])
        if len(selected)==4:break
    pixels=np.concatenate(arrays);reference=None
    report={'scope':'validation-only execution tuning, same frozen checkpoint','selected_validation_ids':selected,
        'frames':len(pixels),'checkpoint_sha256':digest(_fk_paths.relocate((training/'visual/best.pt'), __file__)),'arms':{}}
    for size in (8,16,32,64):
        try:
            with torch.inference_mode():
                warm=torch.from_numpy(pixels[:size].copy()).permute(0,3,1,2).contiguous().float()/255
                model(warm.cuda());torch.cuda.synchronize();times=[];outputs=None
                for repeat in range(2):
                    started=time.perf_counter();outputs=[]
                    for start in range(0,len(pixels),size):
                        x=torch.from_numpy(pixels[start:start+size].copy()).permute(0,3,1,2).contiguous().float()/255
                        logits=model(x.cuda())
                        if not torch.isfinite(logits).all():raise ValueError('Nonfinite inference')
                        outputs.append(logits.cpu())
                    torch.cuda.synchronize();times.append(time.perf_counter()-started)
                observed=torch.cat(outputs)
            if reference is None:reference=observed
            report['arms'][str(size)]={'batch_size':size,'seconds':min(times),'times':times,
                'max_logit_difference':float((observed-reference).abs().max()),
                'all_frame_decisions_identical':bool(torch.equal(observed.argmax(-1),reference.argmax(-1)))}
        except torch.cuda.OutOfMemoryError:
            report['arms'][str(size)]={'batch_size':size,'skipped':'CUDA memory limit'};torch.cuda.empty_cache()
        write_json(_fk_paths.relocate((out/'report.json'), __file__),report);print(report['arms'][str(size)],flush=True)
    valid=[r for r in report['arms'].values() if r.get('all_frame_decisions_identical') and r['max_logit_difference']<=1e-4]
    report['selected_batch_size']=min(valid,key=lambda r:r['seconds'])['batch_size']
    report['status']='COMPLETE';write_json(_fk_paths.relocate((out/'report.json'), __file__),report)


if __name__=='__main__':main()
