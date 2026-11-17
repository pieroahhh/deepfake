"""After the core queue, lock selected models and reuse the original FK test cache."""

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
import sys,time,json,copy
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json


def rebind_manifest(source,source_root,checkpoint,checkpoint_sha,training_sha,expected):
    derived=copy.deepcopy(source);context=derived.get('evaluation_context')
    if not context or context['expected_labels']!=expected:raise ValueError('Missing trusted held-out context')
    if context['training_data_signatures']!=checkpoint['data_signatures']:raise ValueError('New ablation used a different training population')
    for split in ('train','validation','test'):
        for row in derived.get(split,[]):row['cache']=str((source_root/row['cache']).resolve())
    context['checkpoint_sha256']=checkpoint_sha;context['training_run_sha256']=training_sha
    return derived


def main():
    queue=_fk_paths.relocate((ROOT/'runs/remaining-ablations-001'), __file__);run=_fk_paths.relocate((ROOT/'runs/remaining-ablation-evaluation-001'), __file__);run.mkdir(exist_ok=True)
    import msvcrt
    lock=(_fk_paths.relocate((run/'worker.lock'), __file__)).open('a+b');lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    plan=read(_fk_paths.relocate((ROOT/'experiments/remaining-001/plan.json'), __file__));variants=plan['minimum_batch'];finished=[]
    def progress(stage,status='RUNNING',variant=None,processed=0,total=None,**extra):
        write_json(_fk_paths.relocate((run/'progress.json'), __file__),{'status':status,'stage':stage,'variant':variant,'variants':variants,
            'completed_variants':list(finished),'processed':processed,'total':total,'updated_unix':time.time(),**extra})
    try:
        while not (_fk_paths.relocate((queue/'queue-complete.json'), __file__)).exists():
            state=read(_fk_paths.relocate((queue/'progress.json'), __file__)) if (_fk_paths.relocate((queue/'progress.json'), __file__)).exists() else {}
            if state.get('status') in ('FAILED','PAUSED'):
                progress('training_queue_needs_attention','PAUSED',error=state.get('error'));return
            progress('waiting_for_all_training_and_validation');time.sleep(10)
        import torch
        torch.set_num_threads(2)
        from deepfake_detection.training import load_fk_checkpoint
        import deepfake_detection.infrastructure as infrastructure
        from deepfake_detection.study import evaluate_checkpoint
        from deepfake_detection.reporting import write_detection_report
        cache_path=_fk_paths.relocate((ROOT/'runs/heldout-001/fk/cache/training-manifest.json'), __file__);source=read(cache_path)
        heldout=read(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__));expected={r['video_id']:r['label'] for r in heldout}
        if {r['video_id']:r['label'] for r in source['assessments'] if r['split']=='test'}!=expected:raise ValueError('Original test cache population differs')
        checkpoints={name:_fk_paths.relocate((queue/name/'attempt-001/selected.pt'), __file__) for name in variants}
        locked={'checkpoints':{name:digest(path) for name,path in checkpoints.items()},
            'source_test_manifest_sha256':digest(cache_path),'heldout_manifest_sha256':digest(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__)),
            'scope':'Exploratory matched ablations, all variants predeclared; no selection based on held-out performance'}
        if (_fk_paths.relocate((run/'locked-checkpoints.json'), __file__)).exists() and read(_fk_paths.relocate((run/'locked-checkpoints.json'), __file__))!=locked:raise ValueError('Locked inputs changed')
        write_json(_fk_paths.relocate((run/'locked-checkpoints.json'), __file__),locked);models={}
        for variant,path in checkpoints.items():
            output=_fk_paths.relocate((run/variant), __file__);output.mkdir(exist_ok=True);target=_fk_paths.relocate((output/'test.json'), __file__)
            if target.exists():
                report=read(target)
                if report['checkpoint_sha256']!=locked['checkpoints'][variant]:raise ValueError('Completed evaluation model differs')
            else:
                _,checkpoint=load_fk_checkpoint(path,device='cpu')
                derived=rebind_manifest(source,cache_path.parent,checkpoint,locked['checkpoints'][variant],digest(_fk_paths.relocate((queue/variant/'attempt-001/run.json'), __file__)),expected)
                derived['ablation_cache_reuse']={'source_manifest_sha256':digest(cache_path),'operation':'Rebase unchanged cache paths; bind context to selected ablation checkpoint and training run'}
                derived_path=_fk_paths.relocate((output/'evaluation-manifest.json'), __file__);write_json(derived_path,derived)
                original=infrastructure.CachedVideoPopulation
                class Tracked(original):
                    def __getitem__(self,index):
                        if isinstance(index,int) and index%10==0 and index<len(self):
                            progress('evaluate_cached_test',variant=variant,processed=index,total=len(self))
                        return super().__getitem__(index)
                infrastructure.CachedVideoPopulation=Tracked
                try:report=evaluate_checkpoint(derived_path,path,target,split='test',device='cuda')
                finally:infrastructure.CachedVideoPopulation=original
            if len(report['videos'])!=1000 or {r['video_id']:r['label'] for r in report['videos']}!=expected:raise ValueError('Incomplete ablation result')
            models[variant]=report['videos'];finished.append(variant)
            write_json(_fk_paths.relocate((output/'metrics-summary.json'), __file__),{'video_metrics':report['video_metrics'],'window_metrics':report['window_metrics'],
                'checkpoint_sha256':locked['checkpoints'][variant],'localization_available':True})
            progress('variant_evaluation_complete',variant=variant,processed=1000,total=1000)
            torch.cuda.empty_cache()
        progress('paired_ablation_report')
        destination=_fk_paths.relocate((run/'comparison'), __file__)
        if not destination.exists():write_detection_report(models,destination,provenance={'protocol_id':'remaining-001-exploratory-core-ablations','data_kind':'research',
            'locked_models':locked,'budget':'5 epochs maximum, seed 7, same caches and validation grid'},reference='full',replicates=2000)
        if any(digest(path)!=locked['checkpoints'][name] for name,path in checkpoints.items()):raise ValueError('Selected checkpoint changed')
        write_json(_fk_paths.relocate((run/'complete.json'), __file__),{'variants':finished,'source_manifest_unchanged':digest(cache_path)==locked['source_test_manifest_sha256'],
            'files':{str(path.relative_to(run)):digest(path) for path in run.glob('*/test.json')}})
        progress('complete','FINISHED',processed=1000,total=1000)
    except Exception as error:progress('error','FAILED',error=str(error));raise
    finally:lock.close()


if __name__=='__main__':
    run=_fk_paths.relocate((ROOT/'runs/remaining-ablation-evaluation-001'), __file__);run.mkdir(exist_ok=True)
    stream=(_fk_paths.relocate((run/('worker-'+str(time.time_ns())+'.log')), __file__)).open('a',encoding='utf-8',buffering=1)
    sys.stdout=stream;sys.stderr=stream
    main()
