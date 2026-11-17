"""Wait for the existing AV-HuBERT run, verify, fuse, export, and test frozen weights."""

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

import argparse,json,os,shutil,subprocess,sys,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,write,digest,checked_path,fingerprint
from deepfake_detection.quality_fitting import exclusive_run
from deepfake_detection.gpu_queue import gpu_slot
RUN=_fk_paths.relocate((ROOT/'runs/worker-c/avtenet'), __file__)
AUTO=_fk_paths.relocate((ROOT/'runs/auto-next-001'), __file__)
CONFIG=_fk_paths.relocate((ROOT/'baselines/avtenet/configs/avtenet.hf-joint-adaptation-v1.json'), __file__)
INDEX=_fk_paths.relocate((ROOT/'runs/two-day-001/avtenet/shared-index'), __file__)
MANIFEST=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__)


def state(status,stage,**extra):
    write(_fk_paths.relocate((AUTO/'status.json'), __file__),{'status':status,'stage':stage,'updated_unix':time.time(),**extra},replace=True)
    print(status,stage,flush=True)


def copy_verified(source,dest,sha):
    if digest(source)!=sha:raise ValueError('Source changed: '+str(source))
    if dest.exists():
        if digest(dest)!=sha:raise ValueError('Destination differs: '+str(dest))
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.copying');shutil.copy2(source,temp)
    if digest(temp)!=sha:raise ValueError('Copy checksum mismatch')
    temp.replace(dest)


def setup(args):
    bundle=_fk_paths.relocate((Path(args.bundle)), __file__).resolve();meta=read(_fk_paths.relocate((bundle/'bundle.json'), __file__))
    if meta['schema']!='avtenet-machine-a-continuation-v1' or meta['population_sha256']!=digest(MANIFEST) or meta['index_sha256']!=digest(_fk_paths.relocate((INDEX/'index.sqlite'), __file__)):
        raise ValueError('Continuation population/index mismatch')
    # Only completed C expert outputs are imported; the running audiovisual code/output is untouched.
    imported={}
    for folder in ('audio','audio-features','visual','visual-features'):
        prefix='code/runs/worker-c/avtenet/'+folder+'/'
        entries={k[len(prefix):]:v for k,v in meta['files'].items() if k.startswith(prefix)}
        if not entries:raise ValueError('Missing completed stage '+folder)
        dest=_fk_paths.relocate((RUN/folder), __file__);staging=_fk_paths.relocate((RUN/(folder+'.importing')), __file__)
        target=dest if dest.exists() else staging
        for relative,sha in entries.items():
            copy_verified(checked_path(bundle,prefix+relative),checked_path(target,relative),sha)
        if not dest.exists():target.rename(dest)
        imported.update({folder+'/'+rel:sha for rel,sha in entries.items()})
    heldout=_fk_paths.relocate((Path(args.heldout)), __file__).resolve();population=_fk_paths.relocate((heldout/'population'), __file__)
    audit=read(_fk_paths.relocate((population/'audit.json'), __file__));rows=read(_fk_paths.relocate((population/'manifest.json'), __file__));used=read(MANIFEST)
    if digest(_fk_paths.relocate((population/'manifest.json'), __file__))!=audit['manifest_sha256'] or audit['used_manifest_sha256']!=digest(MANIFEST) or len(rows)!=1000 or any(r['split']!='test' for r in rows):
        raise ValueError('Held-out audit/population mismatch')
    for key in ('video_id','source_family','source_sha256'):
        values=[r[key] for r in rows]
        if len(set(values))!=len(values) or set(values)&{r[key] for r in used}:raise ValueError('Held-out leakage: '+key)
    for name in ('manifest.json','audit.json'):
        copy_verified(_fk_paths.relocate((population/name), __file__),_fk_paths.relocate((AUTO/'population'/name), __file__),digest(_fk_paths.relocate((population/name), __file__)))
    for row in rows:
        copy_verified(checked_path(_fk_paths.relocate((heldout/'dataset'), __file__),row['path']),checked_path(_fk_paths.relocate((ROOT.parent/'dataset'), __file__),row['path']),row['source_sha256'])
    header={'schema':'avtenet-auto-next-v1','bundle_sha256':digest(_fk_paths.relocate((bundle/'bundle.json'), __file__)),
        'training_manifest_sha256':digest(MANIFEST),'heldout_manifest_sha256':audit['manifest_sha256'],
        'config_sha256':digest(CONFIG),'fusion_epochs':1,'seed':7,'imported_files':imported}
    if (_fk_paths.relocate((AUTO/'setup.json'), __file__)).exists():
        if read(_fk_paths.relocate((AUTO/'setup.json'), __file__))!=header:raise ValueError('Auto-next setup identity changed')
    else:write(_fk_paths.relocate((AUTO/'setup.json'), __file__),header)
    from baselines.common.population import prepare,Population
    if not (_fk_paths.relocate((AUTO/'index'), __file__)).exists():prepare(_fk_paths.relocate((AUTO/'population/manifest.json'), __file__),_fk_paths.relocate((AUTO/'index'), __file__),'avtenet')
    pop=Population(_fk_paths.relocate((AUTO/'index'), __file__),_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'avtenet')
    if pop.metadata['population_sha256']!=audit['manifest_sha256']:raise ValueError('Held-out index changed')
    print('PRECHECK PASSED: imported C experts and verified all 1000 held-out clips',flush=True)


def verify_expert(stage):
    from baselines.common.training import load_checkpoint
    from baselines.avtenet.feature_resume import validate_rows
    from baselines.common.population import Population,windows
    import sqlite3
    meta=read(_fk_paths.relocate((RUN/(stage+'-features')/'cache.json'), __file__))
    if meta['stage']!=stage or meta['integration_only'] or meta['index_sha256']!=digest(_fk_paths.relocate((INDEX/'index.sqlite'), __file__)):
        raise ValueError('Invalid completed expert cache: '+stage)
    if digest(_fk_paths.relocate((RUN/(stage+'-features')/'features.sqlite'), __file__))!=meta['payload_sha256']:raise ValueError('Feature payload checksum mismatch')
    attempts=sorted((_fk_paths.relocate((RUN/stage), __file__)).glob('attempt-*'),reverse=True)
    checkpoint=next((_fk_paths.relocate((p/'best.pt'), __file__) for p in attempts if (_fk_paths.relocate((p/'best.pt'), __file__)).exists() and digest(_fk_paths.relocate((p/'best.pt'), __file__))==meta['checkpoint_sha256']),None)
    if checkpoint is None:raise ValueError('Selected checkpoint missing: '+stage)
    selected=load_checkpoint(checkpoint);run=read(_fk_paths.relocate((checkpoint.parent/'run.json'), __file__));metrics=read(_fk_paths.relocate((checkpoint.parent/'metrics.json'), __file__))
    if selected['identity']!=meta['identity'] or run['identity']!=meta['identity'] or selected['best_epoch']!=metrics['best_epoch'] or selected['epoch']!=1:
        raise ValueError('Expert selection provenance mismatch')
    if meta['identity']['population_sha256']!=digest(MANIFEST):raise ValueError('Expert population mismatch')
    from contextlib import closing
    with closing(sqlite3.connect((_fk_paths.relocate((RUN/(stage+'-features')/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro',uri=True)) as conn:
        if conn.execute('PRAGMA quick_check').fetchone()[0]!='ok':raise ValueError('Invalid SQLite cache')
        keys,rows,intervals=validate_rows(conn,Population(INDEX,_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'avtenet'))
        if len(keys)!=meta['windows']:raise ValueError('Feature count mismatch')
    return checkpoint,meta


def command(name,argv,*,gpu=False,docker=False):
    if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED by pause.request')
    state('RUNNING',name,log=str(_fk_paths.relocate((AUTO/(name+'.log')), __file__)))
    env=os.environ.copy();env['OPENBLAS_NUM_THREADS']='1'
    with gpu_slot(gpu):
        if gpu:env['FK_GPU_QUEUE_MANAGED']='1'
        if docker:
            argv=['docker','run','--rm','--gpus','all','--shm-size','1g','-e','FK_GPU_QUEUE_MANAGED=1','-e','PYTHONUNBUFFERED=1',
                '-e','OMP_NUM_THREADS=8','-e','OPENBLAS_NUM_THREADS=1','--mount',f'type=bind,source={ROOT},target=/workspace',
                '--mount',f'type=bind,source={_fk_paths.relocate((ROOT.parent/"dataset"), __file__)},target=/dataset,readonly',
                '--entrypoint','python','fk-avhubert:cuda117-v1',*argv]
        else:argv=[sys.executable,'-u',*map(str,argv)]
        with (_fk_paths.relocate((AUTO/(name+'.log')), __file__)).open('a',encoding='utf-8') as stream:
            result=subprocess.run(argv,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(name+' failed; inspect its log')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True);p.add_argument('--heldout',required=True)
    p.add_argument('--check',action='store_true');args=p.parse_args()
    AUTO.mkdir(parents=True,exist_ok=True)
    with exclusive_run(_fk_paths.relocate((AUTO/'controller.lock'), __file__)):
        try:
            setup(args)
            if args.check:state('PRECHECK_PASSED','ready_to_wait');return
            while not (_fk_paths.relocate((RUN/'audiovisual-features/cache.json'), __file__)).exists():
                current=read(_fk_paths.relocate((RUN/'status.json'), __file__))
                if current['status'] in ('FAILED','PAUSED'):raise RuntimeError('Upstream AV-HuBERT '+current['status'])
                if current['status']=='FINISHED':raise RuntimeError('Upstream reports FINISHED but audiovisual cache is missing')
                if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED by pause.request')
                state('WAITING','avhubert_training_and_features',upstream=current)
                time.sleep(30)
            # Wait until the existing controller has released its workflow lock.
            while True:
                try:
                    with exclusive_run(_fk_paths.relocate((RUN/'workflow.lock'), __file__)):pass
                    break
                except ValueError:time.sleep(5)
            state('RUNNING','verify_three_experts')
            experts={stage:verify_expert(stage) for stage in ('audio','visual','audiovisual')}
            for key in ('profile_sha256','preparation_sha256','index_sha256'):
                if len({meta[key] for _,meta in experts.values()})!=1:raise ValueError('Experts differ: '+key)
            write(_fk_paths.relocate((AUTO/'expert-verification.json'), __file__),{stage:{'checkpoint':str(path),'checkpoint_sha256':digest(path),'windows':meta['windows']} for stage,(path,meta) in experts.items()},replace=True)
            # Interrupted merge/evaluation can be regenerated from immutable
            # selected inputs. Preserve partial artifacts before a new attempt.
            for name,marker in (('fusion-features','cache.json'),('evaluation','report.json')):
                partial=_fk_paths.relocate((RUN/name), __file__)
                if partial.exists() and not (_fk_paths.relocate((partial/marker), __file__)).exists():
                    partial.resolve().relative_to(RUN.resolve())
                    partial.rename(_fk_paths.relocate((RUN/(name+'.interrupted-'+str(time.time_ns()))), __file__))
            command('fusion_and_validation',['scripts/run_avtenet_subset.py','--stages','fusion','--index',INDEX,
                '--manifest',MANIFEST,'--dataset-root',_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'--assets','runs/worker-c/assets.local.json',
                '--docker-assets','runs/worker-c/assets.docker.local.json','--output',RUN,'--epochs','1','--threads','8',
                '--checkpoint-seconds','120','--progress-seconds','10'])
            if read(_fk_paths.relocate((RUN/'status.json'), __file__))['status']!='FINISHED':raise RuntimeError('Fusion did not finish')
            fusion=_fk_paths.relocate((sorted((RUN/'fusion').glob('attempt-*'))[-1]/'best.pt'), __file__)
            from baselines.common.training import load_checkpoint
            selected=load_checkpoint(fusion);cache=read(_fk_paths.relocate((RUN/'fusion-features/cache.json'), __file__))
            if selected['identity']['feature_cache']!=cache or digest(_fk_paths.relocate((RUN/'fusion-features/features.sqlite'), __file__))!=cache['payload_sha256']:
                raise ValueError('Fusion checkpoint/cache mismatch')
            report=read(_fk_paths.relocate((RUN/'evaluation/report.json'), __file__))
            if report['checkpoint_sha256']!=digest(fusion) or report['integration_only']:raise ValueError('Validation report mismatch')
            from transfer_worker_results import export,receive
            returned=_fk_paths.relocate((ROOT.parent/'return-c-validation'), __file__)
            if not returned.exists():
                staging=returned.with_name(returned.name+'.building')
                if staging.exists():staging.rename(staging.with_name(staging.name+'.interrupted-'+str(time.time_ns())))
                export(_fk_paths.relocate((ROOT/'runs/worker-c'), __file__),staging,role='C');staging.rename(returned)
            else:
                for rel,sha in read(_fk_paths.relocate((returned/'return.json'), __file__))['files'].items():
                    if digest(checked_path(returned,rel))!=sha:raise ValueError('Export changed')
            relative=lambda path:_fk_paths.relocate((Path(path)), __file__).relative_to(ROOT).as_posix()
            checkpoints={stage:relative(path) for stage,(path,_) in experts.items()};checkpoints['fusion']=relative(fusion)
            plan={'schema':'avtenet-frozen-heldout-plan-v1','manifest':relative(_fk_paths.relocate((AUTO/'population/manifest.json'), __file__)),
                'training_manifest':relative(MANIFEST),'index':relative(_fk_paths.relocate((AUTO/'index'), __file__)),'training_index':relative(INDEX),
                'config':relative(CONFIG),'training_fusion_cache':relative(_fk_paths.relocate((RUN/'fusion-features/cache.json'), __file__)),
                'checkpoints':checkpoints,'output':relative(_fk_paths.relocate((AUTO/'heldout'), __file__)),'files':{}}
            paths=[_fk_paths.relocate((AUTO/'population/manifest.json'), __file__),_fk_paths.relocate((AUTO/'population/audit.json'), __file__),_fk_paths.relocate((AUTO/'index/index.json'), __file__),_fk_paths.relocate((AUTO/'index/index.sqlite'), __file__),
                   MANIFEST,_fk_paths.relocate((INDEX/'index.json'), __file__),_fk_paths.relocate((INDEX/'index.sqlite'), __file__),CONFIG,_fk_paths.relocate((RUN/'fusion-features/cache.json'), __file__),*[_fk_paths.relocate((ROOT/v), __file__) for v in checkpoints.values()]]
            plan['files']={relative(path):digest(path) for path in paths}
            planpath=_fk_paths.relocate((AUTO/'test-plan.json'), __file__)
            if planpath.exists():
                if read(planpath)!=plan:raise ValueError('Selected frozen test plan changed')
            else:write(planpath,plan)
            for stage in ('audio','visual','audiovisual','evaluate'):
                if stage=='audio':
                    warm=_fk_paths.relocate((AUTO/'audio-prewarm/status.json'), __file__)
                    if (_fk_paths.relocate((AUTO/'audio-prewarm-launcher.json'), __file__)).exists():
                        while True:
                            if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED by pause.request')
                            if not warm.exists():raise RuntimeError('Audio prewarm has no status; inspect its log')
                            progress=read(warm)
                            if progress['status']=='FINISHED':break
                            if progress['status']=='FAILED' or time.time()-progress['updated_unix']>180:
                                raise RuntimeError('Audio prewarm failed/stalled; inspect audio-prewarm logs before resume')
                            state('WAITING','cpu_audio_prewarm',progress=progress)
                            time.sleep(10)
                container=stage=='audiovisual';convert=lambda path:'/workspace/'+relative(path) if container else str(path)
                command('heldout_'+stage,[convert(_fk_paths.relocate((ROOT/'scripts/run_avtenet_frozen_test.py'), __file__)),'--plan',convert(planpath),
                    '--stage',stage,'--dataset-root','/dataset' if container else str(_fk_paths.relocate((ROOT.parent/'dataset'), __file__)),
                    '--assets',convert(_fk_paths.relocate((ROOT/'runs/worker-c'/('assets.docker.local.json' if container else 'assets.local.json')), __file__))],gpu=True,docker=container)
            final=read(_fk_paths.relocate((AUTO/'heldout/evaluate/report.json'), __file__))
            if final['metrics']['videos']!=1000:raise ValueError('Held-out report is incomplete')
            write(_fk_paths.relocate((AUTO/'summary.json'), __file__),{'status':'FINISHED','validation':report['metrics'],'heldout':final['metrics'],
                'validation_return':str(returned),'heldout_report':str(_fk_paths.relocate((AUTO/'heldout/evaluate/report.json'), __file__)),
                'heldout_predictions':str(_fk_paths.relocate((AUTO/'heldout/evaluate/predictions.jsonl'), __file__))},replace=True)
            state('FINISHED','all_stages_complete',summary=str(_fk_paths.relocate((AUTO/'summary.json'), __file__)))
        except Exception as error:
            state('PAUSED' if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists() else 'FAILED','stopped',error=str(error))
            raise


if __name__=='__main__':main()
