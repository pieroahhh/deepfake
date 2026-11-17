"""Run the prepared core ablations sequentially; never alters historical results."""

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

import os,sys,time,re,json,subprocess,threading
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
os.environ['FK_PROGRESS_SECONDS']='10'
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json


def main():
    plan_path=_fk_paths.relocate((ROOT/'experiments/remaining-001/plan.json'), __file__);plan=read(plan_path)
    if not _fk_paths.relocate((Path(plan['active_heldout_must_complete_first'])), __file__).exists():raise ValueError('Current test is not complete')
    run=_fk_paths.relocate((ROOT/'runs/remaining-ablations-001'), __file__);run.mkdir(exist_ok=True)
    import msvcrt
    lock=(_fk_paths.relocate((run/'queue.lock'), __file__)).open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    variants=plan['minimum_batch'];completed=[];started=time.monotonic()
    state={'status':'RUNNING','stage':'starting','variants':variants,'completed_variants':completed,
           'variant':None,'epoch':None,'epochs':5,'split':None,'processed':0,'total':None}
    state_lock=threading.Lock();stop=threading.Event()
    def publish():
        with state_lock:
            value={**state,'completed_variants':list(completed),'updated_unix':time.time(),
                'elapsed_seconds_this_session':time.monotonic()-started}
        write_json(_fk_paths.relocate((run/'progress.json'), __file__),value)
    def heartbeat():
        while not stop.wait(10):publish()
    threading.Thread(target=heartbeat,daemon=True).start()
    stamp=str(time.time_ns());log_path=_fk_paths.relocate((run/('queue-'+stamp+'.log')), __file__)
    write_json(_fk_paths.relocate((run/'queue-identity.json'), __file__),{'variants':variants,'plan_sha256':digest(plan_path),
        'queue_script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'runner_sha256':digest(_fk_paths.relocate((ROOT/'scripts/run_prepared_ablation.py'), __file__)),
        'scope':'Train and validation only, five-epoch budget per variant, seed 7','test_inference':'NOT_STARTED'})
    try:
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            for variant in variants:
                finished=_fk_paths.relocate((run/variant/'completed.json'), __file__)
                if finished.exists():
                    _=read(finished);completed.append(variant);continue
                if (_fk_paths.relocate((run/'pause.request'), __file__)).exists():
                    state.update(status='PAUSED',stage='between_variants');publish();return
                with state_lock:state.update(variant=variant,stage='cache_integrity_and_training_initialization',epoch=None,split=None,processed=0,total=None)
                publish();log.write(f'START {variant}\n')
                command=[str(_fk_paths.relocate((ROOT/'.venv-gpu/Scripts/python.exe'), __file__)),'-u',str(_fk_paths.relocate((ROOT/'scripts/run_prepared_ablation.py'), __file__)),'--variant',variant]
                child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
                for line in child.stdout:
                    log.write(line)
                    epoch=re.search(r'^epoch (\d+)/(\d+)',line)
                    clips=re.search(r'^(train|validation): clips (\d+)/(\d+)',line)
                    with state_lock:
                        if epoch:state.update(epoch=int(epoch[1]),epochs=int(epoch[2]),stage='training',processed=0,total=None)
                        if clips:state.update(split=clips[1],processed=int(clips[2]),total=int(clips[3]),stage=clips[1])
                        if 'saved; train loss' in line:state.update(stage='epoch_saved')
                        state['last_log_line']=line.strip()[-800:]
                    publish()
                code=child.wait()
                if code or not finished.exists():raise RuntimeError(f'{variant} stopped (exit {code}); inspect {log_path}. Completed checkpoints retained; no automatic overwrite.')
                completed.append(variant);state.update(stage='variant_complete');publish()
        state.update(status='FINISHED',stage='core_training_and_validation_complete');publish()
        write_json(_fk_paths.relocate((run/'queue-complete.json'), __file__),{'variants':completed,'plan_sha256':digest(plan_path),
            'results':{name:read(_fk_paths.relocate((run/name/'completed.json'), __file__)) for name in completed},
            'test_inference':'NOT_RUN; review validation and lock new checkpoints before scoring held-out'})
    except Exception as error:
        state.update(status='FAILED',stage='error',error=str(error));publish();raise
    finally:stop.set();lock.close()


if __name__=='__main__':main()
