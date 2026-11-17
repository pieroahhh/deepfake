"""Read-only watcher for every remaining experiment, runnable from any folder."""

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

import argparse,json,time,re
import os
from pathlib import Path
from datetime import datetime
ROOT=_fk_paths.relocate((Path(__file__).resolve().parents[1]/'runs'), __file__)
JOBS={'TRAIN':'remaining-ablations-001','EVALUATE':'remaining-ablation-evaluation-001',
      'PREPARE':'remaining-derivatives-001','NEXT':'remaining-supplement-001'}


def worker_is_active(folder):
    if os.name!='nt':return None
    import msvcrt
    try:
        with (_fk_paths.relocate((folder/'worker.lock'), __file__)).open('rb') as stream:
            try:msvcrt.locking(stream.fileno(),msvcrt.LK_NBRLCK,1)
            except OSError:return True
            stream.seek(0);msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            return False
    except OSError:return None


def aggregation_progress(state,folder):
    """Supplement worker logs already contain counts; read without changing it."""
    variant=state.get('stage')
    if state.get('status')!='RUNNING' or variant not in ('window_max','window_mean','video_max','video_mean'):return state
    path=_fk_paths.relocate((folder/'aggregation'), __file__)/(variant+'.log')
    try:
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536));text=stream.read().decode('utf-8',errors='replace')
    except OSError:return state
    value=dict(state);epochs=list(re.finditer(r'^epoch (\d+)/(\d+)',text,re.M))
    if epochs:
        epoch=epochs[-1];value.update(epoch=int(epoch[1]),epochs=int(epoch[2]))
        text=text[epoch.end():]
    counts=list(re.finditer(r'^(train|validation): clips (\d+)/(\d+)[^\r\n]*',text,re.M))
    if counts:
        match=counts[-1]
        value.update(stage=variant+'/'+match[1],processed=int(match[2]),total=int(match[3]),last_log_line=match[0])
    value['updated_unix']=max(value.get('updated_unix',0),path.stat().st_mtime)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=float,default=5)
    parser.add_argument('--once',action='store_true');args=parser.parse_args()
    if args.interval<1:parser.error('interval must be at least one second')
    previous={};rates={}
    print('Read-only watcher; Ctrl+C closes the watcher. ETA is for the current phase only.',flush=True)
    while True:
        for label,folder in JOBS.items():
            if label=='NEXT' and (_fk_paths.relocate((ROOT/'remaining-supplement-active.json'), __file__)).exists():
                active=json.loads((_fk_paths.relocate((ROOT/'remaining-supplement-active.json'), __file__)).read_text())['folder']
                if active not in ('remaining-supplement-001','remaining-supplement-002'):raise ValueError('Unknown continuation')
                folder=active
            try:state=json.loads((_fk_paths.relocate((ROOT/folder/'progress.json'), __file__)).read_text(encoding='utf-8-sig'))
            except (OSError,ValueError):continue
            if label=='NEXT':state=aggregation_progress(state,_fk_paths.relocate((ROOT/folder), __file__))
            stage=state.get('stage','?');done=state.get('processed',0);total=state.get('total')
            age=max(0,time.time()-state.get('updated_unix',time.time()))
            if label=='NEXT' and state.get('status')=='RUNNING' and age>30 and worker_is_active(_fk_paths.relocate((ROOT/folder), __file__)) is False:
                state={**state,'status':'STOPPED','error':'Worker lock is free: no active experiment worker. Saved rows retained; this is not live progress.'}
            # Initialization hashes the entire frozen population before counting.
            threshold=180 if state.get('total') is None else 60
            stale=age>threshold and state.get('status') not in ('FINISHED','FAILED','PAUSED','STOPPED')
            key=(state.get('status'),stage,state.get('variant'),state.get('epoch'),done,total,state.get('last_log_line'),stale)
            if previous.get(label)==key:continue
            previous[label]=key
            text=f'{datetime.now():%H:%M:%S} {label}: {state.get("status","?")} | {stage}'
            if state.get('variant'):text+=' | '+state['variant']
            if state.get('epoch'):text+=f' | epoch {state["epoch"]}/{state.get("epochs","?")}'
            if total:
                text+=f' | {done:,}/{total:,} ({100*done/total:.1f}%)'
                phase=(stage,state.get('variant'),state.get('epoch'),total)
                prior=rates.get(label)
                if prior is None or prior[0]!=phase or done<prior[1]:
                    rates[label]=(phase,done,time.monotonic())
                elif done>prior[1] and time.monotonic()-prior[2]>=20 and not stale:
                    eta=(total-done)*(time.monotonic()-prior[2])/(done-prior[1])
                    text+=f' | phase left ~{eta/60:.1f} min; finish ~{datetime.fromtimestamp(time.time()+eta):%H:%M}'
            if state.get('variants'):
                text+=f' | completed {len(state.get("completed_variants",[]))}/{len(state["variants"])} variants'
            if stale:text+=f' | WARNING: progress file unchanged for {age/60:.1f} min; check worker/log'
            print(text,flush=True)
            if state.get('error'):print('  ERROR: '+str(state['error']),flush=True)
        if args.once:return
        time.sleep(args.interval)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:print('\nWatcher closed; experiment jobs continue.')
