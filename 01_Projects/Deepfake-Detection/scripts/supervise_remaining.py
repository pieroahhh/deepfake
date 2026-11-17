"""Visible progress and bounded recovery for unexplained native worker exits."""

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

import os,sys,json,time,subprocess,signal
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
RUN=_fk_paths.relocate((ROOT/'runs/remaining-supplement-002'), __file__)
sys.path.insert(0,str(_fk_paths.relocate((ROOT/'scripts'), __file__)))


def may_retry(code,state,launched,attempt):
    # Never retry Ctrl+C, an explicit fresh failure, or indefinitely.
    if code in (-1073741510,3221225786,130):return False
    if state.get('updated_unix',0)>=launched and state.get('status') in ('FAILED','PAUSED'):return False
    return attempt<3


def read_state():
    try:return json.loads((_fk_paths.relocate((RUN/'progress.json'), __file__)).read_text(encoding='utf-8-sig'))
    except (OSError,ValueError):return {}


def main():
    RUN.mkdir(exist_ok=True)
    from resume_remaining_checked import preflight
    proof=preflight();print('Dependency/CUDA/RAM checks passed. Supervisor active.',flush=True)
    print('Progress is shown here. Keep this window open; Ctrl+C stops the worker.',flush=True)
    exits=[]
    for attempt in range(1,4):
        launched=time.time();last='';last_print=0.
        log_path=_fk_paths.relocate((RUN/('supervisor-child-'+str(time.time_ns())+'.log')), __file__)
        print(f'Starting worker attempt {attempt}/3; retained clip results will be verified/reused.',flush=True)
        with log_path.open('a',encoding='utf-8',buffering=1) as log:
            child=subprocess.Popen([sys.executable,'-u',str(_fk_paths.relocate((ROOT/'scripts/resume_remaining_checked.py'), __file__))],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=(subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP) if os.name=='nt' else 0)
            try:
                while child.poll() is None:
                    state=read_state();fresh=state.get('updated_unix',0)>=launched
                    key=(state.get('status'),state.get('stage'),state.get('processed'),state.get('total'))
                    if fresh and (key!=last or time.time()-last_print>=30):
                        done,total=state.get('processed',0),state.get('total')
                        counts='' if not total else f' | {done}/{total} ({100*done/total:.1f}%)'
                        age=max(0,time.time()-state.get('updated_unix',0))
                        print(f'{time.strftime("%H:%M:%S")} {state.get("status")} | {state.get("stage")}{counts} | last update {age:.0f}s ago',flush=True)
                        if state.get('error'):print('ERROR: '+str(state['error']),flush=True)
                        last=key;last_print=time.time()
                    time.sleep(3)
                code=child.wait()
            except KeyboardInterrupt:
                if child.poll() is None:
                    child.terminate()
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                print('Stopped by keyboard. Completed outputs retained.',flush=True);return
        state=read_state();entry={'attempt':attempt,'exit_code':code,'exit_hex':hex(code&0xffffffff),
            'state':state,'log':str(log_path),'ended_unix':time.time()};exits.append(entry)
        (_fk_paths.relocate((RUN/'supervisor-exits.json'), __file__)).write_text(json.dumps(exits,indent=2),encoding='utf-8')
        if code==0 and (_fk_paths.relocate((RUN/'complete.json'), __file__)).exists():print('All queued experiments completed.',flush=True);return
        print(f'Worker exited: {code} ({entry["exit_hex"]}). Log: {log_path}',flush=True)
        if not may_retry(code,state,launched,attempt):
            print('Stopped for diagnosis; no further automatic retries.',flush=True)
            print(log_path.read_text(encoding='utf-8')[-2500:],flush=True);return
        print('Unexpected exit; restarting from verified saved outputs in 3 seconds.',flush=True)
        time.sleep(3)


if __name__=='__main__':main()
