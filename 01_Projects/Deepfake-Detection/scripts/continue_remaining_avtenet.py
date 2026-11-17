"""Resume only the unfinished AVTENet study after Docker startup."""

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

import os,sys,time,subprocess
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import run_remaining_supplement as study
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json
RUN=_fk_paths.relocate((ROOT/'runs/remaining-supplement-002'), __file__)


def main():
    from deepfake_detection.quality_fitting import exclusive_run
    with exclusive_run(_fk_paths.relocate((RUN/'worker.lock'), __file__)):
        study.RUN=RUN
        try:
            protected=read(_fk_paths.relocate((RUN/'identity.json'), __file__))['protected']
            if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Protected original input changed')
            for name in ('forensic','profiling-fk','robustness-fk','profiling-avfakenet','robustness-avfakenet'):
                marker=read(_fk_paths.relocate((RUN/name/'complete.json'), __file__))
                if digest(_fk_paths.relocate((RUN/name/'summary.json'), __file__))!=marker['summary_sha256']:raise ValueError('Completed summary changed: '+name)
            write_json(_fk_paths.relocate((RUN/'avtenet-continuation.json'), __file__),{'script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),
                'avtenet_runner_sha256':digest(_fk_paths.relocate((ROOT/'scripts/run_remaining_avtenet.py'), __file__)),
                'completed_stages_retained':['FK','AVFakeNet','forensic','AVTENet AST','AVTENet ViViT'],
                'started_unix':time.time()})
            os.environ['DOCKER_HOST']='npipe:////./pipe/dockerDesktopLinuxEngine'
            started=time.monotonic()
            while True:
                study.progress('waiting_for_docker_linux_engine')
                probe=subprocess.run(['C:/Program Files/Docker/Docker/resources/bin/docker.exe','info','--format','{{.ServerVersion}}'],
                    capture_output=True,text=True,creationflags=subprocess.CREATE_NO_WINDOW,timeout=20)
                if probe.returncode==0:break
                if time.monotonic()-started>600:raise RuntimeError('Docker Linux engine not available after ten minutes: '+probe.stderr)
                time.sleep(5)
            study.avtenet()
            if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Protected original input changed')
            write_json(_fk_paths.relocate((RUN/'complete.json'), __file__),{'historical_inputs_unchanged':True,'aggregation_source':'remaining-supplement-001',
                'transfer':'DEFERRED_BY_USER','avtenet_summary_sha256':digest(_fk_paths.relocate((RUN/'avtenet/summary.json'), __file__))})
            study.progress('complete',status='FINISHED')
        except Exception as error:study.progress('error',status='FAILED',error=str(error));raise


if __name__=='__main__':
    RUN.mkdir(exist_ok=True)
    log=(_fk_paths.relocate((RUN/('worker-avtenet-'+str(time.time_ns())+'.log')), __file__)).open('a',encoding='utf-8',buffering=1)
    sys.stdout=log;sys.stderr=log
    main()
