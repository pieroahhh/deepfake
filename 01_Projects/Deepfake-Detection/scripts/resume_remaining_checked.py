"""Check runtime dependencies before resuming the unchanged precision continuation."""

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

import sys,os,importlib,traceback,json,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]


def preflight():
    versions={};missing=[]
    for name in ('torch','numpy','cv2','av','transformers','safetensors','soundfile','scipy','sklearn','matplotlib','facenet_pytorch','psutil'):
        try:module=importlib.import_module(name)
        except ImportError as error:missing.append(f'{name}: {error}')
        else:versions[name]=getattr(module,'__version__','imported')
    if missing:raise RuntimeError('Missing runtime dependencies:\n'+'\n'.join(missing))
    import torch,psutil
    if not torch.cuda.is_available():raise RuntimeError('CUDA is unavailable')
    rss=psutil.Process().memory_info().rss
    return {'packages':versions,'cuda_device':torch.cuda.get_device_name(),'rss_probe_bytes':rss,'updated_unix':time.time()}


if __name__=='__main__':
    try:
        proof=preflight()
        run=_fk_paths.relocate((ROOT/'runs/remaining-supplement-002'), __file__);run.mkdir(exist_ok=True)
        (_fk_paths.relocate((run/'dependency-preflight.json'), __file__)).write_text(json.dumps(proof,indent=2),encoding='utf-8')
        print('Runtime dependency/CUDA/RAM checks passed.',flush=True)
        if '--check-only' not in sys.argv:
            import continue_remaining_after_precision_fix as continuation
            path=_fk_paths.relocate((run/('worker-checked-'+str(time.time_ns())+'.log')), __file__)
            print('Running pending experiments. Keep this window open.\nLog: '+str(path),flush=True)
            console=sys.stderr
            with path.open('a',encoding='utf-8',buffering=1) as log:
                sys.stdout=log;sys.stderr=log
                try:continuation.main()
                finally:sys.stdout=sys.__stdout__;sys.stderr=console
            print('Experiment queue completed.',flush=True)
    except KeyboardInterrupt:print('Stopped by keyboard; saved outputs retained.',file=sys.stderr)
    except Exception:
        traceback.print_exc();sys.exit(1)
