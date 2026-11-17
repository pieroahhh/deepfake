"""CPU-only raw audio preparation, two bounded workers, resumable verified cache."""

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

import os,sys,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from baselines.common.io import read,write,digest
from baselines.common.population import Population,windows
from baselines.common.media import audio
from baselines.avtenet.heldout_preparation import FastRawCache,audio_cache_identity
from deepfake_detection.quality_fitting import exclusive_run
from baselines.avtenet.input_cache import prefetch_reads
from concurrent.futures import ThreadPoolExecutor

def main():
    root=_fk_paths.relocate((ROOT/'runs/auto-next-001'), __file__);folder=_fk_paths.relocate((root/'audio-prewarm'), __file__);folder.mkdir(exist_ok=True)
    manifest=_fk_paths.relocate((root/'population/manifest.json'), __file__);audit=read(_fk_paths.relocate((root/'population/audit.json'), __file__))
    if digest(manifest)!=audit['manifest_sha256']:raise ValueError('Held-out manifest mismatch')
    population=Population(_fk_paths.relocate((root/'index'), __file__),_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'avtenet')
    rows=list(population.rows('test'))
    if len(rows)!=1000 or population.metadata['population_sha256']!=digest(manifest):raise ValueError('Held-out population mismatch')
    identity=audio_cache_identity(manifest)
    cache=FastRawCache(_fk_paths.relocate((root/'heldout/prepared-inputs'), __file__),identity)
    def prepare(row):
        path=population.path(row);good=bad=0
        for start,end in windows(row['duration']):
            if (_fk_paths.relocate((ROOT/'runs/worker-c/avtenet/pause.request'), __file__)).exists():raise RuntimeError('Paused by request')
            try:cache.get(row,start,lambda:{'waveform':audio(path,start,2)})
            except (ValueError,OSError):bad+=1
            else:good+=1
        return good,bad
    with exclusive_run(_fk_paths.relocate((folder/'worker.lock'), __file__)):
        started=time.monotonic();done=good=bad=0;last=0.
        try:
            with ThreadPoolExecutor(max_workers=2,thread_name_prefix='audio-prewarm') as pool:
                # At most two clips are outstanding. Drain in population order.
                iterator=iter(rows);pending=[]
                for _ in range(2):pending.append(pool.submit(prepare,next(iterator)))
                while pending:
                    result=pending.pop(0).result();good+=result[0];bad+=result[1];done+=1
                    row=next(iterator,None)
                    if row is not None:pending.append(pool.submit(prepare,row))
                    if time.monotonic()-last>=10 or done==len(rows):
                        last=time.monotonic();value={'status':'FINISHED' if done==len(rows) else 'RUNNING',
                            'clips':done,'total_clips':len(rows),'usable_windows':good,'failed_windows':bad,
                            'elapsed_seconds':last-started,'updated_unix':time.time(),'cache_identity':identity}
                        write(_fk_paths.relocate((folder/'status.json'), __file__),value,replace=True)
                        print('CPU audio prewarm',done,'/1000 clips;',good,'usable;',bad,'unavailable',flush=True)
        except Exception as e:
            write(_fk_paths.relocate((folder/'status.json'), __file__),{'status':'FAILED','error':str(e),'clips':done,'updated_unix':time.time()},replace=True)
            raise

if __name__=='__main__':main()
