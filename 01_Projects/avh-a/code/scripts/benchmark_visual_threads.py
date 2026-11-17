"""Small bounded real CUDA preprocessing benchmark; run only while worker paused."""

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

import os,sys,time,json,gc
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from baselines.common.io import read,write
from baselines.common.population import Population,windows
from baselines.common.media import detector
from baselines.avtenet.pipeline import Prepared
from baselines.avtenet.heldout_preparation import FastRawCache
from baselines.avtenet.hf_vivit import HFVivitBackend
from transformers import VivitImageProcessor

def main():
    assets=read(_fk_paths.relocate((ROOT/'runs/worker-c/assets.local.json'), __file__))
    pop=Population(_fk_paths.relocate((ROOT/'runs/auto-next-001/index'), __file__),_fk_paths.relocate((ROOT.parent/'dataset'), __file__),'avtenet')
    rows=list(pop.rows('test'))[:3]
    # Need only the production processor, not a second ViViT transformer.
    class Backend:
        def prepare(self,frames):
            return torch.from_numpy(np.ascontiguousarray(processor(list(frames),return_tensors='np')['pixel_values']))
    processor=VivitImageProcessor.from_pretrained(_fk_paths.relocate_value((assets['experts']['visual']['directory']), __file__),local_files_only=True)
    detect,_=detector('cuda');reference=None;results=[]
    for threads in (8,2,4,1):
        torch.set_num_threads(threads)
        prepared=Prepared(pop,'visual',Backend(),assets,detect,reuse_video_decoder=True,cache_face_detections=True)
        outcomes=[];started=time.perf_counter()
        for row in rows:
            prepared.detection_cache.clear();path=pop.path(row)
            for start,end in windows(row['duration'])[:3]:
                try:
                    raw=prepared.raw_inputs(path,start);value=prepared.transform(raw)
                    outcomes.append((None,raw['faces'],value))
                except (ValueError,OSError) as e:outcomes.append((str(e),None,None))
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        if reference is None:reference=outcomes
        else:
            assert len(reference)==len(outcomes)
            for expected,actual in zip(reference,outcomes):
                assert expected[0]==actual[0],(expected[0],actual[0])
                if expected[0] is None:
                    np.testing.assert_array_equal(expected[1],actual[1]);assert torch.equal(expected[2],actual[2])
        results.append({'threads':threads,'seconds':elapsed,'windows':len(outcomes),
            'usable':sum(v[0] is None for v in outcomes),'input_parity':True})
        prepared.video_reader.close()
        print(json.dumps(results[-1]),flush=True)
    if not any(v[0] is None for v in reference):raise ValueError('No usable input parity sample')
    write(_fk_paths.relocate((ROOT/'runs/auto-next-001/visual-thread-benchmark.json'), __file__),results,replace=True)

if __name__=='__main__':main()
