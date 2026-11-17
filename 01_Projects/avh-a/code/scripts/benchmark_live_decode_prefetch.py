"""Read-only model parity and bounded A/B timing; run only with production paused."""

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

import os,sys,time,json,sqlite3,unittest
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'tests'), __file__))]
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
import torch
from baselines.common.io import read,write
from baselines.common.population import Population,windows
from baselines.common.media import detector
from baselines.common.training import load_checkpoint,move
from baselines.avtenet.pipeline import Prepared,stage_model
from baselines.avtenet.assets import registry
from baselines.avtenet.model import load_component_profile


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,torch.Tensor):assert torch.equal(a,b),'Tensor parity failed'
    else:np.testing.assert_array_equal(a,b)


def main():
    assert (_fk_paths.relocate((ROOT/'runs/worker-c/avtenet/pause.request'), __file__)).exists(),'Production must be paused'
    torch.set_num_threads(1)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromName('test_decode_prefetch'))
    if not result.wasSuccessful():raise RuntimeError('Pinned-runtime CPU tests failed')
    torch.set_num_threads(8)
    plan=read(_fk_paths.relocate((ROOT/'runs/auto-next-001/test-plan.json'), __file__))
    pop=Population(_fk_paths.relocate((ROOT/plan['index']), __file__),'/dataset','avtenet')
    rows={r['video_id']:r for r in pop.rows('test')}
    assets=registry(_fk_paths.relocate((ROOT/'runs/worker-c/assets.docker.local.json'), __file__))
    # Upstream AV-HuBERT uses sibling absolute imports in its user module.
    from native_avhubert_bootstrap import bootstrap
    bootstrap(assets['experts']['audiovisual']['repository'])
    profile=load_component_profile(_fk_paths.relocate((ROOT/plan['config']), __file__))
    model,backend,_=stage_model('audiovisual',profile,assets)
    selected=load_checkpoint(_fk_paths.relocate((ROOT/plan['checkpoints']['audiovisual']), __file__))
    model.load_state_dict(selected['model_state'],strict=True)
    del selected
    model.cuda().eval().requires_grad_(False)
    detect,_=detector('cuda')
    db=_fk_paths.relocate((ROOT/'runs/auto-next-001/heldout/audiovisual/features.sqlite'), __file__)
    samples=[]
    with sqlite3.connect(db.resolve().as_uri()+'?mode=ro',uri=True) as conn:
        seen=set()
        for vid,start,end,blob in conn.execute('SELECT video_id,start,end,features FROM features ORDER BY video_id,start'):
            if vid in seen:continue
            seen.add(vid);samples.append((rows[vid],start,np.frombuffer(blob,dtype='<f4').copy()))
            if len(samples)==3:break
    # Include a known rejected interval so error behavior is also compared.
    events=_fk_paths.relocate((ROOT/'runs/auto-next-001/heldout/audiovisual/preparation-events.jsonl'), __file__)
    for line in events.read_text().splitlines():
        event=json.loads(line)
        if 'start' in event:
            samples.append((rows[event['video_id']],event['start'],None));break
    reference=None;timings=[];feature_checks=0
    with torch.inference_mode():
        for iteration,enabled in enumerate((False,False,True,True,False)):
            prepared=Prepared(pop,'audiovisual',backend,assets,detect,reuse_video_decoder=True,
                cache_face_detections=True,prefetch_video_decode=enabled)
            outputs=[];torch.cuda.synchronize();started=time.perf_counter()
            try:
                for row,start,saved in samples:
                    prepared.detection_cache.clear()
                    try:
                        raw=prepared.raw_inputs(pop.path(row),start)
                        inputs=prepared.transform(raw)
                    except (ValueError,OSError) as error:
                        if saved is not None:raise RuntimeError('Saved usable window was rejected') from error
                        outputs.append({'error':str(error)});continue
                    actual=model.extract_features(move(inputs,'cuda')).cpu().numpy().astype('<f4')
                    if saved is not None:
                        np.testing.assert_array_equal(actual.reshape(-1),saved)
                        feature_checks+=1
                    outputs.append({'raw':raw,'inputs':inputs,'features':actual})
            finally:
                prepared.video_reader.close();prepared.detection_cache.clear()
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            if reference is None:reference=outputs
            else:
                assert len(reference)==len(outputs)
                for a,b in zip(reference,outputs):same(a,b)
            timings.append({'prefetch':enabled,'seconds':elapsed,'windows':len(samples),
                            'warmup':iteration==0})
            print(json.dumps(timings[-1]),flush=True)
    baseline=sum(r['seconds'] for r in timings if not r['prefetch'] and not r['warmup'])
    optimized=sum(r['seconds'] for r in timings if r['prefetch'] and not r['warmup'])
    report={'parity_passed':True,'saved_feature_checks':feature_checks,'pinned_runtime_cpu_tests':result.testsRun,
        'timings':timings,'baseline_seconds':baseline,'optimized_seconds':optimized,
        'ratio':baseline/optimized,'enable_recommended':optimized<baseline*.97,
        'limitation':'Small A/B sample; source content and decode cost vary. No universal speedup claim.'}
    write(_fk_paths.relocate((Path('/host-avh/live-decode-prefetch-001/benchmark.json')), __file__),report,replace=True)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
