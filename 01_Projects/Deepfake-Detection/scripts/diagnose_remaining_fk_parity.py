"""Read-only comparison of retained features, raw encoding and head batching."""

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
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from run_remaining_supplement import FrozenFK,DATA,PACKAGE
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json
import torch
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32='--cudnn-tf32' in sys.argv;torch.backends.cudnn.benchmark=False
OUT=_fk_paths.relocate((ROOT/('runs/remaining-fk-parity-diagnosis-002' if '--cudnn-tf32' in sys.argv else 'runs/remaining-fk-parity-diagnosis-001')), __file__);OUT.mkdir(exist_ok=True)

def main():
    from deepfake_detection.infrastructure import CachedVideoPopulation
    from deepfake_detection.training import forward_video,TENSORS
    import deepfake_detection.preparation as preparation
    rows=read(_fk_paths.relocate((PACKAGE/'forensic-plan.json'), __file__))['selected_source_examples']
    source=_fk_paths.relocate((ROOT/'runs/heldout-001/fk/cache/training-manifest.json'), __file__);manifest=read(source)
    entries={r['video_id']:r for r in manifest['test']}
    old={r['video_id']:r for r in read(_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__))['videos']}
    fk=FrozenFK();results=[]
    for row in rows:
        captured=[];prepared=[];original_encode=preparation.encode_fk_window;original_prepare=preparation.prepare_fk_window
        def capture_encode(value,encoders):
            result=original_encode(value,encoders);captured.append((value['interval'],result,fk.mapper(value['quality_measurements'])));return result
        def capture_prepare(*args,**kwargs):
            value=original_prepare(*args,**kwargs);prepared.append({k:v for k,v in value.items() if k not in ('faces','lips','audio','visual_times')});return value
        preparation.encode_fk_window=capture_encode;preparation.prepare_fk_window=capture_prepare
        try:raw=fk.predict(DATA/row['path'],row)
        finally:preparation.encode_fk_window=original_encode;preparation.prepare_fk_window=original_prepare
        result={'video_id':row['video_id'],'original':old[row['video_id']],'raw':raw,'preparation':prepared}
        if row['video_id'] in entries:
            cached=CachedVideoPopulation([entries[row['video_id']]],source.parent)[0]
            with torch.inference_mode():
                _,score,_=forward_video(fk.model,cached)
                single=[fk.model(**{k:cached[k][i:i+1].cuda() for k in TENSORS})['window_scores'][0] for i in range(len(cached['window_intervals']))]
                result['cached_batch_score']=score.item();result['cached_single_score']=fk.model.aggregate_video(torch.stack(single)).item()
                result['cached_intervals']=cached['window_intervals'];result['raw_intervals']=[item[0] for item in captured]
                if result['cached_intervals']==result['raw_intervals']:
                    values={k:torch.stack([v[k] for _,v,_ in captured]) if k.endswith('_times') else torch.cat([v[k] for _,v,_ in captured]) for k in captured[0][1]}
                    values['quality']=torch.cat([q for _,_,q in captured])
                    result['feature_max_abs_error']={k:float((values[k]-cached[k]).abs().max()) for k in values}
                    outputs=fk.model(**{k:v.cuda() for k,v in values.items()})
                    result['raw_batch_score']=fk.model.aggregate_video(outputs['window_scores']).item()
        results.append(result);write_json(_fk_paths.relocate((OUT/'report.json'), __file__),{'results':results,'completed':len(results),'total':len(rows),'manifest_sha256':digest(source),'checkpoint_sha256':fk.sha})
    write_json(_fk_paths.relocate((OUT/'complete.json'), __file__),{'report_sha256':digest(_fk_paths.relocate((OUT/'report.json'), __file__))})

if __name__=='__main__':
    stream=(_fk_paths.relocate((OUT/'worker.log'), __file__)).open('a',encoding='utf-8',buffering=1);sys.stdout=stream;sys.stderr=stream
    try:main()
    except Exception as error:write_json(_fk_paths.relocate((OUT/'error.json'), __file__),{'error':str(error)});raise
