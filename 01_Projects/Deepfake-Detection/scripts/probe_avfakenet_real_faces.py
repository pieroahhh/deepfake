"""Read-only validation face probes for the previously audited visual weights."""

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

import os,sys,json,argparse,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,write,digest,checked_path
from probe_avfakenet_checkpoints import probe


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--audit-root',type=Path,required=True)
    args=p.parse_args();bundle=args.bundle.resolve();code=args.code_root.resolve();output=args.audit_root.resolve()
    if (_fk_paths.relocate((output/'real-visual-probes.json'), __file__)).exists():raise ValueError('Probe output already exists')
    prior=read(_fk_paths.relocate((output/'report.json'), __file__));sealed=read(_fk_paths.relocate((output/'source-preservation.json'), __file__))['files']
    for path,sha in sealed.items():
        if digest(path)!=sha:raise ValueError('Original evidence changed')
    import numpy as np
    import torch
    from baselines.common.training import load_checkpoint
    from baselines.common.media import faces,detector
    from baselines.common.population import target
    from baselines.avfakenet.model import DSTNet
    from baselines.avfakenet.preprocessing import prepare_face
    torch.set_num_threads(2)
    model_path=next(bundle.glob('avfakenet/visual/attempt-*/best.pt'))
    if digest(model_path)!=prior['branches']['visual']['checkpoint_sha256']:raise ValueError('Wrong visual checkpoint')
    state=load_checkpoint(model_path);profile=state['identity']['profile'];model=DSTNet(profile,'visual').eval().requires_grad_(False)
    model.load_state_dict(state['model_state'],strict=True)
    detect,provenance=detector('cpu')
    if provenance!=state['identity']['detector']:raise ValueError('Face detector provenance differs from training')
    manifest=read(_fk_paths.relocate((code/'runs/two-day-001/population/manifest.json'), __file__))
    returned=[json.loads(line) for line in (_fk_paths.relocate((bundle/'avfakenet/evaluation/predictions.jsonl'), __file__)).read_text().splitlines()]
    eligible={r['video_id'] for r in returned if r['status']=='ASSESSED'}
    inputs=[];details=[];errors=[]
    for wanted in (0,1):
        accepted=0
        for row in sorted(manifest,key=lambda r:r['video_id']):
            if row['split']!='validation' or row['video_id'] not in eligible or row['visual_label']!=wanted:continue
            start=0.
            if wanted:
                intervals=row.get('visual_manipulated_intervals') or []
                if not intervals:continue
                a,b=max(intervals,key=lambda pair:pair[1]-pair[0])
                start=math.floor((a+b)/2*25)/25
                if not a<=start<b:continue
            source=checked_path(_fk_paths.relocate((code.parent/'dataset'), __file__),row['path'])
            if digest(source)!=row['source_sha256']:raise ValueError('Visual probe source differs')
            try:sample=faces(source,start,.04,detect,fps=25,pixels=128)
            except (ValueError,OSError) as error:
                errors.append({'video_id':row['video_id'],'start':start,'error':str(error)})
                if len(errors)>12:break
                continue
            actual_label=target(row,'visual',float(sample['source_times'][0]))
            if actual_label!=wanted:continue
            inputs.append(prepare_face(sample['faces'][0],profile))
            details.append({'video_id':row['video_id'],'sample_time':float(sample['source_times'][0]),'frame_label':actual_label,
                            'source_sha256':row['source_sha256']})
            accepted+=1
            if accepted==2:break
    if len(inputs)<4:raise ValueError('Not enough genuine/manipulated validation face probes')
    result={'selection':'Two genuine and two annotation-modified frames from previously assessed validation clips',
        'device':'cpu','frames':details,'preparation_errors':errors,'observations':probe(model,torch.stack(inputs))}
    write(_fk_paths.relocate((output/'real-visual-probes.json'), __file__),result)
    for path,sha in sealed.items():
        if digest(path)!=sha:raise ValueError('Original evidence changed')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
