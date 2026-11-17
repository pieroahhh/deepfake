"""CPU-only forensic inspection of returned AVFakeNet checkpoints; never trains."""

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

import os,sys,argparse,json,math,importlib.metadata
from pathlib import Path
from collections import Counter
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest,checked_path


def activation_stats(value):
    import torch
    value=value.detach().float().cpu()
    return {'shape':list(value.shape),'zero_fraction':float((value==0).float().mean()),
        'max_abs':float(value.abs().max()),'finite':bool(torch.isfinite(value).all()),
        'spread_across_inputs':float((value-value[:1]).abs().max())}


def probe(model,inputs):
    import torch
    activations={};hooks=[]
    names=['input_block','patch_embedding','stages.0','merges.0','stages.1','merges.1',
           'stages.2','merges.2',*[f'output_block.{i}' for i in range(len(model.output_block))]]
    modules=dict(model.named_modules())
    for name in names:
        hooks.append(modules[name].register_forward_hook(lambda module,values,out,key=name:activations.__setitem__(key,activation_stats(out))))
    try:
        with torch.inference_mode():logits=model(inputs).cpu();probabilities=logits.softmax(-1)
    finally:
        for hook in hooks:hook.remove()
    return {'logits':logits.tolist(),'probability_fake':probabilities[:,1].tolist(),
        'predictions':logits.argmax(-1).tolist(),'max_logit_spread':float((logits-logits[:1]).abs().max()),
        'activations':activations}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();bundle=args.bundle.resolve();code=args.code_root.resolve();output=args.output.resolve()
    if output.exists():raise ValueError('Use a new output folder')
    if output==bundle or bundle in output.parents:raise ValueError('Do not write inside original bundle')
    meta=read(_fk_paths.relocate((bundle/'return.json'), __file__));sealed={_fk_paths.relocate((bundle/'return.json'), __file__):digest(_fk_paths.relocate((bundle/'return.json'), __file__))}
    for relative,sha in meta['files'].items():
        path=checked_path(bundle,relative)
        if digest(path)!=sha:raise ValueError('Returned payload checksum mismatch: '+relative)
        sealed[path]=sha
    report=read(_fk_paths.relocate((bundle/'avfakenet/evaluation/report.json'), __file__))
    trainpath=_fk_paths.relocate((code/'runs/two-day-001/population/manifest.json'), __file__);manifest=read(trainpath)
    if digest(trainpath)!=meta['population_sha256']:raise ValueError('Shared population differs')
    sealed[trainpath]=digest(trainpath)
    validation={r['video_id']:r for r in manifest if r['split']=='validation'}
    val_predictions=[json.loads(line) for line in (_fk_paths.relocate((bundle/'avfakenet/evaluation/predictions.jsonl'), __file__)).read_text().splitlines()]
    if len(val_predictions)!=len(validation):raise ValueError('Validation count differs')
    for row in val_predictions:
        if row['video_id'] not in validation or row['split']!='validation' or row['label']!=validation[row['video_id']]['label']:
            raise ValueError('Validation label mismatch')
    import numpy as np
    import torch
    from baselines.common.training import load_checkpoint
    from baselines.avfakenet.model import DSTNet,profile_hash
    from baselines.avfakenet.preprocessing import prepare_audio
    from baselines.common.media import audio
    torch.set_num_threads(2)
    result={'bundle_files_verified':len(sealed),'training_population_sha256':meta['population_sha256'],
        'runtime':{name:importlib.metadata.version(name) for name in ('torch','numpy','av','librosa')},
        'device':'cpu','heldout_not_used_for_model_selection':True,'branches':{}}
    audio_model=None;audio_profile=None
    for stage in ('visual','audio'):
        matches=[path for path in sealed if path.name=='best.pt' and stage in path.parts]
        if len(matches)!=1:raise ValueError('Selected checkpoint missing/ambiguous')
        checkpoint_path=matches[0]
        if digest(checkpoint_path)!=report['checkpoints'][stage]:raise ValueError('Validation checkpoint differs')
        checkpoint=load_checkpoint(checkpoint_path);run=read(_fk_paths.relocate((checkpoint_path.parent/'run.json'), __file__))
        metrics=read(_fk_paths.relocate((checkpoint_path.parent/'metrics.json'), __file__));identity=checkpoint['identity'];profile=identity['profile']
        if identity!=run['identity'] or checkpoint['identity_sha256']!=run['identity_sha256']:
            raise ValueError('Selected checkpoint identity differs from run')
        if identity['stage']!=stage or identity['population_sha256']!=meta['population_sha256'] or identity['max_steps'] is not None:
            raise ValueError('Wrong stage, population or integration checkpoint')
        if checkpoint['best_epoch']!=metrics['best_epoch'] or checkpoint['epoch']!=metrics['best_epoch']:
            raise ValueError('Selection epoch mismatch')
        model=DSTNet(profile,stage).eval().requires_grad_(False)
        model.load_state_dict(checkpoint['model_state'],strict=True)
        if any(not torch.isfinite(t).all() for t in model.state_dict().values() if t.is_floating_point()):
            raise ValueError('Nonfinite checkpoint tensors')
        torch.manual_seed(identity['seed']);initial=DSTNet(profile,stage).state_dict()
        differences={name:float((tensor-initial[name]).float().norm()) for name,tensor in model.state_dict().items() if tensor.is_floating_point()}
        size=model.pixels;generator=torch.Generator().manual_seed(1729)
        inputs=torch.stack([torch.zeros(3,size,size),torch.ones(3,size,size),
            torch.linspace(0,1,size).expand(3,size,size),torch.rand(3,size,size,generator=generator)])
        observations=probe(model,inputs)
        item={'checkpoint_sha256':digest(checkpoint_path),'selected_epoch':checkpoint['epoch'],
            'profile_sha256':profile_hash(profile),'history':metrics['history'],
            'parameter_tensors_changed_from_recorded_seed_initialization':sum(v>0 for v in differences.values()),
            'parameter_l2_differences':differences,'synthetic_probes':observations,
            'synthetic_probe_caveat':'Zeros, ones, ramp and seeded noise are diagnostic inputs, not a performance test.'}
        result['branches'][stage]=item
        if stage=='audio':audio_model=model;audio_profile=profile
        del checkpoint,initial
        print('Verified selected checkpoint and synthetic probes:',stage,flush=True)
    selected=[]
    for label in (0,1):
        candidates=[r for r in val_predictions if r['status']=='ASSESSED' and validation[r['video_id']]['audio_label']==label]
        candidates.sort(key=lambda r:r['video_id'])
        selected.extend(candidates[:8])
    actual=[];prepared=[]
    for row in selected:
        definition=validation[row['video_id']];path=checked_path(_fk_paths.relocate((code.parent/'dataset'), __file__),definition['path'])
        if digest(path)!=definition['source_sha256']:raise ValueError('Validation audio source checksum differs')
        try:
            image=prepare_audio(audio(path,0,definition['duration']),16000,audio_profile)
        except (ValueError,OSError) as error:
            actual.append({'video_id':row['video_id'],'error':str(error),'audio_label':definition['audio_label']});continue
        with torch.inference_mode():
            logits=audio_model(image.unsqueeze(0));probability=float(logits.softmax(-1)[0,1]);prediction=int(logits.argmax(-1).item())
        actual.append({'video_id':row['video_id'],'audio_label':definition['audio_label'],
            'logits':logits[0].tolist(),'probability_fake':probability,'prediction':prediction,
            'recorded_visual_vote':row.get('visual_prediction'),'recorded_audio_vote':row.get('audio_prediction'),
            'matches_recorded_audio_vote':prediction==row.get('audio_prediction'),
            'image_min':float(image.min()),'image_max':float(image.max()),'image_std':float(image.std())})
        prepared.append(image)
        print('Validation audio probe',len(actual),'/',len(selected),flush=True)
    if prepared:
        result['real_audio_activation_probe']=probe(audio_model,torch.stack(prepared[:8]+prepared[-8:]))
    result['validation_audio_probes']=actual
    result['validation_hard_vote_counts']={stage:dict(Counter(r.get(stage+'_prediction') for r in val_predictions if r['status']=='ASSESSED')) for stage in ('visual','audio')}
    result['validation_labels_probe_selection']='First 8 assessed validation clips with known audio label 0 and first 8 with label 1; no threshold fitting'
    output.mkdir(parents=True)
    write(_fk_paths.relocate((output/'report.json'), __file__),result)
    for path,sha in sealed.items():
        if digest(path)!=sha:raise ValueError('Original returned evidence changed')
    write(_fk_paths.relocate((output/'source-preservation.json'), __file__),{'all_unchanged':True,'files':{str(path):sha for path,sha in sealed.items()}})
    print(json.dumps({'output':str(output),'validation_audio_probes':len(actual),'all_originals_unchanged':True}),flush=True)


if __name__=='__main__':main()
