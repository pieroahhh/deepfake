"""Separate AVTENet expert/fusion training on explicitly prepared inputs.

Input preparation, pretrained checkpoint selection and clip-to-video aggregation
remain separate declared choices. Unknown modality targets never become genuine.
"""

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


import hashlib
import json
import math
from pathlib import Path

import torch

from deepfake_detection.avtenet import EXPERTS, FeatureFusion, FrozenExpertFusion, expert_target, fingerprint, native_binary_loss
from deepfake_detection.cli import save
from deepfake_detection.training import seed_everything


def _audit(rows, split):
    ids, families, hashes = set(), set(), set()
    digest=hashlib.sha256()

    def update(value):
        if isinstance(value,dict):
            for key in sorted(value):
                digest.update(key.encode());update(value[key])
        elif isinstance(value,torch.Tensor):
            if value.requires_grad or not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError('prepared AVTENet inputs must be finite detached floating tensors')
            digest.update(str((tuple(value.shape),value.dtype)).encode())
            digest.update(value.cpu().contiguous().view(torch.uint8).numpy().tobytes())
        else:
            digest.update(json.dumps(value,sort_keys=True,allow_nan=False).encode())
    if not rows:
        raise ValueError('empty training/validation population')
    for row in sorted(rows,key=lambda item:item['video_id']):
        if row['split']!=split:
            raise ValueError('training/selection only accepts designated train/validation records')
        if not row['video_id'] or row['video_id'] in ids or not row['source_family']:
            raise ValueError('missing/duplicate video ID or unknown source family')
        sha=row['source_sha256']
        if not isinstance(sha,str) or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('source SHA-256 required')
        ids.add(row['video_id']);families.add(row['source_family']);hashes.add(sha)
        update(row)
    return digest.hexdigest(),ids,families,hashes


def _to_device(values,device):
    if isinstance(values,dict):
        return {key:_to_device(value,device) for key,value in values.items()}
    return values.to(device)


def _epoch(model,rows,stage,schedule,optimizer=None):
    model.train(optimizer is not None)
    device=next(model.parameters()).device
    losses,known,unknown=[],0,0
    with torch.set_grad_enabled(optimizer is not None):
        for row in rows:
            target=expert_target(
                'audiovisual' if stage=='fusion' else stage,
                visual_label=row['visual_label'],audio_label=row['audio_label'],video_label=row['label'],
            )
            if target is None:
                unknown+=1
                continue
            logits=model(_to_device(row['inputs'],device))
            if logits.shape!=(1,):
                raise ValueError('one prepared video/declared clip per row is required')
            external=torch.tensor([1-target],device=device,dtype=torch.long)
            loss=native_binary_loss(logits,external)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],schedule['gradient_clip_norm'],error_if_nonfinite=True,
                )
                optimizer.step()
            losses.append(loss.detach().item());known+=1
    if not known:
        raise ValueError('no known modality targets in this population')
    return {'loss':sum(losses)/known,'supervised_inputs':known,'unknown_targets':unknown}


def fit_avtenet_stage(model,train_rows,validation_rows,component_profile,schedule,stage,output,*,epochs=None):
    """Train one selected expert or frozen-expert/cached-feature fusion classifier.

    All pretrained backend and preprocessing provenance should be retained by the
    caller. This runner audits split membership/content, not pretraining claims.
    """
    if stage not in (*EXPERTS,'fusion'):
        raise ValueError('unknown AVTENet stage')
    if schedule['profile_id']!='avtenet-training-reconstruction-v1' or schedule['selection_metric']!='mean_validation_loss':
        raise ValueError('unsupported AVTENet training schedule')
    if schedule['checkpoint_ties']!='first_strict_improvement' or schedule['batch_unit']!='one_prepared_video_or_declared_clip':
        raise ValueError('unsupported AVTENet selection/batch policy')
    for key in ('epsilon','gradient_clip_norm'):
        if not math.isfinite(schedule[key]) or schedule[key]<=0:
            raise ValueError('invalid AVTENet optimizer/gradient settings')
    if not math.isfinite(schedule['weight_decay']) or schedule['weight_decay']<0:
        raise ValueError('invalid AVTENet weight decay')
    budget=schedule['maximum_epochs'] if epochs is None else epochs
    if type(budget) is not int or not 1<=budget<=schedule['maximum_epochs'] or schedule['patience']<1:
        raise ValueError('invalid AVTENet epoch budget/patience')
    if stage=='fusion' and not isinstance(model,(FrozenExpertFusion,FeatureFusion)):
        raise ValueError('fusion requires frozen experts or explicit cached feature vectors')
    if stage in EXPERTS and model.feature_dimension!=component_profile['expert_feature_dimensions'][stage]:
        raise ValueError('expert feature dimension disagrees with the component profile')
    train=_audit(train_rows,'train');validation=_audit(validation_rows,'validation')
    if any(train[i]&validation[i] for i in (1,2,3)):
        raise ValueError('source/video/family leakage between training and validation')
    output=_fk_paths.relocate((Path(output)), __file__);output.mkdir(parents=True,exist_ok=False)
    seed_everything(schedule['seed'],True)
    rate=component_profile['training']['learning_rates'][stage]
    optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=rate,
                               betas=tuple(schedule['betas']),eps=schedule['epsilon'],weight_decay=schedule['weight_decay'])
    save(_fk_paths.relocate((output/'run.json'), __file__),{'stage':stage,'component_profile':component_profile,'profile_sha256':fingerprint(component_profile),
                           'schedule':schedule,'learning_rate':rate,'data_signatures':{'train':train[0],'validation':validation[0]},
                           'epoch_budget':budget,'torch_version':str(torch.__version__)})
    best,stale,history=float('inf'),0,[]
    for epoch in range(1,budget+1):
        training_metrics=_epoch(model,train_rows,stage,schedule,optimizer)
        validation_metrics=_epoch(model,validation_rows,stage,schedule)
        history.append({'epoch':epoch,'training':training_metrics,'validation':validation_metrics})
        if validation_metrics['loss']<best:
            best,stale=validation_metrics['loss'],0
            torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                        'epoch':epoch,'stage':stage,'component_profile':component_profile,'schedule':schedule,
                        'data_signatures':{'train':train[0],'validation':validation[0]},
                        'profile_sha256':fingerprint(component_profile)},_fk_paths.relocate((output/'best.pt'), __file__))
        else:
            stale+=1
        if stale>=schedule['patience']:
            break
    checkpoint=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((output/'best.pt'), __file__)), __file__),map_location=next(model.parameters()).device,weights_only=True)
    model.load_state_dict(checkpoint['model_state']);model.eval()
    result={'status':'FINISHED','stage':stage,'selected_epoch':checkpoint['epoch'],
            'best_validation_loss':best,'history':history}
    save(_fk_paths.relocate((output/'metrics.json'), __file__),result)
    return result
