"""Resumable full-network tiny sanity with fixed training-only normalization.

No validation/test access and no automatic full pilot. Every saved boundary
contains model, optimizer, RNG, exact next update and normalization statistics.
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

import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
import sys, argparse, json, time, math
from pathlib import Path
ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT/'src'), __file__))]
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from baselines.common.io import digest, fingerprint, read
from baselines.common.training import load_checkpoint, atomic_checkpoint, rng_state, restore_rng, seed_all
from baselines.avfakenet.model import DSTNet, classification_loss
from baselines.avfakenet.normalized_recovery import FixedNormalizedHead, gradient_norms, trainable_groups, set_training_mode


def write(path, value):
    path = _fk_paths.relocate((Path(path)), __file__); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def batches(labels, seed, steps):
    rng = np.random.default_rng(seed)
    classes = [np.flatnonzero(labels == k) for k in (0,1)]
    queues = [[], []]; result = []
    for _ in range(steps):
        batch = []
        for k in (0,1):
            if len(queues[k]) < 4: queues[k].extend(rng.permutation(classes[k]).tolist())
            batch.extend(queues[k][:4]); del queues[k][:4]
        result.append(np.asarray(batch)[rng.permutation(8)])
    return result


@torch.inference_mode()
def evaluate(model, inputs, labels, smoothing, device):
    model.eval(); logits = []; relu = []
    hook = model.output_block.classifier[1].register_forward_hook(
        lambda module,args,out: relu.append(out.detach().cpu()))
    try:
        for start in range(0, len(inputs), 8): logits.append(model(inputs[start:start+8].to(device)).cpu())
    finally: hook.remove()
    logits = torch.cat(logits); probs = logits.softmax(-1)[:,1].numpy()
    activations = torch.cat(relu)
    if not torch.isfinite(logits).all(): raise ValueError('Nonfinite evaluation')
    return {'loss': float(classification_loss(logits, labels, smoothing)),
        'balanced_accuracy': float(balanced_accuracy_score(labels.numpy(),logits.argmax(-1).numpy())),
        'roc_auc': float(roc_auc_score(labels.numpy(),probs)),
        'confusion': confusion_matrix(labels.numpy(),logits.argmax(-1).numpy(),labels=[0,1]).tolist(),
        'probability_min': float(probs.min()), 'probability_max': float(probs.max()),
        'first_relu_zero_fraction': float((activations==0).float().mean()),
        'first_relu_dead_units': int((activations==0).all(dim=0).sum()),
        'probabilities': probs.tolist()}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    p.add_argument('--backbone-lr',type=float,default=1e-6)
    p.add_argument('--feature-dropout',choices=('on','off'),default='on')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();bundle=args.bundle.resolve();out=args.output.resolve()
    if args.backbone_lr<=0 or not math.isfinite(args.backbone_lr): raise ValueError('Invalid learning rate')
    if out==bundle or bundle in out.parents: raise ValueError('Output overlaps bundle')
    if out.exists() and not args.resume: raise ValueError('Use fresh output or --resume')
    sealed={'SHA256SUMS.txt':digest(_fk_paths.relocate((bundle/'SHA256SUMS.txt'), __file__))}
    for line in (_fk_paths.relocate((bundle/'SHA256SUMS.txt'), __file__)).read_text().splitlines():
        expected,name=line.split('  ',1);path=(bundle/name).resolve()
        if bundle not in path.parents or digest(path)!=expected: raise ValueError('Invalid bundle '+name)
        sealed[name]=expected
    profile=read(_fk_paths.relocate((bundle/'recovery-profile.json'), __file__));code=read(_fk_paths.relocate((bundle/'tiny/attempt-003/code.json'), __file__))
    if digest(_fk_paths.relocate((ROOT/'baselines/avfakenet/model.py'), __file__))!=code['files']['baselines/avfakenet/model.py']:
        raise ValueError('Model source mismatch')
    manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__)
    manifest_hash=digest(manifest_path);manifest={r['video_id']:r for r in read(manifest_path)}
    identity={'protocol':'avfakenet-fixed-normalization-full-network-tiny-v1','bundle':sealed,
        'population_sha256':manifest_hash,'backbone_lr':args.backbone_lr,'head_lr':1e-4,'updates':200,
        'seed':7,'batch_size':8,'clip_norm':1.,'device':args.device,'torch':str(torch.__version__),
        'profile':profile,'training_mode':{'feature_dropout':args.feature_dropout,'head_dropout':'on','all_parameters_trainable':True},
        'source':{str(path.relative_to(ROOT)):digest(path) for path in
            [_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/normalized_recovery.py'), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/model.py'), __file__)]}}
    out.mkdir(parents=True,exist_ok=True)
    if (_fk_paths.relocate((out/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((out/'identity.json'), __file__))!=identity: raise ValueError('Resume identity differs')
    write(_fk_paths.relocate((out/'identity.json'), __file__),identity);signature=fingerprint(identity)
    torch.set_num_threads(2)
    if args.device=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    started=time.monotonic();result={'status':'RUNNING','full_pilot':'ON HOLD','branches':{}}
    def progress(branch,step,state,**extra):
        elapsed=time.monotonic()-started
        value={'status':state,'branch':branch,'update':step,'updates_per_branch':200,
            'elapsed_seconds_this_session':round(elapsed,1),'full_pilot':'ON HOLD',**extra}
        write(_fk_paths.relocate((out/'progress.json'), __file__),value);print(json.dumps(value),flush=True)
    try:
        for branch in ('visual','audio'):
            dest=out/branch;dest.mkdir(exist_ok=True)
            if (_fk_paths.relocate((dest/'complete.json'), __file__)).exists():
                complete=read(_fk_paths.relocate((dest/'complete.json'), __file__))
                if complete['identity']!=signature: raise ValueError('Completed identity differs')
                for name,sha in complete['files'].items():
                    if digest(dest/name)!=sha: raise ValueError('Completed artifact changed')
                result['branches'][branch]=read(_fk_paths.relocate((dest/'metrics.json'), __file__));progress(branch,200,'REUSED');continue
            definitions=read(bundle/f'tiny/{branch}/samples.json');rows=definitions['samples']
            if definitions['population_sha256']!=manifest_hash or definitions['split']!='train': raise ValueError('Wrong samples')
            for sample in rows:
                row=manifest[sample['video_id']]
                if row['split']!='train' or sample['split']!='train' or row['source_family']!=sample['source_family'] or row['source_sha256']!=sample['source_sha256']: raise ValueError('Sample mismatch')
                label=row['audio_label'] if branch=='audio' else int(any(a<=sample['source_timestamp']<b for a,b in row['visual_manipulated_intervals']))
                if label!=sample['annotation_label']: raise ValueError('Annotation mismatch')
            y=np.array([r['annotation_label'] for r in rows],dtype=np.int64);labels=torch.tensor(y)
            pixels=np.load(_fk_paths.relocate_value((bundle/f'tiny/{branch}/pixels.npy'), __file__),allow_pickle=False)
            inputs=torch.from_numpy(pixels.copy()).float()
            if branch=='visual': inputs=inputs.permute(0,3,1,2).contiguous()/255
            parent=load_checkpoint(bundle/f'tiny/attempt-003/{branch}/last.pt')
            run=read(bundle/f'tiny/attempt-003/{branch}/run.json')
            if parent['identity']!=run['identity'] or run['identity']['profile']!=profile: raise ValueError('Parent mismatch')
            prepared=read(bundle/f'tiny/{branch}/prepared.json')
            if prepared!=run['identity']['source']['prepared'] or any(prepared[k+'_sha256']!=digest(bundle/f'tiny/{branch}/{k}.{ext}') for k,ext in [('pixels','npy'),('samples','json')]): raise ValueError('Prepared binding mismatch')
            seed_all(7);model=DSTNet(profile,branch).to(args.device).eval()
            model.load_state_dict(parent['model_state'],strict=True)
            captured=[]
            hook=model.output_block.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu().numpy().copy()))
            try:
                with torch.inference_mode():
                    for i in range(0,len(rows),8):model(inputs[i:i+8].to(args.device))
            finally:hook.remove()
            features=np.concatenate(captured);scaler=StandardScaler().fit(features)
            # Diagnose stochastic feature noise on fixed train rows before fitting.
            probe_rows=np.concatenate([np.flatnonzero(y==k)[:8] for k in (0,1)])
            stochastic=[]
            hook=model.output_block.register_forward_pre_hook(lambda module,args:stochastic.append(args[0].detach().cpu().numpy().copy()))
            try:
                model.train()
                with torch.inference_mode():
                    for repeat in range(5):model(inputs[probe_rows].to(args.device))
            finally:hook.remove()
            stochastic=np.stack(stochastic).astype(np.float64)
            noise=(stochastic-features[probe_rows].astype(np.float64))/scaler.scale_
            write(_fk_paths.relocate((dest/'dropout-comparison.json'), __file__),{'scope':'fixed training examples, parent extractor unchanged',
                'probe_rows':probe_rows.tolist(),'repeats':5,
                'mean_absolute_dropout_change_in_training_std_units':float(np.abs(noise).mean()),
                'rms_dropout_change_in_training_std_units':float(np.sqrt(np.mean(noise**2))),
                'median_per_feature_training_std':float(np.median(scaler.scale_))})
            seed_all(7)
            for module in model.output_block.modules():
                if isinstance(module,torch.nn.Linear):module.reset_parameters()
            model.output_block=FixedNormalizedHead(model.output_block,scaler.mean_,scaler.scale_).to(args.device)
            fixed_mean=model.output_block.mean.detach().cpu().clone();fixed_scale=model.output_block.scale.detach().cpu().clone()
            groups=trainable_groups(model)
            feature_params=[p for k,params in groups.items() if k!='classifier' for p in params]
            config=profile['training']
            optimizer=torch.optim.AdamW([{'params':feature_params,'lr':args.backbone_lr},
                {'params':groups['classifier'],'lr':1e-4}],weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['epsilon'])
            initial=evaluate(model,inputs,labels,config['label_smoothing'],args.device)
            history=[];step=0;latest={};save_path=_fk_paths.relocate((dest/'last.pt'), __file__)
            if save_path.exists():
                if not args.resume:raise ValueError('Checkpoint already exists')
                state=load_checkpoint(save_path)
                if state['identity']!=signature or state['branch']!=branch: raise ValueError('Checkpoint identity mismatch')
                model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
                initial=state['initial'];history=state['history'];step=state['update'];latest=state['latest_gradients']
                if not torch.equal(model.output_block.mean.cpu(),fixed_mean) or not torch.equal(model.output_block.scale.cpu(),fixed_scale): raise ValueError('Normalization changed')
                restore_rng(state['rng'])
            def save():
                atomic_checkpoint(save_path,{'identity':signature,'branch':branch,'update':step,
                    'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'rng':rng_state(),
                    'initial':initial,'history':history,'latest_gradients':latest})
            schedule=batches(y,7,200);branch_started=time.monotonic();resumed_at=step
            progress(branch,step,'TRAINING')
            for index in range(step,200):
                if (_fk_paths.relocate((out/'pause.request'), __file__)).exists():
                    save();progress(branch,step,'PAUSED');return
                set_training_mode(model,args.feature_dropout);optimizer.zero_grad(set_to_none=True)
                ids=schedule[index]
                loss=classification_loss(model(inputs[ids].to(args.device)),labels[ids].to(args.device),config['label_smoothing'])
                if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                loss.backward()
                latest=gradient_norms(model)
                if any(not math.isfinite(v) for v in latest.values()):raise ValueError('Nonfinite gradient')
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
                optimizer.step();step=index+1
                if step%20==0:
                    scores=evaluate(model,inputs,labels,config['label_smoothing'],args.device)
                    history.append({'update':step,'scores':scores,'gradient_norms':latest,'global_gradient_norm_before_clip':norm})
                    save();write(_fk_paths.relocate((dest/'history.json'), __file__),history)
                    elapsed=time.monotonic()-branch_started
                    eta=(200-step)*elapsed/max(1,step-resumed_at)
                    progress(branch,step,'TRAINING',balanced_accuracy=scores['balanced_accuracy'],loss=scores['loss'],branch_eta_seconds=round(eta,1))
            final=evaluate(model,inputs,labels,config['label_smoothing'],args.device)
            if not torch.equal(model.output_block.mean.cpu(),fixed_mean) or not torch.equal(model.output_block.scale.cpu(),fixed_scale):raise ValueError('Fixed normalization changed')
            changed={}
            for name in ('input_block','patch_embedding','stages','merges'):
                changed[name]=sum(not torch.equal(value.detach().cpu(),parent['model_state'][key]) for key,value in model.state_dict().items() if key.startswith(name+'.'))
            reduction=(initial['loss']-final['loss'])/initial['loss']
            gates={'completed_200_updates':step==200,'balanced_accuracy':final['balanced_accuracy']>=.8,
                'loss_reduction':reduction>=.2,'nonconstant_scores':final['probability_max']-final['probability_min']>=1e-5,
                'active_feature_and_head_gradients':all(v>1e-12 for v in latest.values()),
                'all_feature_groups_updated':all(v>0 for v in changed.values()),'classifier_not_fully_dead':final['first_relu_zero_fraction']<1}
            metrics={'branch':branch,'status':'PASSED' if all(gates.values()) else 'FAILED','gates':gates,
                'initial':initial,'final':final,'relative_loss_reduction':reduction,'latest_gradient_norms':latest,
                'changed_feature_parameter_tensors':changed,'normalization_fixed':True,'updates':step,
                'scope':'training-only full-network tiny sanity','full_pilot':'ON HOLD'}
            write(_fk_paths.relocate((dest/'metrics.json'), __file__),metrics)
            write(_fk_paths.relocate((dest/'complete.json'), __file__),{'identity':signature,'files':{name:digest(dest/name) for name in ('metrics.json','last.pt','history.json')}})
            result['branches'][branch]=metrics;write(_fk_paths.relocate((out/'summary.json'), __file__),result)
            del model,optimizer,parent
            if args.device=='cuda':torch.cuda.empty_cache()
        if not all(digest(bundle/name)==sha for name,sha in sealed.items()) or digest(manifest_path)!=manifest_hash:raise ValueError('Original input changed')
        write(_fk_paths.relocate((out/'source-preservation.json'), __file__),{'all_original_files_unchanged':True,'bundle':sealed,'population_sha256':manifest_hash})
        result['status']='PASSED' if all(v['status']=='PASSED' for v in result['branches'].values()) else 'FAILED'
        result['both_branch_sanity_gates_passed']=result['status']=='PASSED'
        write(_fk_paths.relocate((out/'summary.json'), __file__),result);progress('complete',200,'FINISHED',sanity_result=result['status'])
    except Exception as error:
        progress('error',0,'FAILED',error=str(error));raise


if __name__=='__main__':main()
