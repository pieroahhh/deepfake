"""Resumable streamed stage training; compatible with the Linux legacy worker."""

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
from pathlib import Path
import random
import shutil
import time
import gc

import numpy as np
import torch

from .io import write, fingerprint
from .exceptions import TrainingPaused


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(), *state[2:]],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), *value[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def move(value, device):
    return {k: move(v, device) for k, v in value.items()} if isinstance(value, dict) else value.to(device)


def load_checkpoint(path):
    state=torch.load(_fk_paths.relocate_value((path), __file__), map_location='cpu', weights_only=True)
    if state.get('format')=='baseline-safe-scalars-v1':
        return _restore_scalars(state['state'])
    return state


def _safe_scalars(value):
    # Torch 1.13's restricted loader lacks the pickle BINFLOAT opcode. Encode
    # metadata floats explicitly instead of falling back to unrestricted pickle.
    if isinstance(value,float):return {'__baseline_float_hex__':value.hex()}
    if isinstance(value,dict):
        if '__baseline_float_hex__' in value:raise ValueError('reserved checkpoint metadata key')
        return {key:_safe_scalars(item) for key,item in value.items()}
    if isinstance(value,list):return [_safe_scalars(item) for item in value]
    if isinstance(value,tuple):return tuple(_safe_scalars(item) for item in value)
    return value


def _restore_scalars(value):
    if isinstance(value,dict):
        if set(value)=={'__baseline_float_hex__'}:return float.fromhex(value['__baseline_float_hex__'])
        return {key:_restore_scalars(item) for key,item in value.items()}
    if isinstance(value,list):return [_restore_scalars(item) for item in value]
    if isinstance(value,tuple):return tuple(_restore_scalars(item) for item in value)
    return value


def atomic_checkpoint(path, state):
    path = _fk_paths.relocate((Path(path)), __file__)
    temporary = path.with_suffix('.tmp')
    torch.save({'format':'baseline-safe-scalars-v1','state':_safe_scalars(state)}, temporary)
    os.replace(temporary, path)


def fit(model, batches, output, *, identity, schedule, loss_fn, fake_predictions, device='cpu',
        epochs=None, resume=None, max_steps=None, execution=None,
        checkpoint_seconds=0, progress_seconds=10, pause_file=None):
    """batches(split) is re-iterable and yields (inputs, external fake targets)."""
    budget = schedule['maximum_epochs'] if epochs is None else epochs
    if not 1 <= budget <= schedule['maximum_epochs'] or (max_steps is not None and max_steps < 1):
        raise ValueError('invalid epoch/step budget')
    if checkpoint_seconds < 0 or progress_seconds <= 0:
        raise ValueError('invalid checkpoint/progress interval')
    output = _fk_paths.relocate((Path(output)), __file__)
    identity = {**identity, 'schedule': schedule, 'max_steps': max_steps}
    signature = fingerprint(identity)
    model.to(device)
    optimizer_cls = {'Adam': torch.optim.Adam, 'AdamW': torch.optim.AdamW}[schedule['optimizer']]
    optimizer = optimizer_cls([p for p in model.parameters() if p.requires_grad],
                              lr=schedule['learning_rate'], betas=tuple(schedule['betas']),
                              eps=schedule['epsilon'], weight_decay=schedule['weight_decay'])
    maximize = schedule['selection'] == 'accuracy'
    best, stale, first, history = None, 0, 1, []
    saved = None
    if resume:
        saved = load_checkpoint(resume)
        if saved['identity_sha256'] != signature:
            raise ValueError('resume population/config/assets/stage/schedule mismatch')
        if saved['epoch'] >= budget or saved['stale'] >= schedule['patience']:
            raise ValueError('resume has already reached epoch budget or early stopping')
        model.load_state_dict(saved.pop('model_state'), strict=True)
        optimizer.load_state_dict(saved.pop('optimizer_state'))
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)
        best, stale, first, history = saved['best'], saved['stale'], saved['epoch']+1, saved['history']
        previous_best = _fk_paths.relocate((Path(resume).parent / 'best.pt'), __file__)
        if saved['best_epoch'] is not None:
            previous = load_checkpoint(previous_best)
            if previous['identity_sha256'] != signature or previous['epoch'] != saved['best_epoch']:
                raise ValueError('resume needs its matching best.pt alongside last.pt')
            del previous
        elif not saved.get('partial'):
            raise ValueError('completed epoch checkpoint needs a selected best epoch')
    output.mkdir(parents=True, exist_ok=False)
    if saved:
        if saved['best_epoch'] is not None:shutil.copyfile(previous_best, _fk_paths.relocate((output/'best.pt'), __file__))
        restore_rng(saved['rng'])
    run_metadata={'identity': identity, 'identity_sha256': signature,
                             'epochs': budget, 'device': str(device), 'resume': str(resume) if resume else None,
                             'torch_version': str(torch.__version__), 'integration_only': max_steps is not None}
    if execution is not None:run_metadata['execution_options']=execution
    write(_fk_paths.relocate((output/'run.json'), __file__),run_metadata)
    best_epoch = saved['best_epoch'] if saved else None
    partial = saved.get('partial') if saved else None
    if partial and (partial['epoch'] != first or partial['split'] not in ('train','validation')):
        raise ValueError('invalid partial epoch cursor')
    last_checkpoint = time.monotonic()
    checkpoint_path = str(_fk_paths.relocate((Path(resume)), __file__)) if resume else None
    checkpoint_step = partial['steps'] if partial else 0
    for epoch in range(first, budget+1):
        result = dict(partial['result']) if partial else {'epoch': epoch}
        for split in ('train', 'validation'):
            if partial and partial['split']=='validation' and split=='train':continue
            training = split == 'train'
            model.train(training)
            total, count, correct, steps = 0., 0, 0, 0
            gradient_groups={}
            continuing = partial if partial and partial['split']==split else None
            if continuing:
                total,count,correct,steps=(continuing[k] for k in ('total','count','correct','steps'))
                gradient_groups=continuing['gradient_groups']
            started = time.monotonic()
            initial_steps=steps
            last_progress=0.
            if continuing and hasattr(batches,'iter_from'):
                iterator=iter(batches.iter_from(split,continuing['cursor']))
            else:
                iterator=iter(batches(split))
                if continuing:
                    # Generic deterministic/re-iterable loaders can replay inputs;
                    # Prepared uses a source/window cursor and skips media decoding.
                    for _ in range(steps):
                        try:next(iterator)
                        except StopIteration:raise ValueError('resume stream is shorter than saved cursor')
                    restore_rng(saved['rng'])

            def progress(status='RUNNING'):
                nonlocal last_progress
                elapsed=time.monotonic()-started
                rate=(steps-initial_steps)/elapsed if elapsed else 0.
                detail=batches.progress(split) if hasattr(batches,'progress') else {}
                planned=detail.get('windows_total')
                consumed=detail.get('windows_processed')
                # Includes preparation failures in progress, but never in loss metrics.
                estimate_rate=(consumed-detail.get('initial_windows',0))/elapsed if consumed is not None and elapsed else None
                eta=(planned-consumed)/estimate_rate if planned is not None and estimate_rate and estimate_rate>0 else None
                payload={'status':status,'epoch':epoch,'epochs':budget,'split':split,'steps':steps,
                    'samples':count,'loss':total/count if count else None,'accuracy':correct/count if count else None,
                    'elapsed_seconds_this_session':elapsed,'steps_per_second':rate,'eta_seconds_split':eta,
                    'checkpoint':checkpoint_path,'checkpoint_step':checkpoint_step,
                    'updated_unix':time.time(),**detail}
                write(_fk_paths.relocate((output/'progress.json'), __file__),payload,replace=True)
                fraction=f' windows={consumed}/{planned}' if planned is not None else ''
                remaining=f' ETA={eta/60:.1f}min' if eta is not None else ' ETA=warming-up'
                print(f'epoch={epoch} split={split} steps={steps} samples={count}{fraction} '
                      f'rate={rate:.2f}steps/s elapsed={elapsed:.1f}s{remaining} checkpoint_step={checkpoint_step}',flush=True)
                last_progress=time.monotonic()

            with torch.set_grad_enabled(training):
                for inputs, labels in iterator:
                    if max_steps is not None and steps >= max_steps:break
                    inputs, labels = move(inputs,device), labels.to(device)
                    logits = model(inputs)
                    loss = loss_fn(logits, labels)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        if steps==0:
                            for name,parameter in model.named_parameters():
                                if parameter.grad is not None:
                                    group=gradient_groups.setdefault(name.split('.')[0],{'parameters_with_gradient':0,'squared_l2':0.})
                                    group['parameters_with_gradient']+=1
                                    group['squared_l2']+=float(parameter.grad.detach().float().square().sum().item())
                        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                                       schedule.get('gradient_clip_norm', 1.), error_if_nonfinite=True)
                        optimizer.step()
                    count += len(labels)
                    total += loss.item()*len(labels)
                    correct += int((fake_predictions(logits.detach()) == labels).sum().item())
                    steps += 1
                    pause_requested=pause_file is not None and _fk_paths.relocate((Path(pause_file)), __file__).exists()
                    if pause_requested or (checkpoint_seconds and (steps==1 or time.monotonic()-last_checkpoint >= checkpoint_seconds)):
                        cursor=batches.cursor() if hasattr(batches,'cursor') else None
                        state={'schema':'baseline-checkpoint-v1','model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),'identity':identity,'identity_sha256':signature,
                            'epoch':epoch-1,'best_epoch':best_epoch,'best':best,'stale':stale,
                            'history':history,'rng':rng_state(),
                            'partial':{'epoch':epoch,'split':split,'steps':steps,'count':count,
                                'total':total,'correct':correct,'gradient_groups':gradient_groups,
                                'result':dict(result),'cursor':cursor}}
                        atomic_checkpoint(_fk_paths.relocate((output/'last.pt'), __file__),state)
                        last_checkpoint=time.monotonic()
                        checkpoint_path=str(_fk_paths.relocate((output/'last.pt'), __file__));checkpoint_step=steps
                        print(f'CHECKPOINT epoch={epoch} split={split} step={steps} path={checkpoint_path}',flush=True)
                    if pause_requested:
                        progress('PAUSED')
                        if hasattr(iterator,'close'):iterator.close()
                        raise TrainingPaused('Paused after saving '+checkpoint_path)
                    if steps == initial_steps+1 or time.monotonic()-last_progress >= progress_seconds:
                        progress()
                    if max_steps is not None and steps >= max_steps:
                        break
            if hasattr(iterator,'close'):iterator.close()
            progress('SPLIT_COMPLETE')
            partial=None
            if not count:
                raise ValueError('no usable known targets in ' + split + '; inspect preparation failures')
            result[split] = {'loss': total/count, 'accuracy': correct/count, 'samples': count, 'steps': steps}
            if training:
                result[split]['first_step_gradients']=gradient_groups
        history.append(result)
        metric = result['validation'][schedule['selection']]
        improved = best is None or (metric > best if maximize else metric < best)
        best, stale = (metric, 0) if improved else (best, stale+1)
        if improved:
            best_epoch = epoch
        state = {'schema': 'baseline-checkpoint-v1', 'model_state': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'identity': identity, 'identity_sha256': signature,
                 'epoch': epoch, 'best_epoch': best_epoch, 'best': best, 'stale': stale,
                 'history': history, 'rng': rng_state()}
        if improved:
            # Inference/selection does not need Adam states; this also keeps
            # reloading a 95M-parameter expert within the legacy worker's RAM.
            atomic_checkpoint(_fk_paths.relocate((output/'best.pt'), __file__), {key:state[key] for key in
                              ('schema','model_state','identity','identity_sha256','epoch','best_epoch','best')})
        atomic_checkpoint(_fk_paths.relocate((output/'last.pt'), __file__), state)
        checkpoint_path=str(_fk_paths.relocate((output/'last.pt'), __file__));checkpoint_step=steps
        write(_fk_paths.relocate((output/'metrics.json'), __file__), {'best_epoch': best_epoch, 'history': history,
                                    'integration_only': max_steps is not None}, replace=True)
        if stale >= schedule['patience']:
            break
    optimizer.zero_grad(set_to_none=True)
    del optimizer,state
    gc.collect()
    model.load_state_dict(load_checkpoint(_fk_paths.relocate((output/'best.pt'), __file__))['model_state'], strict=True)
    model.eval()
    write(_fk_paths.relocate((output/'progress.json'), __file__),{'status':'FINISHED','epoch':history[-1]['epoch'],
        'epochs':budget,'best_epoch':best_epoch,'checkpoint':checkpoint_path,'updated_unix':time.time()},replace=True)
    return {'best_epoch': best_epoch, 'epochs_completed': history[-1]['epoch'], 'integration_only': max_steps is not None}
