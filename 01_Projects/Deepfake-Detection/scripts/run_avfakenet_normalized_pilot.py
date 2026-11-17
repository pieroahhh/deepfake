"""Gated, resumable AVFakeNet recovery pilot: prepare -> train -> validation.

Uses the shared 3,000/500 population. Never opens held-out data. This is an
explicit adaptation, retaining native majority-frame vote and hard-OR fusion.
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
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import sys,argparse,json,time,math,shutil
from pathlib import Path
from collections import OrderedDict,Counter
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from baselines.common.io import read,digest,fingerprint
from baselines.common.training import load_checkpoint,atomic_checkpoint,seed_all,rng_state,restore_rng
from baselines.common.media import detector,faces,audio
from baselines.common.metrics import metrics
from baselines.common.population import target
from baselines.avfakenet.model import DSTNet,classification_loss
from baselines.avfakenet.preprocessing import prepare_audio
from baselines.avfakenet.normalized_recovery import FixedNormalizedHead,set_training_mode,gradient_norms,trainable_groups
from deepfake_detection.baselines import avfakenet_decision
from recover_avfakenet_normalized_tiny import write,evaluate as evaluate_tiny


class Paused(Exception):pass


def balanced_epoch(labels,seed):
    """Every majority row once; shuffle-cycle minority rows to equal counts."""
    rng=np.random.default_rng(seed);classes=[np.flatnonzero(labels==k) for k in (0,1)]
    if any(len(a)==0 for a in classes):raise ValueError('Both training classes required')
    count=max(map(len,classes));balanced=[]
    for candidates in classes:
        pieces=[];remaining=count
        while remaining:
            part=rng.permutation(candidates)[:remaining];pieces.append(part);remaining-=len(part)
        balanced.append(np.concatenate(pieces))
    # Four of each class per batch, except a balanced final partial batch.
    order=[]
    for start in range(0,count,4):
        block=np.concatenate([part[start:start+4] for part in balanced])
        order.extend(block[rng.permutation(len(block))].tolist())
    return np.asarray(order,dtype=np.int64)


def make_model(profile,branch,state,device):
    model=DSTNet(profile,branch)
    model.output_block=FixedNormalizedHead(model.output_block,state['output_block.mean'],state['output_block.scale'])
    model.load_state_dict(state,strict=True)
    return model.to(device)


class Inputs:
    def __init__(self,cache,records,branch):
        self.cache,self.records,self.branch=cache,records,branch;self.open=OrderedDict()

    def array(self,video):
        key=self.records[video]['key']
        if key not in self.open:
            self.open[key]=np.load(_fk_paths.relocate_value((self.cache/key/(self.branch+'.npy')), __file__),mmap_mode='r',allow_pickle=False)
            if len(self.open)>32:
                _,old=self.open.popitem(last=False);old._mmap.close()
        self.open.move_to_end(key)
        return self.open[key]

    def batch(self,refs,indices):
        arrays=[];labels=[]
        for index in indices:
            video,frame,label=refs[index]
            value=self.array(int(video))
            arrays.append(np.array(value[int(frame)] if self.branch=='visual' else value,copy=True))
            labels.append(label)
        x=torch.from_numpy(np.stack(arrays))
        if self.branch=='visual':x=x.permute(0,3,1,2).contiguous().float()/255
        return x,torch.tensor(labels,dtype=torch.long)

    def close(self):
        for value in self.open.values():value._mmap.close()
        self.open.clear()


def references(records,branch,split,cache):
    values=[]
    for video,record in enumerate(records):
        if record['split']!=split or branch in record['reasons']:continue
        if branch=='audio':
            if record['audio_label'] is not None:values.append((video,0,record['audio_label']))
        else:
            labels=np.load(_fk_paths.relocate_value((_fk_paths.relocate((cache/record['key']/'visual-labels.npy'), __file__)), __file__),allow_pickle=False)
            values.extend((video,i,int(label)) for i,label in enumerate(labels) if label in (0,1))
    return np.asarray(values,dtype=np.int64).reshape(-1,3)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sanity',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume',action='store_true');args=p.parse_args()
    sanity,bundle,dataset,out=[v.resolve() for v in (args.sanity,args.bundle,args.dataset_root,args.output)]
    if any(out==v or v in out.parents for v in (sanity,bundle,dataset)):raise ValueError('Output overlaps originals')
    if out.exists() and not args.resume:raise ValueError('Use new output or --resume')
    out.mkdir(parents=True,exist_ok=True)
    # OS lock is released even after process termination; no stale PID lock file.
    lock=(_fk_paths.relocate((out/'worker.lock'), __file__)).open('a+b');lock.seek(0)
    if os.name=='nt':
        import msvcrt
        if (_fk_paths.relocate((out/'worker.lock'), __file__)).stat().st_size==0:lock.write(b'0');lock.flush();lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();stage_started={}
    def progress(stage,done,total,status='RUNNING',**extra):
        begin=stage_started.setdefault(stage,(time.monotonic(),done));elapsed=time.monotonic()-begin[0]
        rate=(done-begin[1])/elapsed if elapsed else 0
        eta=(total-done)/rate if rate>0 else None
        record={'status':status,'stage':stage,'processed':done,'total':total,
            'percent':100*done/total if total else 0,'rate_per_second':rate,'eta_seconds':eta,
            'estimated_completion_unix':time.time()+eta if eta is not None else None,
            'elapsed_seconds_this_session':round(time.monotonic()-started,1),'updated_unix':time.time(),**extra}
        write(_fk_paths.relocate((out/'progress.json'), __file__),record);print(json.dumps(record),flush=True)
    def check_pause():
        if (_fk_paths.relocate((out/'pause.request'), __file__)).exists():raise Paused('Requested pause; completed preparation/checkpoints retained')
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available():raise ValueError('CUDA is required for this pilot')
        device='cuda';torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        source_identity=read(_fk_paths.relocate((sanity/'identity.json'), __file__));summary=read(_fk_paths.relocate((sanity/'summary.json'), __file__))
        if not summary['both_branch_sanity_gates_passed'] or summary['status']!='PASSED':raise ValueError('Both sanity gates must pass')
        if source_identity['backbone_lr']!=1e-8 or source_identity['training_mode']['feature_dropout']!='off':raise ValueError('Unexpected recovery protocol')
        profile=source_identity['profile'];parents={};protected={}
        for name,sha in source_identity['bundle'].items():
            if digest(bundle/name)!=sha:raise ValueError('Bundle input changed')
            protected[str(bundle/name)]=sha
        for branch in ('visual','audio'):
            complete=read(_fk_paths.relocate((sanity/branch/'complete.json'), __file__))
            if complete['identity']!=fingerprint(source_identity):raise ValueError('Sanity identity mismatch')
            for name,sha in complete['files'].items():
                if digest(sanity/branch/name)!=sha:raise ValueError('Sanity artifact changed')
                protected[str(sanity/branch/name)]=sha
            parent=load_checkpoint(_fk_paths.relocate((sanity/branch/'last.pt'), __file__))
            if parent['update']!=200 or not all(summary['branches'][branch]['gates'].values()):raise ValueError('Incomplete sanity')
            if any(float(v['step'])!=200 for v in parent['optimizer_state']['state'].values()):raise ValueError('Optimizer budget differs')
            model=make_model(profile,branch,parent['model_state'],device)
            pixels=np.load(_fk_paths.relocate_value((bundle/f'tiny/{branch}/pixels.npy'), __file__),allow_pickle=False)
            x=torch.from_numpy(pixels.copy()).float()
            if branch=='visual':x=x.permute(0,3,1,2).contiguous()/255
            samples=read(bundle/f'tiny/{branch}/samples.json')['samples']
            y=torch.tensor([r['annotation_label'] for r in samples],dtype=torch.long)
            verified=evaluate_tiny(model,x,y,profile['training']['label_smoothing'],device)
            recorded=summary['branches'][branch]['final']
            if verified['confusion']!=recorded['confusion'] or max(abs(a-b) for a,b in zip(verified['probabilities'],recorded['probabilities']))>1e-5:raise ValueError('Sanity predictions do not reproduce')
            parents[branch]=parent['model_state'];del model,parent,x
        write(_fk_paths.relocate((out/'sanity-verification.json'), __file__),{'both_branches_reproduced':True,'all_optimizer_steps':200})
        manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__);manifest=read(manifest_path)
        if digest(manifest_path)!=source_identity['population_sha256']:raise ValueError('Population changed')
        if Counter(r['split'] for r in manifest)!={'train':3000,'validation':500}:raise ValueError('Expected shared 3000/500 only')
        protected[str(manifest_path)]=digest(manifest_path)
        detect,detector_info=detector(device)
        identity={'protocol':'avfakenet-normalized-recovery-pilot-v1','population_sha256':digest(manifest_path),
            'parent_sanity_identity':fingerprint(source_identity),'parents':{b:digest(_fk_paths.relocate((sanity/b/'last.pt'), __file__)) for b in parents},
            'profile':profile,'detector':detector_info,'epochs':3,'seed':7,'batch_size':8,
            'optimizer':'AdamW','backbone_lr':1e-8,'head_lr':1e-4,'gradient_clip_norm':1.,
            'feature_dropout':'off','head_dropout':'profile','normalizer':'fixed training-only tiny calibration inherited from sanity',
            'sampler':'all majority examples once per epoch; minority shuffled cycles to equal size',
            'selection':'maximize validation branch balanced_accuracy, earliest strict tie',
            'visual_validation_unit':'annotated frame','audio_validation_unit':'clip',
            'validation_fusion':'native majority frame vote and hard OR; no continuous fused score',
            'source':{str(path.relative_to(ROOT)):digest(path) for path in [_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/model.py'), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/normalized_recovery.py'), __file__),_fk_paths.relocate((ROOT/'baselines/common/media.py'), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/preprocessing.py'), __file__),_fk_paths.relocate((ROOT/'scripts/recover_avfakenet_normalized_tiny.py'), __file__)]}}
        if (_fk_paths.relocate((out/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((out/'identity.json'), __file__))!=identity:raise ValueError('Pilot resume identity differs')
        write(_fk_paths.relocate((out/'identity.json'), __file__),identity);signature=fingerprint(identity)
        if (_fk_paths.relocate((out/'complete.json'), __file__)).exists():
            saved=read(_fk_paths.relocate((out/'complete.json'), __file__))
            if saved['identity']!=signature:raise ValueError('Completed pilot identity differs')
            for path,sha in saved['files'].items():
                if digest(out/path)!=sha:raise ValueError('Completed pilot modified')
            progress('complete',1,1,'FINISHED');return
        cache=_fk_paths.relocate((out/'prepared'), __file__);cache.mkdir(exist_ok=True);records=[]
        predicted=sum(math.ceil(r['duration']*25)*128*128*3+3*175*175*4 for r in manifest)
        if not args.resume and shutil.disk_usage(out).free < predicted*1.1+5*1024**3:raise ValueError('Insufficient free space for conservative prepared-cache estimate')
        progress('prepare',0,len(manifest))
        for number,row in enumerate(manifest,1):
            check_pause();key=fingerprint(row);directory=cache/key;directory.mkdir(exist_ok=True)
            path=(dataset/row['path']).resolve()
            if dataset not in path.parents or digest(path)!=row['source_sha256']:raise ValueError('Media checksum/path mismatch: '+row['video_id'])
            header=_fk_paths.relocate((directory/'record.json'), __file__)
            if header.exists():
                record=read(header)
                if record['source_sha256']!=row['source_sha256']:raise ValueError('Prepared source mismatch')
                for name,sha in record['files'].items():
                    if digest(directory/name)!=sha:raise ValueError('Prepared payload changed')
            else:
                record={'video_id':row['video_id'],'key':key,'split':row['split'],'source_sha256':row['source_sha256'],
                    'label':row['label'],'audio_label':row['audio_label'],'reasons':{},'files':{}}
                def array(name,value):
                    temporary=directory/(name+'.tmp')
                    with temporary.open('wb') as stream:np.save(stream,value,allow_pickle=False)
                    temporary.replace(directory/name);record['files'][name]=digest(directory/name)
                try:
                    sample=faces(path,0,row['duration'],detect)
                    labels=[target(row,'visual',float(t)) for t in sample['source_times']]
                except (ValueError,OSError) as error:record['reasons']['visual']=str(error)
                else:
                    array('visual.npy',sample['faces']);array('visual-labels.npy',np.array([-1 if v is None else v for v in labels],dtype=np.int8))
                try:mel=prepare_audio(audio(path,0,row['duration']),16000,profile).numpy()
                except (ValueError,OSError) as error:record['reasons']['audio']=str(error)
                else:array('audio.npy',mel)
                write(header,record)
            records.append(record)
            if number%5==0 or number==len(manifest):progress('prepare',number,len(manifest),video_id=row['video_id'])
        write(_fk_paths.relocate((out/'prepared-index.json'), __file__),records);del detect
        torch.cuda.empty_cache()
        selected={}
        for branch in ('visual','audio'):
            dest=out/branch;dest.mkdir(exist_ok=True)
            train=references(records,branch,'train',cache);validation=references(records,branch,'validation',cache)
            if len(validation)==0 or len(np.unique(validation[:,2]))<2:raise ValueError('Validation has insufficient known classes')
            write(_fk_paths.relocate((dest/'population.json'), __file__),{'train_examples':len(train),'validation_examples':len(validation),
                'train_class_counts':dict(Counter(map(str,train[:,2]))),'validation_class_counts':dict(Counter(map(str,validation[:,2])))})
            seed_all(7);model=make_model(profile,branch,parents[branch],device)
            groups=trainable_groups(model);config=profile['training']
            optimizer=torch.optim.AdamW([{'params':[p for name,params in groups.items() if name!='classifier' for p in params],'lr':1e-8},
                {'params':groups['classifier'],'lr':1e-4}],weight_decay=config['weight_decay'],betas=tuple(config['betas']),eps=config['epsilon'])
            loader=Inputs(cache,records,branch);epoch=1;cursor=0;history=[];best=None;last_gradients={}
            checkpoint_path=_fk_paths.relocate((dest/'last.pt'), __file__);checkpoint_time=time.monotonic()
            if checkpoint_path.exists():
                state=load_checkpoint(checkpoint_path)
                if state['identity']!=signature or state['branch']!=branch:raise ValueError('Training resume mismatch')
                model.load_state_dict(state['model_state'],strict=True);optimizer.load_state_dict(state['optimizer_state'])
                epoch,cursor,history,best=(state[k] for k in ('epoch','cursor','history','best'))
                restore_rng(state['rng'])
            def save():
                atomic_checkpoint(checkpoint_path,{'identity':signature,'branch':branch,'epoch':epoch,'cursor':cursor,
                    'history':history,'best':best,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'rng':rng_state()})
            while epoch<=3:
                order=balanced_epoch(train[:,2],7+epoch)
                phase=f'train_{branch}_epoch_{epoch}'
                progress(phase,cursor,len(order))
                for start in range(cursor,len(order),8):
                    if (_fk_paths.relocate((out/'pause.request'), __file__)).exists():save();check_pause()
                    x,y=loader.batch(train,order[start:start+8]);set_training_mode(model,'off')
                    optimizer.zero_grad(set_to_none=True);logits=model(x.to(device))
                    loss=classification_loss(logits,y.to(device),config['label_smoothing'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
                    loss.backward()
                    if start==cursor and start%400==0:last_gradients=gradient_norms(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    cursor=min(start+8,len(order))
                    if start==0 or time.monotonic()-checkpoint_time>=60:
                        save();checkpoint_time=time.monotonic()
                        progress(phase,cursor,len(order),loss=float(loss),gradient_norms=last_gradients,checkpoint=str(checkpoint_path))
                save();model.eval();probabilities=[];truth=[];validation_loss=0.
                with torch.inference_mode():
                    for start in range(0,len(validation),8):
                        check_pause();ids=np.arange(start,min(start+8,len(validation)));x,y=loader.batch(validation,ids)
                        logits=model(x.to(device));probabilities.extend(logits.softmax(-1)[:,1].cpu().tolist());truth.extend(y.tolist())
                        validation_loss+=float(classification_loss(logits,y.to(device),config['label_smoothing']))*len(y)
                        if start%800==0:progress(f'validate_{branch}_epoch_{epoch}',start,len(validation))
                score={'epoch':epoch,'balanced_accuracy':float(balanced_accuracy_score(truth,np.asarray(probabilities)>.5)),
                    'roc_auc':float(roc_auc_score(truth,probabilities)),'loss':validation_loss/len(truth),
                    'examples':len(truth),'score_min':min(probabilities),'score_max':max(probabilities)}
                history.append(score)
                if best is None or score['balanced_accuracy']>best['balanced_accuracy']:
                    best=score;atomic_checkpoint(_fk_paths.relocate((dest/'best.pt'), __file__),{'identity':signature,'branch':branch,'epoch':epoch,'metrics':score,'model_state':model.state_dict()})
                epoch+=1;cursor=0;save();write(_fk_paths.relocate((dest/'metrics.json'), __file__),{'history':history,'selected':best})
            selected[branch]=best;loader.close();del model,optimizer
            torch.cuda.empty_cache()
        # Validate the unchanged native video-level decision rule on the same 500 clips.
        models={b:make_model(profile,b,load_checkpoint(_fk_paths.relocate((out/b/'best.pt'), __file__))['model_state'],device).eval() for b in selected}
        loaders={b:Inputs(cache,records,b) for b in models};predictions=[]
        for index,record in enumerate(records):
            if record['split']!='validation':continue
            check_pause();prediction={'video_id':record['video_id'],'split':'validation','label':record['label'],
                'status':'UNASSESSED','prediction':None,'score':None}
            if record['reasons']:prediction['reasons']=record['reasons']
            else:
                crop=loaders['visual'].array(index);votes=[]
                with torch.inference_mode():
                    for start in range(0,len(crop),8):
                        x=torch.from_numpy(np.array(crop[start:start+8],copy=True)).permute(0,3,1,2).contiguous().float()/255
                        votes.extend(models['visual'](x.to(device)).argmax(-1).cpu().tolist())
                    mel=torch.from_numpy(np.array(loaders['audio'].array(index),copy=True)).unsqueeze(0)
                    audio_probability=float(models['audio'](mel.to(device)).softmax(-1)[0,1])
                prediction.update(avfakenet_decision(votes,int(audio_probability>.5),tie_break=profile['decision']['tie_break']))
                prediction['audio_probability']=audio_probability;prediction['visual_fake_vote_fraction']=sum(votes)/len(votes)
            predictions.append(prediction)
            if len(predictions)%10==0:progress('native_validation',len(predictions),500)
        for loader in loaders.values():loader.close()
        (_fk_paths.relocate((out/'validation-predictions.jsonl'), __file__)).write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in predictions),encoding='utf-8')
        report={'model':'AVFakeNet-normalized-recovery-adaptation','split':'validation','selected_branches':selected,
            'native_video_metrics':metrics(predictions),'validation_gate_passed':all(v['balanced_accuracy']>.5 and v['roc_auc']>.5 for v in selected.values()),
            'heldout_evaluation_run':False,'identity_sha256':signature}
        write(_fk_paths.relocate((out/'summary.json'), __file__),report)
        if not all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in protected.items()):raise ValueError('Original inputs changed')
        write(_fk_paths.relocate((out/'source-preservation.json'), __file__),{'all_original_files_unchanged':True,'files':protected})
        write(_fk_paths.relocate((out/'complete.json'), __file__),{'identity':signature,'files':{name:digest(out/name) for name in
            ('summary.json','validation-predictions.jsonl','visual/best.pt','audio/best.pt','source-preservation.json')}})
        progress('complete',500,500,'FINISHED',validation_gate_passed=report['validation_gate_passed'])
    except Paused as error:progress('paused',0,0,'PAUSED',reason=str(error))
    except Exception as error:progress('error',0,0,'FAILED',error=str(error));raise
    finally:lock.close()


if __name__=='__main__':main()
