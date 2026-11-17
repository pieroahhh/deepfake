"""Locked 1,000-clip evaluation; resumable preparation/inference and comparison.

No fitting, threshold selection or model changes. Inputs are the frozen one-epoch
adaptation and existing local held-out manifest. Original results are read-only.
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
import sys,time,json,concurrent.futures,multiprocessing,argparse
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
import numpy as np
import torch
from baselines.common.io import read,digest,fingerprint,checked_path
from baselines.common.training import load_checkpoint
from baselines.common.metrics import metrics
from deepfake_detection.baselines import avfakenet_decision
from avfakenet_progress_io import write_json
from parallel_avfakenet_prepare import worker_init,prepare_one,memory_available
from run_avfakenet_one_epoch import make_model


def validate_cache_record(cache,record,row):
    if record['video_id']!=row['video_id'] or record['source_sha256']!=row['source_sha256']:raise ValueError('Cache row differs')
    directory=_fk_paths.relocate((cache/fingerprint(row)), __file__)
    for name,expected in record['files'].items():
        if digest(checked_path(directory,name))!=expected:raise ValueError('Cached array changed')


@torch.inference_mode()
def predict(row,record,cache,models,tie):
    result={'video_id':row['video_id'],'split':'test','label':row['label'],
        'status':'UNASSESSED','prediction':None,'score':None}
    if record['reasons']:
        result['reasons']=record['reasons'];return result
    directory=_fk_paths.relocate((cache/record['key']), __file__)
    faces=np.load(_fk_paths.relocate_value((_fk_paths.relocate((directory/'visual.npy'), __file__)), __file__),mmap_mode='r',allow_pickle=False);votes=[]
    try:
        for start in range(0,len(faces),8):
            x=torch.from_numpy(np.array(faces[start:start+8],copy=True)).permute(0,3,1,2).contiguous().float()/255
            logits=models['visual'](x.cuda())
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite visual prediction')
            votes.extend(logits.argmax(-1).cpu().tolist())
    finally:faces._mmap.close()
    mel=torch.from_numpy(np.load(_fk_paths.relocate_value((_fk_paths.relocate((directory/'audio.npy'), __file__)), __file__),allow_pickle=False)).unsqueeze(0).cuda()
    logits=models['audio'](mel)
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite audio prediction')
    audio_vote=int(logits.argmax(-1).item())
    result.update(avfakenet_decision(votes,audio_vote,tie_break=tie))
    result.update(audio_probability=float(logits.softmax(-1)[0,1]),visual_fake_vote_fraction=sum(votes)/len(votes),visual_frames=len(votes))
    if result['prediction'] is None:result['reason']='visual_majority_tie'
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=6);args=p.parse_args()
    if not 1<=args.workers<=6:raise ValueError('Expected 1–6 workers')
    run=_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002'), __file__);run.mkdir(exist_ok=True)
    lock=(_fk_paths.relocate((run/'worker.lock'), __file__)).open('a+b');lock.write(b'0');lock.flush();lock.seek(0)
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    started=time.monotonic();phase_starts={}
    def progress(stage,done,total,status='RUNNING',**extra):
        t,initial=phase_starts.setdefault(stage,(time.monotonic(),done));elapsed=time.monotonic()-t
        rate=(done-initial)/elapsed if elapsed else 0;eta=(total-done)/rate if rate>0 else None
        record={'status':status,'stage':stage,'processed':done,'total':total,'percent':done/total*100 if total else 0,
            'rate_per_second':rate,'eta_seconds':eta,'estimated_completion_unix':time.time()+eta if eta is not None else None,
            'updated_unix':time.time(),'elapsed_seconds_this_session':time.monotonic()-started,**extra}
        write_json(_fk_paths.relocate((run/'progress.json'), __file__),record);print(json.dumps(record),flush=True)
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        training=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__);completed=read(_fk_paths.relocate((training/'complete.json'), __file__));training_id=read(_fk_paths.relocate((training/'identity.json'), __file__))
        protected={}
        for name,sha in completed['files'].items():
            if digest(_fk_paths.relocate((training/name), __file__))!=sha:raise ValueError('Locked training artifact changed')
            protected[str(_fk_paths.relocate((training/name), __file__))]=sha
        if not read(_fk_paths.relocate((training/'summary.json'), __file__))['validation_gate_passed']:raise ValueError('Pilot validation gate not passed')
        manifest_path=_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__);audit_path=_fk_paths.relocate((manifest_path.parent/'audit.json'), __file__)
        audit=read(audit_path);manifest=read(manifest_path)
        if digest(manifest_path)!=audit['manifest_sha256'] or len(manifest)!=1000 or len({r['video_id'] for r in manifest})!=1000:raise ValueError('Held-out population mismatch')
        train_rows=read(_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__))
        if {r['video_id'] for r in train_rows}&{r['video_id'] for r in manifest} or {r['source_family'] for r in train_rows}&{r['source_family'] for r in manifest} or {r['source_sha256'] for r in train_rows}&{r['source_sha256'] for r in manifest}:raise ValueError('Held-out overlap')
        if any(r['split']!='test' for r in manifest):raise ValueError('Wrong split')
        protected[str(manifest_path)]=digest(manifest_path);protected[str(audit_path)]=digest(audit_path)
        fk_copy=_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__);fk=read(fk_copy)
        if len(fk['videos'])!=1000:raise ValueError('FK results missing')
        protected[str(fk_copy)]=digest(fk_copy)
        for path in [_fk_paths.relocate((ROOT.parent/'avh-a/code/runs/auto-next-001/heldout/evaluate/predictions.jsonl'), __file__),_fk_paths.relocate((ROOT/'runs/avtenet-analysis-001/heldout-calibrated-predictions.jsonl'), __file__),_fk_paths.relocate((ROOT/'runs/heldout-001/fk/summary.json'), __file__)]:
            protected[str(path)]=digest(path)
        identity={'protocol':'locked-avfakenet-normalized-heldout-v1','heldout_sha256':digest(manifest_path),
            'training_identity_sha256':completed['identity'],'checkpoints':{b:completed['files'][b+'/best.pt'] for b in ('visual','audio')},
            'preparation':'same strict faces/whole-clip Mel; persistent decoder and prefetch; no eligibility changes',
            'detector':training_id['detector'],'decision':training_id['profile']['decision'],
            'threshold_selection':'none; native two-class argmax / majority / hard OR',
            'preparation_device_note':'B cache for training; A prepares held-out using matching declared detector/profile; cross-hardware pixel differences documented',
            'scope':'Exploratory local heldout; adaptations and unequal training budgets',
            'comparison_input_sha256':{path:sha for path,sha in protected.items() if 'predictions' in path or 'fk-heldout-copy' in path},
            'source':{str(path.relative_to(ROOT)):digest(path) for path in [_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((ROOT/'scripts/compare_three_heldout.py'), __file__),_fk_paths.relocate((ROOT/'scripts/parallel_avfakenet_prepare.py'), __file__),_fk_paths.relocate((ROOT/'scripts/avfakenet_progress_io.py'), __file__),_fk_paths.relocate((ROOT/'scripts/run_avfakenet_one_epoch.py'), __file__),_fk_paths.relocate((ROOT/'baselines/common/media.py'), __file__),_fk_paths.relocate((ROOT/'baselines/common/video_reader.py'), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/model.py'), __file__),_fk_paths.relocate((ROOT/'baselines/avfakenet/normalized_recovery.py'), __file__)]}}
        if (_fk_paths.relocate((run/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((run/'identity.json'), __file__))!=identity:raise ValueError('Resume code/model/population changed')
        write_json(_fk_paths.relocate((run/'identity.json'), __file__),identity);signature=fingerprint(identity)
        if (_fk_paths.relocate((run/'complete.json'), __file__)).exists():
            complete=read(_fk_paths.relocate((run/'complete.json'), __file__))
            if complete['identity']!=signature or any(digest(_fk_paths.relocate((run/name), __file__))!=sha for name,sha in complete['files'].items()):raise ValueError('Completed result changed')
            progress('complete',1000,1000,'FINISHED');return
        write_json(_fk_paths.relocate((run/'locked-before-inference.json'), __file__),{'identity':identity,'created_unix':time.time(),'protected_inputs':protected}) if not (_fk_paths.relocate((run/'locked-before-inference.json'), __file__)).exists() else None
        cache=_fk_paths.relocate((run/'prepared'), __file__);cache.mkdir(exist_ok=True);records={};pending=[]
        for row in manifest:
            header=_fk_paths.relocate((cache/fingerprint(row)/'record.json'), __file__)
            if header.exists():
                record=read(header);validate_cache_record(cache,record,row);records[row['video_id']]=record
            else:pending.append(row)
        if pending:
            if memory_available()['available_gib']<3+1.2*args.workers:raise ValueError('Insufficient memory for selected worker count')
            progress('prepare_heldout',len(records),1000)
            iterator=iter(pending);pause=False
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(training_id['profile'],training_id['detector'])) as pool:
                futures={}
                def submit():
                    row=next(iterator,None)
                    if row is not None:futures[pool.submit(prepare_one,row,str(_fk_paths.relocate((ROOT.parent/'Heldout-1000/dataset'), __file__)),str(cache),True)]=row
                for _ in range(args.workers):submit()
                while futures:
                    ready,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                    for future in ready:
                        row=futures.pop(future);record=future.result()['record'];validate_cache_record(cache,record,row);records[row['video_id']]=record
                        progress('prepare_heldout',len(records),1000,video_id=row['video_id'])
                        pause=pause or (_fk_paths.relocate((run/'pause.request'), __file__)).exists()
                        if not pause:submit()
            if pause:progress('prepare_heldout',len(records),1000,'PAUSED');return
        models={}
        for branch in ('visual','audio'):
            state=load_checkpoint(_fk_paths.relocate((training/branch/'best.pt'), __file__))
            if state['identity']!=completed['identity'] or state['branch']!=branch:raise ValueError('Wrong selected checkpoint')
            models[branch]=make_model(training_id['profile'],branch,state['model_state'],'cuda').eval().requires_grad_(False)
        predictions=[];directory=_fk_paths.relocate((run/'per-clip'), __file__);directory.mkdir(exist_ok=True)
        for number,row in enumerate(manifest,1):
            if (_fk_paths.relocate((run/'pause.request'), __file__)).exists():progress('evaluate_heldout',len(predictions),1000,'PAUSED');return
            path=_fk_paths.relocate((directory/(fingerprint(row)+'.json')), __file__)
            if path.exists():
                saved=read(path)
                if saved['identity']!=signature:raise ValueError('Saved prediction identity mismatch')
                prediction=saved['prediction']
            else:
                prediction=predict(row,records[row['video_id']],cache,models,training_id['profile']['decision']['tie_break'])
                write_json(path,{'identity':signature,'prediction':prediction})
            if prediction['video_id']!=row['video_id'] or prediction['label']!=row['label']:raise ValueError('Prediction differs from held-out label/ID')
            predictions.append(prediction)
            if number%10==0 or number==1:progress('evaluate_heldout',number,1000)
        (_fk_paths.relocate((run/'predictions.jsonl'), __file__)).write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in predictions),encoding='utf-8')
        m=metrics(predictions);c=m['confusion'];m['accuracy']=(c['tp']+c['tn'])/m['assessed'] if m['assessed'] else None
        write_json(_fk_paths.relocate((run/'summary.json'), __file__),{'model':'AVFakeNet normalized recovery adaptation','split':'test','metrics':m,
            'identity_sha256':signature,'population_sha256':digest(manifest_path),'fitted_on_heldout':False,
            'training_epochs':1,'tiny_warmups':'documented separately','fused_auc_policy':'hard decisions only; score null'})
        if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Original model/input artifacts changed')
        write_json(_fk_paths.relocate((run/'source-preservation.json'), __file__),{'all_protected_inputs_unchanged':True,'files':protected})
        del models;torch.cuda.empty_cache()
        progress('comparison',0,1)
        from compare_three_heldout import main as compare
        compare(run,_fk_paths.relocate((ROOT/'runs/three-model-heldout-002'), __file__))
        write_json(_fk_paths.relocate((run/'complete.json'), __file__),{'identity':signature,'files':{name:digest(_fk_paths.relocate((run/name), __file__)) for name in ('summary.json','predictions.jsonl','source-preservation.json')}})
        progress('complete',1000,1000,'FINISHED')
    except Exception as error:
        progress('error',0,1000,'FAILED',error=str(error));raise
    finally:lock.close()


if __name__=='__main__':
    if '--supervise' in sys.argv:
        import subprocess
        run=_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002'), __file__);run.mkdir(exist_ok=True)
        for attempt in range(1,4):
            code=subprocess.call([sys.executable,'-u',str(_fk_paths.relocate((Path(__file__)), __file__)),'--workers','6'],cwd=ROOT)
            state=read(_fk_paths.relocate((run/'progress.json'), __file__)) if (_fk_paths.relocate((run/'progress.json'), __file__)).exists() else {}
            write_json(_fk_paths.relocate((run/'supervisor.json'), __file__),{'attempt':attempt,'exit_code':code,'last_status':state.get('status'),'updated_unix':time.time()})
            if (_fk_paths.relocate((run/'complete.json'), __file__)).exists() or state.get('status') in ('FAILED','PAUSED'):sys.exit(code)
            print('Unexpected worker exit; resuming verified completed clips',flush=True)
            time.sleep(2)
        sys.exit(1)
    else:main()
