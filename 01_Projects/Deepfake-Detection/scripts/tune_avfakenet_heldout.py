"""Execution-only tuning around the immutable held-out evaluator."""

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
import sys,time,json,argparse,concurrent.futures,multiprocessing
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
import numpy as np
import torch
from baselines.common.io import read,digest,fingerprint
from avfakenet_progress_io import write_json
import parallel_avfakenet_prepare as preparation
import evaluate_avfakenet_locked_heldout as original
RUN=_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002'), __file__)
EVIDENCE=_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-preparation-benchmark-001'), __file__)


def acquire():
    import msvcrt
    lock=(_fk_paths.relocate((RUN/'worker.lock'), __file__)).open('r+b');msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1);return lock


def benchmark():
    if read(_fk_paths.relocate((RUN/'progress.json'), __file__))['status']!='PAUSED':raise ValueError('Wait for PAUSED')
    lock=acquire()
    try:
        EVIDENCE.mkdir(exist_ok=False);identity=read(_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001/identity.json'), __file__))
        rows=read(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__));available=[]
        for row in rows:
            path=_fk_paths.relocate((RUN/'prepared'/fingerprint(row)/'record.json'), __file__)
            if path.exists() and row['duration']<=14:available.append((row,read(path)))
        available.sort(key=lambda pair:(pair[0]['duration'],pair[0]['video_id']))
        selected=[available[round(i*(len(available)-1)/11)] for i in range(12)]
        report={'scope':'preparation execution only; no outcome selection','sources':{str(_fk_paths.relocate((Path(__file__)), __file__)):digest(_fk_paths.relocate((Path(__file__)), __file__)),str(_fk_paths.relocate((Path(preparation.__file__)), __file__)):digest(_fk_paths.relocate((Path(preparation.__file__)), __file__))},
            'clip_ids':[r['video_id'] for r,_ in selected],'arms':{}}
        for workers in (6,8):
            if preparation.memory_available()['available_gib']<3+1.2*workers:
                report['arms'][str(workers)]={'skipped':'RAM headroom'};continue
            output=_fk_paths.relocate((EVIDENCE/str(workers)), __file__);output.mkdir()
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=preparation.worker_init,initargs=(identity['profile'],identity['detector'])) as pool:
                list(pool.map(preparation.warmup,[1.]*workers*2));started=time.perf_counter()
                futures=[pool.submit(preparation.prepare_one,row,str(_fk_paths.relocate((ROOT.parent/'Heldout-1000/dataset'), __file__)),str(output),True) for row,_ in selected]
                observed=[f.result()['record'] for f in futures];seconds=time.perf_counter()-started
            expected=[r for _,r in selected]
            report['arms'][str(workers)]={'workers':workers,'seconds':seconds,'all_records_and_hashes_identical':observed==expected}
            write_json(_fk_paths.relocate((EVIDENCE/'report.json'), __file__),report);print(report['arms'][str(workers)],flush=True)
        valid=[v for v in report['arms'].values() if v.get('all_records_and_hashes_identical')]
        if not valid:raise ValueError('No equivalent preparation configuration')
        report['selected_workers']=min(valid,key=lambda v:v['seconds'])['workers']
        report['status']='COMPLETE';write_json(_fk_paths.relocate((EVIDENCE/'report.json'), __file__),report)
    finally:lock.close()


@torch.inference_mode()
def optimized_prediction(row,record,cache,models,tie):
    # Reuse original path for all exclusions and original audio inference.
    if record['reasons']:return original_predict(row,record,cache,models,tie)
    directory=_fk_paths.relocate((cache/record['key']), __file__);faces=np.load(_fk_paths.relocate_value((_fk_paths.relocate((directory/'visual.npy'), __file__)), __file__),mmap_mode='r',allow_pickle=False)
    frames=len(faces);votes=[];near=0
    try:
        for start in range(0,frames,BATCH_SIZE):
            x=torch.from_numpy(np.array(faces[start:start+BATCH_SIZE],copy=True)).permute(0,3,1,2).contiguous().float()/255
            logits=models['visual'](x.cuda())
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite prediction')
            # Near ties are recomputed with the exact original batch-of-eight
            # grouping; guard is numerical, never a changed detection threshold.
            uncertain=(logits[:,1]-logits[:,0]).abs()<=1e-3
            if uncertain.any():
                for offset in range(0,len(x),8):
                    if uncertain[offset:offset+8].any():
                        logits[offset:offset+8]=models['visual'](x[offset:offset+8].cuda());near+=1
            votes.extend(logits.argmax(-1).cpu().tolist())
    finally:faces._mmap.close()
    mel=torch.from_numpy(np.load(_fk_paths.relocate_value((_fk_paths.relocate((directory/'audio.npy'), __file__)), __file__),allow_pickle=False)).unsqueeze(0).cuda()
    logits=models['audio'](mel)
    if not torch.isfinite(logits).all():raise ValueError('Nonfinite audio')
    from deepfake_detection.baselines import avfakenet_decision
    result={'video_id':row['video_id'],'split':'test','label':row['label'],**avfakenet_decision(votes,int(logits.argmax(-1).item()),tie_break=tie)}
    result.update(audio_probability=float(logits.softmax(-1)[0,1]),visual_fake_vote_fraction=sum(votes)/frames,visual_frames=frames)
    if result['prediction'] is None:result['reason']='visual_majority_tie'
    return result


def run():
    global BATCH_SIZE,original_predict
    evidence=read(_fk_paths.relocate((EVIDENCE/'report.json'), __file__));inference_path=_fk_paths.relocate((ROOT/'runs/avfakenet-inference-batch-benchmark-001/report.json'), __file__);inference=read(inference_path)
    if evidence['status']!='COMPLETE' or inference['status']!='COMPLETE':raise ValueError('Benchmarks incomplete')
    for path,sha in evidence['sources'].items():
        if digest(_fk_paths.relocate((Path(path)), __file__))!=sha:raise ValueError('Benchmark sources changed')
    # Execution-only tolerance in FP32 logits, with original-batch recomputation
    # near class ties. This is not a fitted classification threshold.
    candidates=[v for v in inference['arms'].values() if v.get('all_frame_decisions_identical') and v['max_logit_difference']<=1e-3]
    BATCH_SIZE=min(candidates,key=lambda v:v['seconds'])['batch_size']
    chosen=inference['arms'][str(BATCH_SIZE)]
    if not chosen['all_frame_decisions_identical'] or chosen['max_logit_difference']>1e-3:raise ValueError('Inference parity failed')
    workers=evidence['selected_workers'];lock=acquire();cache=_fk_paths.relocate((RUN/'prepared'), __file__);manifest=read(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__));training_id=read(_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001/identity.json'), __file__))
    detail={'scope':'execution-only optimization; preserved locked detection identity','sources':{str(_fk_paths.relocate((Path(__file__)), __file__)):digest(_fk_paths.relocate((Path(__file__)), __file__))},
        'preparation_workers':workers,'inference_batch_size':BATCH_SIZE,'preparation_evidence_sha256':digest(_fk_paths.relocate((EVIDENCE/'report.json'), __file__)),
        'inference_evidence_sha256':digest(inference_path),'precision':'FP32, TF32 disabled','logit_execution_tolerance':1e-3,
        'tie_guard':'repeat near-tie chunks with original batch8',
        'numerical_note':'Batch-size changes are not bitwise identical; all validation frame decisions match. Model thresholds and weights are unchanged.'}
    write_json(_fk_paths.relocate((RUN/('execution-tuning-'+str(time.time_ns())+'.json')), __file__),detail)
    try:
        pending=[];done=0
        for row in manifest:
            path=_fk_paths.relocate((cache/fingerprint(row)/'record.json'), __file__)
            if path.exists():original.validate_cache_record(cache,read(path),row);done+=1
            else:pending.append(row)
        initial=done;started=time.monotonic()
        def progress(status='RUNNING'):
            elapsed=time.monotonic()-started;rate=(done-initial)/elapsed if elapsed else 0;eta=(1000-done)/rate if rate>0 else None
            value={'status':status,'stage':'prepare_heldout_optimized','workers':workers,'processed':done,'total':1000,'percent':done/10,
                'eta_seconds':eta,'estimated_completion_unix':time.time()+eta if eta is not None else None,
                'updated_unix':time.time(),'elapsed_seconds_this_session':elapsed,'rate_per_second':rate}
            write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),value);print(json.dumps(value),flush=True)
        progress();pause=False;iterator=iter(pending)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=preparation.worker_init,initargs=(training_id['profile'],training_id['detector'])) as pool:
            futures={}
            def submit():
                row=next(iterator,None)
                if row is not None:futures[pool.submit(preparation.prepare_one,row,str(_fk_paths.relocate((ROOT.parent/'Heldout-1000/dataset'), __file__)),str(cache),True)]=row
            for _ in range(workers):submit()
            while futures:
                finished,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for f in finished:
                    row=futures.pop(f);original.validate_cache_record(cache,f.result()['record'],row);done+=1;progress()
                    pause=pause or (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists()
                    if not pause:submit()
        if pause:progress('PAUSED');return
    finally:lock.close()
    original_predict=original.predict;original.predict=optimized_prediction
    sys.argv=[sys.argv[0],'--workers','6'];original.main()


if __name__=='__main__':
    if '--benchmark' in sys.argv:benchmark()
    else:
        log=_fk_paths.relocate((RUN/('scheduled-'+str(time.time_ns())+'.log')), __file__);stream=log.open('a',buffering=1,encoding='utf-8')
        sys.stdout=stream;sys.stderr=stream
        try:run()
        except Exception as error:
            write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),{'status':'FAILED','stage':'execution_optimization','error':str(error),'updated_unix':time.time()})
            import traceback;traceback.print_exc();sys.exit(1)
