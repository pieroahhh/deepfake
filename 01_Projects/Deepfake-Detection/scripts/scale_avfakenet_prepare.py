"""Measured 2/3/4/6 worker scaling using unchanged preparation primitives."""

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
import sys,time,json,argparse,concurrent.futures,multiprocessing,subprocess
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import parallel_avfakenet_prepare as base
from baselines.common.io import read,digest,fingerprint


def initialize(profile,detector,threads):
    base.worker_init(profile,detector)
    import torch
    torch.set_num_threads(threads)


def warm(delay):
    pid=base.warmup(delay)
    import torch
    return {'pid':pid,'memory':base.memory_available(),'cuda_allocated':torch.cuda.memory_allocated(),
            'cuda_reserved':torch.cuda.memory_reserved()}


def lock_run(run):
    import msvcrt
    lock=(_fk_paths.relocate((run/'worker.lock'), __file__)).open('r+b')
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    return lock


def verify(run):
    identity=read(_fk_paths.relocate((run/'identity.json'), __file__))
    for relative,sha in identity['source'].items():
        if digest(_fk_paths.relocate((ROOT/relative), __file__))!=sha:raise ValueError('Pilot code changed: '+relative)
    manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__)
    if digest(manifest_path)!=identity['population_sha256']:raise ValueError('Population changed')
    return identity,read(manifest_path)


def source_hashes():
    return {str(path.relative_to(ROOT)):digest(path) for path in
        [_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((Path(base.__file__)), __file__),_fk_paths.relocate((ROOT/'baselines/common/video_reader.py'), __file__),_fk_paths.relocate((ROOT/'baselines/common/decode_prefetch.py'), __file__)]}


def benchmark(args,identity,manifest):
    if args.benchmark.exists():raise ValueError('Fresh benchmark output required')
    if read(_fk_paths.relocate((args.run/'progress.json'), __file__))['status']!='PAUSED':raise ValueError('Pause production before uncontended benchmark')
    lock=lock_run(args.run)
    try:
        args.benchmark.mkdir(parents=True)
        candidates=[]
        for row in manifest:
            header=_fk_paths.relocate((args.run/'prepared'/fingerprint(row)/'record.json'), __file__)
            if row['split']=='train' and header.exists():candidates.append((row,read(header)))
        # Same 12 clips for every candidate: duration spread with real outcomes.
        candidates.sort(key=lambda item:(item[0]['duration'],item[0]['video_id']))
        if len(candidates)<12:raise ValueError('At least 12 completed train clips required')
        chosen=[candidates[round(i*(len(candidates)-1)/11)] for i in range(12)]
        protected={str(_fk_paths.relocate((args.run/'prepared'), __file__)/rec['key']/name):sha for _,rec in chosen for name,sha in rec['files'].items()}
        for _,rec in chosen:
            header=_fk_paths.relocate((args.run/'prepared'/rec['key']/'record.json'), __file__);protected[str(header)]=digest(header)
        if not all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in protected.items()):raise ValueError('Source cache changed')
        result={'schema':'avfakenet-worker-scaling-v1','sources':source_hashes(),
            'pilot_identity_sha256':digest(_fk_paths.relocate((args.run/'identity.json'), __file__)),'original_worker_paused':True,
            'selection':'12 completed training clips spread across duration; same clips in all candidates',
            'clips':[{'video_id':r['video_id'],'duration':r['duration'],'reasons':rec['reasons']} for r,rec in chosen],
            'original_files':protected,'initial_memory':base.memory_available(),'arms':{}}
        base.write(_fk_paths.relocate((args.benchmark/'report.json'), __file__),result)
        for workers in (2,3,4,6):
            if base.memory_available()['available_gib'] < 3+1.2*workers:
                result['arms'][str(workers)]={'skipped':'Insufficient RAM headroom'}
                base.write(_fk_paths.relocate((args.benchmark/'report.json'), __file__),result);continue
            directory=args.benchmark/str(workers);directory.mkdir()
            print(json.dumps({'stage':'benchmark','workers':workers,'memory':base.memory_available()}),flush=True)
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(identity['profile'],identity['detector'],2)) as pool:
                warmed=list(pool.map(warm,[1.]*workers*2))
                if len({v['pid'] for v in warmed})!=workers:raise ValueError('Incomplete warmup')
                started=time.perf_counter();futures=[pool.submit(base.prepare_one,row,str(args.dataset_root),str(directory),True) for row,_ in chosen];observed=[];mem=[]
                for future in concurrent.futures.as_completed(futures):
                    observed.append(future.result());mem.append(base.memory_available()['available_gib'])
                    print(json.dumps({'workers':workers,'done':len(observed),'total':len(chosen)}),flush=True)
                seconds=time.perf_counter()-started
            expected={r['video_id']:rec for r,rec in chosen}
            mismatch=[v['record']['video_id'] for v in observed if v['record']!=expected[v['record']['video_id']]]
            result['arms'][str(workers)]={'workers':workers,'threads_per_worker':2,'seconds':seconds,
                'clips_per_second':len(chosen)/seconds,'minimum_observed_free_ram_gib':min(mem),
                'all_records_and_payload_hashes_identical':not mismatch,'mismatches':mismatch,'warmup':warmed}
            base.write(_fk_paths.relocate((args.benchmark/'report.json'), __file__),result)
            print(json.dumps({'workers':workers,'seconds':seconds,'identical':not mismatch,'free_ram_min':min(mem)}),flush=True)
        result['original_files_unchanged']=all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in protected.items())
        qualified=[v for v in result['arms'].values() if v.get('all_records_and_payload_hashes_identical') and v['minimum_observed_free_ram_gib']>=2]
        if not qualified or not result['original_files_unchanged']:raise ValueError('No verified candidate')
        best=min(qualified,key=lambda v:v['seconds'])
        # Avoid selecting a higher-resource candidate for a negligible difference.
        near=[v for v in qualified if v['seconds']<=best['seconds']*1.03]
        selected=min(near,key=lambda v:v['workers'])
        result['selected_workers']=selected['workers'];result['status']='COMPLETE'
        base.write(_fk_paths.relocate((args.benchmark/'report.json'), __file__),result)
        print(json.dumps({'selected_workers':selected['workers'],'speedup_vs_two':result['arms']['2']['seconds']/selected['seconds']}),flush=True)
    finally:lock.close()


def production(args,identity,manifest):
    report=read(_fk_paths.relocate((args.evidence/'report.json'), __file__))
    if report.get('status')!='COMPLETE' or not report['original_files_unchanged']:raise ValueError('Incomplete benchmark')
    if report['sources']!=source_hashes() or report['pilot_identity_sha256']!=digest(_fk_paths.relocate((args.run/'identity.json'), __file__)):raise ValueError('Benchmark source/config changed')
    workers=report['selected_workers'];arm=report['arms'][str(workers)]
    if not arm['all_records_and_payload_hashes_identical']:raise ValueError('Payload equivalence failed')
    if (_fk_paths.relocate((args.run/'pause.request'), __file__)).exists():raise ValueError('Remove acknowledged pause request first')
    if base.memory_available()['available_gib']<3+1.2*workers:raise ValueError('Insufficient memory for selected configuration')
    lock=lock_run(args.run);protected={};pending=[];processed=0;initial=0;started=time.monotonic()
    execution=args.run/('scaled-preparation-'+time.strftime('%Y%m%d-%H%M%S')+'.json')
    detail={'workers':workers,'threads_per_worker':2,'sources':source_hashes(),'evidence_sha256':digest(_fk_paths.relocate((args.evidence/'report.json'), __file__)),'status':'RUNNING'}
    def progress(status='RUNNING',**extra):
        elapsed=time.monotonic()-started;rate=(processed-initial)/elapsed if elapsed else 0
        eta=(len(manifest)-processed)/rate if rate>0 else None
        value={'status':status,'stage':'prepare_scaled','workers':workers,'processed':processed,'total':len(manifest),
            'percent':processed/len(manifest)*100,'newly_prepared':processed-initial,'reused':initial,
            'rate_per_second':rate,'eta_seconds':eta,'estimated_completion_unix':time.time()+eta if eta is not None else None,
            'elapsed_seconds_this_session':elapsed,'updated_unix':time.time(),**extra}
        base.write(_fk_paths.relocate((args.run/'progress.json'), __file__),value);print(json.dumps(value),flush=True)
    try:
        for row in manifest:
            header=_fk_paths.relocate((args.run/'prepared'/fingerprint(row)/'record.json'), __file__)
            if not header.exists():pending.append(row);continue
            rec=read(header)
            if rec['source_sha256']!=row['source_sha256']:raise ValueError('Completed source mismatch')
            for name,sha in rec['files'].items():
                if digest(header.parent/name)!=sha:raise ValueError('Completed payload mismatch')
                protected[str(header.parent/name)]=sha
            protected[str(header)]=digest(header);processed+=1
        initial=processed;detail.update(preserved_existing_records=initial,original_files=protected);base.write(execution,detail)
        iterator=iter(pending);progress()
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(identity['profile'],identity['detector'],2)) as pool:
            futures=set();pause=False
            def submit():
                row=next(iterator,None)
                if row is not None:futures.add(pool.submit(base.prepare_one,row,str(args.dataset_root),str(_fk_paths.relocate((args.run/'prepared'), __file__)),True))
            for _ in range(workers):submit()
            while futures:
                ready,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in ready:
                    futures.remove(future);item=future.result();processed+=1
                    progress(video_id=item['record']['video_id'],memory=base.memory_available())
                    pause=pause or (_fk_paths.relocate((args.run/'pause.request'), __file__)).exists()
                    if not pause:submit()
            if pause:
                detail['status']='PAUSED';base.write(execution,detail);progress('PAUSED');return
        if not all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in protected.items()):raise ValueError('Original cache changed')
        detail.update(status='FINISHED',original_files_unchanged=True);base.write(execution,detail);progress('PREPARATION_COMPLETE')
    except Exception as error:
        detail.update(status='FAILED',error=str(error));base.write(execution,detail);progress('FAILED',error=str(error));raise
    finally:lock.close()
    subprocess.run([sys.executable,'-u',str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_normalized_pilot.py'), __file__)),
        '--sanity',str(_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-full-tiny-004'), __file__)),
        '--bundle',r'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001',
        '--dataset-root',str(args.dataset_root),'--output',str(args.run),'--resume'],cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--benchmark',type=Path);p.add_argument('--evidence',type=Path)
    args=p.parse_args();args.run=args.run.resolve();args.dataset_root=args.dataset_root.resolve()
    identity,manifest=verify(args.run)
    if args.benchmark:benchmark(args,identity,manifest)
    elif args.evidence:production(args,identity,manifest)
    else:raise ValueError('Benchmark or evidence required')


if __name__=='__main__':main()
