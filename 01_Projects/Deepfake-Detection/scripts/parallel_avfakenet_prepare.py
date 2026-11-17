"""Separate, bounded preparation accelerator; original pilot code stays unchanged.

Benchmark in a separate directory first. Production mode requires the original
worker to be paused and holds its OS lock. Completed cache files are read-only.
After filling missing records, release the lock and resume the original runner.
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
import sys,json,time,argparse,concurrent.futures,multiprocessing,subprocess,ctypes
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]


def memory_available():
    class Memory(ctypes.Structure):
        _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(name,ctypes.c_ulonglong) for name in ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended')]
    value=Memory();value.length=ctypes.sizeof(value)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(value)):raise OSError('Memory status unavailable')
    return {'available_gib':value.avail_phys/1024**3,'total_gib':value.total_phys/1024**3}


def worker_init(profile,expected_detector):
    global PROFILE,DETECT
    import torch
    from baselines.common.media import detector
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    PROFILE=profile;DETECT,provenance=detector('cuda')
    if provenance!=expected_detector:raise ValueError('Detector differs from pilot')


def warmup(delay=1.):
    import numpy as np
    DETECT(np.zeros((128,128,3),dtype=np.uint8))
    time.sleep(delay)
    return os.getpid()


def prepare_one(row,dataset,cache,reuse_decoder):
    import numpy as np
    from baselines.common.io import read,digest,fingerprint
    from baselines.common.media import faces,audio
    from baselines.common.population import target
    from baselines.avfakenet.preprocessing import prepare_audio
    from baselines.common.video_reader import VideoWindowReader
    started=time.perf_counter();key=fingerprint(row);directory=_fk_paths.relocate((Path(cache)/key), __file__)
    directory.mkdir(parents=True,exist_ok=True);header=_fk_paths.relocate((directory/'record.json'), __file__)
    path=(_fk_paths.relocate((Path(dataset)/row['path']), __file__)).resolve()
    if _fk_paths.relocate((Path(dataset)), __file__).resolve() not in path.parents or digest(path)!=row['source_sha256']:raise ValueError('Source integrity failure')
    if header.exists():
        saved=read(header)
        if saved['source_sha256']!=row['source_sha256'] or any(digest(_fk_paths.relocate((directory/name), __file__))!=sha for name,sha in saved['files'].items()):raise ValueError('Completed cache differs')
        return {'record':saved,'seconds':time.perf_counter()-started,'reused':True}
    record={'video_id':row['video_id'],'key':key,'split':row['split'],'source_sha256':row['source_sha256'],
            'label':row['label'],'audio_label':row['audio_label'],'reasons':{},'files':{}}
    def array(name,value):
        temp=_fk_paths.relocate((directory/(name+'.tmp')), __file__)
        with temp.open('wb') as stream:np.save(stream,value,allow_pickle=False)
        temp.replace(_fk_paths.relocate((directory/name), __file__));record['files'][name]=digest(_fk_paths.relocate((directory/name), __file__))
    reader=VideoWindowReader() if reuse_decoder else None
    try:
        try:
            sample=faces(path,0,row['duration'],DETECT,sampler=reader.sample if reader else None,prefetch_decode=reuse_decoder)
            labels=[target(row,'visual',float(t)) for t in sample['source_times']]
        except (ValueError,OSError) as error:record['reasons']['visual']=str(error)
        else:
            array('visual.npy',sample['faces']);array('visual-labels.npy',np.array([-1 if v is None else v for v in labels],dtype=np.int8))
    finally:
        if reader:reader.close()
    try:mel=prepare_audio(audio(path,0,row['duration']),16000,PROFILE).numpy()
    except (ValueError,OSError) as error:record['reasons']['audio']=str(error)
    else:array('audio.npy',mel)
    temp=header.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2,allow_nan=False),encoding='utf-8');temp.replace(header)
    return {'record':record,'seconds':time.perf_counter()-started,'reused':False}


def write(path,value):
    path=_fk_paths.relocate((Path(path)), __file__);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def run_benchmark(args,identity,manifest):
    from baselines.common.io import read,fingerprint,digest
    output=args.benchmark.resolve()
    if output.exists():raise ValueError('Use new benchmark directory')
    output.mkdir(parents=True)
    # Select completed train clips across available preparation outcomes.
    candidates=[]
    for row in manifest:
        header=_fk_paths.relocate((args.run/'prepared'/fingerprint(row)/'record.json'), __file__)
        if row['split']=='train' and header.exists():candidates.append((row,read(header)))
    chosen=[];buckets={}
    for row,record in candidates:
        category=tuple(sorted(record['reasons']))
        buckets.setdefault(category,[]).append((row,record))
    while len(chosen)<6 and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(chosen)<6:chosen.append(buckets[category].pop(0))
    if len(chosen)<4:raise ValueError('Need at least four completed clips')
    before={str(_fk_paths.relocate((args.run/'prepared'), __file__)/record['key']/name):sha for _,record in chosen for name,sha in record['files'].items()}
    result={'memory':memory_available(),'original_worker_kept_running':True,'selected_ids':[r['video_id'] for r,_ in chosen],'arms':{}}
    # Warm both processes before measuring steady-state preparation.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(identity['profile'],identity['detector'])) as pool:
        warm=set(pool.map(warmup,[1.,1.,1.,1.]))
        if len(warm)!=2:raise ValueError('Both workers must be warmed')
        for arm,parallel,reuse in [('sequential',False,False),('parallel',True,False),('parallel_reuse_decode',True,True)]:
            directory=output/arm;directory.mkdir();started=time.perf_counter();observed=[]
            if parallel:
                tasks=[pool.submit(prepare_one,row,str(args.dataset_root),str(directory),reuse) for row,_ in chosen]
                for task in concurrent.futures.as_completed(tasks):observed.append(task.result())
            else:
                for row,_ in chosen:observed.append(pool.submit(prepare_one,row,str(args.dataset_root),str(directory),reuse).result())
            seconds=time.perf_counter()-started;expected={r['video_id']:rec for r,rec in chosen}
            mismatches=[item['record']['video_id'] for item in observed if item['record']!=expected[item['record']['video_id']]]
            result['arms'][arm]={'seconds':seconds,'clips_per_second':len(chosen)/seconds,'all_records_and_payload_hashes_identical':not mismatches,'mismatches':mismatches,'observations':observed}
            write(_fk_paths.relocate((output/'report.json'), __file__),result);print(json.dumps({'arm':arm,'seconds':seconds,'identical':not mismatches}),flush=True)
    result['original_completed_payloads_unchanged']=all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in before.items())
    write(_fk_paths.relocate((output/'report.json'), __file__),result)


def run_production(args,identity,manifest):
    from baselines.common.io import read,digest,fingerprint
    benchmark=read(_fk_paths.relocate((args.evidence/'report.json'), __file__));arm='parallel_reuse_decode' if args.reuse_decoder else 'parallel'
    if not benchmark['arms'][arm]['all_records_and_payload_hashes_identical'] or not benchmark['original_completed_payloads_unchanged']:raise ValueError('Benchmark did not prove equality')
    if benchmark['arms'][arm]['seconds']>=benchmark['arms']['sequential']['seconds']:raise ValueError('Candidate was not faster')
    if memory_available()['available_gib']<5:raise ValueError('Need at least 5 GiB free RAM for two preparation workers')
    lock=(_fk_paths.relocate((args.run/'worker.lock'), __file__)).open('r+b')
    import msvcrt
    msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    pending=[];protected={};preserved_headers={};start=time.monotonic();processed=0
    execution=_fk_paths.relocate((args.run/'parallel-preparation.json'), __file__)
    for row in manifest:
        header=_fk_paths.relocate((args.run/'prepared'/fingerprint(row)/'record.json'), __file__)
        if header.exists():
            record=read(header)
            if record['source_sha256']!=row['source_sha256']:raise ValueError('Cache source differs')
            for name,sha in record['files'].items():
                path=_fk_paths.relocate((header.parent/name), __file__)
                if digest(path)!=sha:raise ValueError('Cache payload differs')
                protected[str(path)]=sha
            preserved_headers[str(header)]=digest(header);processed+=1
        else:pending.append(row)
    initial=processed
    execution_info={'schema':'avfakenet-parallel-preparation-v1','workers':2,'threads_per_worker':2,'reuse_decoder':args.reuse_decoder,
        'benchmark_sha256':digest(_fk_paths.relocate((args.evidence/'report.json'), __file__)),'pilot_identity_sha256':digest(_fk_paths.relocate((args.run/'identity.json'), __file__)),
        'accelerator_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'preserved_existing_records':initial,'original_files':{**protected,**preserved_headers},'status':'RUNNING'}
    write(execution,execution_info)
    def progress(status='RUNNING',**extra):
        elapsed=time.monotonic()-start;rate=(processed-initial)/elapsed if elapsed else 0
        eta=(len(manifest)-processed)/rate if rate>0 else None
        event={'status':status,'stage':'prepare_parallel','processed':processed,'total':len(manifest),'percent':processed/len(manifest)*100,
            'newly_prepared':processed-initial,'reused':initial,'rate_per_second':rate,'eta_seconds':eta,
            'estimated_completion_unix':time.time()+eta if eta is not None else None,'elapsed_seconds_this_session':elapsed,'updated_unix':time.time(),**extra}
        write(_fk_paths.relocate((args.run/'progress.json'), __file__),event);print(json.dumps(event),flush=True)
    try:
        if (_fk_paths.relocate((args.run/'pause.request'), __file__)).exists():raise ValueError('Remove acknowledged pause request before starting')
        progress();iterator=iter(pending)
        with concurrent.futures.ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=worker_init,initargs=(identity['profile'],identity['detector'])) as pool:
            futures=set()
            def submit():
                row=next(iterator,None)
                if row is not None:futures.add(pool.submit(prepare_one,row,str(args.dataset_root),str(_fk_paths.relocate((args.run/'prepared'), __file__)),args.reuse_decoder))
            submit();submit()
            pause=False
            while futures:
                finished,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for task in finished:
                    futures.remove(task);item=task.result();processed+=1
                    progress(video_id=item['record']['video_id'])
                    pause=pause or (_fk_paths.relocate((args.run/'pause.request'), __file__)).exists()
                    if not pause:submit()
            if pause:
                execution_info['status']='PAUSED';write(execution,execution_info);progress('PAUSED');return
        if not all(digest(_fk_paths.relocate((Path(path)), __file__))==sha for path,sha in {**protected,**preserved_headers}.items()):raise ValueError('Existing cache changed')
        execution_info['status']='FINISHED';execution_info['original_files_unchanged']=True;write(execution,execution_info)
        progress('PREPARATION_COMPLETE')
    except Exception as error:
        execution_info['status']='FAILED';execution_info['error']=str(error);write(execution,execution_info);progress('FAILED',error=str(error));raise
    finally:lock.close()
    # Original runner verifies every cache record, then executes its original
    # training/validation code and fixed sampler order without modification.
    command=[sys.executable,'-u',str(_fk_paths.relocate((ROOT/'scripts/run_avfakenet_normalized_pilot.py'), __file__)),
        '--sanity',str(_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-full-tiny-004'), __file__)),
        '--bundle',str(args.bundle),'--dataset-root',str(args.dataset_root),'--output',str(args.run),'--resume']
    print('Preparation complete; resuming unchanged pilot runner',flush=True)
    subprocess.run(command,cwd=ROOT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--bundle',type=Path,default=_fk_paths.relocate((Path(r'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001')), __file__))
    p.add_argument('--benchmark',type=Path);p.add_argument('--evidence',type=Path);p.add_argument('--reuse-decoder',action='store_true')
    args=p.parse_args();args.run=args.run.resolve();args.dataset_root=args.dataset_root.resolve()
    from baselines.common.io import read,digest
    identity=read(_fk_paths.relocate((args.run/'identity.json'), __file__))
    for relative,expected in identity['source'].items():
        if digest(_fk_paths.relocate((ROOT/relative), __file__))!=expected:raise ValueError('Original pilot source changed: '+relative)
    manifest_path=_fk_paths.relocate((ROOT/'runs/two-day-001/population/manifest.json'), __file__)
    if digest(manifest_path)!=identity['population_sha256']:raise ValueError('Population changed')
    manifest=read(manifest_path)
    if args.benchmark:run_benchmark(args,identity,manifest)
    elif args.evidence:run_production(args,identity,manifest)
    else:raise ValueError('Benchmark or verified evidence required')


if __name__=='__main__':main()
