"""Independent continuation: aggregation, real records, raw robustness and profiling.

Never modifies the running core queue, its code, or historical artifacts.
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

import os,sys,time,copy,json,math,subprocess,shutil,argparse
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='2'
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
os.environ['PATH']=str(_fk_paths.relocate((ROOT/'.cache/ffmpeg-tools/static_ffmpeg/bin/win32'), __file__))+os.pathsep+os.environ.get('PATH','')
os.environ['FK_PROGRESS_SECONDS']='10'
from baselines.common.io import read,digest,fingerprint
from avfakenet_progress_io import write_json
RUN=_fk_paths.relocate((ROOT/'runs/remaining-supplement-001'), __file__)
PACKAGE=_fk_paths.relocate((ROOT/'experiments/remaining-001'), __file__)
DATA=_fk_paths.relocate((ROOT.parent/'Heldout-1000/dataset'), __file__)
FK_CHECKPOINT=_fk_paths.relocate((ROOT/'runs/two-day-001/experiment/fk/training/attempt-001/selected.pt'), __file__)


def progress(stage,done=0,total=None,**extra):
    write_json(_fk_paths.relocate((RUN/'progress.json'), __file__),{'status':'RUNNING','stage':stage,'processed':done,'total':total,'updated_unix':time.time(),**extra})


def wait_for(directory,marker):
    while not (directory/marker).exists():
        state=read(_fk_paths.relocate((directory/'progress.json'), __file__)) if (_fk_paths.relocate((directory/'progress.json'), __file__)).exists() else {}
        if state.get('status') in ('FAILED','PAUSED'):raise RuntimeError(f'Upstream needs attention: {directory}: {state.get("error")}')
        progress('waiting_for_'+directory.name);time.sleep(10)


def run_command(command,log):
    with log.open('a',encoding='utf-8',buffering=1) as stream:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',creationflags=subprocess.CREATE_NO_WINDOW)
        for line in child.stdout:
            stream.write(line)
            progress(log.stem,last_log_line=line.strip()[-500:])
        if child.wait():raise RuntimeError(f'Command failed; see {log}')


def aggregation():
    import torch
    from evaluate_completed_ablations import rebind_manifest
    from deepfake_detection.training import load_fk_checkpoint
    from deepfake_detection.study import evaluate_checkpoint
    from deepfake_detection.reporting import write_detection_report
    output=_fk_paths.relocate((RUN/'aggregation'), __file__);output.mkdir(exist_ok=True)
    variants=['window_max','window_mean','video_max','video_mean']
    queue=_fk_paths.relocate((ROOT/'runs/remaining-ablations-001'), __file__)
    source_path=_fk_paths.relocate((ROOT/'runs/heldout-001/fk/cache/training-manifest.json'), __file__);source=read(source_path)
    expected={r['video_id']:r['label'] for r in read(_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__))}
    models={'full':read(_fk_paths.relocate((ROOT/'runs/remaining-ablation-evaluation-001/full/test.json'), __file__))['videos']}
    for name in variants:
        if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED between aggregation variants')
        if not (_fk_paths.relocate((queue/name/'completed.json'), __file__)).exists():
            run_command([str(_fk_paths.relocate((ROOT/'.venv-gpu/Scripts/python.exe'), __file__)),'-u',str(_fk_paths.relocate((ROOT/'scripts/run_prepared_ablation.py'), __file__)),'--variant',name],_fk_paths.relocate((output/(name+'.log')), __file__))
        dest=_fk_paths.relocate((output/name), __file__);dest.mkdir(exist_ok=True);cp=_fk_paths.relocate((queue/name/'attempt-001/selected.pt'), __file__);sha=digest(cp)
        lock={'checkpoint_sha256':sha,'training_run_sha256':digest(_fk_paths.relocate((cp.parent/'run.json'), __file__)),'source_manifest_sha256':digest(source_path)}
        if (_fk_paths.relocate((dest/'locked.json'), __file__)).exists() and read(_fk_paths.relocate((dest/'locked.json'), __file__))!=lock:raise ValueError('Aggregation inputs changed')
        write_json(_fk_paths.relocate((dest/'locked.json'), __file__),lock)
        if not (_fk_paths.relocate((dest/'test.json'), __file__)).exists():
            _,metadata=load_fk_checkpoint(cp,device='cpu')
            derived=rebind_manifest(source,source_path.parent,metadata,sha,lock['training_run_sha256'],expected)
            write_json(_fk_paths.relocate((dest/'evaluation-manifest.json'), __file__),derived)
            progress('aggregation_evaluation',variant=name)
            evaluate_checkpoint(_fk_paths.relocate((dest/'evaluation-manifest.json'), __file__),cp,_fk_paths.relocate((dest/'test.json'), __file__),split='test',device='cuda')
        result=read(_fk_paths.relocate((dest/'test.json'), __file__))
        if result['checkpoint_sha256']!=sha or len(result['videos'])!=1000 or {r['video_id']:r['label'] for r in result['videos']}!=expected:raise ValueError('Invalid aggregation evaluation')
        models[name]=result['videos'];torch.cuda.empty_cache()
    if not (_fk_paths.relocate((output/'comparison'), __file__)).exists():
        write_detection_report(models,_fk_paths.relocate((output/'comparison'), __file__),provenance={'protocol_id':'remaining-001-aggregation','data_kind':'research','scope':'Exploratory, five epochs, matched full reference; validation-only calibration'},reference='full',replicates=2000)
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'variants':list(models),'source_cache_manifest_unchanged':digest(source_path)==lock['source_manifest_sha256']})


class FrozenFK:
    def __init__(self):
        import torch
        # Historical FK cache/head inference used cuDNN's default TF32=True.
        # Four source clips reproduce their cached features bit-for-bit with it.
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=True
        torch.backends.cudnn.benchmark=False
        from deepfake_detection.commands import load_fk_assets,fk_asset_hashes
        from deepfake_detection.training import load_fk_checkpoint
        from deepfake_detection.quality import quality_tensor,profile_hash
        from deepfake_detection.preparation import IN_FRAME_CROP_POLICY,STRICT_CROP_POLICY
        self.model,self.metadata=load_fk_checkpoint(FK_CHECKPOINT,device='cuda')
        self.assets=read(_fk_paths.relocate((ROOT/'checkpoints/paper-alignment-001/assets.local.json'), __file__))
        self.encoders,self.detector,self.vad=load_fk_assets(self.assets);self.encoders.to('cuda').eval()
        self.policy=self.metadata['preprocessing']
        if fk_asset_hashes(self.assets,self.encoders)!=self.metadata['asset_sha256']:raise ValueError('FK asset mismatch')
        if self.policy['encoder_device']!='cuda:0' or self.policy['torch_version']!=str(torch.__version__):raise ValueError('FK encoder runtime differs')
        normalization=self.policy['quality_normalization']
        if profile_hash(normalization)!=self.policy['quality_normalization_sha256']:raise ValueError('Quality changed')
        if self.policy['face_crop_policy'] not in (IN_FRAME_CROP_POLICY,STRICT_CROP_POLICY):raise ValueError('Unknown crop policy')
        self.fit=self.policy['face_crop_policy']==IN_FRAME_CROP_POLICY
        self.mapper=lambda measurements:quality_tensor(measurements,normalization)
        self.sha=digest(FK_CHECKPOINT)

    def record(self,path,row,destination):
        from deepfake_detection.inference import run_fk_video
        return run_fk_video(path,row['duration'],self.model,self.encoders,self.detector,self.vad,self.mapper,destination,
            model_version=self.sha,quality_profile=self.policy['quality_profile'],provider_status='pretrained_assets_with_checkpoint',
            fit_to_frame=self.fit,provenance={**self.policy,'asset_sha256':self.metadata['asset_sha256'],
                'execution_precision':{'cuda_matmul_tf32':False,'cudnn_tf32':True,'reference':'remaining-fk-parity-diagnosis-002'}})

    def predict(self,path,row):
        import torch
        from deepfake_detection.preparation import prepare_fk_window,encode_fk_window,fk_source_speech
        from deepfake_detection.inference import record_from_outputs
        timings={'preprocessing_seconds':0.,'encoder_seconds':0.,'head_seconds':0.,'record_seconds':0.}
        def measured(name,fn):
            torch.cuda.synchronize();start=time.perf_counter();result=fn();torch.cuda.synchronize()
            timings[name]+=time.perf_counter()-start;return result
        with torch.inference_mode():
            try:speech=measured('preprocessing_seconds',lambda:fk_source_speech(path,row['duration'],self.vad,pcm16=True))
            except ValueError:speech=[]
            windows=[];outputs={}
            for i in range(math.ceil(row['duration'])):
                prepared=measured('preprocessing_seconds',lambda:prepare_fk_window(path,float(i),row['duration'],self.detector,speech,fit_to_frame=self.fit,pcm16=True))
                windows.append({'window_id':i,'interval':[float(i),float(i)+2.], 'status':prepared['status'],'reasons':prepared['reasons'],'quality':prepared['quality_measurements']})
                if prepared['status']=='USABLE':
                    features=measured('encoder_seconds',lambda:encode_fk_window(prepared,self.encoders))
                    def head():return self.model(**{k:v.cuda() for k,v in features.items()},quality=self.mapper(prepared['quality_measurements']).cuda())
                    outputs[i]=measured('head_seconds',head)
            record=measured('record_seconds',lambda:record_from_outputs(row['source_sha256'],row['duration'],windows,outputs,self.model,model_version=self.sha,preprocessing=self.policy))
        return {'video_id':row['video_id'],'label':row['label'],'status':record['status'],'prediction':record['prediction'],'score':record['score'],'coverage':record['coverage'],'timings':timings}


def mutation_checks(record,trusted,source,artifact_root,destination):
    from deepfake_detection.forensics import verify
    destination.mkdir(parents=True,exist_ok=True)
    results={'original':verify(record,trusted,source=source,artifact_root=artifact_root)}
    usable=next((w for w in record['windows'] if w['status']=='USABLE'),None)
    for name in ('contribution','timestamp','threshold','coverage','source_bytes','artifact_bytes'):
        edited=copy.deepcopy(record);src=source;artifacts=artifact_root
        if name=='contribution' and usable is None:
            results[name]={'status':'NOT_APPLICABLE','reason':'no usable evidence bins'};continue
        if name=='contribution':next(w for w in edited['windows'] if w['status']=='USABLE')['bins'][0]['contributions'][0]+=.1
        elif name=='timestamp':edited['windows'][0]['interval'][0]+=.001
        elif name=='threshold':edited['decision']['video_threshold']+=.001
        elif name=='coverage':edited['coverage']['covered_duration']+=.001
        elif name=='source_bytes':
            src=_fk_paths.relocate((destination/'changed-source.mp4'), __file__)
            if not src.exists():
                with _fk_paths.relocate((Path(source)), __file__).open('rb') as a,src.open('xb') as b:shutil.copyfileobj(a,b);b.write(b'changed')
        elif name=='artifact_bytes':
            if not record.get('derived_evidence'):
                results[name]={'status':'NOT_APPLICABLE','reason':'no derived artifacts'};continue
            # Private complete copy of only the checked artifacts; original untouched.
            artifacts=_fk_paths.relocate((destination/'changed-artifacts'), __file__);target=record['derived_evidence'][0]['path']
            for item in record['derived_evidence']:
                dest=artifacts/item['path'];dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(artifact_root/item['path'],dest)
                    if item['path']==target:
                        with dest.open('ab') as stream:stream.write(b'changed')
        write_json(destination/(name+'.record.json'),edited)
        try:verify(edited,trusted,source=src,artifact_root=artifacts)
        except ValueError as error:results[name]={'status':'REJECTED_AS_EXPECTED','reason':str(error)}
        else:raise ValueError('Tampering accepted: '+name)
    verify(record,trusted,source=source,artifact_root=artifact_root)
    return results


def forensic(fk):
    rows=read(_fk_paths.relocate((PACKAGE/'forensic-plan.json'), __file__))['selected_source_examples'];output=_fk_paths.relocate((RUN/'forensic'), __file__);output.mkdir(exist_ok=True)
    from deepfake_detection.media import media_tool,probe
    real=next(r for r in rows if r['label']==0);sources={r['video_id']:_fk_paths.relocate((DATA/r['path']), __file__) for r in rows}
    for name,options in [('no_audio',['-an','-c:v','copy']),('short',['-t','0.8','-c:v','libx264','-c:a','aac'])]:
        path=_fk_paths.relocate((output/(name+'.mp4')), __file__)
        if not path.exists():subprocess.run([str(media_tool('ffmpeg')),'-v','error','-i',str(_fk_paths.relocate((DATA/real['path']), __file__)),*options,'-threads','1',str(path)],check=True,capture_output=True)
        row={**real,'video_id':'forensic-edge/'+name,'source_sha256':digest(path),'duration':float(probe(path)['format']['duration'])}
        rows.append(row);sources[row['video_id']]=path
    results=[]
    for i,row in enumerate(rows):
        progress('real_forensic_records',i,len(rows));folder=_fk_paths.relocate((output/fingerprint(row)), __file__);source=sources[row['video_id']]
        if digest(source)!=row['source_sha256']:raise ValueError('Forensic source changed')
        if not (_fk_paths.relocate((folder/'record.json'), __file__)).exists():fk.record(source,row,folder)
        record=read(_fk_paths.relocate((folder/'record.json'), __file__));trusted=(_fk_paths.relocate((folder/'trusted-commitment.txt'), __file__)).read_text().strip()
        retained=_fk_paths.relocate((output/(fingerprint(row)+'.trusted.txt')), __file__)
        if retained.exists() and retained.read_text().strip()!=trusted:raise ValueError('Trusted reference changed')
        if not retained.exists():retained.write_text(trusted+'\n')
        tests=mutation_checks(record,retained.read_text().strip(),source,folder,_fk_paths.relocate((output/(fingerprint(row)+'-mutations')), __file__))
        results.append({'video_id':row['video_id'],'source_sha256':row['source_sha256'],'record_status':record['status'],'checks':tests})
    run_command([str(_fk_paths.relocate((ROOT/'.venv-gpu/Scripts/python.exe'), __file__)),'-m','unittest','discover','-s','tests','-p','test_foundations.py','-v'],_fk_paths.relocate((output/'software-edge-cases.log'), __file__))
    write_json(_fk_paths.relocate((output/'summary.json'), __file__),{'real_source_bound_records':results,'claim':'Integrity/reconstruction only, not media authenticity or classifier correctness','additional_edge_cases':'Two transformed genuine source cases: no audio, sub-2-second media; software edge cases separately in log. Partial coverage reported by records.'})
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'summary_sha256':digest(_fk_paths.relocate((output/'summary.json'), __file__))})


def raw_study(fk,name='fk'):
    from baselines.common.metrics import metrics
    from prepare_remaining_derivatives import RUN as derivatives
    wait_for(derivatives,'complete.json');rows=read(_fk_paths.relocate((derivatives/'manifest.json'), __file__))
    if digest(_fk_paths.relocate((derivatives/'manifest.json'), __file__))!=read(_fk_paths.relocate((derivatives/'complete.json'), __file__))['manifest_sha256']:raise ValueError('Derivative manifest changed')
    output=_fk_paths.relocate((RUN/('robustness-'+name)), __file__);output.mkdir(exist_ok=True);groups={}
    for i,row in enumerate(rows):
        if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists():raise RuntimeError('PAUSED between robustness clips')
        progress('robustness_'+name,i,len(rows),condition=row['condition'])
        path=_fk_paths.relocate((Path(row.get('absolute_source',str(derivatives/row['path'])))), __file__)
        if digest(path)!=row['source_sha256']:raise ValueError('Robustness source changed')
        result_path=_fk_paths.relocate((output/(fingerprint(row)+'.json')), __file__)
        if not result_path.exists():
            result=fk.predict(path,row);write_json(result_path,{'checkpoint_sha256':fk.sha,'row':row,'prediction':result})
        saved=read(result_path)
        if saved['row']!=row or saved['checkpoint_sha256']!=fk.sha:raise ValueError('Wrong retained prediction')
        groups.setdefault(row['condition'],[]).append(saved['prediction'])
    write_json(_fk_paths.relocate((output/'summary.json'), __file__),{'model':name,'metrics':{k:metrics(v) for k,v in groups.items()},'note':'Real-only AV delay: report false alarms and coverage, not fake-detection BA; exploratory'})
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'summary_sha256':digest(_fk_paths.relocate((output/'summary.json'), __file__))})


def profiling(fk,name='fk'):
    import torch,statistics,psutil
    output=_fk_paths.relocate((RUN/('profiling-'+name)), __file__);output.mkdir(exist_ok=True);rows=read(_fk_paths.relocate((PACKAGE/'profiling-manifest.json'), __file__));results=[]
    for i,row in enumerate(rows):
        progress('profiling_'+name,i,len(rows));target=_fk_paths.relocate((output/(fingerprint(row)+'.json')), __file__)
        if not target.exists():
            source=_fk_paths.relocate((DATA/row['path']), __file__);samples=[]
            for repeat in range(4):
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                if digest(source)!=row['source_sha256']:raise ValueError('Profile source changed')
                result=fk.predict(source,row);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                samples.append({'repeat':repeat,'seconds':elapsed,'seconds_per_source_second':elapsed/row['duration'],'timings':result['timings'],
                    'prediction':result['prediction'],'status':result['status'],'cuda_peak_allocated':torch.cuda.max_memory_allocated(),
                    'cuda_peak_reserved':torch.cuda.max_memory_reserved(),'process_rss_after':psutil.Process().memory_info().rss})
            write_json(target,{'row':row,'samples':samples,'median_seconds':statistics.median(s['seconds'] for s in samples[1:]),
                'scope':'Loaded model; raw hash/decode/preparation/neural prediction; FK also constructs canonical record. No derived artifact persistence. First pass reported separately, not OS cold-cache guarantee.'})
        results.append(read(target))
    write_json(_fk_paths.relocate((output/'summary.json'), __file__),{'model':name,'clips':results,'warmups':1,'timed_repeats':3,'model_load_excluded':True,'disk_cache':'Uncontrolled OS cache; no claim of cold disk timing'})
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'summary_sha256':digest(_fk_paths.relocate((output/'summary.json'), __file__))})


def parity(model,name):
    from baselines.common.metrics import read_predictions
    path=_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__) if name=='fk' else _fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002/predictions.jsonl'), __file__)
    reference={r['video_id']:r for r in read_predictions(path)}
    results=[]
    for row in read(_fk_paths.relocate((PACKAGE/'forensic-plan.json'), __file__))['selected_source_examples']:
        value=model.predict(_fk_paths.relocate((DATA/row['path']), __file__),row);old=reference[row['video_id']]
        if value['prediction']!=old['prediction'] or (value['score'] is None)!=(old.get('score') is None):raise ValueError('Raw adapter differs from retained held-out result: '+row['video_id'])
        if value['score'] is not None and abs(value['score']-old['score'])>5e-6:raise ValueError('Raw FK score differs from retained result')
        results.append({'video_id':row['video_id'],'matched':True})
    write_json(_fk_paths.relocate((RUN/(name+'-raw-parity.json')), __file__),{'checks':results,'scope':'Four metadata-selected cases; not exhaustive equivalence'})


def avtenet():
    from prepare_remaining_derivatives import RUN as derivatives
    wait_for(derivatives,'complete.json');inputs=[]
    for row in read(_fk_paths.relocate((derivatives/'manifest.json'), __file__)):
        path=_fk_paths.relocate((Path(row.get('absolute_source',str(derivatives/row['path'])))), __file__).resolve()
        inputs.append({'purpose':'robustness','row':{**row,'media_path':path.relative_to(ROOT.parent).as_posix()}})
    for row in read(_fk_paths.relocate((PACKAGE/'profiling-manifest.json'), __file__)):
        inputs.append({'purpose':'profiling','row':{**row,'media_path':(_fk_paths.relocate((DATA/row['path']), __file__)).relative_to(ROOT.parent).as_posix()}})
    manifest=_fk_paths.relocate((RUN/'avtenet-inputs.json'), __file__)
    if manifest.exists() and read(manifest)!=inputs:raise ValueError('AVTENet study inputs changed')
    write_json(manifest,inputs)
    avroot=_fk_paths.relocate((ROOT.parent/'avh-a/code'), __file__);script=_fk_paths.relocate((ROOT/'scripts/run_remaining_avtenet.py'), __file__)
    for stage in ('audio','visual','audiovisual','fusion'):
        if (_fk_paths.relocate((RUN/'avtenet'/stage/'complete.json'), __file__)).exists():continue
        progress('avtenet_'+stage)
        if stage=='audiovisual':
            command=['docker','run','--rm','--gpus','all','--shm-size','1g','-e','OMP_NUM_THREADS=2','-e','OPENBLAS_NUM_THREADS=2',
                '--mount',f'type=bind,source={avroot},target=/workspace,readonly',
                '--mount',f'type=bind,source={ROOT.parent},target=/inputs,readonly',
                '--mount',f'type=bind,source={RUN},target=/study',
                '--mount',f'type=bind,source={script},target=/runner.py,readonly','--entrypoint','python','fk-avhubert:cuda117-v1',
                '/runner.py','--root','/workspace','--study','/study','--input-root','/inputs','--stage',stage,'--assets','/workspace/runs/worker-c/assets.docker.local.json']
        else:
            command=[str(_fk_paths.relocate((ROOT/'.venv-gpu/Scripts/python.exe'), __file__)),'-u',str(script),'--root',str(avroot),'--study',str(RUN),'--input-root',str(ROOT.parent),'--stage',stage,'--assets',str(_fk_paths.relocate((avroot/'runs/worker-c/assets.local.json'), __file__))]
        run_command(command,_fk_paths.relocate((RUN/('avtenet-'+stage+'.log')), __file__))


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('all','aggregation','forensic','robustness','profiling'),default='all');args=p.parse_args()
    RUN.mkdir(exist_ok=True)
    from deepfake_detection.quality_fitting import exclusive_run
    with exclusive_run(_fk_paths.relocate((RUN/'worker.lock'), __file__)):
        protected={str(path):digest(path) for path in (FK_CHECKPOINT,_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__),_fk_paths.relocate((PACKAGE/'SHA256SUMS.json'), __file__))}
        identity={'script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'protected':protected,'scope':'Exploratory remaining study; FakeAVCeleb deferred by user'}
        if (_fk_paths.relocate((RUN/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((RUN/'identity.json'), __file__))!=identity:raise ValueError('Study code/inputs changed; use a new run')
        write_json(_fk_paths.relocate((RUN/'identity.json'), __file__),identity)
        try:
            wait_for(_fk_paths.relocate((ROOT/'runs/remaining-ablation-evaluation-001'), __file__),'complete.json')
            import torch
            torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            if args.stage in ('all','aggregation') and not (_fk_paths.relocate((RUN/'aggregation/complete.json'), __file__)).exists():aggregation()
            if args.stage!='aggregation':
                progress('loading_frozen_fk');fk=FrozenFK()
                parity(fk,'fk')
                for name,fn in [('forensic',forensic),('profiling',profiling),('robustness',raw_study)]:
                    if args.stage in ('all',name):fn(fk)
                del fk
                import gc
                gc.collect();torch.cuda.empty_cache()
                if args.stage=='all':
                    from remaining_avfakenet import FrozenAVFakeNet
                    model=FrozenAVFakeNet(ROOT);parity(model,'avfakenet');profiling(model,'avfakenet');raw_study(model,'avfakenet')
                    del model;gc.collect();torch.cuda.empty_cache()
                    avtenet()
            if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Protected historical input changed')
            write_json(_fk_paths.relocate((RUN/'complete.json'), __file__),{'stage':args.stage,'historical_inputs_unchanged':True,
                'scope':'Aggregation, real FK records, three-method raw profiling and robustness; see measured boundaries and exclusions','transfer':'DEFERRED_BY_USER'})
            progress('complete',status='FINISHED')
        except Exception as error:progress('error',status='FAILED',error=str(error));raise


if __name__=='__main__':
    RUN.mkdir(exist_ok=True)
    log=(_fk_paths.relocate((RUN/('worker-'+str(time.time_ns())+'.log')), __file__)).open('a',encoding='utf-8',buffering=1)
    sys.stdout=log;sys.stderr=log
    main()
