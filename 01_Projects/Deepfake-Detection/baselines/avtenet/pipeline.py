"""Streamed expert fine-tuning, verified feature caches, and frozen-feature fusion."""

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

import json
from pathlib import Path
import sqlite3
import time
from contextlib import closing

import numpy as np
import torch

from baselines.common.io import read,write,digest,fingerprint
from baselines.common.population import target,windows
from baselines.common.media import detector,faces,audio,event
from baselines.common.training import fit,seed_all,load_checkpoint,move
from baselines.common.exceptions import TrainingPaused
from baselines.common.metrics import metrics,compare,read_predictions
from .model import VectorExpert,AudioVisualExpert,FeatureFusion,native_binary_loss,load_component_profile
from .assets import registry,load_backend

CONFIG=_fk_paths.relocate((Path(__file__).parent/'configs/avtenet.components-v1.json'), __file__)
SCHEDULE=_fk_paths.relocate((Path(__file__).parent/'configs/avtenet.training-v1.json'), __file__)
PREPARATION={'version':1,'window_seconds':2,'stride_seconds':1,'tail':'end_aligned_complete_window',
             'short_videos':'UNASSESSED','video_frames':32,'lip_fps':25,'audio_rate':16000,
             'window_targets':'positive_duration_overlap_modality_specific','video_score':'mean_window_fake_probability',
             'threshold':.5,'augmentation':'none_reconstruction_default'}


class Prepared:
    def __init__(self,population,stage,backend,assets,detect,failure_log=None,input_cache=None,
                 reuse_video_decoder=False,prefetch_cached_inputs=0,
                 cache_face_detections=False,fast_vivit_processor=False,
                 prefetch_processed_inputs=0,prefetch_video_decode=False):
        self.population,self.stage,self.backend,self.assets,self.detect=population,stage,backend,assets,detect
        self.failure_log=failure_log
        self.prefetch_video_decode=prefetch_video_decode
        self.input_cache=input_cache
        self._last_cursor=None
        self._progress_info={}
        self.fast_vivit_processor=fast_vivit_processor
        if not 0<=prefetch_processed_inputs<=4:raise ValueError('processed input prefetch must be between 0 and 4')
        self.prefetch_processed_inputs=prefetch_processed_inputs
        if prefetch_processed_inputs:
            from .hf_vivit import HFVivitBackend
            if input_cache is None or not (stage=='audiovisual' or (stage=='visual' and type(backend) is HFVivitBackend)):
                raise ValueError('processed prefetch requires AV-HuBERT or HF ViViT and a prepared cache')
        self.detection_cache=None
        if cache_face_detections and stage in ('visual','audiovisual'):
            from deepfake_detection.faces import MTCNNDetector
            from baselines.common.detection_cache import FrameDetectionCache
            if not isinstance(detect,MTCNNDetector):
                raise ValueError('frame detection reuse is restricted to the stateless MTCNN adapter')
            self.detection_cache=FrameDetectionCache(detect)
            self.detect=self.detection_cache
        if not 0<=prefetch_cached_inputs<=4:raise ValueError('cache prefetch depth must be between 0 and 4')
        self.prefetch_cached_inputs=prefetch_cached_inputs
        self.video_reader=None
        if reuse_video_decoder and stage in ('visual','audiovisual'):
            from baselines.common.video_reader import VideoWindowReader
            self.video_reader=VideoWindowReader()
        if stage=='audio':
            from transformers import ASTFeatureExtractor
            self.extractor=ASTFeatureExtractor.from_pretrained(_fk_paths.relocate_value((assets['experts']['audio']['directory']), __file__),local_files_only=True)
            if self.extractor.sampling_rate!=16000:raise ValueError('AST requires explicit 16 kHz preparation')

    def raw_inputs(self,path,start):
        sampler=self.video_reader.sample if self.video_reader else None
        if self.stage=='visual':
            return {'faces':faces(path,start,2,self.detect,fps=16,pixels=224,sampler=sampler,
                                  prefetch_decode=self.prefetch_video_decode)['faces']}
        waveform=audio(path,start,2)
        if self.stage=='audio':return {'waveform':waveform}
        import cv2
        sample=faces(path,start,2,self.detect,fps=25,pixels=224,sampler=sampler,
                     prefetch_decode=self.prefetch_video_decode)
        return {'waveform':waveform,'lips':np.stack([cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY) for rgb in sample['lips']])}

    def inputs(self,path,start,row=None,raw=None):
        if raw is None:
            raw=(self.input_cache.get(row,start,lambda:self.raw_inputs(path,start))
                 if self.input_cache else self.raw_inputs(path,start))
        return self.transform(raw)

    def transform(self,raw):
        """Deterministic CPU preparation; the model's forward is never called."""
        if self.stage=='visual':
            from deepfake_detection.backbones import prepare_vivit
            if hasattr(self.backend,'prepare'):
                if self.fast_vivit_processor:return self.backend.prepare(raw['faces'],fast=True)
                return self.backend.prepare(raw['faces'])
            return prepare_vivit(raw['faces'],self.backend.config)
        waveform=raw['waveform']
        if self.stage=='audio':
            return self.extractor(waveform,sampling_rate=16000,return_tensors='pt')['input_values']
        from deepfake_detection.avhubert_preparation import prepare_avhubert
        return prepare_avhubert(waveform,raw['lips'],self.assets['experts']['audiovisual']['preparation'])

    def cached_input(self,row,start):
        raw=self.input_cache.load(row,start)
        if self.prefetch_processed_inputs and raw is not None:
            value=self.transform(raw)
            values=value.values() if isinstance(value,dict) else (value,)
            if any(not isinstance(v,torch.Tensor) or v.device.type!='cpu' for v in values):
                raise ValueError('prefetched input must stay on CPU')
            return value
        return raw

    def records(self,split,*,supervised=False,skip_windows=None,resume_after=None):
        try:
            yield from self._records(split,supervised=supervised,skip_windows=skip_windows,resume_after=resume_after)
        finally:
            if self.video_reader:self.video_reader.close()
            if self.detection_cache:self.detection_cache.clear()

    def _records(self,split,*,supervised=False,skip_windows=None,resume_after=None):
        rows=list(self.population.rows(split))
        planned=sum(len(windows(row['duration'])) for row in rows)
        self._progress_info={'windows_total':planned,'windows_processed':0,'initial_windows':0,
                             'clips_total':len(rows),'clip_number':0}
        self._last_cursor=resume_after
        skipping=resume_after is not None
        for number,row in enumerate(rows,1):
            if self.detection_cache:self.detection_cache.clear()
            intervals=windows(row['duration'])
            self._progress_info['clip_number']=number
            if skipping:
                if row['video_id']!=resume_after['video_id']:
                    self._progress_info['windows_processed']+=len(intervals)
                    continue
                pair=(resume_after['start'],resume_after['end'])
                if pair not in intervals:raise ValueError('resume cursor window differs from population')
                offset=intervals.index(pair)+1
                self._progress_info['windows_processed']+=offset
                if self._progress_info['windows_processed']!=resume_after['windows_processed']:
                    raise ValueError('resume cursor order differs from population')
                self._progress_info['initial_windows']=self._progress_info['windows_processed']
                intervals=intervals[offset:]
                skipping=False
                if not intervals:continue
            path=self.population.path(row)
            if not intervals:event(self.failure_log,row,'video_shorter_than_two_seconds')
            if skip_windows:
                remaining=[(start,end) for start,end in intervals if (row['video_id'],start,end) not in skip_windows]
                skipped=len(intervals)-len(remaining)
                self._progress_info['windows_processed']+=skipped
                self._progress_info['initial_windows']+=skipped
                intervals=remaining
            depth=self.prefetch_processed_inputs or self.prefetch_cached_inputs
            if self.input_cache and depth:
                from .input_cache import prefetch_reads
                reads=prefetch_reads(intervals,
                    lambda pair: self.cached_input(row,pair[0]) if not supervised or target(row,self.stage,*pair) is not None else None,
                    depth)
            else:
                reads=((pair,None) for pair in intervals)
            try:
                yield from self._row_records(row,path,reads,supervised)
            finally:
                reads.close()
        if skipping:raise ValueError('resume cursor source not in population')

    def _row_records(self,row,path,reads,supervised):
        for (start,end),future in reads:
            self._progress_info['windows_processed']+=1
            label=target(row,self.stage,start,end)
            if supervised and label is None:
                event(self.failure_log,row,'unknown_modality_target',start=start,end=end)
                continue
            try:
                ready=future.result() if future else None
                inputs=(ready if self.prefetch_processed_inputs and ready is not None
                        else self.inputs(path,start,row,raw=ready))
            except (ValueError,OSError) as error:
                event(self.failure_log,row,error,start=start,end=end)
                continue
            self._last_cursor={'video_id':row['video_id'],'start':start,'end':end,
                'windows_processed':self._progress_info['windows_processed']}
            yield row,start,end,inputs,label

    def __call__(self,split):
        for row,start,end,inputs,label in self.records(split,supervised=True):
            yield inputs,torch.tensor([label],dtype=torch.long)

    def iter_from(self,split,cursor):
        if cursor is None:raise ValueError('prepared training requires its saved source/window cursor')
        for row,start,end,inputs,label in self.records(split,supervised=True,resume_after=cursor):
            yield inputs,torch.tensor([label],dtype=torch.long)

    def cursor(self):return self._last_cursor

    def progress(self,split):return dict(self._progress_info)


def stage_model(stage,profile,assets):
    backend,proof=load_backend(assets,stage)
    if stage=='visual':
        joint=proof['architecture']=='transformers-vivit-joint-adaptation-v1'
        if joint!=(profile['profile_id']=='avtenet-hf-vivit-joint-adaptation-v1'):
            raise ValueError('joint/factorized backbone must match the explicit component profile')
    model=AudioVisualExpert(backend,profile) if stage=='audiovisual' else VectorExpert(backend,768)
    return model,backend,proof


def base_identity(population,profile,stage,seed):
    return {'model':'avtenet','stage':stage,'profile':profile,'preparation':PREPARATION,'seed':seed,
            'index_sha256':population.metadata['index_sha256'],'population_sha256':population.metadata['population_sha256'],
            'source_overlap_policy':population.metadata['source_overlap_policy'],'population_audit':population.metadata['population_audit']}


def stage_identity(population,profile,stage,seed,proof,detector_provenance,assets):
    return {**base_identity(population,profile,stage,seed),'pretrained':proof,'detector':detector_provenance,
            'avhubert_preparation':assets['experts']['audiovisual']['preparation'] if stage=='audiovisual' else None}


def open_cache(directory):
    directory=_fk_paths.relocate((Path(directory)), __file__);metadata=read(_fk_paths.relocate((directory/'cache.json'), __file__))
    if digest(_fk_paths.relocate((directory/'features.sqlite'), __file__))!=metadata['payload_sha256']:
        raise ValueError('feature cache payload checksum mismatch')
    uri=(_fk_paths.relocate((directory/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro'
    return metadata,sqlite3.connect(uri,uri=True)


def check_cache(metadata,population,profile,*,fusion=True):
    if metadata['schema']!=('avtenet-fusion-cache-v1' if fusion else 'avtenet-expert-cache-v1'):
        raise ValueError('cache schema mismatch')
    expected={'index_sha256':population.metadata['index_sha256'],'profile_sha256':fingerprint(profile),
              'preparation_sha256':fingerprint(PREPARATION)}
    if any(metadata.get(k)!=v for k,v in expected.items()):
        raise ValueError('cache population/profile/preprocessing mismatch')


class FusionLoader:
    def __init__(self,population,directory,profile):
        self.population=population
        self.metadata,self.connection=open_cache(directory)
        check_cache(self.metadata,population,profile)

    def close(self):
        self.connection.close()

    def records(self,split):
        for row in self.population.rows(split):
            for start,end,payload,source_hash in self.connection.execute(
                    'SELECT start,end,features,source_sha256 FROM features WHERE video_id=? ORDER BY start',(row['video_id'],)):
                if source_hash!=row['source_sha256']:raise ValueError('cached source mismatch')
                array=np.frombuffer(payload,dtype='<f4').copy().reshape(3,768)
                if not np.isfinite(array).all():raise ValueError('nonfinite cache features')
                values={stage:torch.from_numpy(array[i:i+1]) for i,stage in enumerate(('visual','audio','audiovisual'))}
                yield row,start,end,values,target(row,'audiovisual',start,end)

    def __call__(self,split):
        for row,start,end,values,label in self.records(split):
            if label is not None:yield values,torch.tensor([label],dtype=torch.long)


def execution_options(args):
    reuse=getattr(args,'reuse_video_decoder',False)
    prefetch=getattr(args,'prefetch_cached_inputs',0)
    detections=getattr(args,'cache_face_detections',False)
    processor=getattr(args,'fast_vivit_processor',False)
    processed=getattr(args,'prefetch_processed_inputs',0)
    if not any((reuse,prefetch,detections,processor,processed)):return None
    return {'version':'equivalent-input-io-v3','reuse_video_decoder':reuse,
            'prefetch_cached_inputs':prefetch,'cache_face_detections':detections,
            'fast_vivit_processor':processor,'prefetch_processed_inputs':processed,
            'precision':'unchanged','sample_order':'unchanged'}


def train(args,population):
    profile=load_component_profile(args.config or CONFIG)
    settings=read(args.schedule or SCHEDULE)
    stage=args.stage
    seed_all(args.seed)
    if stage=='fusion':
        if not args.feature_cache:raise ValueError('fusion needs --feature-cache from merge-features')
        model=FeatureFusion(profile)
        batches=FusionLoader(population,args.feature_cache,profile)
        identity={**base_identity(population,profile,stage,args.seed),'feature_cache':batches.metadata}
    else:
        assets=registry(args.assets)
        model,backend,proof=stage_model(stage,profile,assets)
        detect,provenance=detector(args.device)
        identity=stage_identity(population,profile,stage,args.seed,proof,provenance,assets)
        cache=None
        if getattr(args,'prepared_cache',None):
            from .input_cache import InputCache
            cache=InputCache(args.prepared_cache,identity)
        batches=Prepared(population,stage,backend,assets,detect,_fk_paths.relocate((Path(args.output)/'preparation-events.jsonl'), __file__),cache,
            reuse_video_decoder=getattr(args,'reuse_video_decoder',False),
            prefetch_cached_inputs=getattr(args,'prefetch_cached_inputs',0),
            cache_face_detections=getattr(args,'cache_face_detections',False),
            fast_vivit_processor=getattr(args,'fast_vivit_processor',False))
    schedule={**settings,'optimizer':'Adam','learning_rate':profile['training']['learning_rates'][stage],
              'selection':'loss'}
    try:
        return fit(model,batches,args.output,identity=identity,schedule=schedule,loss_fn=native_binary_loss,
                   fake_predictions=lambda logits:(logits<0).long(),device=args.device,
                   epochs=args.epochs,resume=args.resume,max_steps=args.max_steps,
                   execution=execution_options(args),
                   checkpoint_seconds=getattr(args,'checkpoint_seconds',0),
                   progress_seconds=getattr(args,'progress_seconds',10),
                   pause_file=getattr(args,'pause_file',None))
    finally:
        if isinstance(batches,FusionLoader):batches.close()


@torch.inference_mode()
def cache_features(args,population):
    profile=load_component_profile(args.config or CONFIG)
    assets=registry(args.assets)
    model,backend,proof=stage_model(args.stage,profile,assets)
    detect,provenance=detector(args.device)
    selected=load_checkpoint(args.checkpoint)
    identity=stage_identity(population,profile,args.stage,selected['identity']['seed'],proof,provenance,assets)
    if any(selected['identity'].get(k)!=v for k,v in identity.items()):
        raise ValueError('selected expert checkpoint identity mismatch')
    resume=getattr(args,'resume_feature_cache',None)
    resume_metadata=None;handled=set();preserved=set()
    if resume:
        from .feature_resume import validate_snapshot
        expected={'stage':args.stage,'index_sha256':population.metadata['index_sha256'],
                  'profile_sha256':fingerprint(profile),'preparation_sha256':fingerprint(PREPARATION),
                  'checkpoint_sha256':digest(args.checkpoint),'identity':selected['identity']}
        resume_metadata,preserved,failures=validate_snapshot(resume,expected,population)
        handled=preserved|failures
    model.load_state_dict(selected['model_state'],strict=True);model.to(args.device).eval();model.requires_grad_(False)
    output=_fk_paths.relocate((Path(args.output)), __file__);output.mkdir(parents=True,exist_ok=False)
    write(_fk_paths.relocate((output/'producer.json'), __file__),{'schema':'avtenet-owned-feature-producer-v1','stage':args.stage,
        'index_sha256':population.metadata['index_sha256'],'profile_sha256':fingerprint(profile),
        'preparation_sha256':fingerprint(PREPARATION),'checkpoint_sha256':digest(args.checkpoint),
        'identity':selected['identity']})
    (_fk_paths.relocate((output/'preparation-events.jsonl'), __file__)).touch()
    if execution_options(args):write(_fk_paths.relocate((output/'execution-options.json'), __file__),execution_options(args))
    cache=None
    if getattr(args,'prepared_cache',None):
        from .input_cache import InputCache
        cache=InputCache(args.prepared_cache,identity)
    prepared=Prepared(population,args.stage,backend,assets,detect,_fk_paths.relocate((output/'preparation-events.jsonl'), __file__),cache,
        reuse_video_decoder=getattr(args,'reuse_video_decoder',False),
        prefetch_cached_inputs=getattr(args,'prefetch_cached_inputs',0),
        cache_face_detections=getattr(args,'cache_face_detections',False),
        fast_vivit_processor=getattr(args,'fast_vivit_processor',False),
        prefetch_processed_inputs=getattr(args,'prefetch_processed_inputs',0))
    count=len(preserved)
    if resume:
        import shutil
        shutil.copy2(_fk_paths.relocate((Path(resume)/'features.sqlite'), __file__),_fk_paths.relocate((output/'features.sqlite'), __file__))
        shutil.copy2(_fk_paths.relocate((Path(resume)/'preparation-events.jsonl'), __file__),_fk_paths.relocate((output/'preparation-events.jsonl'), __file__))
        write(_fk_paths.relocate((output/'resume-provenance.json'), __file__),{'snapshot_sha256':digest(_fk_paths.relocate((Path(resume)/'resume.json'), __file__)),
            'preserved_windows':count,'preserved_failures':len(handled)-count,'source':str(_fk_paths.relocate((Path(resume)), __file__).resolve())})
        # Fail closed if selected weights + equivalent input path cannot reproduce
        # saved features on this runtime. No existing feature payload is rewritten.
        rows={r['video_id']:r for split in ('train','validation','test') for r in population.rows(split)}
        with closing(sqlite3.connect(_fk_paths.relocate((output/'features.sqlite'), __file__))) as probe:
            for vid,start,blob in probe.execute('SELECT video_id,start,features FROM features ORDER BY video_id,start LIMIT 3'):
                row=rows[vid]
                inputs=prepared.inputs(population.path(row),start,row)
                actual=model.extract_features(move(inputs,args.device)).cpu().numpy().astype('<f4')
                expected=np.frombuffer(blob,dtype='<f4').reshape(1,768)
                if not np.array_equal(actual,expected):raise ValueError('resume feature parity failed; original cache preserved')
        print('Resume feature parity PASSED; preserved',count,'windows and',len(handled)-count,'failure decisions',flush=True)
    started=time.monotonic();initial_count=count;last_progress=0.
    with closing(sqlite3.connect(_fk_paths.relocate((output/'features.sqlite'), __file__))) as conn, conn:
        if not resume:conn.execute('CREATE TABLE features (video_id TEXT, start REAL, end REAL, split TEXT, source_sha256 TEXT, features BLOB, PRIMARY KEY(video_id,start,end))')
        for split in ('train','validation','test'):
            for row,start,end,inputs,label in prepared.records(split,skip_windows=handled):
                features=model.extract_features(move(inputs,args.device)).cpu().numpy().astype('<f4')
                if features.shape!=(1,768) or not np.isfinite(features).all():raise ValueError('invalid expert features')
                conn.execute('INSERT INTO features VALUES (?,?,?,?,?,?)',(row['video_id'],start,end,split,row['source_sha256'],features.tobytes()))
                count+=1
                pause_file=getattr(args,'pause_file',None)
                paused=pause_file and _fk_paths.relocate((Path(pause_file)), __file__).exists()
                if paused or count%100==0 or time.monotonic()-last_progress>=getattr(args,'progress_seconds',10):
                    conn.commit()
                    elapsed=time.monotonic()-started
                    detail=prepared.progress(split) if hasattr(prepared,'progress') else {}
                    write(_fk_paths.relocate((output/'progress.json'), __file__),{'status':'PAUSED' if paused else 'RUNNING','split':split,
                        'cached_windows':count,'new_windows':count-initial_count,'elapsed_seconds':elapsed,
                        'windows_per_second':(count-initial_count)/elapsed if elapsed else None,
                        'updated_unix':time.time(),**detail},replace=True)
                    print('cached',count,'new_windows',count-initial_count,'elapsed_seconds',round(elapsed,2),flush=True)
                    last_progress=time.monotonic()
                if paused:raise TrainingPaused('Feature extraction paused after committing database')
    metadata={'schema':'avtenet-expert-cache-v1','stage':args.stage,'windows':count,
              'index_sha256':population.metadata['index_sha256'],'profile_sha256':fingerprint(profile),
              'preparation_sha256':fingerprint(PREPARATION),'checkpoint_sha256':digest(args.checkpoint),
              'identity':selected['identity'],'payload_sha256':digest(_fk_paths.relocate((output/'features.sqlite'), __file__)),
              'integration_only':selected['identity']['max_steps'] is not None}
    write(_fk_paths.relocate((output/'cache.json'), __file__),metadata)
    write(_fk_paths.relocate((output/'progress.json'), __file__),{'status':'FINISHED','cached_windows':count,'updated_unix':time.time()},replace=True)
    return metadata


def merge_features(args):
    opened={name:open_cache(getattr(args,name)) for name in ('visual','audio','audiovisual')}
    metas={name:item[0] for name,item in opened.items()}
    for name,meta in metas.items():
        if meta['stage']!=name or meta['schema']!='avtenet-expert-cache-v1':raise ValueError('wrong expert cache')
    for key in ('index_sha256','profile_sha256','preparation_sha256'):
        if len({m[key] for m in metas.values()})!=1:raise ValueError('expert caches disagree: '+key)
    output=_fk_paths.relocate((Path(args.output)), __file__);output.mkdir(parents=True,exist_ok=False)
    count=0;missing=0
    with closing(sqlite3.connect(_fk_paths.relocate((output/'features.sqlite'), __file__))) as out, out:
        out.execute('CREATE TABLE features (video_id TEXT, start REAL, end REAL, split TEXT, source_sha256 TEXT, features BLOB, PRIMARY KEY(video_id,start,end))')
        for vid,start,end,split,sha,vector in opened['visual'][1].execute('SELECT * FROM features ORDER BY video_id,start'):
            vectors=[vector]
            for name in ('audio','audiovisual'):
                item=opened[name][1].execute('SELECT split,source_sha256,features FROM features WHERE video_id=? AND start=? AND end=?',(vid,start,end)).fetchone()
                if item is None:break
                if item[:2]!=(split,sha):raise ValueError('expert source/split mismatch')
                vectors.append(item[2])
            if len(vectors)!=3:missing+=1;continue
            if any(len(v)!=768*4 for v in vectors):raise ValueError('incorrect feature dimension')
            out.execute('INSERT INTO features VALUES (?,?,?,?,?,?)',(vid,start,end,split,sha,b''.join(vectors)))
            count+=1
            if count%10000==0:out.commit();print('merged',count,flush=True)
    for meta,conn in opened.values():conn.close()
    metadata={'schema':'avtenet-fusion-cache-v1',**{k:metas['visual'][k] for k in ('index_sha256','profile_sha256','preparation_sha256')},
              'expert_caches':metas,'windows':count,'visual_windows_missing_other_experts':missing,
              'payload_sha256':digest(_fk_paths.relocate((output/'features.sqlite'), __file__)),'integration_only':any(m['integration_only'] for m in metas.values())}
    write(_fk_paths.relocate((output/'cache.json'), __file__),metadata)
    return metadata


@torch.inference_mode()
def evaluate(args,population):
    profile=load_component_profile(args.config or CONFIG)
    with closing(FusionLoader(population,args.feature_cache,profile)) as loader:
        return _evaluate_loaded(args,population,profile,loader)


def _evaluate_loaded(args,population,profile,loader):
    checkpoint=load_checkpoint(args.checkpoint)
    expected={**base_identity(population,profile,'fusion',checkpoint['identity']['seed']),'feature_cache':loader.metadata}
    if any(checkpoint['identity'].get(k)!=v for k,v in expected.items()):raise ValueError('fusion checkpoint/cache/config mismatch')
    model=FeatureFusion(profile).to(args.device)
    model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    output=_fk_paths.relocate((Path(args.output)), __file__);output.mkdir(parents=True,exist_ok=False)
    results=[]
    with (_fk_paths.relocate((output/'predictions.jsonl'), __file__)).open('x',encoding='utf-8') as stream:
        for row in population.rows(args.split):
            scores=[]
            for start,end,blob,sha in loader.connection.execute('SELECT start,end,features,source_sha256 FROM features WHERE video_id=? ORDER BY start',(row['video_id'],)):
                if sha!=row['source_sha256']:raise ValueError('cached source mismatch')
                array=np.frombuffer(blob,dtype='<f4').copy().reshape(3,768)
                inputs={stage:torch.from_numpy(array[i:i+1]).to(args.device) for i,stage in enumerate(('visual','audio','audiovisual'))}
                scores.append(float((-model(inputs)).sigmoid().item()))
            score=sum(scores)/len(scores) if scores else None
            result={'video_id':row['video_id'],'split':args.split,'label':row['label'],'score':score,
                    'prediction':int(score>=.5) if score is not None else None,
                    'status':'ASSESSED' if scores else 'UNASSESSED','usable_windows':len(scores),
                    'expected_windows':len(windows(row['duration'])),'reason':None if scores else 'no_complete_three_expert_window'}
            results.append(result);stream.write(json.dumps(result,allow_nan=False)+'\n')
    method=('AVTENet-adaptation-HF-joint-ViViT' if profile['profile_id']=='avtenet-hf-vivit-joint-adaptation-v1'
            else 'AVTENet-FF-AVHuBERT-paper-guided-reconstruction')
    report={'model':method,'split':args.split,'metrics':metrics(results),
            'population':population.metadata,'checkpoint_sha256':digest(args.checkpoint),'cache':loader.metadata,
            'integration_only':checkpoint['identity']['max_steps'] is not None or loader.metadata['integration_only']}
    if args.compare_with:
        report['comparison']=compare(results,read_predictions(args.compare_with))
    write(_fk_paths.relocate((output/'report.json'), __file__),report)
    return report


def dispatch(args,population):
    return {'train':train,'cache-features':cache_features,'evaluate':evaluate}[args.command](args,population) if args.command!='merge-features' else merge_features(args)
