"""Frozen AVTENet raw studies; AV-HuBERT uses its original Linux runtime."""

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

import argparse,sys,json,time,statistics
from pathlib import Path


def retained_vectors(connection,row,intervals):
    """Original clean expert features on exactly the retained source/windows."""
    import numpy as np
    vectors=[];seen=set();allowed=set(intervals)
    for start,end,sha,blob in connection.execute('SELECT start,end,source_sha256,features FROM features WHERE video_id=? ORDER BY start',(row['video_id'],)):
        if sha!=row['source_sha256'] or (start,end) not in allowed or (start,end) in seen:raise ValueError('Retained clean feature provenance differs')
        array=np.frombuffer(blob,dtype='<f4')
        if array.shape!=(768,) or not np.isfinite(array).all():raise ValueError('Invalid retained clean expert vector')
        seen.add((start,end));vectors.append({'start':start,'end':end,'vector':array.tolist()})
    return vectors


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--study',required=True)
    p.add_argument('--input-root',required=True);p.add_argument('--stage',choices=('audio','visual','audiovisual','fusion'),required=True)
    p.add_argument('--assets',required=True);args=p.parse_args()
    root=_fk_paths.relocate((Path(args.root)), __file__);study=_fk_paths.relocate((Path(args.study)), __file__);media=_fk_paths.relocate((Path(args.input_root)), __file__)
    sys.path[:0]=[str(root),str(_fk_paths.relocate((root/'src'), __file__)),str(_fk_paths.relocate((root/'scripts'), __file__))]
    import torch
    import numpy as np
    from baselines.common.io import read,write,digest,fingerprint
    from baselines.common.training import load_checkpoint,move
    from baselines.common.population import windows
    from baselines.common.metrics import metrics
    from baselines.avtenet.model import load_component_profile,FeatureFusion
    from run_avtenet_frozen_test import validate_plan,check_complete
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    plan=read(_fk_paths.relocate((root/'runs/auto-next-001/test-plan.json'), __file__));validate_plan(plan,root)
    config=load_component_profile(_fk_paths.relocate((root/plan['config']), __file__));inputs=read(_fk_paths.relocate((study/'avtenet-inputs.json'), __file__))
    output=_fk_paths.relocate((study/'avtenet'), __file__);output.mkdir(exist_ok=True);dest=_fk_paths.relocate((output/args.stage), __file__);dest.mkdir(exist_ok=True)
    def progress(done,total,**extra):
        write(_fk_paths.relocate((study/'progress.json'), __file__),{'status':'RUNNING','stage':'avtenet_'+args.stage,'processed':done,'total':total,'updated_unix':time.time(),**extra},replace=True)
    selected=load_checkpoint(_fk_paths.relocate((root/plan['checkpoints'][args.stage]), __file__))
    context={'checkpoint_sha256':digest(_fk_paths.relocate((root/plan['checkpoints'][args.stage]), __file__)),'inputs_sha256':digest(_fk_paths.relocate((study/'avtenet-inputs.json'), __file__)),
        'script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'stage':args.stage,'threshold':.5,'plan_sha256':digest(_fk_paths.relocate((root/'runs/auto-next-001/test-plan.json'), __file__)),
        'stream_verifier_sha256':digest(_fk_paths.relocate((media/'Deepfake-Detection/baselines/common/stream_identity.py'), __file__))}
    if (_fk_paths.relocate((dest/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((dest/'identity.json'), __file__))!=context:raise ValueError('AVTENet resume inputs changed')
    write(_fk_paths.relocate((dest/'identity.json'), __file__),context,replace=True)
    if args.stage=='fusion':
        for stage in ('audio','visual','audiovisual'):
            complete=read(_fk_paths.relocate((output/stage/'complete.json'), __file__))
            if complete['inputs_sha256']!=context['inputs_sha256']:raise ValueError('Incomplete or wrong expert population')
            for name,sha in complete['files'].items():
                if digest(_fk_paths.relocate((output/stage/name), __file__))!=sha:raise ValueError('Expert output changed')
        model=FeatureFusion(config).cuda();model.load_state_dict(selected['model_state'],strict=True);model.eval()
        groups={};profiles=[]
        with torch.inference_mode():
            for i,item in enumerate(inputs):
                row=item['row'];key=fingerprint(item);progress(i,len(inputs))
                experts={stage:read(_fk_paths.relocate((output/stage/(key+'.json')), __file__)) for stage in ('audio','visual','audiovisual')}
                samples=[]
                for rep in range(4 if item['purpose']=='profiling' else 1):
                    lookup={stage:{(v['start'],v['end']):v['vector'] for v in value['samples'][rep]['vectors']} for stage,value in experts.items()}
                    torch.cuda.synchronize();t=time.perf_counter();scores=[]
                    for pair in windows(row['duration']):
                        if all(pair in stream for stream in lookup.values()):
                            tensors={stage:torch.tensor(stream[pair],dtype=torch.float32,device='cuda').reshape(1,768) for stage,stream in lookup.items()}
                            scores.append(float((-model(tensors)).sigmoid().item()))
                    torch.cuda.synchronize();fusion_seconds=time.perf_counter()-t
                    score=sum(scores)/len(scores) if scores else None
                    result={'video_id':row['video_id'],'label':row['label'],'status':'ASSESSED' if scores else 'UNASSESSED',
                        'score':score,'prediction':None if score is None else int(score>=.5),'usable_windows':len(scores)}
                    times=[v['samples'][rep]['seconds'] for v in experts.values()]
                    samples.append({'prediction':result,'fusion_seconds':fusion_seconds,
                        'staged_pipeline_seconds':None if any(v is None for v in times) else fusion_seconds+sum(times),
                        'clean_expert_cache_reused':any(v is None for v in times)})
                write(_fk_paths.relocate((dest/(key+'.json')), __file__),{'item':item,'samples':samples},replace=True)
                if item['purpose']=='robustness':groups.setdefault(row['condition'],[]).append(samples[0]['prediction'])
                else:profiles.append({'video_id':row['video_id'],'samples':samples,'median_staged_seconds':statistics.median(s['staged_pipeline_seconds'] for s in samples[1:])})
        write(_fk_paths.relocate((output/'summary.json'), __file__),{'model':'AVTENet HF joint ViViT adaptation','threshold':.5,'robustness':{k:metrics(v) for k,v in groups.items()},
            'profiling':profiles,'profiling_boundary':'Sum of separately measured raw expert pipelines plus fusion; model load excluded; not observed monolithic wall time. Windows AST/ViViT, Linux AV-HuBERT. Not OS cold-cache timing.'},replace=True)
    else:
        from baselines.avtenet.assets import registry
        from baselines.avtenet.pipeline import Prepared,stage_model
        from baselines.common.media import detector
        assets=registry(args.assets);model,backend,proof=stage_model(args.stage,config,assets)
        model.load_state_dict(selected['model_state'],strict=True);del selected
        model.cuda().eval().requires_grad_(False);detect,provenance=detector('cuda')
        import sqlite3
        retained=_fk_paths.relocate((root/plan['output']/args.stage), __file__)
        expected={'schema':'avtenet-frozen-heldout-v1','plan_sha256':digest(_fk_paths.relocate((root/'runs/auto-next-001/test-plan.json'), __file__)),'stage':args.stage}
        if not check_complete(retained,expected):raise ValueError('Original clean expert cache not verified complete')
        original={r['video_id']:r for r in read(_fk_paths.relocate((root/plan['manifest']), __file__))}
        clean_connection=sqlite3.connect((_fk_paths.relocate((retained/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro',uri=True)
        original_streams={}
        class Single:
            def __init__(self,row):self.row=row;self.root=media
            def rows(self,split):return [self.row]
            def path(self,row):
                path=(_fk_paths.relocate((media/row['media_path']), __file__)).resolve();path.relative_to(media.resolve())
                if digest(path)!=row['source_sha256']:raise ValueError('Study media changed')
                return path
        for i,item in enumerate(inputs):
            progress(i,len(inputs));key=fingerprint(item);target=_fk_paths.relocate((dest/(key+'.json')), __file__);row=item['row']
            if target.exists():
                if read(target)['item']!=item:raise ValueError('Retained expert result differs')
                continue
            if item['purpose']=='robustness' and row.get('condition')=='clean':
                old=original[row['video_id']]
                if row['source_sha256']!=old['source_sha256'] or row['label']!=old['label']:raise ValueError('Clean robustness source differs from locked heldout')
                Single(row).path(row)
                vectors=retained_vectors(clean_connection,row,windows(row['duration']))
                write(target,{'item':item,'samples':[{'repeat':0,'vectors':vectors,'seconds':None,'model_seconds':None,
                    'execution':'reused_verified_clean_expert_features','source_payload_sha256':read(_fk_paths.relocate((retained/'complete.json'), __file__))['files']['features.sqlite']}]})
                continue
            condition=row.get('condition','')
            reuse_kind=('video' if args.stage=='visual' and condition.startswith(('audio_compression-','av_offset-'))
                else 'audio' if args.stage=='audio' and condition.startswith('video_compression-') else None)
            if item['purpose']=='robustness' and reuse_kind:
                # The AV-HuBERT runtime uses its retained code root. Load only
                # this new read-only execution helper from the study workspace.
                import importlib.util
                spec=importlib.util.spec_from_file_location('remaining_stream_identity',_fk_paths.relocate((media/'Deepfake-Detection/baselines/common/stream_identity.py'), __file__))
                verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
                stream_identity=verifier.stream_identity
                old=original[row['video_id']]
                if row.get('original_source_sha256')!=old['source_sha256'] or row['duration']!=old['duration'] or row['label']!=old['label']:raise ValueError('Derivative original source differs')
                original_path=_fk_paths.relocate((media/'Heldout-1000/dataset'/old['path']), __file__)
                cache_key=(old['video_id'],reuse_kind)
                if cache_key not in original_streams:
                    if digest(original_path)!=old['source_sha256']:raise ValueError('Original source changed')
                    original_streams[cache_key]=stream_identity(original_path,reuse_kind)
                current=stream_identity(Single(row).path(row),reuse_kind)
                if current is not None and current==original_streams[cache_key]:
                    vectors=retained_vectors(clean_connection,old,windows(row['duration']))
                    write(target,{'item':item,'samples':[{'repeat':0,'vectors':vectors,'seconds':None,'model_seconds':None,
                        'execution':'reused_identical_modality_stream','stream_identity':current,
                        'source_payload_sha256':read(_fk_paths.relocate((retained/'complete.json'), __file__))['files']['features.sqlite']}]})
                    continue
            samples=[]
            for rep in range(4 if item['purpose']=='profiling' else 1):
                prepared=Prepared(Single(row),args.stage,backend,assets,detect,_fk_paths.relocate((dest/'failures.jsonl'), __file__),None,
                    reuse_video_decoder=True,cache_face_detections=True,fast_vivit_processor=args.stage=='visual',
                    prefetch_video_decode=args.stage in ('visual','audiovisual'))
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();model_seconds=0.;vectors=[]
                with torch.inference_mode():
                    for _,start,end,value,_ in prepared.records('test'):
                        torch.cuda.synchronize();tmodel=time.perf_counter()
                        vector=model.extract_features(move(value,'cuda')).cpu().numpy().astype('<f4')
                        torch.cuda.synchronize();model_seconds+=time.perf_counter()-tmodel
                        if vector.shape!=(1,768) or not np.isfinite(vector).all():raise ValueError('Invalid expert feature')
                        vectors.append({'start':start,'end':end,'vector':vector.reshape(-1).tolist()})
                torch.cuda.synchronize();seconds=time.perf_counter()-t
                try:
                    import psutil
                    rss=psutil.Process().memory_info().rss
                except ImportError:rss=None
                samples.append({'repeat':rep,'vectors':vectors,'seconds':seconds,'model_seconds':model_seconds,
                    'preparation_and_other_seconds':seconds-model_seconds,'process_rss_after':rss,
                    'cuda_peak_allocated':torch.cuda.max_memory_allocated(),'cuda_peak_reserved':torch.cuda.max_memory_reserved()})
            write(target,{'item':item,'samples':samples})
        clean_connection.close()
    write(_fk_paths.relocate((dest/'complete.json'), __file__),{'inputs_sha256':context['inputs_sha256'],'files':{p.name:digest(p) for p in dest.glob('*.json') if p.name!='complete.json'}},replace=True)


if __name__=='__main__':main()
