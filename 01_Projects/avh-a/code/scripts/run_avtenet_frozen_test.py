"""Inference only on the frozen local held-out set; never selects or trains weights."""

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

import argparse,json,sqlite3,sys,time
from pathlib import Path
from contextlib import closing
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest


def validate_plan(plan,root=ROOT):
    required={plan['manifest'],plan['training_manifest'],plan['config'],plan['training_fusion_cache'],
              *plan['checkpoints'].values(),plan['index']+'/index.json',plan['index']+'/index.sqlite',
              plan['training_index']+'/index.json',plan['training_index']+'/index.sqlite'}
    if not required<=set(plan['files']):raise ValueError('Frozen plan has unbound inputs')
    for relative,expected in plan['files'].items():
        path=(root/relative).resolve();path.relative_to(root.resolve())
        if digest(path)!=expected:raise ValueError('Frozen evaluation input changed: '+relative)
    heldout=read(root/plan['manifest']);used=read(root/plan['training_manifest'])
    if len(heldout)!=1000 or any(r['split']!='test' for r in heldout):raise ValueError('Expected held-out 1000')
    for key in ('video_id','source_family','source_sha256'):
        values=[r[key] for r in heldout]
        if len(set(values))!=len(values) or set(values)&{r[key] for r in used}:raise ValueError('Held-out overlap: '+key)
    return heldout


def check_complete(directory,context):
    if not (_fk_paths.relocate((directory/'complete.json'), __file__)).exists():return False
    done=read(_fk_paths.relocate((directory/'complete.json'), __file__))
    if done['context']!=context:raise ValueError('Frozen cache context mismatch')
    for name,sha in done['files'].items():
        if digest(directory/name)!=sha:raise ValueError('Frozen cache payload changed')
    return True


def finish(directory,context,names):
    write(_fk_paths.relocate((directory/'complete.json'), __file__),{'context':context,'files':{n:digest(directory/n) for n in names}},replace=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',required=True);p.add_argument('--stage',choices=('audio','visual','audiovisual','evaluate'),required=True)
    p.add_argument('--dataset-root',required=True);p.add_argument('--assets');p.add_argument('--device',default='cuda')
    p.add_argument('--prefetch-video-decode',action='store_true',
                   help='Opt in to one CPU decode worker; validate throughput before deployment')
    args=p.parse_args();plan=read(args.plan);rows=validate_plan(plan,ROOT)
    import numpy as np
    import torch
    threads=4 if args.stage=='visual' else 8
    torch.set_num_threads(threads)
    from baselines.common.training import load_checkpoint,move
    from baselines.common.population import Population,windows
    from baselines.avtenet.pipeline import Prepared,stage_model,stage_identity,PREPARATION
    from baselines.avtenet.assets import registry
    from baselines.avtenet.model import load_component_profile,FeatureFusion
    from baselines.common.media import detector
    profile=load_component_profile(_fk_paths.relocate((ROOT/plan['config']), __file__))
    output=_fk_paths.relocate((ROOT/plan['output']), __file__);output.mkdir(parents=True,exist_ok=True)
    context={'schema':'avtenet-frozen-heldout-v1','plan_sha256':digest(args.plan),'stage':args.stage}
    dest=_fk_paths.relocate((output/args.stage), __file__);dest.mkdir(exist_ok=True)
    if check_complete(dest,context):print('Already verified complete',args.stage,flush=True);return
    if (_fk_paths.relocate((dest/'context.json'), __file__)).exists():
        if read(_fk_paths.relocate((dest/'context.json'), __file__))!=context:raise ValueError('Incomplete output belongs to a different evaluation')
    else:write(_fk_paths.relocate((dest/'context.json'), __file__),context)
    population=Population(_fk_paths.relocate((ROOT/plan['index']), __file__),args.dataset_root,'avtenet')
    if args.stage!='evaluate':
        assets=registry(args.assets);model,backend,proof=stage_model(args.stage,profile,assets)
        detect,provenance=detector(args.device)
        selected=load_checkpoint(_fk_paths.relocate((ROOT/plan['checkpoints'][args.stage]), __file__))
        training=Population(_fk_paths.relocate((ROOT/plan['training_index']), __file__),args.dataset_root,'avtenet')
        expected=stage_identity(training,profile,args.stage,selected['identity']['seed'],proof,provenance,assets)
        if any(selected['identity'].get(k)!=v for k,v in expected.items()) or selected['identity']['max_steps'] is not None:
            raise ValueError('Expert checkpoint does not match frozen training provenance')
        model.load_state_dict(selected['model_state'],strict=True);del selected
        model.to(args.device).eval().requires_grad_(False)
        from baselines.avtenet.heldout_preparation import FastRawCache,frozen_inputs,audio_cache_identity
        raw_identity=audio_cache_identity(_fk_paths.relocate((ROOT/plan['manifest']), __file__)) if args.stage=='audio' else context
        prepared=frozen_inputs(population,args.stage,backend,assets,detect,
            _fk_paths.relocate((dest/'preparation-events.jsonl'), __file__),FastRawCache(_fk_paths.relocate((output/'prepared-inputs'), __file__),raw_identity),
            prefetch_video_decode=args.prefetch_video_decode)
        write(_fk_paths.relocate((dest/'execution.json'), __file__),{'raw_cache':'uncompressed_npz_lossless','prefetch_depth':4,
            'cpu_audio_miss_preparation':args.stage=='audio','fast_vivit_tensor_packing':args.stage=='visual',
            'precision':'unchanged','batch_size':1,'torch_cpu_threads':threads,
            'prefetch_video_decode':args.prefetch_video_decode},replace=True)
        started=time.monotonic();last=0.
        with closing(sqlite3.connect(_fk_paths.relocate((dest/'features.sqlite'), __file__))) as conn,conn:
            conn.execute('CREATE TABLE IF NOT EXISTS features(video_id TEXT,start REAL,end REAL,split TEXT,source_sha256 TEXT,features BLOB,PRIMARY KEY(video_id,start,end))')
            lookup={r['video_id']:r for r in rows};done=set()
            for vid,start,end,split,sha,blob in conn.execute('SELECT * FROM features'):
                row=lookup.get(vid)
                if row is None or sha!=row['source_sha256'] or split!='test' or (start,end) not in windows(row['duration']) or len(blob)!=3072 or not np.isfinite(np.frombuffer(blob,dtype='<f4')).all():
                    raise ValueError('Invalid partial held-out feature row')
                done.add((vid,start,end))
            count=len(done)
            with torch.inference_mode():
                for row,start,end,inputs,_ in prepared.records('test',skip_windows=done):
                    value=model.extract_features(move(inputs,args.device)).cpu().numpy().astype('<f4')
                    if value.shape!=(1,768) or not np.isfinite(value).all():raise ValueError('Invalid expert features')
                    conn.execute('INSERT INTO features VALUES(?,?,?,?,?,?)',(row['video_id'],start,end,'test',row['source_sha256'],value.tobytes()))
                    count+=1
                    if count%25==0 or time.monotonic()-last>=10:
                        conn.commit();last=time.monotonic()
                        detail=prepared.progress('test') if hasattr(prepared,'progress') else {}
                        processed=detail.get('windows_processed',0)-detail.get('initial_windows',0)
                        rate=processed/(last-started) if last>started else 0
                        remaining=detail.get('windows_total',0)-detail.get('windows_processed',0)
                        write(_fk_paths.relocate((dest/'progress.json'), __file__),{'stage':args.stage,'windows':count,
                            'elapsed_seconds':last-started,'updated_unix':time.time(),
                            'processed_windows_per_second':rate,
                            'eta_seconds':remaining/rate if rate else None,**detail},replace=True)
                        print('heldout',args.stage,'cached_windows',count,flush=True)
                    if (_fk_paths.relocate((ROOT/'runs/worker-c/avtenet/pause.request'), __file__)).exists():
                        conn.commit();raise RuntimeError('PAUSED by pause.request; completed rows retained')
        finish(dest,context,['features.sqlite'])
        return
    # Evaluate only after all expert caches are complete and tied to this plan.
    stores={}
    for stage in ('visual','audio','audiovisual'):
        if not check_complete(_fk_paths.relocate((output/stage), __file__),{**context,'stage':stage}):raise ValueError('Incomplete held-out expert: '+stage)
        stores[stage]=sqlite3.connect((_fk_paths.relocate((output/stage/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro',uri=True)
    checkpoint=load_checkpoint(_fk_paths.relocate((ROOT/plan['checkpoints']['fusion']), __file__))
    train_cache=read(_fk_paths.relocate((ROOT/plan['training_fusion_cache']), __file__))
    if checkpoint['identity']['feature_cache']!=train_cache or checkpoint['identity']['max_steps'] is not None:
        raise ValueError('Fusion checkpoint training cache mismatch')
    model=FeatureFusion(profile).to(args.device);model.load_state_dict(checkpoint['model_state'],strict=True);model.eval()
    from baselines.avtenet.heldout_preparation import aligned_vectors
    results=[]
    with torch.inference_mode():
        for row in rows:
            scores=[]
            for inputs in aligned_vectors(stores,row,windows(row['duration'])):
                scores.append(float((-model(move(inputs,args.device))).sigmoid().item()))
            score=sum(scores)/len(scores) if scores else None
            results.append({'video_id':row['video_id'],'split':'test','label':row['label'],'score':score,
                'prediction':int(score>=.5) if score is not None else None,'status':'ASSESSED' if scores else 'UNASSESSED',
                'usable_windows':len(scores),'expected_windows':len(windows(row['duration'])),
                'reason':None if scores else 'no_complete_three_expert_window'})
    for conn in stores.values():conn.close()
    from baselines.common.metrics import metrics
    result=metrics(results);c=result['confusion'];result['accuracy']=(c['tp']+c['tn'])/result['assessed'] if result['assessed'] else None
    temp=_fk_paths.relocate((dest/'predictions.tmp'), __file__);temp.write_text(''.join(json.dumps(r)+'\n' for r in results),encoding='utf-8');temp.replace(_fk_paths.relocate((dest/'predictions.jsonl'), __file__))
    write(_fk_paths.relocate((dest/'report.json'), __file__),{'model':'AVTENet-adaptation-HF-joint-ViViT','scope':'local held-out 1000 from unused validation; not official test',
        'split':'test','metrics':result,'context':context,'frozen_plan':plan,'threshold':.5,'integration_only':False},replace=True)
    finish(dest,context,['predictions.jsonl','report.json'])
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
