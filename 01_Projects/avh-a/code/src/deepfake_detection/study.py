"""Reproducible shared-cache evaluation and sequential experiment execution."""

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

from pathlib import Path
import torch
from .cli import load_json, save
from .core import coverage
from .forensics import sha256_file


@torch.inference_mode()
def evaluate_checkpoint(manifest_path, checkpoint_path, output, *, split='test', device='cpu'):
    from .infrastructure import CachedVideoPopulation, fk_device
    from .training import load_fk_checkpoint, forward_video, TENSORS
    from .reporting import scored_metrics, timeline_metrics
    if split not in ('validation', 'test'):
        raise ValueError('evaluation split must be validation or test')
    manifest = load_json(manifest_path)
    model, checkpoint = load_fk_checkpoint(checkpoint_path, device=fk_device(device))
    overlap_policy = manifest.get('source_overlap_policy','error')
    if overlap_policy not in ('error','drop-train','keep-official') or checkpoint.get('source_overlap_policy','error') != overlap_policy:
        raise ValueError('evaluation source overlap policy differs from checkpoint')
    if split == 'test' and 'validation_selection' not in checkpoint:
        raise ValueError('test evaluation requires a validation-selected checkpoint')
    for key in ('preprocessing', 'asset_sha256', 'window_label_policy'):
        if checkpoint.get(key) != manifest.get(key):
            raise ValueError('evaluation cache policy differs from checkpoint: '+key)
    population = CachedVideoPopulation(manifest[split], _fk_paths.relocate((Path(manifest_path)), __file__).resolve().parent)
    seen = {key:{} for key in ('video_id','source_family','source_sha256')}
    for partition in ('train','validation','test'):
        for entry in manifest.get(partition, []):
            if entry['split'] != partition:
                raise ValueError('evaluation manifest contains a wrong split')
            for key, owners in seen.items():
                value = entry[key]
                if value in owners and (owners[value] != partition or key=='video_id'):
                    if key == 'video_id' or overlap_policy != 'keep-official':
                        raise ValueError('duplicate video or cross-split family/hash leakage')
                owners[value] = partition
    assessments = [a for a in manifest['assessments'] if a['split']==split]
    by_id = {a['video_id']: a for a in assessments}
    if len(by_id) != len(assessments) or not assessments:
        raise ValueError('complete distinct assessment population required')
    rows, window_rows, timelines, time_coverage = [], [], [], []
    results = {}
    for row in population:
        if row['video_id'] not in by_id:
            raise ValueError('cache video missing assessment')
        if any(row[key] != by_id[row['video_id']][key] for key in ('label','source_family','split')):
            raise ValueError('assessment labels/family/split differ from cache')
        out, score, _ = forward_video(model, row)
        # Retain only small predictions/metadata, never the full feature corpus or GPU graphs.
        results[row['video_id']] = ({k:v for k,v in row.items() if k not in TENSORS},
            {k:out[k].cpu() for k in ('bin_scores','window_scores')}, score.cpu())
    for assessment in assessments:
        vid = assessment['video_id']
        result = results.get(vid)
        expected = [w['interval'] for w in assessment['windows'] if w['status']=='USABLE']
        if (result is None) != (not expected):
            raise ValueError('usable assessment/cache mismatch')
        score = None if result is None else result[2].item()
        rows.append({'video_id':vid, 'source_family':assessment['source_family'], 'label':assessment['label'],
                     'status':'UNASSESSED' if score is None else 'ASSESSED', 'score':score,
                     'prediction':None if score is None else int(score>=model.decision.video_threshold)})
        lookup = {}
        bins = []
        if result is not None:
            row, out, _ = result
            if row['window_intervals'] != expected:
                raise ValueError('assessment intervals differ from cache')
            lookup = {tuple(interval):value for interval,value in zip(expected,out['window_scores'].tolist())}
            for (start,end), scores in zip(expected,out['bin_scores'].tolist()):
                width=(end-start)/len(scores)
                bins.extend({'interval':[start+b*width,start+(b+1)*width],'score':s} for b,s in enumerate(scores))
        annotation = result[0] if result is not None else assessment
        if 'duration' in annotation:
            covered = coverage(annotation['duration'], expected)['covered_duration']
            time_coverage.append({'video_id':vid,'label':assessment['label'], 'duration':annotation['duration'], 'assessed_seconds':covered})
            if annotation.get('temporal_complete') or annotation['label']==0:
                timelines.append({'video_id':vid, **timeline_metrics(annotation['duration'],bins,annotation['manipulated_intervals'],model.decision.window_threshold)})
        for index, window in enumerate(assessment['windows']):
            label = window['labels']['combined']
            if label is None:
                continue
            value = lookup.get(tuple(window['interval']))
            window_rows.append({'video_id':f'{vid}#window-{index}', 'label':label,
                                'status':'UNASSESSED' if value is None else 'ASSESSED', 'score':value,
                                'prediction':None if value is None else int(value>=model.decision.window_threshold)})
    report={'checkpoint_sha256':sha256_file(checkpoint_path),'manifest_sha256':sha256_file(manifest_path),
            'source_overlap_policy':overlap_policy,
            'training_overlap':checkpoint.get('train_validation_overlap'),
            'split':split,'decision':model.decision.to_dict(),'videos':rows,'windows':window_rows,
            'video_metrics':scored_metrics(rows),'window_metrics':scored_metrics(window_rows) if window_rows else None,
            'window_coverage_by_label':{str(y):{'total':sum(w['label']==y for w in window_rows),
                'assessed':sum(w['label']==y and w['status']=='ASSESSED' for w in window_rows)} for y in (0,1)},
            'timeline_metrics_supplement':timelines,
            'timeline_coverage':time_coverage,
            'timeline_coverage_missing_metadata':len(assessments)-len(time_coverage),
            'unknown_window_labels':sum(w['labels']['combined'] is None for a in assessments for w in a['windows'])}
    save(output,report)
    return report


def run_experiments(plan_path, manifest, training_config, output, *, epochs=None, device='cpu'):
    from .infrastructure import train_fk_manifest
    plan=load_json(plan_path)
    root=_fk_paths.relocate((Path(output)), __file__); root.mkdir(parents=True,exist_ok=False)
    result=[]; skipped=[]
    for run in plan['runs']:
        if 'profile' not in run:
            skipped.append({'method':run['method'],'seed':run['seed'],'reason':'requires_external_prepared_input_runner'})
            continue  # External AVFakeNet/AVTENet runners use their own prepared inputs.
        name=f"{run['method']}-{run['variant']}-seed{run['seed']}"
        config=load_json(training_config); config['seed']=run['seed']
        config_path=_fk_paths.relocate((root/(name+'.training.json')), __file__); save(config_path,config)
        metrics=train_fk_manifest(manifest,_fk_paths.relocate((Path(plan_path).parent/run['profile']), __file__),config_path,
                                  _fk_paths.relocate((root/name), __file__),epochs=epochs,calibrate=True,device=device)
        result.append({'run':name,'selected_checkpoint':str(_fk_paths.relocate((root/name/'selected.pt'), __file__)),
                       'selected_epoch':metrics['selected_epoch']})
    save(_fk_paths.relocate((root/'completed.json'), __file__),{'runs':result,'skipped':skipped,'test_scoring':'NOT_RUN'})
    return {'runs':result,'skipped':skipped}
