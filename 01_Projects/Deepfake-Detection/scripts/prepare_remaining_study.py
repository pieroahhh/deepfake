"""Prepare remaining experiments and CPU-only summaries; no GPU work or training."""

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

import sys,json,math,copy
from pathlib import Path
from collections import defaultdict
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,digest,fingerprint
from avfakenet_progress_io import write_json

VARIANTS={'full':{},'no_av':{'disabled_evidence':['av']},'no_temporal':{'disabled_evidence':['temporal']},
          'uniform_weights':{'uniform_weights':True},'window_max':{},'window_mean':{},'video_max':{},'video_mean':{}}


def calibration_grid(variant,bins=10):
    grid={'top_k':[1,3,5],'peak_weight':[0.,.5,1.],'top_ratio':[.1,.2,.5],
          'video_peak_weight':[0.,.5,1.],'reliability_temperature':[.25,.5,1.]}
    if variant=='window_max':grid.update(top_k=[3],peak_weight=[1.])
    if variant=='window_mean':grid.update(top_k=[bins],peak_weight=[0.])
    if variant=='video_max':grid.update(top_ratio=[1.],video_peak_weight=[1.])
    if variant=='video_mean':grid.update(top_ratio=[1.],video_peak_weight=[0.])
    return grid


def select_balanced(rows,per_type,seed=7):
    categories=defaultdict(list)
    for row in rows:categories[row['source_modify_type']].append(row)
    if len(categories)!=4 or any(len(values)<per_type for values in categories.values()):raise ValueError('Four manipulation categories with sufficient rows required')
    selected=[]
    for category,values in sorted(categories.items()):
        selected.extend(sorted(values,key=lambda r:fingerprint([seed,r['video_id']]))[:per_type])
    return selected


def localization_summary(timelines):
    seconds=defaultdict(float)
    for row in timelines:
        values=row['seconds']
        if any(not math.isfinite(v) or v<0 for v in values.values()):raise ValueError('Invalid duration counts')
        if abs(sum(values.values())-row['duration'])>1e-6:raise ValueError('Timeline durations do not partition video')
        for key,value in values.items():seconds[key]+=value
    ratio=lambda a,b:a/b if b else None
    tp,fp,fn,tn=(seconds[k] for k in ('tp','fp','fn','tn'))
    missed=seconds['unassessed_positive'];total=sum(seconds.values());assessed=tp+fp+fn+tn
    return {'videos':len(timelines),'seconds':dict(seconds),
        'duration_weighted_assessed_region':{'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
            'f1':ratio(2*tp,2*tp+fp+fn),'iou':ratio(tp,tp+fp+fn)},
        'time_assessment_coverage':ratio(assessed,total),
        'manipulated_time_assessment_coverage':ratio(tp+fn,tp+fn+missed),
        'detected_fraction_of_all_manipulated_time':ratio(tp,tp+fn+missed),
        'scope':'Summary of existing frozen-threshold FK timeline adapter, not a new localization model',
        'consolidation':'endpoint_partition_mean_of_covering_assessed_bins'}


def main():
    out=_fk_paths.relocate((ROOT/'experiments/remaining-001'), __file__)
    if out.exists():raise ValueError('Preserve prepared protocol; use a new version for changes')
    out.mkdir(parents=True);profiles=_fk_paths.relocate((out/'profiles'), __file__);profiles.mkdir()
    heldout_path=_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__);heldout=read(heldout_path)
    fullrun_path=_fk_paths.relocate((ROOT/'runs/two-day-001/experiment/fk/training/attempt-001/run.json'), __file__);fullrun=read(fullrun_path)
    base=read(_fk_paths.relocate((ROOT/'configs/ours.provisional.json'), __file__));base['decision']=fullrun['decision']
    base['status']='PREPARED_EXPLORATORY_ABLATION_NOT_RUN';base['unresolved']=[]
    settings=copy.deepcopy(fullrun['config']);settings['seed']=7
    write_json(_fk_paths.relocate((out/'training.json'), __file__),settings)
    runs=[]
    for name,ablation in VARIANTS.items():
        model=copy.deepcopy(base);model['ablation']=ablation;model['experiment_variant']=name
        grid=calibration_grid(name,base['decision']['bins'])
        for key in ('top_k','peak_weight','top_ratio','video_peak_weight','reliability_temperature'):
            if len(grid[key])==1:model['decision'][key]=grid[key][0]
        write_json(_fk_paths.relocate((profiles/(name+'.json')), __file__),model);write_json(_fk_paths.relocate((profiles/(name+'.grid.json')), __file__),grid)
        runs.append({'variant':name,'priority':'core' if name in ('full','no_av','no_temporal','uniform_weights') else 'aggregation',
            'model_config':'profiles/'+name+'.json','calibration_grid':'profiles/'+name+'.grid.json',
            'epochs':fullrun['epoch_budget'],'seed':7,'status':'PREPARED_NOT_RUN'})
    cache_path=_fk_paths.relocate((ROOT/'runs/two-day-001/experiment/fk/cache/training-manifest.json'), __file__)
    preflight={'training_cache_manifest':str(cache_path),'tool_access':'NOT_CHECKED','native_user_preflight_required':True}
    try:
        cache=read(cache_path);preflight.update(tool_access='READABLE',counts={k:len(cache[k]) for k in ('train','validation')},sha256=digest(cache_path))
    except OSError as error:preflight.update(tool_access='DENIED',reason=str(error))
    write_json(_fk_paths.relocate((out/'cache-preflight.json'), __file__),preflight)
    robustness=select_balanced(heldout,25);profiling=select_balanced(heldout,3)
    write_json(_fk_paths.relocate((out/'robustness-manifest.json'), __file__),robustness);write_json(_fk_paths.relocate((out/'profiling-manifest.json'), __file__),profiling)
    write_json(_fk_paths.relocate((out/'robustness-plan.json'), __file__),{'status':'PREPARED_NOT_RUN','population':'100 frozen held-out clips, 25 per manipulation type',
        'selection':'lowest SHA256(seed=7, video_id) within type; no prediction-outcome selection',
        'study_status':'exploratory follow-up after clean held-out inspection',
        'conditions':[{'kind':'clean','strength':None},{'kind':'video_compression','strength':28},{'kind':'video_compression','strength':38},
                      {'kind':'audio_compression','strength':64},{'kind':'audio_compression','strength':32}],
        'separate_sync_stress':{'real_clips_only':True,'audio_delay_seconds':[.08,.16,.32],'outcome':'false alarm/assessment rate, not relabeled fake'},
        'unimplemented_bulk_adapter':'Derivative manifest, timestamp/label preservation review, and all-method inference must be completed before bulk scoring',
        'prepared_input_helpers':'deepfake_detection.robustness supports blur/downsample/JPEG/audio noise; these are not raw-media end-to-end tests'})
    write_json(_fk_paths.relocate((out/'profiling-plan.json'), __file__),{'status':'PREPARED_NOT_RUN','clips':12,'unit':'same raw source clips on this hardware',
        'boundaries':['decode/hash/preprocessing','model only','FK record generation/verification','end-to-end'],
        'warmups':1,'measured_repeats':3,'synchronize_cuda_before_and_after_timer':True,
        'report':['median seconds/clip','seconds/source-second','peak CUDA allocated/reserved','process RSS','failure/coverage'],
        'separate_cold_and_warm_cache':True,'do_not_run_during_heldout':True,
        'warning':'Historical concurrent wall times and synthetic head smoke timings are not fair end-to-end benchmarks'})
    examples=select_balanced(heldout,1)
    write_json(_fk_paths.relocate((out/'forensic-plan.json'), __file__),{'status':'PREPARED_NOT_RUN','selected_source_examples':examples,
        'additional_cases':['no usable windows','partial coverage','no audio','short/incomplete media','near-threshold software fixture'],
        'mutations':['source bytes','derived artifact bytes','one evidence contribution','source timestamp','decision threshold','coverage field'],
        'checks':['original commitment matches retained trusted reference','bin/window/video decision reconstruction matches','each corruption is detected','null assessments are preserved'],
        'ground_truth_limit':'Integrity and decision reconstruction do not prove media authenticity or prediction correctness',
        'real_record_status':'Need source-bound raw-media records; synthetic demo/tests are software evidence only'})
    fk_path=_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__);fk=read(fk_path)
    manifest={r['video_id']:r for r in heldout};timelines=fk['timeline_metrics_supplement']
    if len(timelines)!=1000 or {r['video_id'] for r in timelines}!=set(manifest):raise ValueError('Timeline population differs')
    threshold=fk['decision']['window_threshold']
    if any(abs(r['threshold']-threshold)>1e-12 or abs(r['duration']-manifest[r['video_id']]['duration'])>1e-6 for r in timelines):raise ValueError('Timeline thresholds/durations differ')
    summary=localization_summary(timelines);summary.update(source_sha256=digest(fk_path),window_threshold=threshold)
    write_json(_fk_paths.relocate((out/'localization-summary.json'), __file__),summary)
    write_json(_fk_paths.relocate((out/'plan.json'), __file__),{'schema':'remaining-study-preparation-v1','status':'PREPARED_NOT_EXECUTED',
        'active_heldout_must_complete_first':str(_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002/complete.json'), __file__)),
        'training_cache_manifest':str(cache_path),'heldout_manifest_sha256':digest(heldout_path),
        'base_training_run_sha256':digest(fullrun_path),'ablation_runs':runs,
        'minimum_batch':['full','no_av','no_temporal','uniform_weights'],'optional_training_seeds':[17,27],
        'bootstrap':{'replicates':2000,'seed':7,'group':'source_family','reference':'FK','metrics':['balanced_accuracy','f1'],
            'timing':'after all primary predictions complete','scope':'exploratory evaluation-sampling uncertainty, not training-seed variance'},
        'controls':'Already trained/evaluated in runs/shared-controls-001; do not retrain without a separate reason',
        'transfer':{'dataset':'FakeAVCeleb','status':'DEFERRED_DATASET_ACCESS_AND_OVERLAP_AUDIT_REQUIRED'},
        'no_gpu_work_started':True,'no_original_artifacts_modified':True})
    write_json(_fk_paths.relocate((out/'SHA256SUMS.json'), __file__),{str(p.relative_to(out)):digest(p) for p in out.rglob('*.json')})
    print(str(out));print('Prepared 8 ablation profiles, constrained calibration grids, robustness/profiling/forensic plans and existing localization summary. No training or GPU inference started.')


if __name__=='__main__':main()
