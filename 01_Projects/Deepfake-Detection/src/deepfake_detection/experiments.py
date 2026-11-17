"""Predeclared, unexecuted experiment/ablation plans; no placeholder results."""

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


import copy
from pathlib import Path

from .cli import load_json,save


def prepare_experiments(base_config,output,*,seeds=(7,17,27)):
    if not seeds or any(type(seed) is not int or seed<0 for seed in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('distinct nonnegative integer seeds required')
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    variants={
        'full':{},'no_av':{'disabled_evidence':['av']},
        'no_temporal':{'disabled_evidence':['temporal']},'uniform_weights':{'uniform_weights':True},
        'window_max':{},'window_mean':{},'video_max':{},'video_mean':{},
    }
    plans=[]
    for kind in ('visual_only', 'audio_only', 'concatenation', 'late_fusion'):
        profile = copy.deepcopy(base_config)
        profile['model_kind'] = kind
        save(_fk_paths.relocate((root/(kind+'.json')), __file__), profile)
        for seed in seeds:
            plans.append({'method': kind, 'variant': 'primary', 'seed': seed,
                          'profile': kind+'.json', 'status': 'PLANNED_NOT_RUN', 'requires_retraining': True})
    for name,ablation in variants.items():
        profile=copy.deepcopy(base_config)
        profile['ablation']=ablation
        if name=='window_max':profile['decision']['peak_weight']=1.
        if name=='window_mean':profile['decision'].update(peak_weight=0.,top_k=profile['decision']['bins'])
        if name=='video_max':profile['decision']['video_peak_weight']=1.
        if name=='video_mean':profile['decision'].update(video_peak_weight=0.,top_ratio=1.)
        profile['experiment_variant']=name
        save(_fk_paths.relocate((root/(name+'.json')), __file__),profile)
        for seed in seeds:
            plans.append({'method':'FK','variant':name,'seed':seed,'profile':name+'.json',
                          'status':'PLANNED_NOT_RUN','requires_retraining':True})
    for method in ('AVFakeNet-reconstruction-v1','AVTENet-ff-AVHuBERT-reconstruction'):
        for seed in seeds:
            plans.append({'method':method,'variant':'primary','seed':seed,'status':'PLANNED_NOT_RUN',
                          'requires_retraining':True})
    manifest={'schema':'experiment-plan-v1','status':'AWAITING_DATA_AND_FROZEN_PROTOCOL',
              'runs':plans,'selection':'validation_only','test_access':'after_protocol_freeze',
              'budget_note':'three-seed target; feasibility and equal tuning budgets require pilot review',
              'ablation_note':'retrain each variant; test-time zeroing is not an ablation result'}
    save(_fk_paths.relocate((root/'plan.json'), __file__),manifest)
    return manifest
