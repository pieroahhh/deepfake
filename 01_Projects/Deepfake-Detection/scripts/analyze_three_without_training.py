"""Read-only three-model subgroup and error-overlap analysis; no fitting."""

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

import os,sys,json,hashlib
from pathlib import Path
from collections import Counter
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'scripts'), __file__)),str(_fk_paths.relocate((ROOT/'src'), __file__))]
os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
from compare_three_heldout import checked_rows,measured
from baselines.common.metrics import read_predictions
from baselines.common.io import read,digest
OUT=_fk_paths.relocate((ROOT/'runs/three-model-no-training-001'), __file__)
NAMES=['FK','AVTENet adaptation (0.5)','AVFakeNet adaptation']


def subgroup(rows):
    n=len(rows);assessed=[r for r in rows if r['prediction'] is not None]
    correct=sum(r['prediction']==r['label'] for r in assessed)
    return {'attempted':n,'assessed':len(assessed),'unassessed':n-len(assessed),
        'coverage':len(assessed)/n if n else None,'correct':correct,'wrong':len(assessed)-correct,
        'correct_rate_assessed':correct/len(assessed) if assessed else None,
        'false_positive_rate_assessed':sum(r['prediction']==1 and r['label']==0 for r in assessed)/len(assessed) if assessed and all(r['label']==0 for r in assessed) else None}


def main():
    OUT.mkdir(exist_ok=False)
    locked=read(_fk_paths.relocate((ROOT/'runs/three-model-heldout-002/comparison.json'), __file__))
    population=_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__);manifest={r['video_id']:r for r in read(population)}
    if digest(population)!=locked['population_sha256']:raise ValueError('Population changed')
    paths={'FK':_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__),
        'AVTENet adaptation (0.5)':_fk_paths.relocate((ROOT.parent/'avh-a/code/runs/auto-next-001/heldout/evaluate/predictions.jsonl'), __file__),
        'AVFakeNet adaptation':_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002/predictions.jsonl'), __file__)}
    models={}
    for name,path in paths.items():
        if digest(path)!=locked['source_sha256'][name]:raise ValueError('Locked predictions changed')
        models[name]=checked_rows(read_predictions(path),manifest)
    lookup={name:{r['video_id']:r for r in rows} for name,rows in models.items()}
    common=set.intersection(*[{r['video_id'] for r in rows if r['prediction'] is not None} for rows in models.values()])
    if len(common)!=locked['common_assessed_count']:raise ValueError('Common population differs')
    types={kind:{vid for vid,r in manifest.items() if r['source_modify_type']==kind} for kind in ('real','audio_modified','visual_modified','both_modified')}
    duration={'under_8s':{v for v,r in manifest.items() if r['duration']<8},
        '8_to_under_16s':{v for v,r in manifest.items() if 8<=r['duration']<16},
        '16s_and_over':{v for v,r in manifest.items() if r['duration']>=16}}
    def groups(definitions,paired):
        return {group:{name:subgroup([row for row in rows if row['video_id'] in ids and (not paired or row['video_id'] in common)]) for name,rows in models.items()} for group,ids in definitions.items()}
    overlap=Counter();examples={}
    for vid in sorted(common):
        wrong=tuple(name for name in NAMES if lookup[name][vid]['prediction']!=manifest[vid]['label'])
        label='all_correct' if not wrong else 'wrong: '+' + '.join(wrong)
        overlap[label]+=1;examples.setdefault(label,[]).append(vid)
    result={'scope':'Exploratory post-inspection analysis; no new inference, training, threshold/model selection',
        'primary_AVTENet_threshold':.5,'calibrated_AVTENet':'Separate exploratory result retained in original report; not substituted here',
        'sources':{name:digest(path) for name,path in paths.items()},'population_sha256':digest(population),
        'by_type_own_assessed':groups(types,False),'by_type_common_assessed':groups(types,True),
        'by_duration_own_assessed':groups(duration,False),'by_duration_common_assessed':groups(duration,True),
        'duration_bins':'Metadata-defined descriptive bins: <8, 8<=duration<16, >=16 seconds; not optimized using predictions',
        'common_assessed':len(common),'common_assessed_error_patterns':dict(overlap),'pattern_ids':examples,
        'limitations':['Own-assessed subgroup denominators differ; unassessed is not silently counted as a correct classification.',
            'Within a single manipulation type the combined label is constant; report fake recall/real specificity instead of BA or AUC.',
            'Common-assessed subsets are selected by model availability and do not remove missing-data bias.',
            'Duration subgroup label composition may differ; accuracy alone does not identify duration as a cause.',
            'Baselines are local adaptations with unequal budgets; no claim of author-method reproduction.']}
    (_fk_paths.relocate((OUT/'analysis.json'), __file__)).write_text(json.dumps(result,indent=2),encoding='utf-8')
    def table(definitions,metric):
        text=['| Group | Model | Assessed / attempted | Coverage | Correct on assessed |','|---|---|---:|---:|---:|']
        for group,models in definitions.items():
            for name,m in models.items():
                value=m['correct_rate_assessed'];rate='N/A' if value is None else f'{value*100:.2f}%'
                cov='N/A' if m['coverage'] is None else f'{m["coverage"]*100:.2f}%'
                text.append(f'| {group} | {name} | {m["assessed"]}/{m["attempted"]} | {cov} | {rate} |')
        return '\n'.join(text)
    report=['# Three-model comparisons added without training\n',
        'All results use the locked existing 1,000 predictions. Primary AVTENet uses threshold 0.5; its calibrated post-inspection analysis stays separate. No weights, thresholds or predictions changed.\n',
        '## Manipulation types\n\nCorrect classification on fake types is recall; on genuine clips it is specificity. AUC/BA are undefined within these constant-label groups.\n',
        table(result['by_type_own_assessed'],None),
        '\n## Same assessed clips in each type\n',table(result['by_type_common_assessed'],None),
        '\n## Clip duration, method-specific assessed populations\n',table(result['by_duration_own_assessed'],None),
        '\n## Clip duration, common assessed populations\n',table(result['by_duration_common_assessed'],None),
        '\n## Error overlap on 415 common assessed clips\n\n| Pattern | Clips |\n|---|---:|\n'+'\n'.join(f'| {name} | {count} |' for name,count in sorted(overlap.items())),
        '\nThese overlap counts are descriptive, not an evaluated ensemble. Combining models would require a validation-defined decision rule and new held-out predictions.\n',
        '\n## Figures\n\n![Type-specific performance and coverage](manipulation-types.png)\n\n![Error overlap](error-overlap.png)\n',
        '\n## Limits\n\n'+'\n'.join('- '+v for v in result['limitations'])]
    (_fk_paths.relocate((OUT/'REPORT.md'), __file__)).write_text('\n'.join(report),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained');x=np.arange(4);width=.25
    for i,name in enumerate(NAMES):
        values=[result['by_type_own_assessed'][kind][name] for kind in types]
        axes[0].bar(x+(i-1)*width,[m['correct_rate_assessed'] for m in values],width,label=name)
        axes[1].bar(x+(i-1)*width,[m['coverage'] for m in values],width,label=name)
    for ax,title in zip(axes,['Correct rate on assessed clips (real: specificity; fake: recall)','Assessment coverage; 250 attempted per type']):
        ax.set_xticks(x,['Real','Audio fake','Visual fake','Both fake']);ax.set_ylim(0,1);ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[1].legend(fontsize=8);fig.suptitle('Existing locked predictions; own-assessed denominators differ; exploratory',fontsize=11)
    fig.savefig(_fk_paths.relocate((OUT/'manipulation-types.png'), __file__),dpi=180);plt.close(fig)
    labels=list(overlap);fig,ax=plt.subplots(figsize=(10,5),layout='constrained');ax.barh(labels,[overlap[k] for k in labels]);ax.invert_yaxis();ax.set_xlabel('Clips');ax.set_title('Error overlap on 415 common assessed clips; not an ensemble result')
    fig.savefig(_fk_paths.relocate((OUT/'error-overlap.png'), __file__),dpi=160);plt.close(fig)
    (_fk_paths.relocate((OUT/'complete.json'), __file__)).write_text(json.dumps({'files':{p.name:digest(p) for p in OUT.iterdir() if p.is_file()},'predictions_unchanged':all(digest(p)==locked['source_sha256'][name] for name,p in paths.items())},indent=2))
    print(OUT)


if __name__=='__main__':main()
