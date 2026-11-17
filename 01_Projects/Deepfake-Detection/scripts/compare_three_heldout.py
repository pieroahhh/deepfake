"""Fixed-result comparison; never fits thresholds or changes predictions."""

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

import os,sys,json
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,digest
from baselines.common.metrics import metrics
from avfakenet_progress_io import write_json


def checked_rows(rows,manifest):
    if len(rows)!=len(manifest) or len({r['video_id'] for r in rows})!=len(manifest):raise ValueError('Missing or duplicate comparison IDs')
    for row in rows:
        if row['video_id'] not in manifest or row['label']!=manifest[row['video_id']]['label']:raise ValueError('Comparison labels/IDs differ')
        if row['prediction'] not in (None,0,1):raise ValueError('Invalid prediction')
    return rows


def measured(rows):
    result=metrics(rows);c=result['confusion'];n=result['assessed']
    result['accuracy']=(c['tp']+c['tn'])/n if n else None
    return result


def compare(models,manifest):
    validated={name:checked_rows(rows,manifest) for name,rows in models.items()}
    assessed=[{r['video_id'] for r in rows if r['prediction'] is not None} for rows in validated.values()]
    common=set.intersection(*assessed)
    return {'all_models_attempt_same_ids':True,'attempted_clips':len(manifest),
        'own_assessed':{name:measured(rows) for name,rows in validated.items()},
        'common_assessed_ids':sorted(common),'common_assessed_count':len(common),
        'common_assessed':{name:measured(r for r in rows if r['video_id'] in common) for name,rows in validated.items()}}


def main(avfake_run,output):
    output=_fk_paths.relocate((Path(output)), __file__);output.mkdir(exist_ok=True)
    manifest_path=_fk_paths.relocate((ROOT.parent/'Heldout-1000/population/manifest.json'), __file__)
    manifest={r['video_id']:r for r in read(manifest_path)}
    sources={'FK':_fk_paths.relocate((ROOT.parent/'fk-heldout-copy.json'), __file__),
        'AVTENet adaptation (0.5)':_fk_paths.relocate((ROOT.parent/'avh-a/code/runs/auto-next-001/heldout/evaluate/predictions.jsonl'), __file__),
        'AVFakeNet adaptation':_fk_paths.relocate((Path(avfake_run)/'predictions.jsonl'), __file__),
        'AVTENet adaptation (calibrated, exploratory)':_fk_paths.relocate((ROOT/'runs/avtenet-analysis-001/heldout-calibrated-predictions.jsonl'), __file__)}
    models={}
    for name,path in sources.items():
        if name=='FK':models[name]=read(path)['videos']
        else:models[name]=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result=compare(models,manifest)
    expected=read(_fk_paths.relocate((ROOT/'runs/heldout-001/fk/summary.json'), __file__))['video_metrics'];actual=result['own_assessed']['FK']
    if actual['confusion']!={k:expected[k] for k in ('tp','tn','fp','fn')} or actual['assessed']!=expected['assessed']:raise ValueError('FK copy does not match original summary')
    for key in ('accuracy','balanced_accuracy','f1','roc_auc'):
        if abs(actual[key]-expected[key])>1e-10:raise ValueError('FK copied metrics mismatch')
    result.update(scope='Exploratory local held-out from unused official validation; not official test',
        source_sha256={name:digest(path) for name,path in sources.items()},population_sha256=digest(manifest_path),
        limitations=['AVFakeNet and AVTENet are adaptations, not author-verified reproductions.',
            'FK trained five epochs; AVTENet stages one epoch; AVFakeNet one pilot epoch plus documented tiny recovery warmups.',
            'AVTENet calibrated threshold was chosen from validation after the first held-out results were seen; show as exploratory.',
            'Different coverage changes evaluated populations; own-assessed metrics alone do not establish fair superiority.',
            'Common-assessed results describe a selected subset; they do not remove missing-data bias for the full population.',
            'AVFakeNet uses hard majority/OR; no continuous fused ROC AUC is defined.'])
    write_json(_fk_paths.relocate((output/'comparison.json'), __file__),result)
    def percent(value):return 'N/A' if value is None else f'{value*100:.2f}%'
    def table(title,rows):
        text=[title,'','| Model | Assessed | Coverage | Accuracy | Balanced accuracy | F1 | ROC AUC |',
            '|---|---:|---:|---:|---:|---:|---:|']
        for name,m in rows.items():
            auc='N/A' if m['roc_auc'] is None else f'{m["roc_auc"]:.4f}'
            text.append(f'| {name} | {m["assessed"]}/{m["videos"]} | {percent(m["coverage"])} | {percent(m["accuracy"])} | {percent(m["balanced_accuracy"])} | {percent(m["f1"])} | {auc} |')
        return '\n'.join(text)
    report=['# Preliminary comparison on the same 1,000 held-out clips','',
        'Models and decisions were frozen before the new AVFakeNet held-out inference. No test-fitted threshold or retraining was performed.',
        '',table('## Each model on its own assessed clips',result['own_assessed']),
        '',f'These rows have different denominators. Coverage is part of the result, not a reason to silently discard failures.',
        '',table(f'## Paired comparison on {len(result["common_assessed_ids"])} common assessed clips',result['common_assessed']),
        '', 'The calibrated AVTENet row is a supplementary exploratory analysis. It is not an additional model or an independent confirmation.',
        '', '## Limits','',*['- '+s for s in result['limitations']],
        '', 'Source hashes and common clip IDs are saved in `comparison.json`. Predictions remain in their original result folders.']
    (_fk_paths.relocate((output/'COMPARISON.md'), __file__)).write_text('\n'.join(report)+'\n',encoding='utf-8')
    os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((ROOT/'.cache/matplotlib'), __file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    names=list(models);short=['FK','AVTENet\n0.5','AVFakeNet','AVTENet\ncalibrated*']
    fig,axes=plt.subplots(1,3,figsize=(13,4.8));colors=['#2563eb','#d97706','#059669','#a16207']
    for ax,key,title in [(axes[0],'coverage','Coverage on all 1,000 clips'),(axes[1],'balanced_accuracy','Balanced accuracy: own assessed'),(axes[2],'balanced_accuracy',f'Balanced accuracy: common n={result["common_assessed_count"]}')]:
        values=result['common_assessed'] if ax is axes[2] else result['own_assessed']
        heights=[(values[name][key] or 0)*100 for name in names]
        bars=ax.bar(np.arange(4),heights,color=colors)
        bars[3].set_hatch('//');ax.set_ylim(0,110);ax.set_xticks(np.arange(4),short,fontsize=9);ax.set_title(title,fontsize=10)
        ax.set_ylabel('%');ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Preliminary adaptation comparison — unequal training budgets',fontsize=12)
    fig.text(.5,.015,'* Validation-calibrated AVTENet after initial held-out inspection; exploratory. Missing clips are reported separately.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.05,1,.92));fig.savefig(_fk_paths.relocate((output/'comparison.png'), __file__),dpi=180);fig.savefig(_fk_paths.relocate((output/'comparison.pdf'), __file__));plt.close(fig)
    write_json(_fk_paths.relocate((output/'complete.json'), __file__),{'files':{name:digest(_fk_paths.relocate((output/name), __file__)) for name in ('comparison.json','COMPARISON.md','comparison.png','comparison.pdf')}})


if __name__=='__main__':
    main(_fk_paths.relocate((ROOT/'runs/avfakenet-heldout-002'), __file__),_fk_paths.relocate((ROOT/'runs/three-model-heldout-002'), __file__))
