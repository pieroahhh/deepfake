"""Audit frozen AVTENet results, fit validation threshold, write separate analysis."""

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

import argparse,json,os,sys,sqlite3,math
from contextlib import closing
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
from baselines.common.io import read,write,digest
from baselines.common.threshold_analysis import checked_rows,fit_validation_threshold,apply_threshold,summarize


def predictions(path):
    return [json.loads(line) for line in _fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_same(path,value):
    """Resume analysis publication without replacing an already frozen artifact."""
    if path.exists():
        if read(path)!=value:raise ValueError('Existing analysis artifact differs: '+str(path))
    else:write(path,value)


def verify_population(rows,manifest,split):
    expected={r['video_id']:r for r in manifest if r['split']==split}
    checked_rows(rows,split)
    if {r['video_id'] for r in rows}!=set(expected):raise ValueError('Population differs: '+split)
    for row in rows:
        if row['label']!=expected[row['video_id']]['label']:raise ValueError('Clip label differs')
    return expected


def validation_feature_audit(code,plan,rows,manifest):
    import numpy as np
    import torch
    from baselines.common.training import load_checkpoint
    from baselines.common.population import target
    from baselines.avtenet.model import FeatureFusion,load_component_profile,fake_scores,native_binary_loss
    torch.set_num_threads(1)
    profile=load_component_profile(code/plan['config'])
    checkpoint=load_checkpoint(code/plan['checkpoints']['fusion'])
    model=FeatureFusion(profile).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model_state'],strict=True)
    cache=_fk_paths.relocate((code/'runs/worker-c/avtenet/fusion-features'), __file__)
    metadata=read(_fk_paths.relocate((cache/'cache.json'), __file__))
    if checkpoint['identity']['feature_cache']!=metadata or digest(_fk_paths.relocate((cache/'features.sqlite'), __file__))!=metadata['payload_sha256']:
        raise ValueError('Fusion weights/cache mismatch')
    # Independent numerical check of declared real-logit/fake-label convention.
    assert fake_scores(torch.tensor([-3.,3.]))[0]>.5
    assert native_binary_loss(torch.tensor([-3.,3.]),torch.tensor([1,0])).item()<.1
    video_scores={};window_predictions=[];positive_fractions={};max_error=0.
    with closing(sqlite3.connect((_fk_paths.relocate((cache/'features.sqlite'), __file__)).resolve().as_uri()+'?mode=ro',uri=True)) as conn,torch.inference_mode():
        for row in rows:
            definition=manifest[row['video_id']];scores=[];known=[]
            for start,end,source_sha,blob in conn.execute('SELECT start,end,source_sha256,features FROM features WHERE video_id=? ORDER BY start',(row['video_id'],)):
                if source_sha!=definition['source_sha256']:raise ValueError('Validation feature source mismatch')
                array=np.frombuffer(blob,dtype='<f4').copy().reshape(3,768)
                inputs={name:torch.from_numpy(array[i:i+1]) for i,name in enumerate(('visual','audio','audiovisual'))}
                score=float(fake_scores(model(inputs)).item());scores.append(score)
                label=target(definition,'audiovisual',start,end)
                if label is not None:
                    known.append(label)
                    window_predictions.append({'video_id':row['video_id']+':'+str(start),'label':label,
                        'score':score,'prediction':int(score>=.5),'status':'ASSESSED'})
            if len(scores)!=row['usable_windows']:raise ValueError('Validation usable window count mismatch')
            score=sum(scores)/len(scores) if scores else None
            if (score is None)!=(row['score'] is None):raise ValueError('Validation assessment mismatch')
            if score is not None:
                error=abs(score-row['score']);max_error=max(error,max_error)
                if error>2e-5:raise ValueError('Recomputed validation video score differs')
            video_scores[row['video_id']]=score
            positive_fractions[row['video_id']]=sum(known)/len(known) if known else None
    def group(values):
        return {'clips':len(values),'median':float(np.median(values)) if values else None,
                'q10':float(np.quantile(values,.1)) if values else None,'q90':float(np.quantile(values,.9)) if values else None}
    fractions={}
    for title,predicate in [('fake_false_negative',lambda r:r['label']==1 and r['status']=='ASSESSED' and r['score']<.5),
                            ('fake_true_positive',lambda r:r['label']==1 and r['status']=='ASSESSED' and r['score']>=.5)]:
        values=[positive_fractions[r['video_id']] for r in rows if predicate(r) and positive_fractions[r['video_id']] is not None]
        fractions[title]=group(values)
    return {'score_direction':'verified: external fake probability = sigmoid(-native real logit)',
        'checkpoint_sha256':digest(code/plan['checkpoints']['fusion']),
        'validation_clip_scores_recomputed':len(rows),'maximum_cpu_vs_saved_score_difference':max_error,
        'score_parity_tolerance':2e-5,'window_metrics_at_0_5':summarize(window_predictions),
        'fake_positive_window_fraction_by_video_decision':fractions,
        'aggregation':'unweighted mean of usable window fake probabilities; no aggregation change made'}


def charts(output,validation,heldout,threshold):
    os.environ['MPLCONFIGDIR']=str(_fk_paths.relocate((output/'.matplotlib-cache'), __file__))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    all_values=[math.log10(max(r['score'],1e-12)) for r in validation if r['status']=='ASSESSED']
    lower,upper=min(all_values),max(all_values)
    bins=[lower+(upper-lower)*i/25 for i in range(26)]
    for label,name,color in [(0,'Real','#2670ad'),(1,'Fake','#c45537')]:
        values=[math.log10(max(r['score'],1e-12)) for r in validation if r['status']=='ASSESSED' and r['label']==label]
        axes[0].hist(values,bins=bins,alpha=.5,label=name,color=color)
    axes[0].axvline(math.log10(max(threshold,1e-12)),color='black',ls='--',label='Validation threshold')
    axes[0].axvline(math.log10(.5),color='gray',ls=':',label='Original 0.5')
    axes[0].set(xlabel='log10(fake probability)',ylabel='Validation clips',title='Validation score distribution')
    axes[0].legend(fontsize=8)
    for rows,name in [(validation,'Validation'),(heldout,'Held-out (exploratory)')]:
        a=[r for r in rows if r['status']=='ASSESSED'];fpr,tpr,_=roc_curve([r['label'] for r in a],[r['score'] for r in a])
        axes[1].plot(fpr,tpr,label=name)
    axes[1].plot([0,1],[0,1],':',color='gray');axes[1].set(xlabel='False positive rate',ylabel='True positive rate',title='Fixed model: ROC curves')
    axes[1].legend(fontsize=8)
    fig.savefig(_fk_paths.relocate((output/'diagnostics.png'), __file__),dpi=180);fig.savefig(_fk_paths.relocate((output/'diagnostics.pdf'), __file__));plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--code-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-analysis',action='store_true',help='Finish an interrupted analysis; frozen artifacts must match')
    args=p.parse_args()
    code=args.code_root.resolve();output=args.output.resolve()
    if output==code or code in output.parents:raise ValueError('Analysis must be outside the completed run checkout')
    if output.exists() and not args.resume_analysis:raise ValueError('Output exists; choose a new analysis folder')
    auto=_fk_paths.relocate((code/'runs/auto-next-001'), __file__);planpath=_fk_paths.relocate((auto/'test-plan.json'), __file__);plan=read(planpath)
    valdir=_fk_paths.relocate((code/'runs/worker-c/avtenet/evaluation'), __file__);testdir=_fk_paths.relocate((auto/'heldout/evaluate'), __file__)
    sealed={code/name:sha for name,sha in plan['files'].items()}
    for name,sha in read(_fk_paths.relocate((testdir/'complete.json'), __file__))['files'].items():sealed[testdir/name]=sha
    # Completed validation files are also bound by the already exported return.
    returned=read(_fk_paths.relocate((code.parent/'return-c-validation/return.json'), __file__))
    for name in ('predictions.jsonl','report.json'):
        keys=[k for k in returned['files'] if k.endswith('/evaluation/'+name)]
        if len(keys)!=1:raise ValueError('Validation export seal missing')
        sealed[valdir/name]=returned['files'][keys[0]]
    for path,sha in sealed.items():
        if digest(path)!=sha:raise ValueError('Frozen source checksum mismatch: '+str(path))
    validation=predictions(_fk_paths.relocate((valdir/'predictions.jsonl'), __file__));training=read(code/plan['training_manifest'])
    definitions=verify_population(validation,training,'validation')
    valreport=read(_fk_paths.relocate((valdir/'report.json'), __file__))
    if valreport['checkpoint_sha256']!=digest(code/plan['checkpoints']['fusion']):raise ValueError('Validation checkpoint differs')
    audit=validation_feature_audit(code,plan,validation,definitions)
    selection=fit_validation_threshold(validation)
    selection['validation_predictions_sha256']=digest(_fk_paths.relocate((valdir/'predictions.jsonl'), __file__))
    selection['fusion_checkpoint_sha256']=audit['checkpoint_sha256']
    selection['study_status']='Exploratory follow-up after original held-out results were inspected; requires fresh confirmation'
    output.mkdir(parents=True,exist_ok=args.resume_analysis)
    # Freeze selection BEFORE loading held-out labels or scores in this run.
    write_same(_fk_paths.relocate((output/'threshold-selection.json'), __file__),selection)
    threshold=read(_fk_paths.relocate((output/'threshold-selection.json'), __file__))['threshold']
    heldout=predictions(_fk_paths.relocate((testdir/'predictions.jsonl'), __file__));heldout_manifest=read(code/plan['manifest'])
    verify_population(heldout,heldout_manifest,'test')
    for name in ('video_id','source_family','source_sha256'):
        if {r[name] for r in training}&{r[name] for r in heldout_manifest}:raise ValueError('Held-out overlap: '+name)
    recalibrated=apply_threshold(checked_rows(heldout,'test'),threshold)
    report={'model':'AVTENet adaptation','study_status':selection['study_status'],'threshold':threshold,
        'threshold_fit_split':'validation','aggregation_changed':False,'weights_changed':False,
        'validation_default':selection['default_validation'],'validation_calibrated':selection['selected_validation'],
        'heldout_default':summarize(heldout),'heldout_calibrated':summarize(recalibrated),'validation_audit':audit}
    payload=''.join(json.dumps(row,allow_nan=False)+'\n' for row in recalibrated)
    prediction_path=_fk_paths.relocate((output/'heldout-calibrated-predictions.jsonl'), __file__)
    if prediction_path.exists():
        if prediction_path.read_text(encoding='utf-8')!=payload:raise ValueError('Existing calibrated predictions differ')
    else:
        with prediction_path.open('x',encoding='utf-8') as stream:stream.write(payload)
    write_same(_fk_paths.relocate((output/'report.json'), __file__),report)
    charts(output,validation,heldout,threshold)
    table=['| Split / decision | Accuracy | Balanced accuracy | Recall | F1 | AUC | Coverage |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,keyname in [('Validation / original','validation_default'),('Validation / calibrated','validation_calibrated'),
                          ('Held-out / original','heldout_default'),('Held-out / calibrated','heldout_calibrated')]:
        values=report[keyname];table.append('| '+label+' | '+' | '.join(f'{values[k]:.4f}' for k in ('accuracy','balanced_accuracy','recall','f1','roc_auc','coverage'))+' |')
    text=['# AVTENet validation and threshold analysis','',
        'Completed model outputs and weights are preserved. This is a separate, exploratory analysis.',
        'The original held-out result had already been inspected, so the follow-up is not a fresh confirmatory benchmark.','',
        f'Validation-only threshold: **{threshold:.12g}** (maximize balanced accuracy; deterministic tie policy).',
        'Selection was frozen before this script loaded held-out predictions. No held-out threshold search was performed.','',
        *table,'','## Findings','',
        f"- Score direction and label conversion passed numerical checks; {audit['validation_clip_scores_recomputed']} saved validation video scores were reproduced from frozen features and the selected fusion weights.",
        f"- Maximum CPU versus saved score difference: {audit['maximum_cpu_vs_saved_score_difference']:.3g}.",
        '- Window-level training accuracy and clip-level mean-score decisions measure different targets and denominators.',
        f"- Recomputed validation window accuracy is {audit['window_metrics_at_0_5']['accuracy']:.4f}; this matches the completed training log.",
        '- Among validation fake clips missed at 0.5, the median fraction of known manipulated usable windows was '+str(audit['fake_positive_window_fraction_by_video_decision']['fake_false_negative']['median'])+'. Mean aggregation can dilute short manipulations. This supports a threshold/aggregation explanation; it does not establish the only cause.',
        '- AUC and coverage do not change when only the decision threshold changes.',
        '- The model had a one-epoch budget and is the user-approved joint-ViViT adaptation; it is not an exact AVTENet reproduction.','',
        '## Next','',
        'Keep original 0.5-threshold results in the primary comparison; label calibrated results separately.',
        'For a fair calibrated comparison, use the same validation-only selection policy for every compared model.',
        'Confirm any revised protocol on a fresh untouched split before making confirmatory performance claims.',
        'Review AVFakeNet coverage/constant predictions before final three-model comparison plots.','',
        '![Validation score and ROC diagnostics](diagnostics.png)','']
    (_fk_paths.relocate((output/'ANALYSIS.md'), __file__)).write_text('\n'.join(text),encoding='utf-8')
    after={str(path):digest(path)==sha for path,sha in sealed.items()}
    if not all(after.values()):raise ValueError('Completed source changed during analysis')
    write_same(_fk_paths.relocate((output/'source-preservation.json'), __file__),{'all_unchanged':True,'files':{str(path):sha for path,sha in sealed.items()}})
    print(json.dumps({'output':str(output),'threshold':threshold,'heldout_original':report['heldout_default'],
                      'heldout_calibrated':report['heldout_calibrated'],'audit':audit},indent=2))


if __name__=='__main__':main()
