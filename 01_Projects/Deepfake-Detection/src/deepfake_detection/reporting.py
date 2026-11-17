"""Detection/localization reporting with explicit coverage and paired group CIs."""

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


import csv
from pathlib import Path

import numpy as np

from .cli import save
from .core import finite, interval, probability, union
from .evaluation import common_assessed, detection_metrics


def scored_metrics(rows):
    """Continuous metrics are conditional on assessment and justified scores."""
    result=detection_metrics(rows)
    assessed=[row for row in rows if row['status']=='ASSESSED']
    available=[row for row in assessed if row['score'] is not None]
    result.update(roc_auc=None,average_precision=None,pr_auc_trapezoid=None,
                  continuous_scored=len(available),positive_prevalence=None)
    if assessed:
        result['positive_prevalence']=sum(row['label'] for row in assessed)/len(assessed)
    if len(available)!=len(assessed) or {row['label'] for row in available}!={0,1}:
        return result
    from sklearn.metrics import average_precision_score,auc,precision_recall_curve,roc_auc_score
    labels=[row['label'] for row in available];scores=[row['score'] for row in available]
    precision,recall,_=precision_recall_curve(labels,scores)
    result.update(roc_auc=float(roc_auc_score(labels,scores)),
                  average_precision=float(average_precision_score(labels,scores)),
                  pr_auc_trapezoid=float(auc(recall,precision)))
    return result


def paired_group_bootstrap(method_rows, *, reference, replicates=2000, seed=7,
                           group_field='source_family', metrics=('balanced_accuracy','f1')):
    """Use identical sampled source families across all methods on shared assessments.

    These intervals describe evaluation-population sampling, not training-seed
    variability. Undefined replicates are counted and excluded per metric.
    """
    if reference not in method_rows or type(replicates) is not int or replicates<2:
        raise ValueError('explicit reference and at least two resamples required')
    shared=common_assessed(method_rows)['video_ids']
    if not shared:
        raise ValueError('no commonly assessed population')
    indices={name:{row['video_id']:row for row in rows} for name,rows in method_rows.items()}
    families={}
    for vid in shared:
        group=indices[reference][vid].get(group_field)
        if not isinstance(group,str) or not group:
            raise ValueError('known source groups required for clustered bootstrap')
        if any(index[vid].get(group_field)!=group for index in indices.values()):
            raise ValueError('source-group mappings differ across methods')
        families.setdefault(group,[]).append(vid)
    groups=sorted(families)
    estimates={name:scored_metrics([index[vid] for vid in shared]) for name,index in indices.items()}
    generator=np.random.default_rng(seed)
    distributions={name:{metric:[] for metric in metrics} for name in indices}
    for _ in range(replicates):
        sampled=generator.choice(groups,len(groups),replace=True)
        values={}
        for name,index in indices.items():
            rows=[]
            for occurrence,group in enumerate(sampled):
                for vid in families[group]:
                    rows.append({**index[vid],'video_id':f'{occurrence}:{vid}'})
            values[name]=scored_metrics(rows)
        for name in indices:
            for metric in metrics:
                if metric not in values[name] or metric not in values[reference]:
                    raise ValueError('unsupported bootstrap metric: '+metric)
                a,b=values[name][metric],values[reference][metric]
                if a is not None and b is not None:
                    distributions[name][metric].append(a-b)
    output={}
    for name,metric_values in distributions.items():
        output[name]={}
        for metric,values in metric_values.items():
            output[name][metric]={'difference_from':reference,'valid_resamples':len(values),
                                 'undefined_resamples':replicates-len(values),
                                 'observed_difference':None if estimates[name][metric] is None or estimates[reference][metric] is None else estimates[name][metric]-estimates[reference][metric],
                                 'ci95':None if not values else np.quantile(values,[.025,.975]).tolist()}
    return {'population':'common_assessed','videos':len(shared),'groups':len(groups),
            'group_field':group_field,'replicates':replicates,'seed':seed,
            'interval_method':'percentile_of_paired_cluster_resamples',
            'uncertainty':'evaluation_group_sampling_not_training_variability','paired_differences':output}


def timeline_metrics(duration, predictions, manipulated, threshold):
    """Average overlapping bin scores on endpoint-partitioned source time.

    Uncovered time remains unassessed; full-timeline manipulation coverage is
    reported separately instead of treating unassessed regions as genuine.
    """
    if finite(duration)<=0:
        raise ValueError('duration must be positive')
    probability(threshold)
    events={0.:[0.,0,0],duration:[0.,0,0]}
    for prediction in predictions:
        start,end=interval(prediction['interval']);score=probability(prediction['score'])
        if end>duration:
            raise ValueError('prediction interval exceeds source duration')
        for at,sign in ((start,1),(end,-1)):
            value=events.setdefault(at,[0.,0,0]);value[0]+=sign*score;value[1]+=sign
    for start,end in union(manipulated):
        if end>duration:
            raise ValueError('annotation exceeds source duration')
        events.setdefault(start,[0.,0,0])[2]+=1
        events.setdefault(end,[0.,0,0])[2]-=1
    score_sum,count,truth=0.,0,0
    totals=dict(tp=0.,tn=0.,fp=0.,fn=0.,unassessed_positive=0.,unassessed_negative=0.)
    segments=[];points=sorted(events)
    for start,end in zip(points,points[1:]):
        delta=events[start];score_sum+=delta[0];count+=delta[1];truth+=delta[2]
        width=end-start
        if not count:
            totals['unassessed_positive' if truth else 'unassessed_negative']+=width
            segments.append({'interval':[start,end],'score':None,'label':int(bool(truth))})
            continue
        score=float(np.clip(score_sum/count,0,1))
        predicted=score>=threshold
        key=('tp' if predicted else 'fn') if truth else ('fp' if predicted else 'tn')
        totals[key]+=width
        segments.append({'interval':[start,end],'score':score,'label':int(bool(truth))})
    def ratio(a,b):return a/b if b else None
    tp,fp,fn=totals['tp'],totals['fp'],totals['fn']
    assessed_positive=tp+fn;positive=assessed_positive+totals['unassessed_positive']
    return {'consolidation':'endpoint_partition_mean_of_covering_assessed_bins',
            'threshold':threshold,'duration':duration,'seconds':totals,
            'assessment_coverage':1-(totals['unassessed_positive']+totals['unassessed_negative'])/duration,
            'assessed_region':{'precision':ratio(tp,tp+fp),'recall':ratio(tp,tp+fn),
                               'f1':ratio(2*tp,2*tp+fp+fn),'iou':ratio(tp,tp+fp+fn)},
            'manipulated_time_assessment_coverage':ratio(assessed_positive,positive),
            'detected_fraction_of_all_manipulated_time':ratio(tp,positive),'segments':segments}


def write_detection_report(method_rows, output, *, provenance, reference=None, replicates=2000):
    """Write per-method and common-assessed results; preserve hard-only baseline scores."""
    if not provenance.get('protocol_id') or provenance.get('data_kind') not in ('synthetic','research'):
        raise ValueError('explicit protocol ID and synthetic/research provenance required')
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    shared=common_assessed(method_rows)
    common=set(shared['video_ids'])
    result={'provenance':provenance,'per_method_assessed':{name:scored_metrics(rows) for name,rows in method_rows.items()},
            'common_assessed_ids':shared['video_ids'],
            'common_assessed':{name:scored_metrics([row for row in rows if row['video_id'] in common]) if common else None
                               for name,rows in method_rows.items()}}
    if reference is not None:
        result['bootstrap']=paired_group_bootstrap(method_rows,reference=reference,replicates=replicates)
    save(_fk_paths.relocate((root/'metrics.json'), __file__),result)
    columns=['population','method','attempted','assessed','coverage','tp','tn','fp','fn',
             'balanced_accuracy','precision','sensitivity','specificity','f1','roc_auc','average_precision','pr_auc_trapezoid']
    with (_fk_paths.relocate((root/'metrics.csv'), __file__)).open('x',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for population in ('per_method_assessed','common_assessed'):
            for name,values in result[population].items():
                if values is not None:
                    writer.writerow({'population':population,'method':name,**{key:values[key] for key in columns[2:]}})
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        save(_fk_paths.relocate((root/'plot-status.json'), __file__),{'status':'NOT_RENDERED','reason':'matplotlib unavailable; numeric reports are complete'})
    else:
        for index,(name,metrics) in enumerate(result['per_method_assessed'].items()):
            matrix=np.array([[metrics['tn'],metrics['fp']],[metrics['fn'],metrics['tp']]])
            figure,axis=plt.subplots(figsize=(4,4))
            axis.imshow(matrix,cmap='Blues')
            for row in range(2):
                for column in range(2):axis.text(column,row,str(matrix[row,column]),ha='center',va='center')
            axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['genuine','fake'],yticklabels=['genuine','fake'],
                     xlabel='Prediction',ylabel='Ground truth',title=name)
            figure.tight_layout();figure.savefig(_fk_paths.relocate((root/f'confusion-{index}.svg'), __file__));plt.close(figure)
        save(_fk_paths.relocate((root/'plot-status.json'), __file__),{'status':'RENDERED'})
    return result
