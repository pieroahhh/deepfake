"""Validation-only decision threshold analysis on immutable prediction scores."""
import math
from .io import fingerprint
from .metrics import metrics


def checked_rows(rows,split):
    rows=list(rows);seen=set()
    for row in rows:
        if row.get('split')!=split:raise ValueError('Expected '+split+' predictions')
        vid=row['video_id']
        if vid in seen:raise ValueError('Duplicate video ID')
        seen.add(vid)
        if type(row.get('label')) is not int or row['label'] not in (0,1):
            raise ValueError('Known binary clip labels are required')
        assessed=row.get('status')=='ASSESSED'
        score=row.get('score')
        if assessed:
            if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError('Assessed score must be finite and in [0,1]')
        elif score is not None or row.get('prediction') is not None:
            raise ValueError('Unassessed predictions must retain null scores and decisions')
    return rows


def apply_threshold(rows,threshold):
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid threshold')
    return [{**row,'prediction':int(row['score']>=threshold) if row['status']=='ASSESSED' else None}
            for row in rows]


def summarize(rows):
    result=metrics(rows);c=result['confusion']
    result['accuracy']=(c['tp']+c['tn'])/result['assessed'] if result['assessed'] else None
    return result


def fit_validation_threshold(rows,default=.5):
    """Held-out rows are not accepted. Maximize BA, then prefer default proximity."""
    rows=checked_rows(rows,'validation')
    assessed=[r for r in rows if r['status']=='ASSESSED']
    scores=[r['score'] for r in assessed];labels=[r['label'] for r in assessed]
    if set(labels)!={0,1}:raise ValueError('Validation selection needs both classes')
    positives=labels.count(1);negatives=labels.count(0);best=None
    for threshold in sorted({0.,1.,default,*scores}):
        tpr=sum(s>=threshold and y==1 for s,y in zip(scores,labels))/positives
        tnr=sum(s<threshold and y==0 for s,y in zip(scores,labels))/negatives
        rank=((tpr+tnr)/2,-abs(threshold-default),-threshold)
        if best is None or rank>best[0]:best=(rank,threshold)
    return {'schema':'validation-only-video-threshold-v1','fit_split':'validation',
        'objective':'balanced_accuracy','default_threshold':default,'threshold':best[1],
        'tie_policy':'closest_to_default_then_lower','assessed_validation_clips':len(assessed),
        'validation_prediction_fingerprint':fingerprint(sorted(rows,key=lambda r:r['video_id'])),
        'default_validation':summarize(apply_threshold(rows,default)),
        'selected_validation':summarize(apply_threshold(rows,best[1]))}
