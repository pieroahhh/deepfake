"""Video metrics always retain the denominator of unassessed videos."""

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

import math


def read_predictions(path):
    """Accept comparator JSONL or the FK evaluator's JSON `videos` array."""
    import json
    from pathlib import Path
    text=_fk_paths.relocate((Path(path)), __file__).read_text(encoding='utf-8')
    try:
        value=json.loads(text)
    except json.JSONDecodeError:
        value=[json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value,dict):value=value['videos']
    if not isinstance(value,list):raise ValueError('expected predictions list or FK report')
    return value


def metrics(rows):
    rows = list(rows)
    assessed = [r for r in rows if r['prediction'] is not None and r['label'] is not None]
    tp = sum(r['prediction'] == 1 and r['label'] == 1 for r in assessed)
    tn = sum(r['prediction'] == 0 and r['label'] == 0 for r in assessed)
    fp = sum(r['prediction'] == 1 and r['label'] == 0 for r in assessed)
    fn = sum(r['prediction'] == 0 and r['label'] == 1 for r in assessed)
    ratio = lambda a, b: a / b if b else None
    recall, specificity = ratio(tp, tp+fn), ratio(tn, tn+fp)
    result = {'videos': len(rows), 'assessed': len(assessed), 'unassessed': len(rows)-len(assessed),
              'coverage': ratio(len(assessed), len(rows)),
              'confusion': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
              'precision': ratio(tp,tp+fp), 'recall': recall, 'specificity': specificity,
              'f1': ratio(2*tp,2*tp+fp+fn),
              'balanced_accuracy': None if recall is None or specificity is None else (recall+specificity)/2,
              'roc_auc': None}
    if assessed and all(r.get('score') is not None and math.isfinite(r['score']) for r in assessed) and tp+fn and tn+fp:
        from sklearn.metrics import roc_auc_score
        result['roc_auc'] = float(roc_auc_score([r['label'] for r in assessed], [r['score'] for r in assessed]))
    return result


def compare(left, right):
    left, right = list(left), list(right)
    a, b = ({r['video_id']: r for r in rows} for rows in (left, right))
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError('duplicate prediction IDs')
    for key in a.keys() & b.keys():
        if a[key]['label'] != b[key]['label']:
            raise ValueError('comparison labels disagree: ' + key)
    common = sorted(k for k in a.keys() & b.keys() if a[k]['prediction'] is not None and b[k]['prediction'] is not None)
    return {'left_all': metrics(left), 'right_all': metrics(right), 'common_assessed': len(common),
            'left_common': metrics(a[k] for k in common), 'right_common': metrics(b[k] for k in common),
            'only_left_ids': len(a.keys()-b.keys()), 'only_right_ids': len(b.keys()-a.keys())}
