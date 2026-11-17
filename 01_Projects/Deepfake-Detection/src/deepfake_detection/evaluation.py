"""Detection counts with explicit assessment denominators, never null-as-real."""
from .core import probability


def detection_metrics(rows):
    if not rows:
        raise ValueError("empty evaluation population")
    tp = tn = fp = fn = assessed = 0
    ids = set()
    for row in rows:
        if row["video_id"] in ids:
            raise ValueError("duplicate prediction ID")
        ids.add(row["video_id"])
        label = row["label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("invalid ground-truth label")
        if row["status"] in ("UNASSESSED", "ERROR"):
            if row["prediction"] is not None or row["score"] is not None:
                raise ValueError("unassessed/error predictions must be null")
            continue
        pred = row["prediction"]
        if row["status"] != "ASSESSED" or type(pred) is not int or pred not in (0, 1):
            raise ValueError("invalid prediction/status")
        if row["score"] is not None:
            probability(row["score"])
        assessed += 1
        tp += int(label == 1 and pred == 1)
        tn += int(label == 0 and pred == 0)
        fp += int(label == 0 and pred == 1)
        fn += int(label == 1 and pred == 0)
    def divide(a,b):
        return a/b if b else None
    sensitivity, specificity = divide(tp,tp+fn), divide(tn,tn+fp)
    return dict(attempted=len(rows), assessed=assessed, unassessed_or_error=len(rows)-assessed,
                coverage=assessed/len(rows), tp=tp, tn=tn, fp=fp, fn=fn,
                accuracy=divide(tp+tn,assessed), sensitivity=sensitivity, specificity=specificity,
                balanced_accuracy=None if sensitivity is None or specificity is None else (sensitivity+specificity)/2,
                precision=divide(tp,tp+fp), f1=divide(2*tp,2*tp+fp+fn))


def common_assessed(method_rows):
    """All methods must contain exactly the same full manifest IDs and labels."""
    if not method_rows:
        raise ValueError("no methods")
    indices = {}
    expected = None
    labels = {}
    for method, rows in method_rows.items():
        detection_metrics(rows)  # validate each population first
        index = {r["video_id"]: r for r in rows}
        if expected is None:
            expected = set(index)
            labels = {k:v["label"] for k,v in index.items()}
        if set(index) != expected or any(index[k]["label"] != labels[k] for k in expected):
            raise ValueError("methods must share full manifest IDs and labels")
        indices[method] = index
    shared = sorted(k for k in expected if all(index[k]["status"] == "ASSESSED" for index in indices.values()))
    return {"video_ids": shared,
            "metrics": {method: detection_metrics([index[k] for k in shared]) if shared else None
                        for method,index in indices.items()}}
