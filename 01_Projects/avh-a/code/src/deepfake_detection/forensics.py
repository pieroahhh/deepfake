"""Model-independent record construction and trusted-commitment verification."""

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

import hashlib
import json
import math
import shutil
from pathlib import Path
from .core import Decision, coverage, finite, hybrid, interval, probability, sigmoid, video_score


# Allow a few float32 ULPs at 1.0 for the four softmax weights.
_FLOAT32_WEIGHT_SUM_TOLERANCE = 4 * 2.0**-23


def sha256_file(path):
    digest = hashlib.sha256()
    with open(_fk_paths.relocate_value((path), __file__), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preserve(source, destination):
    """Exclusive destination creation; originals are never overwritten."""
    source, destination = _fk_paths.relocate((Path(source)), __file__), _fk_paths.relocate((Path(destination)), __file__)
    original = sha256_file(source)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha256_file(destination) != original or sha256_file(source) != original:
        raise ValueError("source/working-copy integrity mismatch")
    return original


def canonical(record):
    # FK Python JSON profile v1: sorted keys, UTF-8, compact separators,
    # finite Python binary64 shortest-roundtrip numbers; no Unicode normalization.
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def commitment(record):
    source_hash = record["source_sha256"]
    if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
        raise ValueError("invalid source SHA-256")
    return hashlib.sha256(b"FAC-v1" + bytes.fromhex(source_hash) + canonical(record)).hexdigest()


def build_record(source_hash, duration, windows, decision, *, model_version, preprocessing):
    """Windows include quality plus bins with alpha and signed contributions."""
    output = json.loads(json.dumps(windows, allow_nan=False))
    for window in output:
        if window["status"] == "USABLE":
            for b in window["bins"]:
                b["score"] = sigmoid(math.fsum(b["contributions"]))
            window["score"] = hybrid([b["score"] for b in window["bins"]], decision.top_k, decision.peak_weight)
        else:
            window["bins"], window["score"] = None, None
    usable = [w for w in output if w["status"] == "USABLE"]
    score = video_score([w["score"] for w in usable], decision)
    result = dict(schema_version="fk-record-v1", source_sha256=source_hash, duration=duration,
                  model_version=model_version, preprocessing=preprocessing, decision=decision.to_dict(), windows=output,
                  status="ASSESSED" if usable else "UNASSESSED", score=score,
                  prediction=None if score is None else int(score >= decision.video_threshold),
                  candidates=[w["window_id"] for w in usable if w["score"] >= decision.window_threshold],
                  coverage=coverage(duration, [w["interval"] for w in usable]))
    verify(result, commitment(result))
    return result


def verify(record, trusted_commitment, *, source=None, tolerance=1e-12, artifact_root=None):
    """Requires an independently retained commitment, not one trusted from the record."""
    if finite(tolerance) < 0:
        raise ValueError("verification tolerance must be nonnegative")
    weight_sum_tolerance = max(tolerance, _FLOAT32_WEIGHT_SUM_TOLERANCE)
    if commitment(record) != trusted_commitment:
        raise ValueError("commitment mismatch")
    if source is not None and sha256_file(source) != record["source_sha256"]:
        raise ValueError("source hash mismatch")
    if record["schema_version"] not in ("fk-record-v1", "fk-record-v2"):
        raise ValueError("unsupported schema")
    checked = 0
    if record['schema_version'] == 'fk-record-v2':
        if not isinstance(record.get('source_metadata'), dict) or not isinstance(record.get('derived_evidence'), list):
            raise ValueError('v2 source metadata and derived evidence required')
        paths = set()
        for item in record['derived_evidence']:
            if item['source_sha256'] != record['source_sha256'] or item['path'] in paths:
                raise ValueError('invalid derived source binding or duplicate artifact')
            interval(item['interval'])
            paths.add(item['path'])
            if artifact_root is not None:
                root = _fk_paths.relocate((Path(artifact_root)), __file__).resolve()
                target = (_fk_paths.relocate((root/item['path']), __file__)).resolve()
                target.relative_to(root)
                if sha256_file(target) != item['sha256']:
                    raise ValueError('derived artifact hash mismatch')
                checked += 1
    decision = Decision(**record["decision"])
    duration = finite(record["duration"])
    errors = []

    def check(actual, expected):
        if expected is None:
            if actual is not None:
                raise ValueError("unassessed output must be null")
        else:
            probability(actual)
            error = abs(actual - expected)
            errors.append(error)
            if error > tolerance:
                raise ValueError("decision reconstruction mismatch")

    scores, assessed, candidates, ids = [], [], [], set()
    for w in record["windows"]:
        wid = w["window_id"]
        if type(wid) is not int or wid < 0 or wid in ids:
            raise ValueError("invalid/duplicate window ID")
        ids.add(wid)
        start, end = interval(w["interval"])
        if start >= duration or not isinstance(w["quality"], dict):
            raise ValueError("invalid source interval/quality")
        if w["status"] == "UNASSESSED":
            if not w["reasons"] or w["bins"] is not None or w["score"] is not None:
                raise ValueError("invalid unassessed window")
            continue
        if w["status"] != "USABLE" or end > duration or w["reasons"] or len(w["bins"]) != decision.bins:
            raise ValueError("invalid usable window")
        bins = []
        for i, b in enumerate(w["bins"]):
            expected_interval = (start + i*(end-start)/decision.bins, start + (i+1)*(end-start)/decision.bins)
            if any(abs(x-y) > tolerance for x,y in zip(interval(b["interval"]), expected_interval)):
                raise ValueError("bin timestamp mismatch")
            if len(b["alpha"]) != 4 or len(b["contributions"]) != 4:
                raise ValueError("four evidence channels required")
            if abs(math.fsum(probability(x) for x in b["alpha"]) - 1) > weight_sum_tolerance:
                raise ValueError("weights must sum to one")
            score = sigmoid(math.fsum(finite(x) for x in b["contributions"]))
            check(b["score"], score)
            bins.append(score)
        score = hybrid(bins, decision.top_k, decision.peak_weight)
        check(w["score"], score)
        scores.append(score)
        assessed.append((start,end))
        if score >= decision.window_threshold:
            candidates.append(wid)
    score = video_score(scores, decision)
    check(record["score"], score)
    prediction = None if score is None else int(score >= decision.video_threshold)
    if record["prediction"] != prediction or record["status"] != ("ASSESSED" if scores else "UNASSESSED"):
        raise ValueError("verdict/status mismatch")
    if record["candidates"] != candidates:
        raise ValueError("candidate mismatch")
    if canonical(record["coverage"]) != canonical(coverage(duration, assessed)):
        raise ValueError("coverage mismatch")
    return {"valid": True, "max_absolute_error": max(errors, default=0.0), "prediction": prediction,
            "derived_artifacts_checked": checked, "artifact_verification": 'CHECKED' if artifact_root is not None else 'NOT_REQUESTED'}
