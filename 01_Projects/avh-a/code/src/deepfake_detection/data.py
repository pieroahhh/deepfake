"""Strict shared manifest audit for dataset-specific adapters."""
import hashlib
from .core import finite, interval


def audit(records):
    if not records:
        raise ValueError("empty manifest")
    policies = {row.get('source_overlap_policy','error') for row in records}
    if len(policies) != 1 or not policies <= {'error','drop-train','keep-official'}:
        raise ValueError('mixed or invalid source overlap policies')
    policy = next(iter(policies))
    keep_official = policy == 'keep-official'
    if keep_official and any(row.get('dataset') != 'AV-Deepfake1M' for row in records):
        raise ValueError('keep-official requires AV-Deepfake1M rows')
    seen, families, hashes, counts = set(), {}, {}, {}
    shared_families, shared_hashes = set(), set()
    for row in records:
        vid = row["video_id"]
        if not isinstance(vid, str) or not vid or vid in seen:
            raise ValueError("missing/duplicate video_id")
        seen.add(vid)
        for key in ("dataset", "release", "source_family", "path"):
            if not isinstance(row[key], str) or not row[key]:
                raise ValueError(f"{key} must be a known nonempty string")
        split = row["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("invalid split")
        family = row["source_family"]
        if families.setdefault(family, split) != split:
            shared_families.add(family)
            if not keep_official:
                raise ValueError(f"source-family leakage: {family}")
        source_hash = row["source_sha256"]
        if source_hash is not None:
            if not isinstance(source_hash, str) or len(source_hash) != 64 or any(c not in "0123456789abcdef" for c in source_hash):
                raise ValueError("invalid source hash")
            if hashes.setdefault(source_hash, split) != split:
                shared_hashes.add(source_hash)
                if not keep_official:
                    raise ValueError("content-hash leakage")
        if finite(row["duration"]) <= 0:
            raise ValueError("duration must be positive")
        for key in ("label", "audio_label", "visual_label"):
            value = row[key]
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError("labels must be 0, 1 or null")
        if row["label"] is None:
            raise ValueError("video label required")
        known = [row["audio_label"], row["visual_label"]]
        if (row["label"] == 0 and 1 in known) or (None not in known and row["label"] != int(any(known))):
            raise ValueError("contradictory modality labels")
        if type(row["temporal_complete"]) is not bool:
            raise ValueError("temporal_complete must be explicit Boolean")
        edits = [interval(x) for x in row["manipulated_intervals"]]
        if any(b > row["duration"] for a,b in edits):
            raise ValueError("annotation exceeds source duration")
        if row["label"] == 0 and edits:
            raise ValueError("real video has manipulation annotations")
        if row["label"] == 1 and row["temporal_complete"] and not edits:
            raise ValueError("complete positive annotation cannot be empty")
        counts[split] = counts.get(split, 0) + 1
    return {"videos": len(records), "source_families": len(families), "counts": counts,
            'source_overlap_policy':policy, 'cross_split_source_families':len(shared_families),
            'cross_split_content_hashes':len(shared_hashes),
            "missing_source_hashes": sum(r["source_sha256"] is None for r in records),
            "identity_disjointness": "not established"}


def assign_groups(records, seed="fk-provisional-v1", train=0.7, validation=0.15):
    """Provisional hash assignment, group-safe but NOT stratified or benchmark-frozen."""
    if not 0 < train < 1 or not 0 < validation < 1 or train + validation >= 1:
        raise ValueError("invalid split fractions")
    result = []
    for row in records:
        family = row["source_family"]
        if not isinstance(family, str) or not family:
            raise ValueError("resolve source families before assigning groups")
        value = int.from_bytes(hashlib.sha256((seed + "\0" + family).encode()).digest(), "big") / 2**256
        split = "train" if value < train else "validation" if value < train+validation else "test"
        result.append({**row, "split": split})
    audit(result)
    return result
