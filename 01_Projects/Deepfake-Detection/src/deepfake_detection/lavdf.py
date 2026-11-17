"""Normalize LAV-DF annotations and select a reproducible development pilot."""

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


from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import re

from .cli import load_json, save
from .core import finite, interval
from .data import audit
from .forensics import sha256_file


SPLITS = {"train": "train", "dev": "validation", "test": "test"}
CATEGORIES = {(0, 0): "genuine", (1, 0): "audio_only", (0, 1): "visual_only", (1, 1): "audio_visual"}


def normalize_lavdf(metadata, release):
    """Preserve source-time labels and report rows with invalid temporal annotations."""
    sources = {}
    for item in metadata:
        path = item["file"]
        split = item["split"]
        if (split not in SPLITS or not isinstance(path, str)
                or not re.fullmatch(rf"{split}/[A-Za-z0-9_-]+\.mp4", path) or path in sources):
            raise ValueError(f"invalid or duplicate LAV-DF path: {path!r}")
        sources[path] = item

    rows, excluded = [], []
    for path, item in sources.items():
        audio, visual = item["modify_audio"], item["modify_video"]
        if type(audio) is not bool or type(visual) is not bool:
            raise ValueError(f"LAV-DF modality flags must be Boolean: {path}")
        label = int(audio or visual)
        original = item["original"]
        if label:
            parent = sources.get(original)
            if (parent is None or parent["original"] is not None or parent["modify_audio"]
                    or parent["modify_video"] or parent["split"] != item["split"]):
                raise ValueError(f"LAV-DF original must be a genuine clip in the same split: {path}")
        elif original is not None:
            raise ValueError(f"genuine LAV-DF clip has an original reference: {path}")

        duration = finite(item["duration"])
        periods = item["fake_periods"]
        count = item["n_fakes"]
        if duration <= 0 or type(count) is not int or not isinstance(periods, list):
            raise ValueError(f"invalid LAV-DF duration or interval metadata: {path}")
        if count != len(periods) or int(count > 0) != label:
            raise ValueError(f"LAV-DF segment count disagrees with labels: {path}")
        try:
            edits = sorted(interval(period) for period in periods)
            if any(end > duration for _, end in edits):
                raise ValueError("manipulation interval exceeds annotated duration")
        except (ValueError, TypeError) as error:
            excluded.append({"path": path, "duration": duration, "fake_periods": periods, "reason": str(error)})
            continue

        # The release provides a single interval list and clip-level modality flags.
        audio_edits = edits if audio else []
        visual_edits = edits if visual else []
        if audio and visual:
            audio_edits = visual_edits = None
        rows.append({
            "video_id": path, "dataset": "LAV-DF", "release": release,
            "path": path, "source_family": original or path, "source_sha256": None,
            "split": SPLITS[item["split"]], "duration": duration,
            "label": label, "audio_label": int(audio), "visual_label": int(visual),
            "temporal_complete": True, "manipulated_intervals": edits,
            "audio_manipulated_intervals": audio_edits,
            "visual_manipulated_intervals": visual_edits,
        })
    return rows, excluded


def prepare_lavdf_pilot(dataset_root, output, *, train_families=8, validation_families=4, seed=7):
    """Choose one clip per category in each selected original/variant family."""
    limits = {"train": train_families, "validation": validation_families}
    if any(type(limit) is not int or limit < 1 for limit in limits.values()):
        raise ValueError("pilot family counts must be positive integers")
    root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)
    metadata_path = _fk_paths.relocate((root / "metadata.min.json"), __file__)
    metadata_sha256 = sha256_file(metadata_path)
    metadata = load_json(metadata_path)
    rows, excluded = normalize_lavdf(metadata, f"metadata-sha256:{metadata_sha256}")
    annotation_audit = audit(rows)

    families = {split: defaultdict(dict) for split in limits}
    for row in sorted(rows, key=lambda row: row["video_id"]):
        if row["split"] in families:
            family = families[row["split"]][row["source_family"]]
            family.setdefault((row["audio_label"], row["visual_label"]), row)

    def family_rank(source_family):
        return hashlib.sha256(f"{seed}\0{source_family}".encode()).digest()

    selected, category_counts = [], {}
    for split, limit in limits.items():
        eligible = [family for family, clips in families[split].items() if clips.keys() == CATEGORIES.keys()]
        if len(eligible) < limit:
            raise ValueError(f"only {len(eligible)} complete LAV-DF families available in {split}")
        chosen = sorted(eligible, key=family_rank)[:limit]
        counts = Counter()
        for family in chosen:
            for category, name in CATEGORIES.items():
                row = families[split][family][category]
                source = (_fk_paths.relocate((root / row["path"]), __file__)).resolve(strict=True)
                source.relative_to(root)
                selected.append({**row, "source_sha256": sha256_file(source)})
                counts[name] += 1
        category_counts[split] = dict(counts)

    report = {
        "schema": "lavdf-pilot-v2", "dataset_root": str(root),
        "metadata": {"path": str(metadata_path), "sha256": metadata_sha256, "rows": len(metadata)},
        "annotation_audit": annotation_audit, "excluded_annotations": excluded,
        "selection": {"seed": seed, "families_per_split": limits, "category_counts": category_counts},
        "pilot_audit": audit(selected),
    }
    destination = _fk_paths.relocate((Path(output)), __file__)
    destination.mkdir(parents=True, exist_ok=False)
    save(_fk_paths.relocate((destination / "manifest.json"), __file__), selected)
    save(_fk_paths.relocate((destination / "audit.json"), __file__), report)
    return report


def prepare_lavdf_study(dataset_root, output, *, clip_limits, seed=7):
    """Freeze a seeded prefix of whole families within each official split."""
    if set(clip_limits) != set(SPLITS.values()) or any(type(n) is not int or n < 1 for n in clip_limits.values()):
        raise ValueError('provide positive clip limits for train, validation, and test')
    root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)
    metadata_path = _fk_paths.relocate((root / 'metadata.min.json'), __file__)
    metadata_digest = sha256_file(metadata_path)
    rows, excluded = normalize_lavdf(load_json(metadata_path), f'metadata-sha256:{metadata_digest}')
    annotation_audit = audit(rows)
    families = {split: defaultdict(list) for split in SPLITS.values()}
    for row in sorted(rows, key=lambda item: item['video_id']):
        families[row['split']][row['source_family']].append(row)

    selected, category_counts = [], {}
    for split, groups in families.items():
        ranked = sorted(groups, key=lambda family: hashlib.sha256(f'{seed}\0{family}'.encode()).digest())
        count = 0
        categories = Counter()
        for family in ranked:
            group = groups[family]
            if count + len(group) > clip_limits[split]:
                break
            for row in group:
                source = (_fk_paths.relocate((root / row['path']), __file__)).resolve(strict=True)
                source.relative_to(root)
                selected.append({**row, 'source_sha256': sha256_file(source)})
                categories[CATEGORIES[row['audio_label'], row['visual_label']]] += 1
            count += len(group)
        if count == 0:
            raise ValueError(f'clip limit too small for the first family in {split}')
        category_counts[split] = dict(categories)

    report = {
        'schema': 'lavdf-study-v1', 'source_metadata_sha256': metadata_digest,
        'seed': seed, 'clip_limits': clip_limits, 'category_counts': category_counts,
        'sampling': 'official splits; SHA-256-ranked whole families; stop before exceeding each clip cap; retain natural class mix',
        'annotation_audit': annotation_audit, 'excluded_annotations': excluded, 'study_audit': audit(selected),
    }
    destination = _fk_paths.relocate((Path(output)), __file__)
    destination.mkdir(parents=True, exist_ok=False)
    save(_fk_paths.relocate((destination / 'manifest.json'), __file__), selected)
    report['manifest_sha256'] = sha256_file(_fk_paths.relocate((destination / 'manifest.json'), __file__))
    save(_fk_paths.relocate((destination / 'audit.json'), __file__), report)
    return report
