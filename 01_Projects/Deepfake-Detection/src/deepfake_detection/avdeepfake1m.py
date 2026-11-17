"""AV-Deepfake1M (2024, not ++) metadata adapter for the shared FK pipeline.

Schema source: https://github.com/ControlNet/AV-Deepfake1M#metadata-structure
Duration comes from media timestamps, never a guessed frame rate. Official
unlabeled test lists remain inference inventories, not labeled manifests.
"""

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
from pathlib import Path, PurePosixPath
import re
import sys
from time import perf_counter

from .cli import load_json, save
from .core import finite, interval, union
from .data import audit
from .forensics import sha256_file


SPLITS = {'train': 'train', 'val': 'validation', 'test': 'test'}
MODALITIES = {'real': (0, 0), 'audio_modified': (1, 0),
              'visual_modified': (0, 1), 'both_modified': (1, 1)}


def relative_path(value, *, source_reference=False):
    if (not isinstance(value, str) or not value or '\\' in value
            or re.match(r'^[A-Za-z]:', value) or '\x00' in value):
        raise ValueError('expected a safe relative POSIX dataset path')
    # Official original IDs include VoxCeleb2//dev/... . Canonicalize these
    # aliases before resolving families, while keeping media paths strict.
    if source_reference and not value.startswith('/'):
        value = re.sub('/+', '/', value)
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise ValueError('expected a safe relative POSIX dataset path')
    return path.as_posix()


def media_path(value, split):
    path = relative_path(value)
    prefix = path.split('/')[0]
    if prefix in SPLITS:
        if prefix != split:
            raise ValueError('media path contradicts the official split')
        return path
    return f'{split}/{path}'


def missing_original_source(reference, split):
    """Recover the release's VoxCeleb2 ID when real.mp4 was not distributed.

    This identifies a source family only; it neither creates a real example nor
    assumes that the missing original media is available.
    """
    path = media_path(reference, split).removeprefix(split+'/')
    match = re.fullmatch(r'(id\d+)/([^/]+)/(\d+)/real\.mp4', path)
    if match is None:
        raise ValueError('missing original must use the AV-Deepfake1M source layout: '+reference)
    person, video, clip = match.groups()
    return f'VoxCeleb2/dev/mp4/{person}/{video}/{clip}.mp4'


def index_metadata(metadata, *, source_overlap_policy='error', exclusions=None, overlaps=None):
    """Resolve genuine source IDs before sampling, including cross-split leaks."""
    if source_overlap_policy not in ('error', 'drop-train', 'keep-official'):
        raise ValueError('unsupported source overlap policy')
    sources, originals = {}, defaultdict(list)
    for item in metadata:
        split = item['split']
        if split not in SPLITS or item.get('modify_type') not in MODALITIES:
            raise ValueError('unsupported AV-Deepfake1M split or modify_type')
        path = media_path(item['file'], split)
        if path in sources:
            raise ValueError('duplicate AV-Deepfake1M file: '+path)
        original = relative_path(item['original'], source_reference=True)
        sources[path] = {**item, 'original': original, 'path': path}
        if item['modify_type'] == 'real':
            originals[original].append(path)
    groups, ownership = defaultdict(list), defaultdict(set)
    for path, item in sources.items():
        if item['modify_type'] == 'real':
            source = item['original']
        else:
            original = item['original']
            parent = sources.get(media_path(original, item['split']))
            if parent is None:
                matches = originals.get(original, [])
                if len(matches) > 1:
                    raise ValueError('ambiguous genuine source: '+path)
                if matches:
                    parent = sources[matches[0]]
            if parent is None:
                source = missing_original_source(original, item['split'])
            else:
                if parent['modify_type'] != 'real' or parent['split'] != item['split']:
                    raise ValueError('fake original must be genuine and in the same split: '+path)
                source = parent['original']
        family = 'voxceleb2:'+source
        split = SPLITS[item['split']]
        ownership[family].add(split)
        groups[(split, family)].append({**item, 'source_family': family})
    for family, splits in ownership.items():
        if len(splits) < 2:
            continue
        if overlaps is not None:
            overlaps.append({'source_family':family,
                             'clips_by_split':{s:len(groups[s,family]) for s in sorted(splits)}})
        if source_overlap_policy == 'keep-official':
            continue
        if source_overlap_policy != 'drop-train' or len(splits - {'train'}) > 1:
            raise ValueError('source-family leakage: '+family)
        removed = groups.pop(('train', family))
        if exclusions is not None:
            exclusions.append({'source_family': family, 'removed_split': 'train',
                               'retained_splits': sorted(splits - {'train'}),
                               'video_ids': [row['path'] for row in removed]})
    return groups


def normalize_clip(item, duration, release):
    duration = finite(duration)
    if duration <= 0:
        raise ValueError('media duration must be positive')
    audio, visual = MODALITIES[item['modify_type']]

    def segments(key, required=False):
        values = item.get(key)
        if values is None and not required:
            return None
        if not isinstance(values, list):
            raise ValueError('full temporal annotations required: '+key)
        result = union(interval(value) for value in values)
        return [list(value) for value in result]

    edits = segments('fake_segments', required=True)
    if bool(edits) != bool(audio or visual):
        raise ValueError('modify_type disagrees with fake_segments')
    modality_edits, modality_labels, conflicts = [], [], {}
    for key, modified in (('audio_fake_segments', audio), ('visual_fake_segments', visual)):
        values = segments(key)
        if values is not None:
            if union([*edits, *values]) != union(edits):
                raise ValueError('modality segments extend outside combined fake_segments')
            if bool(values) != bool(modified):
                # The release contains conflicting type/interval annotations.
                # Mask that modality's supervision; retain the supplied values
                # for audit rather than inventing a corrected target.
                conflicts[key] = values
                modality_edits.append(None)
                modality_labels.append(None)
                continue
        modality_edits.append(([] if not modified else None) if values is None else values)
        modality_labels.append(modified)
    if all(values is not None for values in modality_edits):
        if union([*modality_edits[0], *modality_edits[1]]) != union(edits):
            raise ValueError('combined and modality-specific segments disagree')
    # Validate the supplied annotations above, then intersect with observed media.
    # Keep the exact out-of-bounds annotations for audit, including large errors.
    bounds_adjustments = {}
    def observed(values, key):
        if values is None:
            return None
        clipped = [[start, min(end,duration)] for start,end in values if start < duration]
        if clipped != values:
            bounds_adjustments[key] = {'original':item[key], 'observed':clipped,
                                      'maximum_overrun_seconds':max(end-duration for _,end in values)}
        return clipped
    edits = observed(edits,'fake_segments')
    modality_edits = [observed(values,key) for values,key in zip(modality_edits,
                        ('audio_fake_segments','visual_fake_segments'))]
    temporal_complete = not (bool(audio or visual) and not edits)
    if not temporal_complete:
        modality_edits = [None if modified else [] for modified in (audio,visual)]
    return {'video_id': item['path'], 'dataset':'AV-Deepfake1M', 'release':release,
            'path':item['path'], 'source_family':item['source_family'], 'source_sha256':None,
            'split':SPLITS[item['split']], 'duration':duration, 'label':int(audio or visual),
            'audio_label':modality_labels[0], 'visual_label':modality_labels[1], 'temporal_complete':temporal_complete,
            'annotation_bounds_adjustments':bounds_adjustments,
            'source_modify_type':item['modify_type'], 'modality_annotation_conflicts':conflicts,
            'manipulated_intervals':edits, 'audio_manipulated_intervals':modality_edits[0],
            'visual_manipulated_intervals':modality_edits[1]}


def media_duration(path):
    import av
    with av.open(str(path)) as container:
        if not container.streams.video or container.duration is None:
            raise ValueError('video stream and timestamp duration required')
        return container.duration / av.time_base


def prepare_avdeepfake1m(dataset_root, output, *, clip_limits, seed=7, test_metadata=None,
                        source_overlap_policy='error'):
    """Freeze whole source families from official train/val, optionally labeled test.

    Omitted limits mean all rows. A provided limit takes a deterministic family
    prefix. Metadata contradictions fail explicitly; no labels are invented.
    """
    expected = {'train','validation'} | ({'test'} if test_metadata else set())
    if set(clip_limits) != expected or any(n is not None and (type(n) is not int or n<1) for n in clip_limits.values()):
        raise ValueError('provide positive caps or None for exactly the labeled splits')
    if type(seed) is not int or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)
    files = {'train':_fk_paths.relocate((root/'train_metadata.json'), __file__), 'val':_fk_paths.relocate((root/'val_metadata.json'), __file__)}
    if test_metadata:
        files['test'] = _fk_paths.relocate((Path(test_metadata)), __file__).resolve(strict=True)
    metadata, digests = [], {}
    started = perf_counter()
    print('Loading metadata and resolving official source families...',file=sys.stderr,flush=True)
    for split, path in files.items():
        records = load_json(path)
        if not isinstance(records, list) or any(r.get('split') != split for r in records):
            raise ValueError('metadata file must contain only its official split: '+split)
        metadata.extend(records)
        digests[split] = sha256_file(path)
    exclusions, overlaps = [], []
    groups = index_metadata(metadata, source_overlap_policy=source_overlap_policy,
                            exclusions=exclusions, overlaps=overlaps)
    selected, counts = [], Counter()
    release = 'metadata-sha256:'+hashlib.sha256(str(sorted(digests.items())).encode()).hexdigest()
    total_processed = 0
    expected_total = (sum(len(group) for group in groups.values())
                      if all(limit is None for limit in clip_limits.values()) else None)
    for split in ('train','validation','test'):
        if split not in clip_limits:
            continue
        families = sorted((family for s,family in groups if s==split),
                          key=lambda family:hashlib.sha256(f'{seed}\0{family}'.encode()).digest())
        print(f'{split}: checking media timestamps and SHA-256...',file=sys.stderr,flush=True)
        for family in families:
            group = groups[split,family]
            if clip_limits[split] is not None and counts[split]+len(group)>clip_limits[split]:
                break
            for item in sorted(group, key=lambda r:r['path']):
                source = (_fk_paths.relocate((root/item['path']), __file__)).resolve(strict=True)
                source.relative_to(root)
                try:
                    row = normalize_clip(item, media_duration(source), release)
                except ValueError as exc:
                    raise ValueError(f"{item['path']}: {exc}") from exc
                row['source_sha256'] = sha256_file(source)
                row['source_overlap_policy'] = source_overlap_policy
                selected.append(row)
                total_processed += 1
                if total_processed % 1000 == 0:
                    progress = (f'{total_processed:,}/{expected_total:,} ({100*total_processed/expected_total:.1f}%)'
                                if expected_total else f'{total_processed:,}')
                    print(f'Prepared {progress} clips; elapsed {(perf_counter()-started)/60:.1f} min; '
                          f"last: {item['path']}",file=sys.stderr,flush=True)
            counts[split] += len(group)
        if not counts[split]:
            raise ValueError('no complete family fits the requested split cap: '+split)
    report = {'schema':'avdeepfake1m-study-v1', 'dataset':'AV-Deepfake1M',
              'metadata_sha256':digests, 'seed':seed, 'clip_limits':clip_limits,
              'source_overlap_policy':source_overlap_policy,
              'excluded_train_families':len(exclusions),
              'excluded_train_clips':sum(len(item['video_ids']) for item in exclusions),
              'overlapping_source_families':len(overlaps),
              'modality_conflict_policy':'mask conflicting modality labels/intervals; retain source annotations',
              'selected_clips_with_modality_conflicts':sum(bool(r['modality_annotation_conflicts']) for r in selected),
              'annotation_bounds_policy':'intersect annotations with observed duration; retain originals; mask temporal labels if no fake time remains',
              'selected_clips_with_bounds_adjustments':sum(bool(r['annotation_bounds_adjustments']) for r in selected),
              'sampling':'official splits; seeded whole-family prefix; no identity-disjointness claim',
              'source_reference_policy':'collapse repeated slashes in original IDs; strict relative media paths',
              'missing_original_policy':'source ID from id/video/clip/real.mp4 layout; no synthetic real rows',
              'duration_source':'PyAV container timestamps', 'study_audit':audit(selected),
              'test_labels':'SUPPLIED' if test_metadata else 'UNAVAILABLE_NOT_USED'}
    inventory = []
    test_list = _fk_paths.relocate((root/'test_files.txt'), __file__)
    if test_list.is_file() and not test_metadata:
        for name in test_list.read_text(encoding='utf-8').splitlines():
            if name.strip():
                inventory.append({'path':media_path(name.strip(),'test'),'split':'test','label':None})
        if len({r['path'] for r in inventory}) != len(inventory):
            raise ValueError('duplicate test inventory path')
        report['test_inventory_sha256'] = sha256_file(test_list)
    report['unlabeled_test_files'] = len(inventory)
    destination = _fk_paths.relocate((Path(output)), __file__)
    destination.mkdir(parents=True, exist_ok=False)
    save(_fk_paths.relocate((destination/'manifest.json'), __file__), selected)
    save(_fk_paths.relocate((destination/'source-overlap-exclusions.json'), __file__), exclusions)
    save(_fk_paths.relocate((destination/'source-overlaps.json'), __file__), overlaps)
    report['source_overlaps_sha256'] = sha256_file(_fk_paths.relocate((destination/'source-overlaps.json'), __file__))
    report['source_overlap_exclusions_sha256'] = sha256_file(_fk_paths.relocate((destination/'source-overlap-exclusions.json'), __file__))
    if inventory:
        save(_fk_paths.relocate((destination/'test-inference-files.json'), __file__), inventory)
    report['manifest_sha256'] = sha256_file(_fk_paths.relocate((destination/'manifest.json'), __file__))
    save(_fk_paths.relocate((destination/'audit.json'), __file__), report)
    return report
