"""Prepare or verify a frozen, labeled local held-out population (no model execution)."""

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

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = _fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(_fk_paths.relocate((ROOT / 'src'), __file__))]
from baselines.common.io import checked_path, digest, read, write
from baselines.common.subset import stream_rows
from deepfake_detection.data import audit

KINDS = ('real', 'visual_modified', 'audio_modified', 'both_modified')
SCHEMA = 'local-heldout-population-v1'


def progress(message):
    print('held-out: ' + message, flush=True)


def audited_manifest(path):
    path = _fk_paths.relocate((Path(path)), __file__)
    report = read(_fk_paths.relocate((path.parent / 'audit.json'), __file__))
    checksum = digest(path)
    if report.get('manifest_sha256') != checksum:
        raise ValueError('manifest checksum does not match audit: ' + str(path))
    return checksum, report


def check_separation(rows, used):
    report = audit(rows)
    audit(used)
    if not used or any(r['split'] not in ('train', 'validation') for r in used):
        raise ValueError('used population must contain train/validation rows only')
    if any(r['split'] != 'test' or r.get('official_split') != 'validation' for r in rows):
        raise ValueError('held-out rows must be logical test from official validation')
    for key in ('video_id', 'source_family', 'source_sha256'):
        selected = {r[key] for r in rows}
        if None in selected or len(selected) != len(rows):
            raise ValueError('missing or duplicate held-out ' + key)
        if selected & {r[key] for r in used}:
            raise ValueError('held-out overlaps used population: ' + key)
    return report


def verify_media(rows, dataset_root):
    root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)
    for index, row in enumerate(rows, 1):
        path = checked_path(root, row['path'])
        if digest(path) != row['source_sha256']:
            raise ValueError('media checksum mismatch: ' + row['path'])
        if index % 100 == 0 or index == len(rows):
            progress(f'verified media {index}/{len(rows)}')


def select_rows(source, used, per_type, seed):
    if per_type < 1:
        raise ValueError('clips per type must be positive')
    blocked = {key: {r[key] for r in used} for key in ('video_id', 'source_family', 'source_sha256')}
    representatives = {}
    counts = Counter()
    for index, row in enumerate(stream_rows(source), 1):
        if index % 100000 == 0:
            progress(f'scanned {index} source rows')
        if row['split'] != 'validation':
            continue
        counts['official_validation'] += 1
        if any(row[key] in blocked[key] for key in blocked):
            counts['excluded_overlap'] += 1
            continue
        if row.get('source_modify_type') not in KINDS or not row.get('source_sha256'):
            raise ValueError('eligible row has missing hash or unknown manipulation type')
        rank = (hashlib.sha256(f'{seed}\0{row["video_id"]}'.encode()).hexdigest(), row['video_id'])
        family = row['source_family']
        if family not in representatives or rank < representatives[family][0]:
            representatives[family] = (rank, row)
    counts['eligible_families'] = len(representatives)
    candidates = {kind: [] for kind in KINDS}
    seen_hashes = set()
    for _, row in sorted(representatives.values(), key=lambda item: item[0]):
        if row['source_sha256'] in seen_hashes:
            counts['excluded_duplicate_representative_content'] += 1
            continue
        seen_hashes.add(row['source_sha256'])
        candidates[row['source_modify_type']].append(row)
    available = {kind: len(rows) for kind, rows in candidates.items()}
    if any(n < per_type for n in available.values()):
        raise ValueError(f'insufficient independent candidates for {per_type} per type: {available}')
    chosen = [dict(row, official_split='validation', split='test')
              for kind in KINDS for row in candidates[kind][:per_type]]
    return sorted(chosen, key=lambda r: r['video_id']), dict(counts), available


def prepare(source_manifest, used_manifest, dataset_root, output, *, per_type=250, seed=17):
    output = _fk_paths.relocate((Path(output)), __file__)
    if output.exists():
        raise FileExistsError('output exists; use --verify or choose a new output directory')
    source_manifest, used_manifest = _fk_paths.relocate((Path(source_manifest)), __file__), _fk_paths.relocate((Path(used_manifest)), __file__)
    progress('verifying source and used manifest checksums')
    source_hash, _ = audited_manifest(source_manifest)
    used_hash, used_audit = audited_manifest(used_manifest)
    if used_audit.get('source_manifest_sha256') != source_hash:
        raise ValueError('used population was not sampled from this source manifest')
    used = read(used_manifest)
    audit(used)
    rows, selection_counts, available = select_rows(source_manifest, used, per_type, seed)
    study = check_separation(rows, used)
    verify_media(rows, dataset_root)
    # Recheck inputs after scanning, before publishing a completed population.
    if digest(source_manifest) != source_hash or digest(used_manifest) != used_hash:
        raise ValueError('input manifest changed during preparation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.heldout-', dir=output.parent) as temporary:
        stage = _fk_paths.relocate((Path(temporary) / 'population'), __file__)
        stage.mkdir()
        write(_fk_paths.relocate((stage / 'manifest.json'), __file__), rows)
        shutil.copyfile(used_manifest, _fk_paths.relocate((stage / 'used-manifest.json'), __file__))
        write(_fk_paths.relocate((stage / 'used-audit.json'), __file__), used_audit)
        report = {
            'schema': SCHEMA, 'manifest_sha256': digest(_fk_paths.relocate((stage / 'manifest.json'), __file__)),
            'source_manifest_sha256': source_hash, 'used_manifest_sha256': used_hash,
            'used_audit_sha256': digest(_fk_paths.relocate((stage / 'used-audit.json'), __file__)),
            'seed': seed, 'clips_per_type': per_type,
            'selected_counts': dict(Counter(r['source_modify_type'] for r in rows)),
            'selection_counts': selection_counts, 'available_counts': available,
            'selection': 'lowest seeded SHA256 representative per family; global content deduplication; lowest ranks per type',
            'study_audit': study, 'media_verified': len(rows),
            'separation': {'video_id_overlap': 0, 'source_family_overlap': 0, 'source_sha256_overlap': 0},
            'scope': 'local held-out from unused official validation; not official test',
            'identity_disjointness': 'not established',
            'official_test_access': 'none',
        }
        write(_fk_paths.relocate((stage / 'audit.json'), __file__), report)
        if output.exists():
            raise FileExistsError('output appeared during preparation')
        stage.rename(output)
    progress(f'prepared {len(rows)} verified clips: {output}')
    return report


def verify(output, dataset_root):
    output = _fk_paths.relocate((Path(output)), __file__)
    _, report = audited_manifest(_fk_paths.relocate((output / 'manifest.json'), __file__))
    if report.get('schema') != SCHEMA:
        raise ValueError('unsupported held-out audit schema')
    if digest(_fk_paths.relocate((output / 'used-manifest.json'), __file__)) != report['used_manifest_sha256']:
        raise ValueError('used manifest checksum mismatch')
    if digest(_fk_paths.relocate((output / 'used-audit.json'), __file__)) != report['used_audit_sha256']:
        raise ValueError('used audit checksum mismatch')
    used_audit = read(_fk_paths.relocate((output / 'used-audit.json'), __file__))
    if (used_audit['manifest_sha256'] != report['used_manifest_sha256'] or
            used_audit['source_manifest_sha256'] != report['source_manifest_sha256']):
        raise ValueError('source provenance mismatch')
    rows, used = read(_fk_paths.relocate((output / 'manifest.json'), __file__)), read(_fk_paths.relocate((output / 'used-manifest.json'), __file__))
    study = check_separation(rows, used)
    expected = {kind: report['clips_per_type'] for kind in KINDS}
    if (type(report['clips_per_type']) is not int or report['clips_per_type'] < 1 or
            dict(Counter(r['source_modify_type'] for r in rows)) != expected or
            report['selected_counts'] != expected or report['study_audit'] != study or
            report['media_verified'] != len(rows)):
        raise ValueError('held-out counts or study audit mismatch')
    verify_media(rows, dataset_root)
    progress(f'VERIFIED {len(rows)} held-out clips; zero overlap with used population')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-manifest')
    parser.add_argument('--used-manifest')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--clips-per-type', type=int, default=250)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--verify', action='store_true', help='verify an existing population and all its media')
    args = parser.parse_args()
    if not args.verify and (not args.source_manifest or not args.used_manifest):
        parser.error('preparation requires --source-manifest and --used-manifest')
    try:
        if args.verify:
            verify(args.output, args.dataset_root)
        else:
            prepare(args.source_manifest, args.used_manifest, args.dataset_root, args.output,
                    per_type=args.clips_per_type, seed=args.seed)
    except (ValueError, OSError, KeyError) as error:
        print('error: ' + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
