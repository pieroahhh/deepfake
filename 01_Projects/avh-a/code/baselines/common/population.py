"""Portable audited metadata indices. Media is opened lazily during training."""

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

from collections import Counter
from contextlib import closing
from pathlib import Path
import sqlite3

from .io import read, write, digest, fingerprint, checked_path


def prepare(manifest, output, model, *, limit=None, seed=7):
    from deepfake_detection.data import audit
    manifest = _fk_paths.relocate((Path(manifest)), __file__)
    if manifest.is_dir():
        manifest = _fk_paths.relocate((manifest / 'manifest.json'), __file__)
    rows = read(manifest)
    report = audit(rows)
    if report['missing_source_hashes']:
        raise ValueError('population must contain source SHA-256 for every video')
    neighbor = _fk_paths.relocate((manifest.parent / 'audit.json'), __file__)
    source_hash = digest(manifest)
    if neighbor.is_file() and read(neighbor).get('manifest_sha256') != source_hash:
        raise ValueError('population manifest differs from its audit')
    if limit is not None:
        if limit < 1:
            raise ValueError('limit per split must be positive')
        selected = []
        for split in ('train', 'validation', 'test'):
            candidates = [r for r in rows if r['split'] == split]
            candidates.sort(key=lambda r: fingerprint([seed, r['video_id']]))
            selected.extend(candidates[:limit])
        rows = selected
    for row in rows:
        checked_path('.', row['path'])
    output = _fk_paths.relocate((Path(output)), __file__)
    output.mkdir(parents=True, exist_ok=False)
    import json
    with closing(sqlite3.connect(_fk_paths.relocate((output / 'index.sqlite'), __file__))) as connection, connection:
        connection.execute('CREATE TABLE clips (video_id TEXT PRIMARY KEY, split TEXT NOT NULL, row_json TEXT NOT NULL)')
        connection.executemany('INSERT INTO clips VALUES (?,?,?)',
                              ((r['video_id'], r['split'], json.dumps(r, allow_nan=False)) for r in rows))
        connection.execute('CREATE INDEX split_order ON clips(split,video_id)')
    metadata = {'schema': 'baseline-population-v1', 'model': model,
                'population_sha256': source_hash, 'index_sha256': digest(_fk_paths.relocate((output / 'index.sqlite'), __file__)),
                'source_overlap_policy': report['source_overlap_policy'], 'population_audit': report,
                'counts': dict(Counter(r['split'] for r in rows)), 'limit_per_split': limit,
                'seed': seed, 'selection': 'all official rows' if limit is None else 'explicit smoke subset'}
    write(_fk_paths.relocate((output / 'index.json'), __file__), metadata)
    return metadata


class Population:
    def __init__(self, directory, dataset_root, model):
        self.directory = _fk_paths.relocate((Path(directory)), __file__)
        self.metadata = read(_fk_paths.relocate((self.directory / 'index.json'), __file__))
        if self.metadata['schema'] != 'baseline-population-v1' or self.metadata['model'] != model:
            raise ValueError('index schema or comparator mismatch')
        if digest(_fk_paths.relocate((self.directory / 'index.sqlite'), __file__)) != self.metadata['index_sha256']:
            raise ValueError('index checksum mismatch')
        self.root = _fk_paths.relocate((Path(dataset_root)), __file__).resolve(strict=True)

    def rows(self, split):
        if split not in ('train', 'validation', 'test'):
            raise ValueError('unknown split')
        import json
        # URI mode=ro also prevents accidental creation/mutation of an index.
        with closing(sqlite3.connect((_fk_paths.relocate((self.directory / 'index.sqlite'), __file__)).resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            for (payload,) in conn.execute('SELECT row_json FROM clips WHERE split=? ORDER BY video_id', (split,)):
                yield json.loads(payload)

    def path(self, row):
        path = checked_path(self.root, row['path'])
        if digest(path) != row['source_sha256']:
            raise ValueError('source checksum mismatch: ' + row['video_id'])
        return path


def windows(duration):
    """Complete 2-second windows, stride 1, with one end-aligned final view."""
    if duration < 2:
        return []
    count = int(duration - 2 + 1e-9) + 1
    starts = [float(i) for i in range(count)]
    if duration - 2 - starts[-1] > 1e-6:
        starts.append(duration - 2)
    return [(start, start + 2) for start in starts]


def target(row, modality, start=None, end=None):
    key = 'label' if modality == 'audiovisual' else modality + '_label'
    label = row[key]
    if label is None or start is None:
        return label
    if label == 0:
        return 0
    intervals = row.get('manipulated_intervals' if modality == 'audiovisual' else modality + '_manipulated_intervals')
    if intervals is None or (modality == 'audiovisual' and not row['temporal_complete']):
        return None
    if end is None:
        return int(any(a <= start < b for a, b in intervals))
    return int(any(max(a, start) < min(b, end) for a, b in intervals))
