"""Inspect retained result schemas without loading models or reference-paper data."""

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

import json
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)


def describe(value, depth=0):
    if depth > 2:
        return list(value)[:12] if isinstance(value, dict) else type(value).__name__
    if isinstance(value, dict):
        return {k: describe(v, depth + 1) for k, v in list(value.items())[:14]}
    if isinstance(value, list):
        return {'length': len(value), 'first': describe(value[0], depth + 1) if value else None}
    return value


sources = [
    _fk_paths.relocate((ROOT / 'fk-heldout-copy.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/three-model-heldout-002/comparison.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/three-model-no-training-001/analysis.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/remaining-offline-001/bootstrap/metrics.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/remaining-offline-001/localization/localization-examples.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/remaining-ablation-evaluation-001/comparison/metrics.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/remaining-supplement-001/aggregation/comparison/metrics.json'), __file__),
    _fk_paths.relocate((REPO / 'runs/remaining-supplement-002/forensic/summary.json'), __file__),
]
for p in sources:
    print('\nSOURCE', str(p.relative_to(ROOT)))
    if not p.exists():
        print('MISSING')
        continue
    value = json.loads(p.read_text(encoding='utf-8'))
    print(json.dumps(describe(value), ensure_ascii=True)[:7500])
for folder in ('robustness-fk', 'robustness-avfakenet', 'avtenet/fusion'):
    directory = _fk_paths.relocate((REPO / 'runs/remaining-supplement-002' / folder), __file__)
    for p in directory.glob('*.json'):
        value = json.loads(p.read_text(encoding='utf-8'))
        if 'row' in value or 'item' in value:
            print('\nRAW', folder, json.dumps(describe(value), ensure_ascii=True)[:3500])
            break
