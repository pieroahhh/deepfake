"""Attempt public pretrained downloads; report every failure without fake readiness."""

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
import json
from pathlib import Path
import sys

sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'src'), __file__)))
from deepfake_detection.asset_download import acquire_assets

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True,help='new artifact directory, e.g. checkpoints/pretrained-001')
parser.add_argument('--assets',nargs='+',choices=['ast','wav2vec2_base','efficientnet_b0','silero','yunet','factorized_vivit','avhubert_native'],
                    default=['ast','wav2vec2_base','efficientnet_b0','silero','yunet','factorized_vivit','avhubert_native'])
args=parser.parse_args()
report=acquire_assets(args.output,args.assets)
print(json.dumps(report,indent=2))
raise SystemExit(2 if any(value['status']=='FAILED' for value in report['assets'].values()) else 0)
