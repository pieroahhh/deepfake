"""Download public backbone candidates and emit a machine-local comparator registry."""

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
from pathlib import Path

from baselines.common.io import read,write


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='new checkpoint directory')
    parser.add_argument('--avhubert-repository',help='pinned official checkout in the Linux worker')
    args=parser.parse_args(argv)
    from deepfake_detection.asset_download import acquire_assets
    report=acquire_assets(args.output,['ast','factorized_vivit','avhubert_native'])
    native=read(_fk_paths.relocate((Path(args.output)/'assets.local.json'), __file__))
    result=read(_fk_paths.relocate((Path(__file__).parent/'configs/assets.example.json'), __file__))
    result['experts']['audio']=native['artifacts']['ast']
    result['experts']['audiovisual'].update({k:v for k,v in native['avhubert'].items() if k in ('commit','checkpoint','checkpoint_sha256')})
    result['experts']['audiovisual']['repository']=args.avhubert_repository
    write(_fk_paths.relocate((Path(args.output)/'baseline-assets.local.json'), __file__),result)
    print('Registry:',_fk_paths.relocate((Path(args.output)/'baseline-assets.local.json'), __file__))
    print('ViViT still requires conversion and independently recorded reference parity.')
    for name,status in report['assets'].items():print(name,status)
    return 2 if any(a['status']=='FAILED' for a in report['assets'].values()) else 0


if __name__=='__main__':raise SystemExit(main())
