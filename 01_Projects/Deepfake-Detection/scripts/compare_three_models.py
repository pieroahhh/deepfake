"""Compare identical validation IDs with full and common-assessed coverage."""

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
import sys
from itertools import combinations
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.io import read,write,digest
from baselines.common.metrics import read_predictions,metrics,compare


def compare_models(manifest,paths):
    expected={r['video_id']:r['label'] for r in read(manifest) if r['split']=='validation'}
    models={name:read_predictions(path) for name,path in paths.items()}
    for name,rows in models.items():
        mapping={r['video_id']:r['label'] for r in rows}
        if len(mapping)!=len(rows) or mapping!=expected:raise ValueError(name+' predictions do not cover the same validation population/labels')
    common=set(expected)
    for rows in models.values():common&={r['video_id'] for r in rows if r['prediction'] is not None}
    return {'schema':'three-model-subset-comparison-v1','population_sha256':digest(manifest),
        'scope':'shared subset validation; one seed; declared unequal epoch budgets; AVTENet joint-ViViT adaptation',
        'full_population':{name:metrics(rows) for name,rows in models.items()},
        'pairwise':{a+' vs '+b:compare(models[a],models[b]) for a,b in combinations(models,2)},
        'all_three_assessed':{'videos':len(common),'metrics':{name:metrics(r for r in rows if r['video_id'] in common) for name,rows in models.items()}}}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('manifest','fk','avfakenet','avtenet','output'):p.add_argument('--'+flag,required=True)
    args=p.parse_args();write(args.output,compare_models(args.manifest,{'FK':args.fk,'AVFakeNet':args.avfakenet,'AVTENet adaptation':args.avtenet}))
