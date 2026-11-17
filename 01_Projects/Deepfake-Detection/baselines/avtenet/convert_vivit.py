"""Convert explicitly mapped Scenic arrays and require independent reference parity.

The reference must come from the pinned official Scenic backend on the SAME
checkpoint. This command cannot authenticate an operator-supplied reference.
It never downloads substitutes when the official checkpoint is inaccessible.
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

import argparse
from pathlib import Path
import numpy as np
import torch

from baselines.common.io import read,write,digest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('arrays','mapping','config','reference','reference-manifest','output'):
        parser.add_argument('--'+name,required=True)
    args=parser.parse_args(argv)
    from deepfake_detection.checkpoint_conversion import convert_factorized_arrays
    from deepfake_detection.backbones import FactorizedViViT
    proof=read(args.reference_manifest)
    if proof.get('backend')!='official-scenic-factorized-vivit' or len(proof.get('revision',''))!=40:
        raise ValueError('reference requires pinned official Scenic backend provenance')
    if digest(args.reference)!=proof.get('reference_sha256') or digest(args.arrays)!=proof.get('arrays_sha256'):
        raise ValueError('exported arrays/reference checksum mismatch')
    config=read(args.config)
    with np.load(_fk_paths.relocate_value((args.arrays), __file__),allow_pickle=False) as payload:
        arrays={key:payload[key] for key in payload.files}
    manifest=convert_factorized_arrays(arrays,read(args.mapping),config,args.output,
                                       source=proof['source'],revision=proof['revision'],
                                       source_checkpoint_sha256=proof['source_checkpoint_sha256'])
    root=_fk_paths.relocate((Path(args.output)), __file__)
    model=FactorizedViViT.from_local(root,manifest,require_pretrained=False).eval()
    torch.set_num_threads(2)
    with np.load(_fk_paths.relocate_value((args.reference), __file__),allow_pickle=False) as payload:
        inputs,expected=payload['inputs'],payload['features']
    if len(inputs)<2 or expected.shape!=(len(inputs),768) or not np.isfinite(inputs).all() or not np.isfinite(expected).all():
        raise ValueError('reference needs at least two finite full-size inputs and penultimate vectors')
    with torch.inference_mode():
        actual=np.concatenate([model(torch.from_numpy(value[None]).float()).numpy() for value in inputs])
    passed=bool(np.allclose(actual,expected,rtol=1e-4,atol=1e-5))
    parity={'status':'PASSED' if passed else 'FAILED','reference_revision':proof['revision'],
            'reference_manifest':proof,'maximum_absolute_error':float(np.max(np.abs(actual-expected))),
            'rtol':1e-4,'atol':1e-5,'samples':len(inputs)}
    write(_fk_paths.relocate((root/'parity.json'), __file__),parity)
    manifest['parity']=parity
    manifest['files']['parity.json']=digest(_fk_paths.relocate((root/'parity.json'), __file__))
    if passed:manifest['weights_status']='pretrained'
    write(_fk_paths.relocate((root/'artifact.json'), __file__),manifest,replace=True)
    if not passed:raise ValueError('conversion parity failed; converted weights remain unverified')
    print('Verified conversion:',root)


if __name__=='__main__':main()
