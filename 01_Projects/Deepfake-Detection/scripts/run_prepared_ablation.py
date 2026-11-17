"""Explicitly run one prepared variant only after held-out completion."""

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

import sys,argparse
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',required=True)
    p.add_argument('--preflight-only',action='store_true');args=p.parse_args()
    package=_fk_paths.relocate((ROOT/'experiments/remaining-001'), __file__);plan=read(_fk_paths.relocate((package/'plan.json'), __file__))
    for relative,sha in read(_fk_paths.relocate((package/'SHA256SUMS.json'), __file__)).items():
        if digest(_fk_paths.relocate((package/relative), __file__))!=sha:raise ValueError('Prepared protocol changed; version it instead')
    candidates=[r for r in plan['ablation_runs'] if r['variant']==args.variant]
    if len(candidates)!=1:raise ValueError('Unknown prepared variant')
    selected=candidates[0];manifest=_fk_paths.relocate((Path(plan['training_cache_manifest'])), __file__);data=read(manifest)
    from deepfake_detection.infrastructure import CachedVideoPopulation
    for split in ('train','validation'):
        if not data[split]:raise ValueError('Empty training/validation population')
        _=CachedVideoPopulation(data[split],manifest.parent)[0]
    if args.preflight_only:
        write_json(_fk_paths.relocate((package/'native-cache-preflight.json'), __file__),{'status':'PASSED','manifest_sha256':digest(manifest),
            'verified_sample_cache_per_split':True,'counts':{split:len(data[split]) for split in ('train','validation')},
            'scope':'Read-only first-cache probe, not a full cache audit or training run','gpu_work_started':False})
    print('Native-user manifest and first verified cache per split readable.',flush=True)
    if args.preflight_only:return
    if not _fk_paths.relocate((Path(plan['active_heldout_must_complete_first'])), __file__).exists():raise ValueError('Held-out workflow still active; do not compete for GPU')
    import torch
    if not torch.cuda.is_available():raise ValueError('CUDA unavailable')
    destination=_fk_paths.relocate((ROOT/'runs/remaining-ablations-001'/args.variant), __file__)
    destination.mkdir(parents=True,exist_ok=False)
    grid=read(_fk_paths.relocate((package/selected['calibration_grid']), __file__))
    import deepfake_detection.calibration as calibration
    original=calibration.select_decision
    def constrained(model,validation_rows,grid_argument=None):
        return original(model,validation_rows,grid=grid)
    calibration.select_decision=constrained
    write_json(_fk_paths.relocate((destination/'prepared-run.json'), __file__),{'variant':args.variant,'plan_sha256':digest(_fk_paths.relocate((package/'plan.json'), __file__)),
        'manifest_sha256':digest(manifest),'epochs':selected['epochs'],'grid':grid,'scope':'exploratory retrained ablation; no test access'})
    from deepfake_detection.infrastructure import train_fk_manifest
    result=train_fk_manifest(manifest,_fk_paths.relocate((package/selected['model_config']), __file__),_fk_paths.relocate((package/'training.json'), __file__),_fk_paths.relocate((destination/'attempt-001'), __file__),
        epochs=selected['epochs'],calibrate=True,device='cuda')
    decision=result['validation_selection']['decision']
    for key,values in grid.items():
        if len(values)==1 and decision[key]!=values[0]:raise ValueError('Calibration violated the fixed ablation parameter')
    write_json(_fk_paths.relocate((destination/'completed.json'), __file__),result)


if __name__=='__main__':
    try:main()
    except Exception as error:
        if '--preflight-only' in sys.argv:
            write_json(_fk_paths.relocate((ROOT/'experiments/remaining-001/native-cache-preflight.json'), __file__),{'status':'FAILED','error':str(error),'gpu_work_started':False})
        raise
