"""Resume the verified one-epoch runner with input I/O overlap only."""

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

import os,sys,time
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
os.environ['HF_HOME']=str(_fk_paths.relocate((ROOT/'.cache/huggingface'), __file__))
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json


def main():
    run=_fk_paths.relocate((ROOT/'runs/avfakenet-one-epoch-001'), __file__)
    evidence=_fk_paths.relocate((ROOT/'runs/avfakenet-training-prefetch-benchmark-001/report.json'), __file__);report=read(evidence)
    candidate=report['arms']['fast'];baseline=report['arms']['baseline']
    if not candidate['bitwise_logits_identical'] or not candidate['all_final_weights_bitwise_identical'] or candidate['seconds']>=baseline['seconds']:
        raise ValueError('Prefetch equivalence/speed checks failed')
    identity=read(_fk_paths.relocate((run/'identity.json'), __file__))
    for relative,sha in identity['source'].items():
        if digest(_fk_paths.relocate((ROOT/relative), __file__))!=sha:raise ValueError('Original runner changed: '+relative)
    from avfakenet_training_fastpath import input_class
    import run_avfakenet_one_epoch as pilot
    pilot.Inputs=input_class(pilot.Inputs,pilot.balanced_epoch)
    write_json(_fk_paths.relocate((run/('input-prefetch-'+str(time.time_ns())+'.json')), __file__),{
        'policy':'One CPU input worker, one batch ahead, 1024 read-only memmaps; original model/loss unchanged',
        'evidence_sha256':digest(evidence),'original_runner_identity':identity,
        'sources':{str(path.relative_to(ROOT)):digest(path) for path in [_fk_paths.relocate((Path(__file__)), __file__),_fk_paths.relocate((ROOT/'scripts/avfakenet_training_fastpath.py'), __file__)]}})
    sys.argv=[sys.argv[0],'--sanity',str(_fk_paths.relocate((ROOT/'runs/avfakenet-normalized-full-tiny-004'), __file__)),
        '--bundle',r'C:\Users\teeku\Downloads\LANDrop\return-b-recovery-probe-inputs-001',
        '--dataset-root',str(_fk_paths.relocate((ROOT.parent/'avh-a/dataset'), __file__)),'--output',str(run),
        '--prepared-cache',str(_fk_paths.relocate((ROOT/'runs/avfakenet-machine-b-prepared-001'), __file__)),'--resume']
    pilot.main()


if __name__=='__main__':main()
