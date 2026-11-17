"""Continue only pending raw studies after proving historical FK precision parity."""

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

import sys,time,gc
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
import run_remaining_supplement as study
from baselines.common.io import read,digest
from avfakenet_progress_io import write_json
RUN=_fk_paths.relocate((ROOT/'runs/remaining-supplement-002'), __file__);RUN.mkdir(exist_ok=True)


def main():
    import torch
    from deepfake_detection.quality_fitting import exclusive_run
    with exclusive_run(_fk_paths.relocate((RUN/'worker.lock'), __file__)):
        report_path=_fk_paths.relocate((ROOT/'runs/remaining-fk-parity-diagnosis-002/report.json'), __file__)
        done=read(_fk_paths.relocate((report_path.parent/'complete.json'), __file__))
        if digest(report_path)!=done['report_sha256']:raise ValueError('Precision diagnostic changed')
        report=read(report_path)
        if report['completed']!=4:raise ValueError('Incomplete precision diagnostic')
        for row in report['results']:
            if any(row.get('feature_max_abs_error',{}).values()):raise ValueError('Raw/cached features differ')
            if row['raw']['prediction']!=row['original']['prediction']:raise ValueError('Diagnostic predictions differ')
            if abs(row['raw']['score']-row['original']['score'])>5e-6:raise ValueError('Diagnostic scores differ')
        if not (_fk_paths.relocate((ROOT/'runs/remaining-supplement-001/aggregation/complete.json'), __file__)).exists():raise ValueError('Aggregation not complete')
        protected=read(_fk_paths.relocate((ROOT/'runs/remaining-supplement-001/identity.json'), __file__))['protected']
        if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Historical inputs changed')
        identity={'predecessor':'remaining-supplement-001','diagnostic_sha256':digest(report_path),'protected':protected,
            'script_sha256':digest(_fk_paths.relocate((Path(__file__)), __file__)),'study_helper_sha256':digest(_fk_paths.relocate((ROOT/'scripts/run_remaining_supplement.py'), __file__)),
            'scope':'Pending raw studies only; completed aggregation and earlier results retained'}
        if (_fk_paths.relocate((RUN/'identity.json'), __file__)).exists() and read(_fk_paths.relocate((RUN/'identity.json'), __file__))!=identity:raise ValueError('Continuation code changed')
        write_json(_fk_paths.relocate((RUN/'identity.json'), __file__),identity)
        study.RUN=RUN
        write_json(_fk_paths.relocate((ROOT/'runs/remaining-supplement-active.json'), __file__),{'folder':RUN.name,'predecessor':'remaining-supplement-001'})
        torch.set_num_threads(2)
        try:
            study.progress('loading_frozen_fk');fk=study.FrozenFK()
            study.parity(fk,'fk')
            study.forensic(fk);study.profiling(fk);study.raw_study(fk)
            del fk;gc.collect();torch.cuda.empty_cache()
            # AVFakeNet's retained pilot used explicit TF32=False.
            torch.backends.cudnn.allow_tf32=False
            from remaining_avfakenet import FrozenAVFakeNet
            model=FrozenAVFakeNet(ROOT);study.parity(model,'avfakenet')
            study.profiling(model,'avfakenet');study.raw_study(model,'avfakenet')
            del model;gc.collect();torch.cuda.empty_cache()
            study.avtenet()
            if any(digest(_fk_paths.relocate((Path(path)), __file__))!=sha for path,sha in protected.items()):raise ValueError('Historical inputs changed')
            write_json(_fk_paths.relocate((RUN/'complete.json'), __file__),{'historical_inputs_unchanged':True,'aggregation_source':'remaining-supplement-001','transfer':'DEFERRED_BY_USER'})
            study.progress('complete',status='FINISHED')
        except Exception as error:study.progress('error',status='FAILED',error=str(error));raise


if __name__=='__main__':
    log=(_fk_paths.relocate((RUN/('worker-'+str(time.time_ns())+'.log')), __file__)).open('a',encoding='utf-8',buffering=1)
    sys.stdout=log;sys.stderr=log
    main()
