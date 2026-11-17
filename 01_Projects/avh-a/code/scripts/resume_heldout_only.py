"""Resume only unfinished frozen test work; completed training is read-only."""

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

import sys,sqlite3
from pathlib import Path
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(_fk_paths.relocate((ROOT/'scripts'), __file__))]
from baselines.common.io import read,write,digest
from deepfake_detection.quality_fitting import exclusive_run
from auto_after_avhubert import AUTO,RUN,command,state
from run_avtenet_frozen_test import validate_plan,check_complete
EVIDENCE=_fk_paths.relocate((ROOT.parent/'live-decode-prefetch-001'), __file__)


def verify_sealed():
    for item in read(_fk_paths.relocate((EVIDENCE/'completed-before.json'), __file__)):
        if digest(_fk_paths.relocate((ROOT/item['path']), __file__))!=item['sha256']:
            raise ValueError('Completed output changed: '+item['path'])


def verify_saved_features():
    paths=[_fk_paths.relocate((EVIDENCE/'paused-features.sqlite'), __file__),_fk_paths.relocate((AUTO/'heldout/audiovisual/features.sqlite'), __file__)]
    connections=[sqlite3.connect(p.resolve().as_uri()+'?mode=ro',uri=True) for p in paths]
    count=0
    try:
        for row in connections[0].execute('SELECT * FROM features'):
            current=connections[1].execute('SELECT * FROM features WHERE video_id=? AND start=? AND end=?',row[:3]).fetchone()
            if row!=current:raise ValueError('Previously saved audiovisual feature changed')
            count+=1
    finally:
        for conn in connections:conn.close()
    return count


def main():
    with exclusive_run(_fk_paths.relocate((AUTO/'controller.lock'), __file__)):
        try:
            verify_sealed();preserved=verify_saved_features()
            planpath=_fk_paths.relocate((AUTO/'test-plan.json'), __file__);plan=read(planpath);validate_plan(plan)
            context={'schema':'avtenet-frozen-heldout-v1','plan_sha256':digest(planpath)}
            for stage in ('audio','visual'):
                if not check_complete(_fk_paths.relocate((AUTO/'heldout'), __file__)/stage,{**context,'stage':stage}):
                    raise ValueError('Expected completed expert: '+stage)
            benchmark=read(_fk_paths.relocate((EVIDENCE/'benchmark.json'), __file__))
            if not benchmark['parity_passed']:raise ValueError('Decode parity not verified')
            enabled=benchmark['enable_recommended']
            write(_fk_paths.relocate((EVIDENCE/'resume-decision.json'), __file__),{'prefetch_video_decode':enabled,
                'preserved_feature_rows':preserved,'benchmark':benchmark},replace=True)
            relative=lambda path:_fk_paths.relocate((Path(path)), __file__).relative_to(ROOT).as_posix()
            for stage in ('audiovisual','evaluate'):
                if check_complete(_fk_paths.relocate((AUTO/'heldout'), __file__)/stage,{**context,'stage':stage}):continue
                container=stage=='audiovisual'
                convert=lambda path:'/workspace/'+relative(path) if container else str(path)
                worker='native_frozen_worker.py' if container else 'run_avtenet_frozen_test.py'
                argv=[convert(_fk_paths.relocate((ROOT/'scripts'/worker), __file__)),'--plan',convert(planpath),
                    '--stage',stage,'--dataset-root','/dataset' if container else str(_fk_paths.relocate((ROOT.parent/'dataset'), __file__)),
                    '--assets',convert(_fk_paths.relocate((ROOT/'runs/worker-c'/('assets.docker.local.json' if container else 'assets.local.json')), __file__))]
                if container and enabled:argv.append('--prefetch-video-decode')
                command('heldout_'+stage,argv,gpu=True,docker=container)
                verify_sealed();verify_saved_features()
            final=read(_fk_paths.relocate((AUTO/'heldout/evaluate/report.json'), __file__))
            if final['metrics']['videos']!=1000:raise ValueError('Incomplete held-out report')
            report=read(_fk_paths.relocate((RUN/'evaluation/report.json'), __file__))
            summary={'status':'FINISHED','validation':report['metrics'],'heldout':final['metrics'],
                'validation_return':str(_fk_paths.relocate((ROOT.parent/'return-c-validation'), __file__)),
                'heldout_report':str(_fk_paths.relocate((AUTO/'heldout/evaluate/report.json'), __file__)),
                'heldout_predictions':str(_fk_paths.relocate((AUTO/'heldout/evaluate/predictions.jsonl'), __file__))}
            if (_fk_paths.relocate((AUTO/'summary.json'), __file__)).exists():
                if read(_fk_paths.relocate((AUTO/'summary.json'), __file__))!=summary:raise ValueError('Existing final summary differs')
            else:write(_fk_paths.relocate((AUTO/'summary.json'), __file__),summary)
            write(_fk_paths.relocate((EVIDENCE/'completed-after.json'), __file__),{'all_completed_checksums_match':True,
                'previous_feature_rows_preserved':verify_saved_features()},replace=True)
            state('FINISHED','all_stages_complete',summary=str(_fk_paths.relocate((AUTO/'summary.json'), __file__)))
        except Exception as error:
            state('PAUSED' if (_fk_paths.relocate((RUN/'pause.request'), __file__)).exists() else 'FAILED','stopped',error=str(error))
            raise


if __name__=='__main__':main()
