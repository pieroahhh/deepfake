"""Recover frozen fusion after a Windows progress-replacement failure only."""

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

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = _fk_paths.workspace_root(__file__)
REPO = _fk_paths.relocate((ROOT / 'Deepfake-Detection'), __file__)
RUN = _fk_paths.relocate((REPO / 'runs/remaining-supplement-002'), __file__)
AVROOT = _fk_paths.relocate((ROOT / 'avh-a/code'), __file__)
sys.path[:0] = [str(AVROOT), str(_fk_paths.relocate((AVROOT / 'src'), __file__)), str(_fk_paths.relocate((REPO / 'scripts'), __file__))]
from baselines.common.io import read, digest
import baselines.common.io as io
from avfakenet_progress_io import write_json
from deepfake_detection.quality_fitting import exclusive_run


def main():
    with exclusive_run(_fk_paths.relocate((RUN / 'worker.lock'), __file__)):
        if (_fk_paths.relocate((RUN / 'complete.json'), __file__)).exists():
            print('Original study already complete; nothing to recover', flush=True)
            return
        protected = read(_fk_paths.relocate((RUN / 'identity.json'), __file__))['protected']
        if any(digest(_fk_paths.relocate((Path(path)), __file__)) != sha for path, sha in protected.items()):
            raise ValueError('Protected original changed')
        for name in ('forensic', 'profiling-fk', 'robustness-fk', 'profiling-avfakenet', 'robustness-avfakenet'):
            marker = read(_fk_paths.relocate((RUN / name / 'complete.json'), __file__))
            if digest(_fk_paths.relocate((RUN / name / 'summary.json'), __file__)) != marker['summary_sha256']:
                raise ValueError('Completed summary changed: ' + name)
        for stage in ('audio', 'visual', 'audiovisual'):
            marker = read(_fk_paths.relocate((RUN / 'avtenet' / stage / 'complete.json'), __file__))
            for name, sha in marker['files'].items():
                if digest(_fk_paths.relocate((RUN / 'avtenet' / stage / name), __file__)) != sha:
                    raise ValueError('Completed expert changed: ' + stage + '/' + name)
        existing = {str(p): digest(p) for p in (_fk_paths.relocate((RUN / 'avtenet/fusion'), __file__)).glob('*.json')}
        evidence = {'wrapper_sha256': digest(_fk_paths.relocate((Path(__file__)), __file__)), 'original_runner_sha256': digest(_fk_paths.relocate((REPO / 'scripts/run_remaining_avtenet.py'), __file__)),
                    'prior_progress': read(_fk_paths.relocate((RUN / 'progress.json'), __file__)), 'started_unix': time.time(),
                    'scope': 'Only progress throttling/retries and preservation of equal existing fusion samples; original inference/checkpoints/threshold unchanged',
                    'existing_files': existing}
        write_json(_fk_paths.relocate((RUN / 'fusion-progress-recovery.json'), __file__), evidence)
        original_write = io.write
        last_progress = [0.]
        def resilient_write(path, value, **kwargs):
            path = _fk_paths.relocate((Path(path)), __file__)
            if path == _fk_paths.relocate((RUN / 'progress.json'), __file__):
                if time.monotonic() - last_progress[0] >= 2:
                    last_progress[0] = time.monotonic()
                    write_json(path, value, timeout=.2)
                    write_json(_fk_paths.relocate((RUN / 'fusion-recovery-progress.json'), __file__), value)
                    print('Fusion %s/%s' % (value.get('processed'), value.get('total')), flush=True)
                return
            if path.parent == _fk_paths.relocate((RUN / 'avtenet/fusion'), __file__) and len(path.stem) == 64 and path.exists():
                old = read(path)
                if old['item'] != value['item'] or len(old['samples']) != len(value['samples']):
                    raise ValueError('Existing fusion sample identity changed')
                for a, b in zip(old['samples'], value['samples']):
                    pa, pb = a['prediction'], b['prediction']
                    if any(pa.get(k) != pb.get(k) for k in ('video_id', 'label', 'status', 'prediction', 'usable_windows')):
                        raise ValueError('Existing fusion decision changed')
                    if (pa.get('score') is None) != (pb.get('score') is None) or (pa.get('score') is not None and abs(pa['score'] - pb['score']) > 5e-6):
                        raise ValueError('Existing fusion score changed')
                value['samples'][:] = old['samples']
                return
            if path.exists() and read(path) == value:
                return
            return original_write(path, value, **kwargs)
        io.write = resilient_write
        script = _fk_paths.relocate((REPO / 'scripts/run_remaining_avtenet.py'), __file__)
        spec = importlib.util.spec_from_file_location('original_frozen_fusion', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.argv = [str(script), '--root', str(AVROOT), '--study', str(RUN), '--input-root', str(ROOT),
                    '--stage', 'fusion', '--assets', str(_fk_paths.relocate((AVROOT / 'runs/worker-c/assets.local.json'), __file__))]
        module.main()
        if any(digest(_fk_paths.relocate((Path(path)), __file__)) != sha for path, sha in protected.items()):
            raise ValueError('Protected original changed during recovery')
        if any(digest(_fk_paths.relocate((Path(path)), __file__)) != sha for path, sha in existing.items()):
            raise ValueError('Retained partial fusion file was overwritten')
        write_json(_fk_paths.relocate((RUN / 'complete.json'), __file__), {'historical_inputs_unchanged': True, 'aggregation_source': 'remaining-supplement-001',
                   'transfer': 'DEFERRED_BY_USER', 'avtenet_summary_sha256': digest(_fk_paths.relocate((RUN / 'avtenet/summary.json'), __file__)),
                   'fusion_recovery_sha256': digest(_fk_paths.relocate((RUN / 'fusion-progress-recovery.json'), __file__))})
        write_json(_fk_paths.relocate((RUN / 'progress.json'), __file__), {'status': 'FINISHED', 'stage': 'complete', 'processed': 587, 'total': 587, 'updated_unix': time.time()})
        write_json(_fk_paths.relocate((RUN / 'fusion-recovery-complete.json'), __file__), {'existing_files_unchanged': True, 'summary_sha256': digest(_fk_paths.relocate((RUN / 'avtenet/summary.json'), __file__))})
        print('FINISHED: fusion recovered; original files retained', flush=True)


if __name__ == '__main__':
    main()
