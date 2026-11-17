
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

import sys,unittest,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
import run_avfakenet_deadline as dispatcher
import scale_avfakenet_prepare as scale


class DeadlineDispatchTests(unittest.TestCase):
    def test_only_next_training_command_changes_and_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as directory:
            root=_fk_paths.relocate((Path(directory)), __file__);run=_fk_paths.relocate((root/'runs/avfakenet-normalized-pilot-001'), __file__);run.mkdir(parents=True)
            original_base_write=scale.base.write;argv=list(sys.argv)
            command=[sys.executable,'-u',str(_fk_paths.relocate((root/'scripts/run_avfakenet_normalized_pilot.py'), __file__)),
                     '--sanity','existing-sanity','--bundle','existing-bundle','--dataset-root','existing-data',
                     '--output',str(run),'--resume']
            try:
                with patch.object(dispatcher,'ROOT',root),patch.object(scale.subprocess,'run') as called:
                    with patch.object(scale,'main',side_effect=lambda:scale.subprocess.run(command,cwd=root,check=True)):
                        dispatcher.main()
                    forwarded=called.call_args.args[0]
                    self.assertEqual(forwarded[2],str(_fk_paths.relocate((root/'scripts/run_avfakenet_one_epoch.py'), __file__)))
                    self.assertEqual(forwarded[forwarded.index('--output')+1],str(_fk_paths.relocate((root/'runs/avfakenet-one-epoch-001'), __file__)))
                    self.assertEqual(forwarded[forwarded.index('--prepared-cache')+1],str(_fk_paths.relocate((run/'prepared'), __file__)))
                    self.assertEqual(forwarded[forwarded.index('--dataset-root')+1],'existing-data')
                    self.assertEqual(forwarded[forwarded.index('--sanity')+1],'existing-sanity')
            finally:scale.base.write=original_base_write;sys.argv=argv


if __name__=='__main__':unittest.main()
