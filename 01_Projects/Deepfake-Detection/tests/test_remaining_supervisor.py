
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

import unittest,sys
from pathlib import Path
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
from supervise_remaining import may_retry


class SupervisorTests(unittest.TestCase):
    def test_retry_only_unexplained_exit_with_limit(self):
        self.assertTrue(may_retry(3221225477,{'status':'RUNNING','updated_unix':12},10,1))
        self.assertFalse(may_retry(1,{'status':'FAILED','updated_unix':12},10,1))
        self.assertFalse(may_retry(1,{'status':'PAUSED','updated_unix':12},10,1))
        self.assertFalse(may_retry(3221225786,{},10,1))
        self.assertFalse(may_retry(3221225477,{},10,3))


if __name__=='__main__':unittest.main()
