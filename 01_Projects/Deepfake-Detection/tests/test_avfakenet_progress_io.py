
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

import json
import sys
import tempfile
import os
import ctypes
from ctypes import wintypes
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'scripts'), __file__)))
from avfakenet_progress_io import write_json
import avfakenet_progress_io as io


class ProgressIOTests(unittest.TestCase):
    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_actual_windows_reader_without_delete_sharing(self):
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        create=kernel.CreateFileW
        create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
        create.restype=wintypes.HANDLE
        close=kernel.CloseHandle;close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
        with tempfile.TemporaryDirectory() as directory:
            path=_fk_paths.relocate((Path(directory)/'progress.json'), __file__);path.write_text('{"before":true}')
            handle=create(str(path),0x80000000,1|2,None,3,0x80,None)
            self.assertNotEqual(handle,ctypes.c_void_p(-1).value)
            try:
                self.assertFalse(write_json(path,{'after':True},timeout=.1))
                self.assertEqual(json.loads(path.read_text()),{'before':True})
            finally:close(handle)
            self.assertTrue(write_json(path,{'after':True}))
            self.assertEqual(json.loads(path.read_text()),{'after':True})

    def test_transient_lock_retries_and_preserves_atomic_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path=_fk_paths.relocate((Path(directory)/'progress.json'), __file__);path.write_text('{"old":true}')
            replace=io.os.replace;attempts=[]
            def flaky(source,target):
                attempts.append(source)
                if len(attempts)<3:
                    self.assertEqual(json.loads(path.read_text()),{'old':True})
                    raise PermissionError(5,'temporarily locked')
                return replace(source,target)
            with patch.object(io.os,'replace',side_effect=flaky):
                self.assertTrue(write_json(path,{'new':True}))
            self.assertEqual(len(attempts),3)
            self.assertEqual(json.loads(path.read_text()),{'new':True})
            self.assertEqual(list(_fk_paths.relocate((Path(directory)), __file__).glob('*.tmp')),[])

    def test_persistent_progress_lock_does_not_stop_work_and_next_update_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path=_fk_paths.relocate((Path(directory)/'progress.json'), __file__);path.write_text('{"processed":4}')
            with patch.object(io.os,'replace',side_effect=PermissionError(5,'locked')):
                self.assertFalse(write_json(path,{'processed':5},timeout=0))
            self.assertEqual(json.loads(path.read_text()),{'processed':4})
            self.assertTrue(write_json(path,{'processed':6}))
            self.assertEqual(json.loads(path.read_text()),{'processed':6})
            self.assertEqual(list(_fk_paths.relocate((Path(directory)), __file__).glob('*.tmp')),[])

    def test_persistent_nonprogress_failure_is_not_suppressed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=_fk_paths.relocate((Path(directory)/'record.json'), __file__)
            with patch.object(io.os,'replace',side_effect=PermissionError(5,'locked')):
                with self.assertRaises(PermissionError):write_json(path,{'ok':True},timeout=0)
            self.assertFalse(path.exists())


if __name__=='__main__':unittest.main()
