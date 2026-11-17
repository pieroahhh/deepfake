
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

import sys,unittest,sqlite3,tempfile,hashlib
from pathlib import Path
import numpy as np
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(_fk_paths.relocate((ROOT/'scripts'), __file__)),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(ROOT)]
from run_remaining_avtenet import retained_vectors
from remaining_avfakenet import FrozenAVFakeNet


class CleanReuseTests(unittest.TestCase):
    def test_expert_vectors_keep_exact_values_and_reject_wrong_source(self):
        conn=sqlite3.connect(':memory:')
        conn.execute('CREATE TABLE features(video_id TEXT,start REAL,end REAL,source_sha256 TEXT,features BLOB)')
        array=np.arange(768,dtype='<f4')
        conn.execute('INSERT INTO features VALUES(?,?,?,?,?)',('v',0,2,'sha',array.tobytes()))
        value=retained_vectors(conn,{'video_id':'v','source_sha256':'sha'},[(0,2)])
        np.testing.assert_array_equal(value[0]['vector'],array)
        with self.assertRaises(ValueError):retained_vectors(conn,{'video_id':'v','source_sha256':'changed'},[(0,2)])
        with self.assertRaises(ValueError):retained_vectors(conn,{'video_id':'v','source_sha256':'sha'},[(1,3)])
        conn.close()

    def test_clean_prediction_reuse_rejects_changed_file_without_model_call(self):
        model=object.__new__(FrozenAVFakeNet)
        with tempfile.TemporaryDirectory() as temp:
            path=_fk_paths.relocate((Path(temp)/'original'), __file__);path.write_bytes(b'original')
            row={'video_id':'v','source_sha256':hashlib.sha256(b'original').hexdigest(),'label':0,'condition':'clean'}
            model.original={'v':row};model.clean={'v':{'video_id':'v','label':0,'prediction':None,'score':None,'status':'UNASSESSED'}}
            result=model.predict(path,row)
            self.assertIsNone(result['prediction']);self.assertIsNone(result['timings'])
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):model.predict(path,row)


if __name__=='__main__':unittest.main()
