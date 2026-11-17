
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

import json,sys,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
import numpy as np
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1]
sys.path[:0]=[str(_fk_paths.relocate((ROOT/'scripts'), __file__)),str(_fk_paths.relocate((ROOT/'src'), __file__)),str(ROOT)]
from remaining_avfakenet import FrozenAVFakeNet
from baselines.common.io import fingerprint,digest


class ModalityInputTests(unittest.TestCase):
    def test_exact_stream_reuses_hash_checked_array_and_profiles_stay_raw(self):
        model=object.__new__(FrozenAVFakeNet)
        with tempfile.TemporaryDirectory() as temporary:
            root=_fk_paths.relocate((Path(temporary)), __file__);source=_fk_paths.relocate((root/'source.mp4'), __file__);source.write_bytes(b'source')
            old={'video_id':'v','path':source.name,'duration':2.,'source_sha256':digest(source),'label':1}
            row={**old,'condition':'video_compression-28','original_source_sha256':old['source_sha256']}
            model.original={'v':old};model.dataset=root;model.prepared=_fk_paths.relocate((root/'prepared'), __file__);model.stream_cache={}
            directory=model.prepared/fingerprint(old);directory.mkdir(parents=True)
            expected=np.arange(6,dtype=np.float32).reshape(1,2,3);np.save(_fk_paths.relocate((directory/'audio.npy'), __file__),expected)
            record={'video_id':'v','source_sha256':old['source_sha256'],'reasons':{},'files':{'audio.npy':digest(_fk_paths.relocate((directory/'audio.npy'), __file__))}}
            (_fk_paths.relocate((directory/'record.json'), __file__)).write_text(json.dumps(record))
            with patch('remaining_avfakenet.stream_identity',return_value={'sha256':'same'}):
                value=model.retained_input(source,row,'audio');np.testing.assert_array_equal(value['array'],expected)
                self.assertIsNone(model.retained_input(source,old,'audio'))
                np.save(_fk_paths.relocate((directory/'audio.npy'), __file__),expected+1)
                with self.assertRaises(ValueError):model.retained_input(source,row,'audio')
            with patch('remaining_avfakenet.stream_identity',return_value={'sha256':'different'}):
                self.assertIsNone(model.retained_input(source,row,'audio'))


if __name__=='__main__':unittest.main()
