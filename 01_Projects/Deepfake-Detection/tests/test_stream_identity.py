
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

import unittest,tempfile,sys,struct
from pathlib import Path
import numpy as np
import av
ROOT=_fk_paths.relocate((Path(__file__)), __file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(_fk_paths.relocate((ROOT/'src'), __file__))]
from baselines.common.stream_identity import stream_identity


def video(path,value=32):
    with av.open(str(path),'w') as output:
        stream=output.add_stream('mpeg4',rate=25);stream.width=32;stream.height=32;stream.pix_fmt='yuv420p'
        for i in range(3):
            frame=av.VideoFrame.from_ndarray(np.full((32,32,3),value,dtype=np.uint8),format='rgb24')
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)


class StreamIdentityTests(unittest.TestCase):
    def test_identical_copy_matches_changed_visual_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            a=_fk_paths.relocate((Path(directory)/'a.mp4'), __file__);b=_fk_paths.relocate((Path(directory)/'b.mp4'), __file__);c=_fk_paths.relocate((Path(directory)/'c.mp4'), __file__)
            video(a);b.write_bytes(a.read_bytes());video(c,220)
            self.assertEqual(stream_identity(a,'video'),stream_identity(b,'video'))
            self.assertNotEqual(stream_identity(a,'video'),stream_identity(c,'video'))
            self.assertIsNone(stream_identity(a,'audio'))
            with self.assertRaises(ValueError):stream_identity(a,'other')


if __name__=='__main__':unittest.main()
