"""Convert a prepared window into paired official-backend input in the legacy env."""

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


import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,help='prepared.npz from FK source-time preparation')
    parser.add_argument('--settings',required=True,help='explicit checkpoint-compatible preprocessing JSON')
    parser.add_argument('--pcm16',required=True,help='retained source-aligned audio-16k-pcm16.wav')
    parser.add_argument('--output',required=True,help='new output NPZ')
    args=parser.parse_args()
    import cv2
    import numpy as np
    from scipy.io import wavfile
    sys.path.insert(0,str(_fk_paths.relocate((Path(__file__).resolve().parents[1]/'src'), __file__)))
    from deepfake_detection.avhubert_preparation import prepare_avhubert
    settings=json.loads(_fk_paths.relocate((Path(args.settings)), __file__).read_text())
    sample_rate,pcm=wavfile.read(args.pcm16)
    if sample_rate!=16000 or pcm.dtype!=np.int16 or pcm.ndim!=1:
        raise ValueError('retained 16kHz mono PCM16 WAV is required')
    with np.load(_fk_paths.relocate_value((args.prepared), __file__),allow_pickle=False) as archive:
        lips=np.stack([cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY) for frame in archive['lips']])
        if len(pcm)!=len(archive['audio']):
            raise ValueError('retained PCM and prepared source window differ in duration')
        source=prepare_avhubert(pcm,lips,settings)
    with _fk_paths.relocate((Path(args.output)), __file__).open('xb') as stream:
        np.savez_compressed(stream,**{key:value.numpy() for key,value in source.items()})


if __name__=='__main__':
    main()
