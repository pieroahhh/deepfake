"""Timestamped media adapters with separate preparation errors and integrity errors."""

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

import importlib.metadata
from pathlib import Path
import json
import math

import numpy as np

from .io import digest, fingerprint


class PreparationError(ValueError):
    pass


def detector(device='cpu'):
    from facenet_pytorch import MTCNN
    from deepfake_detection.faces import MTCNNDetector
    import facenet_pytorch
    directory = _fk_paths.relocate((Path(facenet_pytorch.__file__).parent/'data'), __file__)
    files = {p.name: digest(p) for p in sorted(directory.glob('*net.pt'))}
    if len(files) != 3:
        raise ValueError('MTCNN must provide PNet/RNet/ONet weight files')
    provenance = {'provider': 'facenet-pytorch', 'version': importlib.metadata.version('facenet-pytorch'),
                  'files': files, 'threshold': .9, 'minimum_iou': .3,
                  'crop': 'eye-aligned face and mouth-centered lips; fit_to_frame'}
    return MTCNNDetector(MTCNN(keep_all=True, device=device)), provenance


def faces(path, start, duration, detect, *, fps=25, pixels=128, sampler=None, prefetch_decode=False):
    from deepfake_detection.media import sample_video
    from deepfake_detection.faces import track_and_crop
    # Duration is rounded down to the last requested 25-Hz sample, never padded.
    count = max(1, math.ceil(duration*fps-1e-9))
    source_times=[];target_times=[]
    def frames():
        from .decode_prefetch import decoded_chunks
        # Decode at most one second of full-resolution RGB at a time while the
        # tracker retains its identity across chunks of the same video/view.
        chunk=max(1,round(fps))
        requests=((start+offset/fps,min(chunk,count-offset)/fps) for offset in range(0,count,chunk))
        samples=decoded_chunks(sampler or sample_video,path,requests,fps=fps,enabled=prefetch_decode)
        try:
            for sampled in samples:
                source_times.extend(sampled['source_times']);target_times.extend(sampled['target_times'])
                yield from sampled['rgb']
        finally:
            samples.close()
    stream=frames()
    try:
        crops = track_and_crop(stream, detect, face_pixels=pixels, lip_pixels=96, fit_to_frame=True)
    finally:
        stream.close()
    if any(r['status'] != 'USABLE' for r in crops):
        reasons = sorted({r['reason'] for r in crops if r['status'] != 'USABLE'})
        raise PreparationError('; '.join(reasons))
    return {'faces': np.stack([r['face'] for r in crops]), 'lips': np.stack([r['lips'] for r in crops]),
            'source_times': np.asarray(source_times), 'target_times': np.asarray(target_times)}


def audio(path, start, duration):
    from deepfake_detection.media import aligned_audio
    return aligned_audio(path, start, duration)['samples']


def event(path, row, reason, **extra):
    if path is not None:
        with _fk_paths.relocate((Path(path)), __file__).open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'video_id': row['video_id'], 'split': row['split'], 'reason': str(reason), **extra})+'\n')
