"""Equivalent input preparation with bounded CPU overlap for frozen inference."""

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

import os
from pathlib import Path
import shutil
import numpy as np
import torch
from baselines.common.io import fingerprint,digest,write
from .input_cache import InputCache
from .pipeline import Prepared


def audio_cache_identity(manifest):
    """Raw CPU audio is independent of selected model weights, but decoder-bound."""
    import av
    from deepfake_detection import media
    from baselines.common import media as common_media
    return {'schema':'heldout-raw-audio-v1','manifest_sha256':digest(manifest),
        'pyav_version':av.__version__,'libav':{k:list(v) for k,v in av.library_versions.items()},
        'media_source_sha256':digest(_fk_paths.relocate((Path(media.__file__)), __file__)),
        'adapter_source_sha256':digest(_fk_paths.relocate((Path(common_media.__file__)), __file__)),
        'sample_rate':16000,'duration':2,'precision':'float32-reference'}


class FastRawCache(InputCache):
    """Lossless, uncompressed NPZ: trade SSD space for CPU time, retaining hashes."""
    def get(self,row,start,provider):
        cached=self.load(row,start)
        if cached is not None:return cached
        if shutil.disk_usage(self.root).free<8*1024**3:
            raise RuntimeError('Less than 8 GiB free: preserve prepared cache and free disk before resume')
        values=provider()
        key=fingerprint([row,start]);payload=self.root/(key+'.npz');header=self.root/(key+'.json')
        temporary=payload.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez(stream,**values)
        os.replace(temporary,payload)
        write(header,{'sha256':digest(payload),'source_sha256':row['source_sha256']})
        # Same return semantics as the reference cache: materialize saved arrays.
        with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as data:return {key:data[key] for key in data.files}


class FastPrepared(Prepared):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.stage=='audio':
            if self.input_cache is None:raise ValueError('AST CPU preparation requires a cache')
            self.prefetch_processed_inputs=4

    def cached_input(self,row,start):
        if self.stage!='audio':return super().cached_input(row,start)
        # Audio preparation is CPU-only. GPU inference never runs on this worker.
        path=self.population.root/row['path']
        raw=self.input_cache.get(row,start,lambda:self.raw_inputs(path,start))
        with torch.inference_mode():value=self.transform(raw)
        if value.device.type!='cpu':raise ValueError('AST prefetch must remain on CPU')
        return value


def frozen_inputs(population,stage,backend,assets,detect,failure_log,cache,*,prefetch_video_decode=False):
    return FastPrepared(population,stage,backend,assets,detect,failure_log,cache,
        reuse_video_decoder=True,cache_face_detections=True,prefetch_cached_inputs=4,
        fast_vivit_processor=stage=='visual',
        prefetch_processed_inputs=4 if stage in ('visual','audiovisual') else 0,
        prefetch_video_decode=prefetch_video_decode)


def aligned_vectors(stores,row,intervals):
    """Read each expert once per clip, preserving reference window/expert order."""
    streams={}
    for stage,conn in stores.items():
        values={}
        for start,end,sha,blob in conn.execute(
                'SELECT start,end,source_sha256,features FROM features WHERE video_id=? ORDER BY start',
                (row['video_id'],)):
            if sha!=row['source_sha256']:raise ValueError('Held-out source mismatch')
            array=np.frombuffer(blob,dtype='<f4')
            if array.size!=768 or not np.isfinite(array).all():raise ValueError('Invalid held-out feature vector')
            if (start,end) in values:raise ValueError('Duplicate held-out feature window')
            values[start,end]=array.copy().reshape(1,768)
        streams[stage]=values
    for pair in intervals:
        if all(pair in stream for stream in streams.values()):
            yield {stage:torch.from_numpy(stream[pair]) for stage,stream in streams.items()}
