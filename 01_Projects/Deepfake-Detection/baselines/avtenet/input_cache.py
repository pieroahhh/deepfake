"""Lossless raw prepared views; avoid repeating decoding/detection each epoch."""

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

from pathlib import Path
import os
import shutil
import numpy as np
from baselines.common.io import read,write,digest,fingerprint


class InputCache:
    def __init__(self,directory,identity):
        self.root=_fk_paths.relocate((Path(directory)/fingerprint(identity)), __file__)
        self.root.mkdir(parents=True,exist_ok=True)
        header=_fk_paths.relocate((self.root/'identity.json'), __file__)
        if header.exists():
            if read(header)!=identity:raise ValueError('AVTENet input cache identity mismatch')
        else:write(header,identity)

    def load(self,row,start):
        """Read only: safe for bounded CPU prefetch; never invokes the provider."""
        key=fingerprint([row,start]);payload=self.root/(key+'.npz');header=self.root/(key+'.json')
        if not header.exists():return None
        metadata=read(header)
        if digest(payload)!=metadata['sha256']:raise ValueError('AVTENet input cache checksum mismatch')
        if metadata['source_sha256']!=row['source_sha256']:
            raise ValueError('AVTENet input cache source mismatch')
        with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as values:return {key:values[key] for key in values.files}

    def get(self,row,start,provider):
        cached=self.load(row,start)
        if cached is not None:return cached
        key=fingerprint([row,start]);payload=self.root/(key+'.npz');header=self.root/(key+'.json')
        if not header.exists():
            if shutil.disk_usage(self.root).free<5*1024**3:
                raise RuntimeError('AVTENet input cache needs at least 5 GiB of free disk; preserve cache and resume on larger disk')
            values=provider()
            temporary=payload.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**values)
            os.replace(temporary,payload)
            write(header,{'sha256':digest(payload),'source_sha256':row['source_sha256']})
        with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as values:return {key:values[key] for key in values.files}


def prefetch_reads(items, loader, depth):
    """Yield ordered (item, future) pairs; workers only read existing CPU caches.

    Providers, face detection, model preparation and RNG stay on the main thread.
    The caller resolves each future inside its normal per-window error handler.
    """
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    if not 1 <= depth <= 4:raise ValueError('cache prefetch depth must be between 1 and 4')
    iterator=iter(items)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='avtenet-cache') as pool:
        pending=deque()
        for _ in range(depth):
            item=next(iterator,None)
            if item is None:break
            pending.append((item,pool.submit(loader,item)))
        try:
            while pending:
                item,future=pending.popleft()
                yield item,future
                item=next(iterator,None)
                if item is not None:pending.append((item,pool.submit(loader,item)))
        finally:
            for _,future in pending:future.cancel()
