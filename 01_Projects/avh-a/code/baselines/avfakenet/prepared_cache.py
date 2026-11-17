"""Resumable lossless per-video inputs; cache detections once across epochs."""

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
import numpy as np
from baselines.common.io import read,write,digest,fingerprint
from baselines.common.media import faces,audio
from .preprocessing import prepare_audio


class PreparedCache:
    def __init__(self,directory,identity):
        self.directory=_fk_paths.relocate((Path(directory)), __file__);self.directory.mkdir(parents=True,exist_ok=True)
        self.identity=identity
        header=_fk_paths.relocate((self.directory/'identity.json'), __file__)
        if header.exists():
            if read(header)!=identity:raise ValueError('AVFakeNet prepared cache identity mismatch')
        else:write(header,identity)

    def get(self,population,row,detect,profile):
        name=fingerprint(row)
        payload=self.directory/(name+'.npz');header=self.directory/(name+'.json')
        if header.exists():
            recorded=read(header)
            if recorded['source_sha256']!=row['source_sha256'] or recorded['payload_sha256']!=digest(payload):
                raise ValueError('AVFakeNet prepared cache checksum mismatch')
        else:
            source=population.path(row)
            arrays={};reasons={}
            try:
                sample=faces(source,0,row['duration'],detect)
                arrays.update(faces=sample['faces'],source_times=sample['source_times'])
            except (ValueError,OSError) as error:reasons['visual']=str(error)
            try:arrays['mel']=prepare_audio(audio(source,0,row['duration']),16000,profile).numpy()
            except (ValueError,OSError) as error:reasons['audio']=str(error)
            temporary=payload.with_suffix('.tmp')
            with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
            os.replace(temporary,payload)
            recorded={'source_sha256':row['source_sha256'],'payload_sha256':digest(payload),'reasons':reasons}
            write(header,recorded)
        with np.load(_fk_paths.relocate_value((payload), __file__),allow_pickle=False) as data:
            return {k:data[k] for k in data.files},recorded['reasons']
