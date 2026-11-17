"""Explicit Flax-array-to-factorized-ViViT tensor conversion; parity remains a gate.

This consumes exported named arrays and a complete operator-reviewed mapping.
It does not guess Scenic training-state formats or invent checkpoint key names.
Converted files are deliberately NOT marked pretrained until reference parity.
"""

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

import numpy as np
import torch

from .backbones import FactorizedViViT
from .cli import save
from .forensics import sha256_file


def convert_factorized_arrays(arrays,mapping,config,output,*,source,revision,source_checkpoint_sha256):
    if (mapping.get('schema')!='explicit-flax-tensor-map-v1' or not source or not revision
            or not isinstance(source_checkpoint_sha256,str) or len(source_checkpoint_sha256)!=64
            or any(character not in '0123456789abcdef' for character in source_checkpoint_sha256)):
        raise ValueError('explicit mapping and source/checkpoint provenance required')
    model=FactorizedViViT(config)
    expected=model.state_dict();rules=mapping['tensors']
    if set(rules)!=set(expected):
        raise ValueError('mapping must cover every target tensor exactly')
    converted={};used=set()
    for target,rule in rules.items():
        names=rule['sources']
        if not names or any(name not in arrays for name in names):
            raise ValueError('missing source tensor for '+target)
        values=[np.asarray(arrays[name]) for name in names]
        used.update(names)
        operation=rule['operation']
        if operation=='identity' and len(values)==1:value=values[0]
        elif operation=='dense_transpose' and len(values)==1 and values[0].ndim==2:value=values[0].T
        elif operation=='conv3d_flax_to_torch' and len(values)==1 and values[0].ndim==5:
            value=values[0].transpose(4,3,0,1,2)
        elif operation=='qkv_weight' and len(values)==3:
            value=np.concatenate([value.reshape(value.shape[0],-1).T for value in values],axis=0)
        elif operation=='qkv_bias' and len(values)==3:
            value=np.concatenate([value.reshape(-1) for value in values])
        elif operation=='attention_output' and len(values)==1 and values[0].ndim==3:
            value=values[0].reshape(-1,values[0].shape[-1]).T
        else:raise ValueError('unsupported source shape/operation for '+target)
        if tuple(value.shape)!=tuple(expected[target].shape) or not np.issubdtype(value.dtype,np.floating) or not np.isfinite(value).all():
            raise ValueError('invalid converted shape/type/value for '+target)
        converted[target]=torch.from_numpy(np.ascontiguousarray(value)).to(expected[target].dtype)
        if not torch.isfinite(converted[target]).all():
            raise ValueError('converted tensor overflows target precision: '+target)
    ignored=set(mapping.get('ignored_sources',[]))
    if used & ignored or set(arrays)!=used|ignored:
        raise ValueError('all unused source arrays must be explicitly declared ignored')
    model.load_state_dict(converted,strict=True)
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    from safetensors.torch import save_file
    save_file(converted,str(_fk_paths.relocate((root/'model.safetensors'), __file__)))
    save(_fk_paths.relocate((root/'config.json'), __file__),config);save(_fk_paths.relocate((root/'tensor-map.json'), __file__),mapping)
    manifest={'schema':'model-artifact-v1','source':source,'revision':revision,
              'source_checkpoint_sha256':source_checkpoint_sha256,
              'architecture':'factorized-vivit-reconstruction-v1','weights_status':'converted_unverified',
              'parity_status':'NOT_EXECUTED_ON_MATCHING_OFFICIAL_BACKEND',
              'dtype_policy':'cast_to_target_model_dtype_after_explicit_layout_conversion',
              'files':{path.name:sha256_file(path) for path in root.iterdir()}}
    save(_fk_paths.relocate((root/'artifact.json'), __file__),manifest)
    return manifest
