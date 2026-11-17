"""Explicit Google joint-attention ViViT adaptation, never a factorized encoder."""

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
import torch
from torch import nn
from torch.nn import functional as F


class SDPAAttention(nn.Module):
    """Same Q/K/V parameters and equations as HF 4.44; no attention-map materialization."""
    def __init__(self,attention):
        super().__init__()
        self.query,self.key,self.value=attention.query,attention.key,attention.value
        self.num_attention_heads=attention.num_attention_heads
        self.attention_head_size=attention.attention_head_size
        self.all_head_size=attention.all_head_size
        self.dropout=attention.dropout

    def forward(self,hidden_states,head_mask=None,output_attentions=False):
        if head_mask is not None or output_attentions:
            raise ValueError('adaptation SDPA does not support head masks or attention-map output')
        batch,tokens,_=hidden_states.shape
        def shape(layer):
            return layer(hidden_states).reshape(batch,tokens,self.num_attention_heads,self.attention_head_size).transpose(1,2)
        result=F.scaled_dot_product_attention(shape(self.query),shape(self.key),shape(self.value),
                    dropout_p=self.dropout.p if self.training else 0.,is_causal=False)
        return (result.transpose(1,2).reshape(batch,tokens,self.all_head_size),)


class HFVivitBackend(nn.Module):
    def __init__(self,model,processor=None,*,efficient=True):
        super().__init__()
        self.model,self.processor=model,processor
        self.feature_dimension=model.config.hidden_size
        if efficient:
            for layer in model.encoder.layer:
                layer.attention.attention=SDPAAttention(layer.attention.attention)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def forward(self,values):
        if values.ndim!=5 or not torch.isfinite(values).all():
            raise ValueError('expected finite [B,T,C,H,W] ViViT pixels')
        return self.model(pixel_values=values,return_dict=True).last_hidden_state[:,0]

    def prepare(self,frames,*,fast=False):
        if fast:
            # Use the same processor transformations, then construct an ndarray
            # in one operation instead of torch.tensor(nested lists of arrays).
            import numpy as np
            values=self.processor(list(frames),return_tensors='np')['pixel_values']
            return torch.from_numpy(np.ascontiguousarray(values))
        return self.processor(list(frames),return_tensors='pt')['pixel_values']

    @classmethod
    def from_local(cls,directory,manifest):
        from deepfake_detection.artifacts import verified_files,require_recorded
        from transformers import VivitModel,VivitImageProcessor
        metadata=verified_files(directory,manifest)
        if metadata['architecture']!='transformers-vivit-joint-adaptation-v1':
            raise ValueError('expected explicit joint-attention adaptation artifact')
        require_recorded(metadata,'config.json','model.safetensors','preprocessor_config.json')
        if metadata.get('sdpa_parity',{}).get('status')!='PASSED':
            raise ValueError('joint ViViT needs recorded eager-versus-SDPA parity')
        model,info=VivitModel.from_pretrained(_fk_paths.relocate_value((directory), __file__),add_pooling_layer=False,
            local_files_only=True,use_safetensors=True,output_loading_info=True)
        if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
            raise ValueError('ViViT checkpoint did not load exactly')
        if model.config.hidden_size!=768 or model.config.num_frames!=32:
            raise ValueError('adaptation requires 32-frame 768-dimensional ViViT')
        result=cls(model,VivitImageProcessor.from_pretrained(_fk_paths.relocate_value((directory), __file__),local_files_only=True))
        result.artifact=metadata
        return result
