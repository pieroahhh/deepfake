"""Opt-in validated CPU batches and identical FP32 math with fewer GPU syncs."""

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

import types
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.nn import functional as F


def forward(self, images, *, return_shapes=False):
    if return_shapes:return self._checked_forward(images,return_shapes=True)
    features=self.input_block(images.permute(0,2,3,1))
    patch=self.patch_size;height,width=features.shape[1:3]
    features=F.pad(features,(0,0,0,(-width)%patch,0,(-height)%patch))
    batch,height,width,channels=features.shape
    patches=features.reshape(batch,height//patch,patch,width//patch,patch,channels)
    patches=patches.permute(0,1,3,2,4,5).reshape(batch,height//patch,width//patch,patch*patch*channels)
    features=self.patch_embedding(patches)
    for stage,merge in zip(self.stages,self.merges):features=merge(stage(features))
    return self.output_block(features.mean(dim=(1,2)))


def loss(logits,labels,smoothing):
    if logits.ndim!=2 or logits.shape[1]!=2 or labels.shape!=logits.shape[:1] or labels.dtype!=torch.long or not 0<=smoothing<1:
        raise ValueError('Invalid two-class loss inputs')
    # Image/target values are checked on CPU in Inputs.batch. The live loop
    # still checks loss finiteness and gradient finiteness before every update.
    targets=F.one_hot(labels,num_classes=2).to(logits.dtype)
    targets=targets*(1-smoothing)+smoothing/2
    log_probabilities=logits.log_softmax(-1)
    return -(targets*log_probabilities+(1-targets)*log_probabilities.flip(-1)).mean()


def enable(model):
    model._checked_forward=model.forward
    model.forward=types.MethodType(forward,model)
    return model


def input_class(base,epoch_order):
    class Prefetched(base):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='avfake-input')
            self.pending=None;self.rows=None;self.order=None;self.position=0

        def array(self,video):
            key=self.records[video]['key']
            if key not in self.open:
                self.open[key]=np.load(_fk_paths.relocate_value((self.cache/key/(self.branch+'.npy')), __file__),mmap_mode='r',allow_pickle=False)
                if len(self.open)>1024:
                    _,old=self.open.popitem(last=False);old._mmap.close()
            self.open.move_to_end(key);return self.open[key]

        def checked(self,refs,indices):
            x,y=super().batch(refs,indices)
            if not torch.isfinite(x).all() or ((x<0)|(x>1)).any() or not ((y==0)|(y==1)).all():
                raise ValueError('Invalid prepared batch')
            return x,y

        def batch(self,refs,indices):
            if self.rows is not refs:
                if self.pending:self.pending[1].result();self.pending=None
                self.rows=refs;self.position=0
                # Exactly the existing epoch-1 sampler; validation has no prefetch
                # because its indexed frame traversal differs from training.
                split=self.records[int(refs[0,0])]['split'] if len(refs) else None
                self.order=epoch_order(refs[:,2],8) if split=='train' else None
            if self.pending and np.array_equal(self.pending[0],indices):
                value=self.pending[1].result();self.pending=None
            else:
                if self.pending:self.pending[1].result();self.pending=None
                value=self.pool.submit(self.checked,refs,indices.copy()).result()
                if self.order is not None:
                    # Locate resume cursor once without consuming any RNG state.
                    candidates=np.flatnonzero(self.order==indices[0])
                    matches=[int(i) for i in candidates if i%8==0 and np.array_equal(self.order[i:i+len(indices)],indices)]
                    self.position=matches[0] if len(matches)==1 else -len(indices)
            self.position+=len(indices)
            if self.order is not None and 0<self.position<len(self.order):
                following=self.order[self.position:self.position+8]
                self.pending=(following.copy(),self.pool.submit(self.checked,refs,following.copy()))
            return value

        def close(self):
            self.pool.shutdown(wait=True)
            super().close()
    return Prefetched
