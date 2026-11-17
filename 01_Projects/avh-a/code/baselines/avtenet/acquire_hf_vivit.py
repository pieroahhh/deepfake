"""Acquire and verify the user-approved joint-attention Google ViViT adaptation."""

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
import copy
from pathlib import Path
import torch
from baselines.common.io import write,digest
from .hf_vivit import HFVivitBackend


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args(argv)
    from huggingface_hub import HfApi,snapshot_download
    from transformers import VivitForVideoClassification,VivitModel,VivitImageProcessor
    root=_fk_paths.relocate((Path(args.output)), __file__);root.mkdir(parents=True,exist_ok=False)
    repo='google/vivit-b-16x2-kinetics400'
    revision=HfApi(token=False).model_info(repo).sha
    snapshot=snapshot_download(repo,revision=revision,token=False,cache_dir=_fk_paths.relocate((root.parent/'.hf-download-cache'), __file__),
        local_dir=_fk_paths.relocate((root/'upstream'), __file__),local_dir_use_symlinks=False,
        allow_patterns=['config.json','model.safetensors','pytorch_model.bin','preprocessor_config.json','README.md','LICENSE*'])
    wrapper,info=VivitForVideoClassification.from_pretrained(_fk_paths.relocate_value((snapshot), __file__),local_files_only=True,output_loading_info=True)
    if any(info.get(k) for k in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('official model checkpoint loading incomplete')
    base=wrapper.vivit.eval();base.save_pretrained(root,safe_serialization=True)
    processor=VivitImageProcessor.from_pretrained(_fk_paths.relocate_value((snapshot), __file__),local_files_only=True);processor.save_pretrained(root)
    for filename in ('README.md','LICENSE','LICENSE.txt'):
        path=_fk_paths.relocate((Path(snapshot)/filename), __file__)
        if path.is_file():(_fk_paths.relocate((root/('upstream-'+filename)), __file__)).write_bytes(path.read_bytes())
    del wrapper
    torch.set_num_threads(2)
    generator=torch.Generator().manual_seed(7)
    inputs=torch.rand(1,32,3,224,224,generator=generator)*2-1
    base.to(args.device)
    with torch.inference_mode():reference=base(inputs.to(args.device)).last_hidden_state[:,0].cpu()
    base.cpu()
    efficient=HFVivitBackend(base,processor).to(args.device).eval()
    with torch.inference_mode():actual=efficient(inputs.to(args.device)).cpu()
    torch.testing.assert_close(actual,reference,rtol=1e-4,atol=2e-5)
    parity={'status':'PASSED','reference':'official Transformers VivitModel eager CLS output on same checkpoint',
            'samples':1,'input':'seed7 full 32x224x224 tensor','rtol':1e-4,'atol':2e-5,
            'maximum_absolute_error':float((actual-reference).abs().max()),'device':args.device}
    metadata={'schema':'model-artifact-v1','architecture':'transformers-vivit-joint-adaptation-v1',
              'source':'https://huggingface.co/'+repo,'revision':revision,'weights_status':'pretrained',
              'source_files':{p.name:digest(p) for p in _fk_paths.relocate((Path(snapshot)), __file__).iterdir() if p.is_file()},
              'adaptation':'joint spatiotemporal attention; NOT the paper-guided factorized encoder',
              'sdpa_parity':parity,'files':{p.name:digest(p) for p in root.iterdir() if p.is_file()}}
    write(_fk_paths.relocate((root/'artifact.json'), __file__),metadata)
    print(metadata,flush=True)


if __name__=='__main__':main()
