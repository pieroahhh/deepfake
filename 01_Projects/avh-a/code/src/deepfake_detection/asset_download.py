"""Public pretrained asset acquisition with recorded revisions and conversion checks.

No credentials, dataset access or random model fallback. Network failures are
reported per asset; partially fetched directories never count as ready models.
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


import json
from pathlib import Path
import re
import urllib.request

from .artifacts import verified_files
from .cli import save
from .forensics import sha256_file

SOURCES = {
    'ast': 'https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593',
    'wav2vec2_base': 'https://huggingface.co/facebook/wav2vec2-base',
    'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth',
    'silero': 'https://github.com/snakers4/silero-vad',
    'yunet': 'https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet',
    'factorized_vivit': 'https://storage.googleapis.com/scenic-bucket/vivit/kinetics_400/vivit_base_16x2_fe/checkpoint',
    'avhubert_native': 'https://dl.fbaipublicfiles.com/avhubert/model/lrs3/clean-pretrain/base_lrs3_iter5.pt',
}


def _download(url, path):
    request = urllib.request.Request(url, headers={'User-Agent':'FK-research-asset-preparation/1'})
    with urllib.request.urlopen(request, timeout=15) as source, _fk_paths.relocate((Path(path)), __file__).open('xb') as target:
        while True:
            block = source.read(1024*1024)
            if not block:
                break
            target.write(block)


def _github_revision(repository, ref):
    request = urllib.request.Request(f'https://api.github.com/repos/{repository}/commits/{ref}',
                                     headers={'User-Agent':'FK-research-asset-preparation/1'})
    with urllib.request.urlopen(request, timeout=15) as stream:
        revision = json.load(stream)['sha']
    if not re.fullmatch('[0-9a-f]{40}', revision):
        raise ValueError('GitHub did not return an immutable revision')
    return revision


def _manifest(root, source, revision, architecture, *, parity=None):
    metadata = {'schema':'model-artifact-v1', 'source':source, 'revision':revision,
                'architecture':architecture, 'weights_status':'pretrained',
                'files':{path.name:sha256_file(path) for path in root.iterdir() if path.is_file()},
                'conversion_parity':parity}
    save(_fk_paths.relocate((root/'artifact.json'), __file__),metadata)
    verified_files(root,metadata)
    return metadata


def _hf_asset(name, root):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import (ASTForAudioClassification, ASTModel, ASTFeatureExtractor,
                              Wav2Vec2ForPreTraining, Wav2Vec2Model, Wav2Vec2FeatureExtractor)
    repo = 'MIT/ast-finetuned-audioset-10-10-0.4593' if name=='ast' else 'facebook/wav2vec2-base'
    revision = HfApi(token=False).model_info(repo,timeout=5).sha
    snapshot = snapshot_download(repo, revision=revision, token=False, cache_dir=_fk_paths.relocate((root.parent/'.download-cache'), __file__), allow_patterns=[
        'config.json','preprocessor_config.json','model.safetensors','pytorch_model.bin','LICENSE*','README.md',
    ])
    wrapper_cls = ASTForAudioClassification if name=='ast' else Wav2Vec2ForPreTraining
    wrapper, loading = wrapper_cls.from_pretrained(_fk_paths.relocate_value((snapshot), __file__),local_files_only=True,output_loading_info=True)
    if any(loading.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')):
        raise ValueError('upstream checkpoint loading was incomplete')
    wrapper.eval()
    base = wrapper.audio_spectrogram_transformer if name=='ast' else wrapper.wav2vec2
    base.save_pretrained(root, safe_serialization=True)
    extractor_cls = ASTFeatureExtractor if name=='ast' else Wav2Vec2FeatureExtractor
    extractor_cls.from_pretrained(_fk_paths.relocate_value((snapshot), __file__),local_files_only=True).save_pretrained(root)
    for filename in ('README.md','LICENSE','LICENSE.txt','LICENSE.md'):
        source=_fk_paths.relocate((Path(snapshot)/filename), __file__)
        if source.is_file():
            (_fk_paths.relocate((root/('upstream-'+filename)), __file__)).write_bytes(source.read_bytes())
    reloaded_cls = ASTModel if name=='ast' else Wav2Vec2Model
    reloaded = reloaded_cls.from_pretrained(_fk_paths.relocate_value((root), __file__),local_files_only=True,use_safetensors=True).eval()
    generator=torch.Generator().manual_seed(7)
    inputs = (torch.randn(1,base.config.max_length,base.config.num_mel_bins,generator=generator)
              if name=='ast' else torch.randn(1,32000,generator=generator))
    with torch.inference_mode():
        reference=base(inputs).last_hidden_state
        actual=reloaded(inputs).last_hidden_state
        torch.testing.assert_close(actual,reference,rtol=1e-6,atol=1e-6)
    parity={'status':'BASE_EXTRACTION_AND_RELOAD_MATCH','input':'seed7 synthetic tensor',
            'max_absolute_error':float((actual-reference).abs().max()),
            'scope':'same implementation base state extraction; not independent implementation equivalence'}
    architecture='transformers-ast-base' if name=='ast' else 'transformers-wav2vec2-base'
    return _manifest(root,SOURCES[name],revision,architecture,parity=parity)


def _efficientnet(root):
    import torch
    from safetensors.torch import save_file
    from torchvision.models import efficientnet_b0,EfficientNet_B0_Weights
    weights=EfficientNet_B0_Weights.IMAGENET1K_V1
    raw=_fk_paths.relocate((root/'upstream.pth'), __file__)
    _download(weights.url,raw)
    actual_hash=sha256_file(raw)
    if not actual_hash.startswith('7f5810bc'):
        raise ValueError('torchvision published hash prefix mismatch')
    state=torch.load(_fk_paths.relocate_value((raw), __file__),map_location='cpu',weights_only=True)
    model=efficientnet_b0(weights=None)
    model.load_state_dict(state,strict=True)
    save_file(state,str(_fk_paths.relocate((root/'model.safetensors'), __file__)))
    transform=weights.transforms()
    save(_fk_paths.relocate((root/'preprocessor_config.json'), __file__),{'image_size':224,'mean':transform.mean,'std':transform.std,
                                          'source_crop_note':'FK receives already aligned 224px crops'})
    # State conversion is lossless and checked tensor-for-tensor.
    from safetensors.torch import load_file
    loaded=load_file(_fk_paths.relocate_value((str(_fk_paths.relocate((root/'model.safetensors'), __file__))), __file__))
    if set(loaded)!=set(state) or any(not torch.equal(loaded[key],state[key]) for key in state):
        raise ValueError('EfficientNet state conversion mismatch')
    return _manifest(root,weights.url,actual_hash,'torchvision-efficientnet-b0',
                     parity={'status':'EXACT_STATE_TENSOR_MATCH','scope':'state serialization conversion'})


def _github_file(name,root):
    if name=='silero':
        repo,ref,relative='snakers4/silero-vad','master','src/silero_vad/data/silero_vad.jit'
    else:
        repo,ref,relative='opencv/opencv_zoo','main','models/face_detection_yunet/face_detection_yunet_2023mar.onnx'
    revision=_github_revision(repo,ref)
    raw=_fk_paths.relocate((root/Path(relative).name), __file__)
    _download(f'https://raw.githubusercontent.com/{repo}/{revision}/{relative}',raw)
    if raw.stat().st_size<1024:
        pointer=raw.read_text()
        match=re.search(r'oid sha256:([0-9a-f]{64})',pointer)
        if match:
            raw.rename(_fk_paths.relocate((root/'upstream-lfs-pointer.txt'), __file__))
            _download(f'https://media.githubusercontent.com/media/{repo}/{revision}/{relative}',raw)
            if sha256_file(raw)!=match.group(1):
                raise ValueError('Git LFS checksum mismatch')
    license_relative='LICENSE' if name=='silero' else 'models/face_detection_yunet/LICENSE'
    _download(f'https://raw.githubusercontent.com/{repo}/{revision}/{license_relative}',_fk_paths.relocate((root/'LICENSE'), __file__))
    if name=='silero':
        from .vad import SileroVAD
        SileroVAD(raw,sha256_file(raw))
    else:
        from .faces import YuNetDetector
        YuNetDetector(raw,sha256_file(raw))
    return _manifest(root,SOURCES[name],revision,name)


def _native_checkpoint(name,root):
    """Native files remain gated until the appropriate conversion/runtime checks."""
    repo='google-research/scenic' if name=='factorized_vivit' else 'facebookresearch/av_hubert'
    revision=_github_revision(repo,'main')
    filename='scenic-checkpoint' if name=='factorized_vivit' else 'base_lrs3_iter5.pt'
    _download(SOURCES[name],_fk_paths.relocate((root/filename), __file__))
    _download(f'https://raw.githubusercontent.com/{repo}/{revision}/LICENSE',_fk_paths.relocate((root/'LICENSE'), __file__))
    metadata={'schema':'model-artifact-v1','source':SOURCES[name],'revision':revision,
              'architecture':'scenic-factorized-vivit-native' if name=='factorized_vivit' else 'official-avhubert-core-native',
              'weights_status':'upstream_download_runtime_or_conversion_pending',
              'files':{path.name:sha256_file(path) for path in root.iterdir()},
              'note':'candidate backbone checkpoint; exact AVTENet author checkpoint unverified'}
    save(_fk_paths.relocate((root/'artifact.json'), __file__),metadata)
    verified_files(root,metadata,require_pretrained=False)
    return metadata


def acquire_assets(output, names):
    """Acquire selected assets into new directories; retain successes and failures."""
    root=_fk_paths.relocate((Path(output)), __file__)
    root.mkdir(parents=True,exist_ok=False)
    report={'status':'ASSET_ACQUISITION_ATTEMPT','assets':{}}
    registry={'schema':'local-model-assets-v1','artifacts':{
        name:{'directory':None,'manifest':None} for name in ('factorized_vivit','ast','efficientnet_b0','wav2vec2_base')},
        'silero':{'path':None,'sha256':None},'detector':{'kind':None,'path':None,'sha256':None},
        'avhubert':{'python':None,'repository':None,'commit':None,'checkpoint':None,'checkpoint_sha256':None}}
    for name in names:
        if name not in ('ast','wav2vec2_base','efficientnet_b0','silero','yunet','factorized_vivit','avhubert_native'):
            raise ValueError('unsupported automatic asset: '+name)
        destination=_fk_paths.relocate((root/name), __file__)
        destination.mkdir()
        try:
            if name in ('ast','wav2vec2_base'):
                metadata=_hf_asset(name,destination)
            elif name=='efficientnet_b0':
                metadata=_efficientnet(destination)
            elif name in ('factorized_vivit','avhubert_native'):
                metadata=_native_checkpoint(name,destination)
            else:
                metadata=_github_file(name,destination)
            ready=metadata['weights_status']=='pretrained'
            report['assets'][name]={'status':'ACQUIRED_AND_CHECKED' if ready else 'DOWNLOADED_RUNTIME_OR_CONVERSION_PENDING','manifest':str(_fk_paths.relocate((destination/'artifact.json'), __file__)),
                                    'revision':metadata['revision']}
            if ready and name in registry['artifacts']:
                registry['artifacts'][name]={'directory':str(destination.resolve()),'manifest':str((_fk_paths.relocate((destination/'artifact.json'), __file__)).resolve())}
            elif name=='silero':
                path=_fk_paths.relocate((destination/'silero_vad.jit'), __file__)
                registry['silero']={'path':str(path.resolve()),'sha256':sha256_file(path)}
            elif name=='yunet':
                path=_fk_paths.relocate((destination/'face_detection_yunet_2023mar.onnx'), __file__)
                registry['detector']={'kind':'yunet','path':str(path.resolve()),'sha256':sha256_file(path)}
            elif name=='avhubert_native':
                path=_fk_paths.relocate((destination/'base_lrs3_iter5.pt'), __file__)
                registry['avhubert'].update(checkpoint=str(path.resolve()),checkpoint_sha256=sha256_file(path),commit=metadata['revision'])
        except Exception as error:
            # A failed artifact never gets promoted into a usable registry entry.
            report['assets'][name]={'status':'FAILED','error_type':type(error).__name__,'reason':str(error)}
    save(_fk_paths.relocate((root/'assets.local.json'), __file__),registry)
    report['asset_registry']=str(_fk_paths.relocate((root/'assets.local.json'), __file__))
    save(_fk_paths.relocate((root/'acquisition.json'), __file__),report)
    return report
