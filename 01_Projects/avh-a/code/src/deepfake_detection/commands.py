"""High-level commands joining explicit assets with reusable research tooling."""

from pathlib import Path

import torch

from .cli import load_json,save
from .forensics import sha256_file
from .provenance import git_provenance


def load_fk_assets(assets):
    from .encoders import FrozenFKEncoders
    from .faces import YuNetDetector,Dlib68Detector
    from .vad import SileroVAD

    artifacts = assets['artifacts']
    for name in ('efficientnet_b0','wav2vec2_base'):
        if not artifacts[name]['directory'] or not artifacts[name]['manifest']:
            raise ValueError('actual pretrained artifact required: '+name)
    encoders=FrozenFKEncoders.from_local(
        artifacts['efficientnet_b0']['directory'],artifacts['efficientnet_b0']['manifest'],
        artifacts['wav2vec2_base']['directory'],artifacts['wav2vec2_base']['manifest'],
    )
    detector_settings=assets['detector']
    if detector_settings['kind']=='yunet':
        detector=YuNetDetector(detector_settings['path'],detector_settings['sha256'])
    elif detector_settings['kind']=='dlib68':
        detector=Dlib68Detector(detector_settings['path'],detector_settings['sha256'])
    else:raise ValueError('configure an explicit supported learned detector; no synthetic fallback')
    vad=SileroVAD(assets['silero']['path'],assets['silero']['sha256'])
    return encoders, detector, vad


def fk_asset_hashes(assets, encoders):
    """Identify the verified weights and encoder configuration consumed by FK."""
    required = {
        'visual': ('model.safetensors', 'preprocessor_config.json'),
        'audio': ('model.safetensors', 'config.json', 'preprocessor_config.json'),
    }
    return {
        **{name: {file: encoders.artifacts[name]['files'][file] for file in files}
           for name, files in required.items()},
        'detector': {'kind': assets['detector']['kind'], 'sha256': assets['detector']['sha256']},
        'silero': assets['silero']['sha256'],
    }


def infer_fk_from_assets(path, duration, checkpoint, assets_path, output, quality_profile, *, fit_to_frame=None):
    from .inference import run_fk_video
    from .preparation import (FK_QUALITY_PROFILE, IN_FRAME_CROP_POLICY, STRICT_CROP_POLICY,
                              fk_availability_quality)
    from .training import load_fk_checkpoint

    from .quality import QUALITY_V2, quality_tensor, profile_hash
    if quality_profile not in (FK_QUALITY_PROFILE, QUALITY_V2):
        raise ValueError('unknown FK quality profile')
    assets = load_json(assets_path)
    encoders, detector, vad = load_fk_assets(assets)
    model, metadata = load_fk_checkpoint(checkpoint)
    if metadata.get('model_spec', {}).get('model_kind', 'fk') != 'fk':
        raise ValueError('forensic inference requires an FK checkpoint; use evaluate-fk for baselines')
    policy = metadata.get('preprocessing', {})
    recorded_quality = policy.get('quality_profile', FK_QUALITY_PROFILE)
    if recorded_quality != quality_profile:
        raise ValueError('inference quality profile differs from checkpoint')
    mapper = fk_availability_quality
    if quality_profile == QUALITY_V2:
        normalization = policy['quality_normalization']
        if profile_hash(normalization) != policy['quality_normalization_sha256']:
            raise ValueError('checkpoint quality normalization hash mismatch')
        mapper = lambda measurements: quality_tensor(measurements, normalization)
    asset_hashes = fk_asset_hashes(assets, encoders)
    if 'asset_sha256' in metadata and asset_hashes != metadata['asset_sha256']:
        raise ValueError('inference assets differ from the checkpoint training assets')
    if model.quality_dim != 2:
        raise ValueError('provisional quality policy requires quality dimension 2')
    recorded_crop = metadata.get('preprocessing', {}).get('face_crop_policy')
    if recorded_crop is not None:
        if recorded_crop not in (IN_FRAME_CROP_POLICY, STRICT_CROP_POLICY):
            raise ValueError('unknown checkpoint face crop policy')
        recorded_fit = recorded_crop == IN_FRAME_CROP_POLICY
        if fit_to_frame is not None and fit_to_frame != recorded_fit:
            raise ValueError('requested face crop policy differs from checkpoint training policy')
        fit_to_frame = recorded_fit
    return run_fk_video(
        path,duration,model,encoders,detector,vad,mapper,output,
        model_version=sha256_file(checkpoint),quality_profile=quality_profile,
        provider_status='pretrained_assets_with_checkpoint',fit_to_frame=bool(fit_to_frame),
        provenance={**git_provenance(), **policy, 'asset_sha256': asset_hashes,
                    'checkpoint_git_commit': metadata.get('git_commit')},
    )


def profile_synthetic_fk(output,*,iterations=20,threads=2):
    from .model import EvidenceModel
    from .profiling import profile_stage,parameter_storage
    if type(threads) is not int or threads<1:raise ValueError('positive CPU thread count required')
    previous=torch.get_num_threads();torch.set_num_threads(threads)
    try:
        torch.manual_seed(7)
        model=EvidenceModel().eval()
        inputs=(torch.randn(1,50,1280),torch.randn(1,99,768),torch.arange(50)/25,
                (torch.arange(99)*320+199.5)/16000,torch.ones(1,2))
        result=profile_stage(lambda:model(*inputs),name='FK-head-only-synthetic-features',
                             unit='one_synthetic_two_second_window',iterations=iterations)
        result['parameters']=parameter_storage(model)
        result['data_kind']='synthetic';result['end_to_end']=False
        save(output,result)
        return result
    finally:torch.set_num_threads(previous)
