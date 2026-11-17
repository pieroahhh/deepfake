"""Readiness reporting and explicit frozen-cache training entry points."""

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


import hashlib
import importlib.util
from pathlib import Path
import platform
import subprocess
from collections.abc import Sequence

from .artifacts import verified_files
from .cli import load_json, save
from .forensics import canonical, sha256_file


class CachedVideoPopulation(Sequence):
    """Load one video's verified frozen features at a time, not the entire dataset."""

    def __init__(self, metadata, root):
        self.metadata, self.root = metadata, _fk_paths.relocate((Path(root)), __file__)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        from .feature_cache import read_cache
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        entry = self.metadata[index]
        cache = self.root / entry['cache']
        if sha256_file(_fk_paths.relocate((cache / 'metadata.json'), __file__)) != entry['cache_metadata_sha256']:
            raise ValueError('cache metadata digest differs from trusted training manifest')
        provenance = entry['cache_provenance']
        if provenance['source_sha256'] != entry['source_sha256'] or provenance['window_intervals'] != entry['window_intervals']:
            raise ValueError('cache source/windows differ from training labels')
        values = read_cache(cache, provenance)
        row = {key: value for key, value in entry.items() if key not in ('cache', 'cache_metadata_sha256', 'cache_provenance')}
        if set(values) & set(row):
            raise ValueError('feature cache must not override trusted metadata/labels')
        row.update(values)
        row['feature_provenance'] = provenance
        return row


def readiness(assets):
    """Report missing artifacts as missing; architecture tests are a separate gate."""
    if assets['schema'] != 'local-model-assets-v1':
        raise ValueError('unsupported local asset registry')
    report = {'python':platform.python_version(), 'assets':{},
              'status':'NOT_READY_FOR_PRETRAINED_MEDIA_INFERENCE'}
    for name, entry in assets['artifacts'].items():
        if not entry['directory'] or not entry['manifest']:
            report['assets'][name]={'status':'MISSING_ARTIFACT'}
            continue
        try:
            metadata=verified_files(entry['directory'],entry['manifest'])
            report['assets'][name]={'status':'CHECKSUMS_VERIFIED_RUNTIME_NOT_TESTED',
                                    'revision':metadata['revision']}
        except (ValueError,OSError,KeyError) as error:
            report['assets'][name]={'status':'INVALID_ARTIFACT','reason':str(error)}
    for name, field in (('silero','path'),('detector','path')):
        entry=assets[name]
        if not entry[field] or not entry['sha256']:
            report['assets'][name]={'status':'MISSING_ARTIFACT'}
        else:
            try:
                matched=sha256_file(entry[field])==entry['sha256']
                report['assets'][name]={'status':'CHECKSUM_VERIFIED_RUNTIME_NOT_TESTED' if matched else 'INVALID_ARTIFACT'}
            except OSError as error:
                report['assets'][name]={'status':'INVALID_ARTIFACT','reason':str(error)}
    legacy=assets['avhubert']
    if all(legacy.values()):
        try:
            version=subprocess.run([legacy['python'],'-c','import sys; print("%d.%d" % sys.version_info[:2])'],
                                   capture_output=True,text=True,check=True,timeout=10).stdout.strip()
            matched=sha256_file(legacy['checkpoint'])==legacy['checkpoint_sha256']
            report['assets']['avhubert']={'status':'ARTIFACT_CONFIGURED_EXTRACTION_NOT_TESTED' if version=='3.8' and matched else 'INVALID_ARTIFACT'}
        except (OSError,subprocess.SubprocessError) as error:
            report['assets']['avhubert']={'status':'LEGACY_ENVIRONMENT_UNAVAILABLE','reason':str(error)}
    else:
        report['assets']['avhubert']={'status':'LEGACY_ENVIRONMENT_OR_CHECKPOINT_MISSING'}
    report['packages']={name:importlib.util.find_spec(name) is not None for name in
                        ('torch','transformers','torchvision','av','cv2','soundfile','safetensors')}
    return report


def fk_device(requested):
    import torch

    if requested not in ('auto', 'cuda', 'cpu'):
        raise ValueError('FK device must be auto, cuda, or cpu')
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable in this PyTorch environment')
    return torch.device('cuda:0' if requested == 'cuda' else 'cpu')


def train_fk_manifest(manifest_path, model_config_path, training_config_path, output,
                      *, epochs=None, resume_from=None, calibrate=False, device='cpu'):
    """Load an independently retained manifest and verified detached feature caches."""
    import torch
    from .model import EvidenceModel
    from .core import Decision
    from .training import fit_fk, seed_everything
    from .calibration import select_decision
    manifest=load_json(manifest_path)
    device = fk_device(device)
    root=_fk_paths.relocate((Path(manifest_path)), __file__).resolve().parent
    if manifest.get('schema') in ('fk-cache-manifest-v1', 'fk-cache-manifest-v2'):
        preprocessing = manifest['preprocessing']
        if manifest['schema'] == 'fk-cache-manifest-v2':
            from .quality import QUALITY_V2, profile_hash
            normalization = preprocessing['quality_normalization']
            if (preprocessing['quality_profile'] != QUALITY_V2
                    or normalization.get('schema') != QUALITY_V2 or normalization.get('fit_split') != 'train'
                    or profile_hash(normalization) != preprocessing['quality_normalization_sha256']):
                raise ValueError('invalid quality normalization profile or hash')
        policy_digest = hashlib.sha256(canonical(preprocessing)).hexdigest()
        for split in ('train', 'validation'):
            for row in manifest[split]:
                provenance = row['cache_provenance']
                if (provenance['preprocessing_sha256'] != policy_digest
                        or provenance['face_crop_policy'] != preprocessing['face_crop_policy']
                        or provenance['window_label_policy'] != manifest['window_label_policy']
                        or provenance.get('asset_sha256') != manifest.get('asset_sha256')):
                    raise ValueError('cache preprocessing or window labels differ from manifest policy')
    populations={split: CachedVideoPopulation(manifest[split], root) for split in ('train','validation')}
    settings=load_json(training_config_path)
    model_config=load_json(model_config_path)
    choices=model_config['reconstruction_choices']
    seed_everything(settings['seed'],settings['deterministic'])
    model=EvidenceModel(
        visual_dim=choices['visual_feature_dimension'],audio_dim=choices['audio_feature_dimension'],
        hidden=choices['gru_hidden'],dimension=model_config['paper_fixed']['dimension'],
        quality_dim=choices['quality_dimension'],decision=Decision(**model_config['decision']),
        **model_config.get('ablation', {}),
    ).to(device)
    if model_config.get('model_kind', 'fk') != 'fk':
        from .shared_baselines import SharedBaseline
        model = SharedBaseline(model_kind=model_config['model_kind'],
            visual_dim=choices['visual_feature_dimension'], audio_dim=choices['audio_feature_dimension'],
            hidden=choices['gru_hidden'], dimension=model_config['paper_fixed']['dimension'],
            quality_dim=choices['quality_dimension'], decision=Decision(**model_config['decision'])).to(device)
    policy = {key: manifest[key] for key in ('preprocessing', 'window_label_policy', 'asset_sha256') if key in manifest}
    policy['source_overlap_policy'] = manifest.get('source_overlap_policy','error')
    if 'git_commit' in manifest:
        policy['cache_git_commit'] = manifest['git_commit']
    result=fit_fk(model,populations['train'],populations['validation'],settings,output,
                  epochs=epochs,resume_from=resume_from,manifest_policy=policy)
    if calibrate:
        selection=select_decision(model,populations['validation'])
        selection['source_checkpoint_sha256']=sha256_file(_fk_paths.relocate((Path(output)/'best.pt'), __file__))
        checkpoint=torch.load(_fk_paths.relocate_value((_fk_paths.relocate((Path(output)/'best.pt'), __file__)), __file__),map_location='cpu',weights_only=True)
        checkpoint['decision']=selection['decision']
        checkpoint['validation_selection']=selection
        torch.save(checkpoint,_fk_paths.relocate((Path(output)/'selected.pt'), __file__))
        save(_fk_paths.relocate((Path(output)/'selection.json'), __file__),selection)
        result['validation_selection']=selection
    return result
