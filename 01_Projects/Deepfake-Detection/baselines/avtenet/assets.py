"""Load actual expert weights with pinned provenance; never fall back to random experts."""

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
import subprocess
import sys
from types import SimpleNamespace

from baselines.common.io import read, digest, fingerprint


def registry(path):
    if path is None:
        raise ValueError('AVTENet needs --assets pointing to a local asset registry')
    value=read(path)
    if value.get('schema')!='baseline-assets-v1':
        raise ValueError('expected baseline-assets-v1 registry')
    base=_fk_paths.relocate((Path(path)), __file__).resolve().parent
    for entry in value['experts'].values():
        for key in ('directory','manifest','repository','checkpoint'):
            if entry.get(key):
                candidate=_fk_paths.relocate((Path(entry[key])), __file__)
                entry[key]=str(candidate if candidate.is_absolute() else (_fk_paths.relocate((base/candidate), __file__)).resolve())
    return value


def evidence(assets,stage):
    entry=assets['experts'][stage]
    if stage=='audiovisual':
        if not entry.get('checkpoint') or digest(entry['checkpoint'])!=entry.get('checkpoint_sha256'):
            raise ValueError('AV-HuBERT checkpoint missing or checksum mismatch')
        if not entry.get('commit') or len(entry['commit'])!=40:
            raise ValueError('AV-HuBERT requires an immutable commit')
        return {k:v for k,v in entry.items() if k not in ('repository','checkpoint')}
    from deepfake_detection.artifacts import verified_files
    metadata=verified_files(entry['directory'],entry['manifest'])
    if stage=='visual' and metadata['architecture']!='transformers-vivit-joint-adaptation-v1':
        parity=metadata.get('parity',{})
        if parity.get('status')!='PASSED' or not parity.get('reference_revision'):
            raise ValueError('factorized ViViT requires recorded official-reference conversion parity')
    return metadata


def load_backend(assets,stage):
    from deepfake_detection.backbones import FactorizedViViT,ASTBackend,AVHubertBackend
    proof=evidence(assets,stage)
    entry=assets['experts'][stage]
    if stage=='visual':
        if proof['architecture']=='transformers-vivit-joint-adaptation-v1':
            from .hf_vivit import HFVivitBackend
            backend=HFVivitBackend.from_local(entry['directory'],entry['manifest'])
        else:backend=FactorizedViViT.from_local(entry['directory'],entry['manifest'])
    elif stage=='audio':
        backend=ASTBackend.from_local(entry['directory'],entry['manifest'],token='cls')
    else:
        if sys.version_info[:2]!=(3,8) or sys.platform=='win32':
            raise RuntimeError('AV-HuBERT training runs in the isolated Linux Python 3.8 runtime; use runtime/invoke-wsl.ps1')
        repo=_fk_paths.relocate((Path(entry['repository'])), __file__)
        actual=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
        if actual!=entry['commit']:
            raise ValueError('AV-HuBERT source revision mismatch')
        for extra in ([],['--cached']):
            subprocess.run(['git','-C',str(repo),'diff','--exit-code',*extra],check=True,capture_output=True)
        expected=subprocess.run(['git','-C',str(repo),'ls-tree','HEAD','fairseq'],check=True,capture_output=True,text=True).stdout.split()[2]
        actual=subprocess.run(['git','-C',str(_fk_paths.relocate((repo/'fairseq'), __file__)),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
        if expected!=actual:
            raise ValueError('Fairseq submodule revision mismatch')
        from fairseq import checkpoint_utils,utils
        utils.import_user_module(SimpleNamespace(user_dir=str(_fk_paths.relocate((repo/'avhubert'), __file__))))
        models,cfg,task=checkpoint_utils.load_model_ensemble_and_task([entry['checkpoint']])
        if len(models)!=1:raise ValueError('expected a single AV-HuBERT core checkpoint')
        for configured,saved in (('image_mean','image_mean'),('image_std','image_std'),('crop_size','image_crop_size'),
                                 ('stack_order_audio','stack_order_audio'),('normalize_audio','normalize')):
            # Legacy checkpoints omit defaults. The pinned upstream task merges
            # its dataclass defaults when constructed; compare that resolved cfg.
            if entry['preparation'][configured]!=getattr(task.cfg,saved,None):
                raise ValueError('AV-HuBERT preparation disagrees with checkpoint: '+configured)
        backend=AVHubertBackend(models[0])
    if backend.feature_dimension!=768:
        raise ValueError('this AVTENet profile requires 768-dimensional experts')
    return backend,proof


def check_assets(path):
    result={}
    try:assets=registry(path)
    except Exception as error:return {'assets':{'ok':False,'reason':str(error)}}
    for stage in ('visual','audio','audiovisual'):
        try:
            proof=evidence(assets,stage)
            result[stage]={'ok':True,'provenance_sha256':fingerprint(proof),'runtime_exercised':False}
        except Exception as error:result[stage]={'ok':False,'reason':str(error)}
    if sys.platform=='win32' or sys.version_info[:2]!=(3,8):
        result['avhubert_runtime']={'ok':False,'reason':'use the isolated Linux Python 3.8 worker for AVN training/extraction'}
    else:
        try:
            import fairseq
            result['avhubert_runtime']={'ok':True,'runtime_exercised':False}
        except Exception as error:result['avhubert_runtime']={'ok':False,'reason':str(error)}
    return result
