"""Exercise actual checksummed pretrained assets on explicitly synthetic inputs.

A successful forward pass validates runtime compatibility, not detection accuracy
or positive real-face recognition. Missing assets are reported, never substituted.
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


import argparse
import gc
from pathlib import Path

import numpy as np
import torch

from .cli import load_json,save


def validate_pretrained_assets(assets,output):
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    report={'data_kind':'synthetic_signals_with_actual_pretrained_weights',
            'purpose':'runtime_compatibility_not_accuracy','assets':{}}
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        try:
            from .backbones import ASTBackend,prepare_ast
            entry=assets['artifacts']['ast']
            if not entry['directory'] or not entry['manifest']:raise ValueError('AST artifact is missing')
            audio=(.1*np.sin(2*np.pi*440*np.arange(32000)/16000)).astype(np.float32)
            values=prepare_ast(audio,16000,entry['directory'],entry['manifest'])
            backend=ASTBackend.from_local(entry['directory'],entry['manifest'],token='cls').eval()
            with torch.inference_mode():features=backend(values)
            if not torch.isfinite(features).all():raise ValueError('nonfinite AST output')
            report['assets']['ast']={'status':'RUNTIME_CHECK_PASSED','prepared_shape':list(values.shape),
                                     'feature_shape':list(features.shape),'token':'cls',
                                     'revision':backend.artifact['revision']}
            del backend,features,values;gc.collect()
        except Exception as error:
            report['assets']['ast']={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
        try:
            from .encoders import FrozenFKEncoders
            from .model import EvidenceModel
            from .inference import record_from_outputs
            from .forensics import commitment,verify
            visual=assets['artifacts']['efficientnet_b0'];audio=assets['artifacts']['wav2vec2_base']
            if not all((visual['directory'],visual['manifest'],audio['directory'],audio['manifest'])):
                raise ValueError('both FK pretrained encoder artifacts are required')
            encoders=FrozenFKEncoders.from_local(visual['directory'],visual['manifest'],audio['directory'],audio['manifest'])
            generator=torch.Generator().manual_seed(7)
            pixels=torch.rand(50,3,224,224,generator=generator)
            waveform=.1*torch.sin(2*torch.pi*440*torch.arange(32000)/16000)
            values=encoders.encode_window(pixels,waveform,torch.arange(50)/25)
            torch.manual_seed(7)
            model=EvidenceModel().eval()
            with torch.inference_mode():result=model(**values,quality=torch.ones(1,2))
            windows=[{'window_id':0,'interval':[0.,2.],'status':'USABLE','reasons':[],
                      'quality':{'provenance':'synthetic_pixels_and_tone_actual_pretrained_encoders'}}]
            record=record_from_outputs('0'*64,2.,windows,{0:result},model,
                                       model_version='SYNTHETIC_UNTRAINED_FK_HEAD',
                                       preprocessing={'provider_status':'actual_pretrained_encoders_synthetic_inputs',
                                                      'source_hash_note':'synthetic tensor fixture; no source media file'})
            verification=verify(record,commitment(record))
            save(_fk_paths.relocate((root/'pretrained-encoder-synthetic-record.json'), __file__),record)
            save(_fk_paths.relocate((root/'pretrained-encoder-synthetic-verification.json'), __file__),verification)
            report['assets']['fk_encoders']={'status':'RUNTIME_CHECK_PASSED',
                'visual_shape':list(values['visual_features'].shape),'audio_shape':list(values['audio_features'].shape),
                'audio_feature_centers':len(values['audio_times']),'record_verification':verification,
                'classifier':'random_FK_head_not_trained_detector',
                'visual_revision':encoders.artifacts['visual']['revision'],'audio_revision':encoders.artifacts['audio']['revision']}
            del encoders,model,result,values,pixels;gc.collect()
        except Exception as error:
            report['assets']['fk_encoders']={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
        try:
            from .vad import SileroVAD
            entry=assets['silero']
            if not entry['path'] or not entry['sha256']:raise ValueError('Silero artifact is missing')
            vad=SileroVAD(entry['path'],entry['sha256'])
            intervals=vad(np.zeros(32000,dtype=np.float32))
            report['assets']['silero']={'status':'RUNTIME_CHECK_PASSED','input':'two_seconds_synthetic_silence',
                                        'speech_intervals':intervals,'model_sha256':entry['sha256']}
            del vad;gc.collect()
        except Exception as error:
            report['assets']['silero']={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
        try:
            from .faces import YuNetDetector
            entry=assets['detector']
            if entry['kind']!='yunet' or not entry['path']:raise ValueError('YuNet artifact is missing')
            detector=YuNetDetector(entry['path'],entry['sha256'])
            detections=detector(np.zeros((320,320,3),dtype=np.uint8))
            report['assets']['yunet']={'status':'RUNTIME_CHECK_PASSED','input':'blank_synthetic_image',
                                       'detections':len(detections),'real_face_validation':False}
        except Exception as error:
            report['assets']['yunet']={'status':'FAILED','reason':str(error),'error_type':type(error).__name__}
    finally:torch.set_num_threads(previous)
    report['all_supplied_runtime_checks_passed']=all(item['status']=='RUNTIME_CHECK_PASSED' for item in report['assets'].values())
    report['still_pending']=['factorized ViViT access/conversion/parity','AV-HuBERT download/legacy runtime',
                              'consented real-face video validation','dataset training/evaluation']
    save(_fk_paths.relocate((root/'runtime-checks.json'), __file__),report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    import json
    print(json.dumps(validate_pretrained_assets(load_json(args.assets),args.output),indent=2))


if __name__=='__main__':main()
