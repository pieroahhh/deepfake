"""FK raw-video orchestration and float32-to-canonical forensic record export.

Dependencies/quality policy are explicit. Synthetic providers exercise mechanics
but must never be described as pretrained or accurate detectors.
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


import math
from pathlib import Path

import torch

from .cli import save
from .forensics import build_record,commitment,preserve,sha256_file,verify
from .preparation import (IN_FRAME_CROP_POLICY, STRICT_CROP_POLICY, encode_fk_window,
                          fk_source_speech, prepare_fk_window, save_prepared_window, FK_QUALITY_PROFILE)


def record_from_outputs(source_hash,duration,windows,outputs,model,*,model_version,preprocessing):
    """Preserve signed float32 contributions; recompute canonical scores in Python.

    Record arithmetic is binary64 from stored contributions, so score reconstruction
    remains strict. Raw model-vs-canonical rounding differences are recorded.
    """
    usable={window['window_id'] for window in windows if window['status']=='USABLE'}
    if set(outputs)!=usable:
        raise ValueError('model outputs must match exactly the usable window IDs')
    records=[];raw_bins=[];raw_windows=[]
    for window in windows:
        record={**window,'quality':dict(window['quality'])}
        if window['status']=='USABLE':
            result=outputs[window['window_id']]
            bins=model.decision.bins
            if (tuple(result['alpha'].shape)!=(1,bins,4)
                    or tuple(result['contributions'].shape)!=(1,bins,4)
                    or tuple(result['bin_scores'].shape)!=(1,bins)
                    or tuple(result['window_scores'].shape)!=(1,)):
                raise ValueError('one correctly shaped model result required per window')
            start,end=window['interval'];width=(end-start)/bins
            alpha=result['alpha'].detach().cpu().double().tolist()[0]
            contributions=result['contributions'].detach().cpu().double().tolist()[0]
            record['bins']=[{'interval':[start+index*width,start+(index+1)*width],
                             'alpha':alpha[index],'contributions':contributions[index]}
                            for index in range(bins)]
            raw_bins.extend(result['bin_scores'].detach().cpu().double().tolist()[0])
            raw_windows.append(result['window_scores'].detach().reshape(()))
        records.append(record)
    record=build_record(source_hash,duration,records,model.decision,
                        model_version=model_version,preprocessing=preprocessing)
    exported_bins=[item['score'] for window in record['windows'] if window['status']=='USABLE' for item in window['bins']]
    exported_windows=[window['score'] for window in record['windows'] if window['status']=='USABLE']
    errors=[abs(a-b) for a,b in zip(raw_bins,exported_bins)]
    errors.extend(abs(a.item()-b) for a,b in zip(raw_windows,exported_windows))
    raw_video=None if not raw_windows else model.aggregate_video(torch.stack(raw_windows)).item()
    if raw_video is not None:
        errors.append(abs(raw_video-record['score']))
    maximum=max(errors,default=0.)
    if maximum>5e-6:
        raise ValueError('model output differs materially from stored-contribution reconstruction')
    raw_prediction=None if raw_video is None else int(raw_video>=model.decision.video_threshold)
    record['preprocessing']={**preprocessing,'numeric_export':{
        'canonical_arithmetic':'binary64_from_stored_signed_contributions',
        'raw_model_video_score':raw_video,'maximum_model_rounding_difference':maximum,
        'raw_model_prediction':raw_prediction,
        'prediction_changed_by_canonical_rounding':raw_prediction!=record['prediction'],
    }}
    verify(record,commitment(record))
    return record


@torch.inference_mode()
def run_fk_video(path,source_duration,model,encoders,detector,vad,quality_mapper,output,
                 *,model_version,quality_profile,provider_status,fit_to_frame=False,provenance=None):
    """Source -> preparation -> frozen features -> FK -> committed verified record.

    quality_mapper receives measured quality and must return one [1,quality_dim]
    tensor. Its named policy is recorded, never silently calibrated on test data.
    provider_status is 'synthetic' or 'pretrained_assets_with_checkpoint'.
    """
    if (not math.isfinite(source_duration) or source_duration<=0 or not quality_profile
            or provider_status not in ('synthetic','pretrained_assets_with_checkpoint')):
        raise ValueError('valid source duration, named quality policy and provider status required')
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    working=_fk_paths.relocate((root/('working-'+Path(path).name)), __file__)
    source_hash=preserve(path,working)
    from .media import source_metadata
    metadata = source_metadata(working)
    metadata['filename'] = _fk_paths.relocate((Path(path)), __file__).name
    if abs(metadata['duration'] - source_duration) > .1:
        raise ValueError('supplied duration differs from source metadata')
    derived = []
    model.eval();encoders.eval()
    use_pcm16 = quality_profile != FK_QUALITY_PROFILE
    try:
        speech = fk_source_speech(working, source_duration, vad, pcm16=use_pcm16)
        speech_error = None
    except ValueError as error:
        speech, speech_error = [], str(error)
    windows=[];outputs={}
    device=next(model.parameters()).device
    for index in range(math.ceil(source_duration)):
        prepared=prepare_fk_window(working,float(index),source_duration,detector,speech,
                                   fit_to_frame=fit_to_frame, pcm16=use_pcm16)
        window={'window_id':index,'interval':[float(index),float(index)+2.],
                'status':prepared['status'],'reasons':prepared['reasons'],
                'quality':prepared['quality_measurements']}
        if prepared['status']=='USABLE':
            directory = _fk_paths.relocate((root/'derived'/f'{index:06d}'), __file__)
            saved = save_prepared_window(prepared, directory)
            import cv2
            representative = _fk_paths.relocate((directory/'representative.png'), __file__)
            if not cv2.imwrite(str(representative), cv2.cvtColor(prepared['faces'][25], cv2.COLOR_RGB2BGR)):
                raise OSError('could not save representative frame')
            for name in (*saved['derived_sha256'], 'metadata.json', 'representative.png'):
                artifact = _fk_paths.relocate((directory/name), __file__)
                derived.append({'path': artifact.relative_to(root).as_posix(), 'sha256': sha256_file(artifact),
                                'source_sha256': source_hash, 'interval': prepared['interval'],
                                'window_id': index, 'type': name, 'preprocessing': saved['derivation']})
            window['representative_frame'] = representative.relative_to(root).as_posix()
            features = encode_fk_window(prepared, encoders)
            quality=quality_mapper(prepared['quality_measurements'])
            if tuple(quality.shape)!=(1,model.quality_dim) or not torch.isfinite(quality).all():
                raise ValueError('quality mapping produced incompatible values')
            features={key:value.to(device) for key,value in features.items()}
            outputs[index]=model(**features,quality=quality.to(device))
        windows.append(window)
    preprocessing={'pipeline':'fk-source-to-record-v1','provider_status':provider_status,
                   'detector':detector.method,'quality_profile':quality_profile,
                   'source_time_speech_intervals':speech,'window_seconds':2.,'stride_seconds':1.,
                   'fps':25, 'face_pixels':224, 'audio_sample_rate':16000,
                   'audio_pcm_policy':'pcm16-roundtrip-v1' if use_pcm16 else 'legacy_float_npz_with_pcm16_derivative',
                   'vad_settings':{'threshold':.5,'minimum_speech':.25,'merge_gap':.3,'padding':.1},
                   'torch_version':str(torch.__version__),
                   'ablations':{'disabled_evidence':list(model.disabled_evidence),'uniform_weights':model.uniform_weights}}
    preprocessing['face_crop_policy'] = IN_FRAME_CROP_POLICY if fit_to_frame else STRICT_CROP_POLICY
    if speech_error is not None:
        preprocessing['speech_detection_error'] = speech_error
    if provenance is not None:
        preprocessing.update(provenance)
    record=record_from_outputs(source_hash,source_duration,windows,outputs,model,
                               model_version=model_version,preprocessing=preprocessing)
    record.update(schema_version='fk-record-v2', source_metadata=metadata, derived_evidence=derived,
                  optional_review={'caption_transcript': 'NOT_RUN', 'perturbation': 'NOT_RUN'})
    trusted=commitment(record)
    result=verify(record,trusted,source=path,artifact_root=root)
    save(_fk_paths.relocate((root/'record.json'), __file__),record)
    with (_fk_paths.relocate((root/'trusted-commitment.txt'), __file__)).open('x') as stream:stream.write(trusted+'\n')
    save(_fk_paths.relocate((root/'verification.json'), __file__),result)
    return {'record':str(_fk_paths.relocate((root/'record.json'), __file__)),'verification':result,'status':record['status'],
            'provider_status':provider_status}
