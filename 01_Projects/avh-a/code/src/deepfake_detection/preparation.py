"""Connect source-time decoding, explicit tracking and FK window assessment."""

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

import cv2
import numpy as np
import soundfile as sf

from .cli import save
from .core import coverage, union
from .faces import track_and_crop
from .forensics import sha256_file
from .media import aligned_audio, sample_video


IN_FRAME_CROP_POLICY = 'in-frame-landmark-preserving-v1'
STRICT_CROP_POLICY = 'strict-observed-square-v1'
FK_QUALITY_PROFILE = 'availability-speech-v1-provisional'


def pcm16_samples(samples):
    """The float representation of the exact signed PCM16 samples retained."""
    return (np.clip(np.rint(samples * 32768), -32768, 32767).astype(np.int16).astype(np.float32) / 32768)


def fk_source_speech(path, duration, vad, *, pcm16=True):
    audio = aligned_audio(path, 0., duration, require_complete=False)
    samples = pcm16_samples(audio['samples']) if pcm16 else audio['samples']
    return [(start, min(end, duration)) for start, end in vad(samples) if start < duration]


def encode_fk_window(prepared, encoders):
    import torch

    device = next(encoders.parameters()).device
    faces = torch.from_numpy(prepared['faces'].copy()).permute(0, 3, 1, 2).float() / 255
    audio = torch.from_numpy(prepared['audio'].copy()).float()
    times = torch.from_numpy(prepared['visual_times'].copy()).float()
    return encoders.encode_window(faces.to(device), audio.to(device), times.to(device))


def fk_availability_quality(measurements):
    import torch

    return torch.tensor([[measurements['face_coverage'], measurements['speech_coverage']]], dtype=torch.float32)


def prepare_fk_window(path, start, source_duration, detector, source_speech_intervals, *, fit_to_frame=False, pcm16=True):
    """Prepare one 2s window using previously computed source-time VAD intervals.

    Detector and source VAD are mandatory inputs. No fixture/heuristic is silently
    installed as a real model. Quality measures are recorded, not calibrated gates.
    """
    if not np.isfinite(source_duration) or source_duration <= 0 or not np.isfinite(start) or start < 0:
        raise ValueError('explicit valid source duration and start are required')
    source_speech_intervals = union(source_speech_intervals)
    if any(end > source_duration for _, end in source_speech_intervals):
        raise ValueError('source VAD intervals must be clipped to the source duration')
    result = {'status': 'UNASSESSED', 'interval': [start, start+2.], 'reasons': [],
              'source_sha256': sha256_file(path), 'detector': detector.method,
              'quality_measurements': {}, 'derivation': 'fk-window-preparation-reconstruction-v1'}
    if fit_to_frame:
        result['face_crop_policy'] = IN_FRAME_CROP_POLICY
    if start + 2 > source_duration + 1e-9:
        result['reasons'].append('incomplete_window')
        return result
    spoken = coverage(2., [(max(0., a-start), min(2., b-start))
                           for a, b in source_speech_intervals if min(2., b-start) > max(0., a-start)])
    speech_fraction = spoken['covered_duration']/2
    result['quality_measurements']['speech_coverage'] = speech_fraction
    if speech_fraction < .5:
        result['reasons'].append('insufficient_speech')
    video = audio = records = None
    try:
        video = sample_video(path, start, 2., fps=25, maximum_time_error=.06)
        relative = video['source_times'] - start
        if ((relative < 0) | (relative >= 2)).any():
            raise ValueError('sampled source timestamps fall outside this half-open window')
        records = track_and_crop(video['rgb'], detector, fit_to_frame=fit_to_frame)
        fraction = sum(row['status']=='USABLE' for row in records)/len(records)
        result['quality_measurements']['face_coverage'] = fraction
        result['sampled_source_times'] = video['source_times'].tolist()
        result['duplicate_frame_samples'] = video['duplicate_samples']
        if fraction < 1:
            result['reasons'].append('unusable_face_track')
            result['face_failures'] = [{'frame': index, 'reason': row['reason']}
                                       for index,row in enumerate(records) if row['status']!='USABLE']
    except ValueError as error:
        result['reasons'].append(str(error))
    try:
        audio = aligned_audio(path, start, 2., sample_rate=16000, require_complete=False)
        result['quality_measurements']['audio_coverage'] = audio['coverage']
        result['quality_measurements']['audio_rms'] = float(np.sqrt(np.mean(audio['samples']**2)))
        result['quality_measurements']['audio_clipping_fraction'] = float((np.abs(audio['samples'])>=1).mean())
        if pcm16:
            audio['samples'] = pcm16_samples(audio['samples'])
        speech_mask = np.zeros(len(audio['samples']), dtype=bool)
        for a, b in spoken['assessed']:
            speech_mask[round(a*16000):round(b*16000)] = True
        result['quality_measurements']['estimated_snr_db'] = None
        result['quality_measurements']['snr_unavailable_reason'] = 'insufficient_speech_or_noise_reference'
        if speech_mask.sum() >= 1600 and (~speech_mask).sum() >= 1600:
            signal = float(np.mean(audio['samples'][speech_mask].astype(np.float64)**2))
            noise = float(np.mean(audio['samples'][~speech_mask].astype(np.float64)**2))
            if noise > 1e-12 and signal > noise:
                result['quality_measurements']['estimated_snr_db'] = float(10*np.log10((signal-noise)/noise))
                result['quality_measurements']['snr_unavailable_reason'] = None
        if not audio['present'].all():
            result['reasons'].append('incomplete_audio_coverage')
    except ValueError as error:
        result['reasons'].append(str(error))
    if sha256_file(path) != result['source_sha256']:
        result['reasons'].append('source_changed_during_preparation')
    if result['reasons']:
        return result
    result['status'] = 'USABLE'
    result['faces'] = np.stack([row['face'] for row in records])
    result['lips'] = np.stack([row['lips'] for row in records])
    result['audio'] = audio['samples']
    result['visual_times'] = video['source_times'] - start
    result['quality_measurements']['mean_detection_confidence'] = float(np.mean([r['detection']['score'] for r in records]))
    result['quality_measurements']['mean_tracking_iou'] = float(np.mean([r['tracking_iou'] for r in records]))
    result['quality_measurements']['mean_face_laplacian_variance'] = float(np.mean([
        cv2.Laplacian(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),cv2.CV_64F).var() for image in result['faces']
    ]))
    return result


def save_prepared_window(result, output):
    """Write a new derivation directory; retain 16-bit mono PCM and array hashes."""
    output=_fk_paths.relocate((Path(output)), __file__)
    output.mkdir(parents=True,exist_ok=False)
    metadata={key:value for key,value in result.items() if key not in ('faces','lips','audio','visual_times')}
    if result['status']=='USABLE':
        with (_fk_paths.relocate((output/'prepared.npz'), __file__)).open('xb') as stream:
            np.savez_compressed(stream, **{key:result[key] for key in ('faces','lips','audio','visual_times')})
        sf.write(_fk_paths.relocate((output/'audio-16k-pcm16.wav'), __file__),np.clip(result['audio'],-1,1),16000,subtype='PCM_16',format='WAV')
        metadata['derived_sha256']={name:sha256_file(_fk_paths.relocate((output/name), __file__)) for name in ('prepared.npz','audio-16k-pcm16.wav')}
        metadata['audio_pcm_policy']='clip to [-1,1], quantize to PCM16; retain float decode separately in NPZ'
    save(_fk_paths.relocate((output/'metadata.json'), __file__),metadata)
    return metadata
