"""Deterministic prepared-input corruptions and immutable FFmpeg derivatives.

Strength grids are provisional; select realistic levels on train/validation later.
AV offset is a synchronization stress test, not automatically an authenticity label.
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
import subprocess
import sys

import cv2
import numpy as np

from .cli import save
from .forensics import sha256_file
from .media import probe, media_tool


def corrupt_frames(frames,kind,strength,*,seed=7):
    frames=np.asarray(frames)
    if frames.ndim!=4 or frames.shape[-1]!=3 or frames.dtype!=np.uint8 or frames.shape[0]==0:
        raise ValueError('expected nonempty uint8 RGB frames')
    if not math.isfinite(strength):raise ValueError('finite corruption strength required')
    outputs=[];generator=np.random.default_rng(seed)
    for frame in frames:
        if kind=='blur':
            if strength<=0:raise ValueError('blur sigma must be positive')
            image=cv2.GaussianBlur(frame,(0,0),strength)
        elif kind=='downsample':
            if not 0<strength<=1:raise ValueError('downsample ratio must lie in (0,1]')
            height,width=frame.shape[:2]
            image=cv2.resize(frame,(max(1,round(width*strength)),max(1,round(height*strength))),interpolation=cv2.INTER_AREA)
            image=cv2.resize(image,(width,height),interpolation=cv2.INTER_LINEAR)
        elif kind=='jpeg':
            if type(strength) is not int or not 1<=strength<=100:raise ValueError('JPEG quality must be integer1..100')
            ok,payload=cv2.imencode('.jpg',cv2.cvtColor(frame,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,strength])
            if not ok:raise ValueError('JPEG encoding failed')
            image=cv2.cvtColor(cv2.imdecode(payload,cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB)
        elif kind=='gaussian_noise':
            if strength<0:raise ValueError('noise standard deviation must be nonnegative')
            image=np.clip(frame.astype(np.float32)+generator.normal(0,strength,frame.shape),0,255).round().astype(np.uint8)
        else:raise ValueError('unknown visual corruption')
        outputs.append(image)
    return np.stack(outputs)


def audio_noise(samples,snr_db,*,seed=7):
    values=np.asarray(samples)
    if values.ndim!=1 or not np.issubdtype(values.dtype,np.floating) or not np.isfinite(values).all() or len(values)<2:
        raise ValueError('finite nonempty mono audio required')
    if not math.isfinite(snr_db):raise ValueError('finite SNR required')
    signal_power=float(np.mean(values.astype(np.float64)**2))
    if signal_power==0:raise ValueError('SNR is undefined for silent audio')
    noise=np.random.default_rng(seed).normal(size=len(values));noise-=noise.mean()
    noise*=math.sqrt(signal_power/(10**(snr_db/10)*float(np.mean(noise**2))))
    combined=values.astype(np.float64)+noise
    # Do not clip silently: clipping is a separate corruption with different SNR.
    return combined.astype(np.float32),{'requested_snr_db':snr_db,'seed':seed,
                                      'samples_outside_pcm_unit_range':int((np.abs(combined)>1).sum()),
                                      'alignment':'unchanged','label_policy':'review_before_benchmark_use'}


def transcode_media(path,output,kind,strength):
    """Save one explicitly described derivative with original/derived stream metadata."""
    root=_fk_paths.relocate((Path(output)), __file__);root.mkdir(parents=True,exist_ok=False)
    info=probe(path)
    video=next((stream for stream in info['streams'] if stream['codec_type']=='video'),None)
    if video is None:raise ValueError('video stream required')
    if not isinstance(strength,(int,float)) or not math.isfinite(strength):raise ValueError('finite strength required')
    ffmpeg=media_tool('ffmpeg')
    if not ffmpeg.is_file():raise RuntimeError('activate environment with FFmpeg')
    destination=_fk_paths.relocate((root/'derived.mkv'), __file__)
    arguments=[str(ffmpeg),'-v','error','-copyts','-start_at_zero','-i',str(path),'-map','0:v:0','-map','0:a:0?']
    label_policy='review_label_preservation_before_evaluation'
    if kind=='video_compression':
        if type(strength) is not int or not 0<=strength<=51:raise ValueError('CRF must be integer0..51')
        arguments+=['-c:v','libx264','-crf',str(strength),'-c:a','copy']
    elif kind=='video_blur':
        if strength<=0:raise ValueError('blur sigma must be positive')
        arguments+=['-vf',f'gblur=sigma={strength}','-c:v','ffv1','-c:a','copy']
    elif kind=='video_downsample':
        if not 0<strength<=1:raise ValueError('downsample ratio must lie in (0,1]')
        width,height=int(video['width']),int(video['height'])
        small_w,small_h=max(2,round(width*strength)),max(2,round(height*strength))
        arguments+=['-vf',f'scale={small_w}:{small_h}:flags=area,scale={width}:{height}:flags=bilinear',
                    '-c:v','ffv1','-c:a','copy']
    elif kind=='audio_compression':
        if type(strength) is not int or strength<8:raise ValueError('audio bitrate must be integer kbps >=8')
        if not any(stream['codec_type']=='audio' for stream in info['streams']):raise ValueError('audio stream required')
        arguments+=['-c:v','copy','-c:a','aac','-b:a',f'{strength}k']
    elif kind=='av_offset':
        if not any(stream['codec_type']=='audio' for stream in info['streams']):raise ValueError('audio stream required')
        if strength<0:raise ValueError('this profile explicitly supports nonnegative audio delay only')
        arguments+=['-c:v','copy','-af',f'adelay={round(strength*1000)}:all=1','-c:a','pcm_s16le']
        label_policy='synchronization_stress_do_not_inherit_original_authenticity_labels'
    else:raise ValueError('unknown media corruption')
    arguments+=['-fps_mode','passthrough','-threads','1',str(destination)]
    original_hash=sha256_file(path)
    subprocess.run(arguments,check=True,capture_output=True,timeout=300)
    if sha256_file(path)!=original_hash:raise ValueError('source changed during transcode')
    metadata={'schema':'controlled-media-derivative-v1','kind':kind,'strength':strength,
              'source_sha256':original_hash,'derived_sha256':sha256_file(destination),
              'arguments':arguments,'source_streams':info,'derived_streams':probe(destination),
              'label_policy':label_policy,'derivative':str(destination)}
    save(_fk_paths.relocate((root/'transformation.json'), __file__),metadata)
    return metadata
