"""Explicit paired-input conversion, usable in the isolated AV-HuBERT environment.

Requires python-speech-features from that environment. Crop/alignment policy is
our reconstruction; dataset normalization must match the selected checkpoint.
"""

import numpy as np
import torch


def prepare_avhubert(waveform, lips_gray, settings):
    """Return unpadded aligned audio[B,F,T] and video[B,1,T,H,W].

    waveform is retained mono int16 PCM or normalized 16kHz float mono;
    lips_gray is already aligned uint8 [T,H,W].
    No silent substitution of librosa/Kaldi filterbanks or per-stream time shifts.
    """
    try:
        from python_speech_features import logfbank
    except ImportError as error:
        raise RuntimeError('AV-HuBERT preparation requires the isolated python-speech-features environment') from error
    samples=np.asarray(waveform)
    lips=np.asarray(lips_gray)
    floating=np.issubdtype(samples.dtype,np.floating)
    if (samples.ndim!=1 or not (floating or samples.dtype==np.int16) or not np.isfinite(samples).all()
            or samples.size==0 or (floating and np.abs(samples).max()>1)
            or lips.ndim!=3 or lips.dtype!=np.uint8 or lips.shape[0]==0):
        raise ValueError('expected mono int16/normalized float PCM and uint8 grayscale lip sequence')
    if abs(len(samples)/16000-len(lips)/25)>1e-6:
        raise ValueError('audio/lip source intervals are not aligned at 16kHz/25fps')
    fb=settings['filterbank_settings']
    if set(fb)!= {'winlen','winstep','nfilt','nfft','lowfreq','highfreq','preemph'}:
        raise ValueError('all log-filterbank parameters must be explicit')
    pcm=(np.clip(np.floor(samples*32768),-32768,32767).astype(np.int16) if floating else samples)
    # python-speech-features 0.6 logfbank does not expose winfunc. Its fbank
    # implementation already uses the required rectangular window by default.
    features=logfbank(pcm,samplerate=16000,**fb)
    stack=settings['stack_order_audio']
    if type(stack) is not int or stack<1:
        raise ValueError('invalid audio frame stacking')
    missing=(-len(features))%stack
    features=np.pad(features,((0,missing),(0,0))).reshape(-1,features.shape[1]*stack)
    if len(features)!=len(lips):
        raise ValueError('stacked filterbanks and lips differ in length; automatic time repair forbidden')
    audio=torch.from_numpy(features.astype(np.float32))
    if settings['normalize_audio']:
        audio=torch.nn.functional.layer_norm(audio,(audio.shape[-1],))
    size=settings['crop_size']
    height,width=lips.shape[1:]
    if type(size) is not int or size<1 or min(height,width)<size or settings['image_std']<=0:
        raise ValueError('invalid lip center crop/normalization')
    top,left=(height-size)//2,(width-size)//2
    video=torch.from_numpy(lips[:,top:top+size,left:left+size].copy()).float()/255
    video=(video-settings['image_mean'])/settings['image_std']
    return {'audio':audio.T.unsqueeze(0),'video':video.unsqueeze(0).unsqueeze(0)}
