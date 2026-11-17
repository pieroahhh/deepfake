"""Declared AVFakeNet preparation choices; face detection remains external.

These helpers implement reconstruction-v1 image conversion, not the unknown
original spectrogram renderer. They do not download media or detector weights.
"""

import numpy as np
import torch
from torch.nn import functional as F


def prepare_face(rgb_crop, profile):
    """Resize an already detected uint8 RGB crop and scale to [0, 1].

    MTCNN detection, tracking, crop margins and frame selection are not supplied.
    An OpenCV BGR frame must be converted to RGB by the caller first.
    """
    crop = np.asarray(rgb_crop)
    if crop.ndim != 3 or crop.shape[-1] != 3 or 0 in crop.shape or crop.dtype != np.uint8:
        raise ValueError("face crop must be a nonempty uint8 RGB image")
    image = torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float() / 255
    pixels = profile["preprocessing"]["visual_pixels"]
    return F.interpolate(
        image.unsqueeze(0), size=(pixels, pixels), mode="bilinear", align_corners=False
    ).squeeze(0)


def prepare_audio(waveform, sample_rate, profile):
    """Return [3, 175, 175] from normalized floating-point PCM.

    Input shape is [samples] or [samples, channels]. Mix channels before
    resampling. Empty, nonfinite, or effectively silent audio is rejected,
    allowing callers to report an assessment failure rather than invent evidence.
    """
    import librosa

    settings = profile["preprocessing"]
    samples = np.asarray(waveform)
    if (samples.ndim not in (1, 2) or 0 in samples.shape
            or not np.issubdtype(samples.dtype, np.floating)
            or not np.isfinite(samples).all()):
        raise ValueError("audio must be nonempty finite floating-point PCM")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample rate must be a positive integer")
    if np.abs(samples).max() > 1:
        raise ValueError("PCM must be normalized to [-1, 1]")
    samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    target_rate = settings["audio_sample_rate"]
    if sample_rate != target_rate:
        samples = librosa.resample(
            samples, orig_sr=sample_rate, target_sr=target_rate,
            res_type=settings["resample_type"],
        )
    spectrum = librosa.feature.melspectrogram(
        y=samples, sr=target_rate, n_fft=settings["n_fft"],
        hop_length=settings["hop_length"], n_mels=settings["n_mels"],
        win_length=settings["n_fft"], window=settings["stft_window"],
        center=settings["stft_center"], pad_mode=settings["stft_pad_mode"],
        power=settings["mel_power"], htk=settings["mel_htk"],
        norm=settings["mel_norm"], fmin=settings["mel_fmin"], fmax=settings["mel_fmax"],
    )
    if spectrum.max() <= settings["minimum_peak_power"]:
        raise ValueError("audio has insufficient energy for this preparation profile")
    decibels = librosa.power_to_db(
        spectrum, ref=np.max, amin=settings["power_to_db_amin"], top_db=-settings["db_floor"]
    )
    grayscale = np.clip((decibels - settings["db_floor"]) / -settings["db_floor"], 0, 1)
    image = torch.from_numpy(grayscale).unsqueeze(0).unsqueeze(0)
    pixels = settings["audio_pixels"]
    image = F.interpolate(image, size=(pixels, pixels), mode="bilinear", align_corners=False)
    return image.squeeze(0).expand(3, -1, -1).contiguous()
