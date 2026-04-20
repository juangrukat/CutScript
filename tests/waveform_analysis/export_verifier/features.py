"""Audio feature helpers for the experimental export verifier."""

from __future__ import annotations

import librosa
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    n = min(len(a), len(b))
    if n == 0:
        return float("nan")
    a = a[:n]
    b = b[:n]
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom <= 1e-12:
        return float("nan")
    return float(np.dot(a, b) / denom)


def rms(y: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y)) + 1e-12))


def rms_envelope(y: np.ndarray, sr: int) -> np.ndarray:
    frame = max(256, int(round(0.025 * sr)))
    hop = max(80, int(round(0.010 * sr)))
    return librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=False)[0]


def logmel(y: np.ndarray, sr: int) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=1024,
        hop_length=max(80, int(round(0.010 * sr))),
        n_mels=48,
        power=2.0,
        center=False,
    )
    return librosa.power_to_db(mel, ref=np.max)
