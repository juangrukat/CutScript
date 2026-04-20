"""Media decoding for the experimental export verifier."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from .models import MediaAudio


def decode_audio(path: Path, sr: int) -> MediaAudio:
    """Decode any media file to mono PCM with ffmpeg, then read it."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(path),
                "-ac",
                "1",
                "-ar",
                str(sr),
                tmp.name,
            ],
            check=True,
        )
        y, actual_sr = sf.read(tmp.name, dtype="float32")

    if y.ndim > 1:
        y = np.mean(y, axis=1)

    return MediaAudio(path=path, samples=y.astype(np.float32), sr=int(actual_sr))
