"""Shared data models for the experimental export verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class VerifierConfig:
    """Configuration for the current EOF-tail verifier probe."""

    sr: int = 16000
    tail_window_s: float = 1.2
    terminal_window_s: float = 0.6


@dataclass(frozen=True)
class MediaAudio:
    """Decoded mono PCM audio and source metadata."""

    path: Path
    samples: np.ndarray
    sr: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr


@dataclass(frozen=True)
class ProbeResult:
    """Scores for one exported cut against the source EOF tail."""

    cut: Path
    duration: float
    waveform_cosine: float
    envelope_cosine: float
    logmel_cosine: float
    rms_ratio: float
    terminal_envelope_cosine: float
    terminal_logmel_cosine: float
    terminal_rms_ratio: float
    verdict: str
