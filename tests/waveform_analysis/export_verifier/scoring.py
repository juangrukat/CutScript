"""Scoring for the current EOF-tail verifier probe."""

from __future__ import annotations

import math
from pathlib import Path

from .decode import decode_audio
from .features import cosine, logmel, rms, rms_envelope
from .models import MediaAudio, ProbeResult, VerifierConfig
from .verdicts import classify_tail_similarity
from .windows import tail


def probe_source(source: Path, config: VerifierConfig) -> MediaAudio:
    return decode_audio(source, config.sr)


def probe_cut(
    source_audio: MediaAudio,
    cut: Path,
    config: VerifierConfig,
) -> ProbeResult:
    source_tail = tail(source_audio.samples, source_audio.sr, config.tail_window_s)
    cut_audio = decode_audio(cut, config.sr)
    cut_tail = tail(cut_audio.samples, cut_audio.sr, config.tail_window_s)
    source_terminal = tail(source_audio.samples, source_audio.sr, config.terminal_window_s)
    cut_terminal = tail(cut_audio.samples, cut_audio.sr, config.terminal_window_s)

    source_env = rms_envelope(source_tail, source_audio.sr)
    cut_env = rms_envelope(cut_tail, cut_audio.sr)
    source_mel = logmel(source_tail, source_audio.sr)
    cut_mel = logmel(cut_tail, cut_audio.sr)
    source_terminal_env = rms_envelope(source_terminal, source_audio.sr)
    cut_terminal_env = rms_envelope(cut_terminal, cut_audio.sr)
    source_terminal_mel = logmel(source_terminal, source_audio.sr)
    cut_terminal_mel = logmel(cut_terminal, cut_audio.sr)

    source_rms = rms(source_tail)
    cut_rms = rms(cut_tail)
    source_terminal_rms = rms(source_terminal)
    cut_terminal_rms = rms(cut_terminal)
    envelope_cosine = cosine(source_env, cut_env)
    logmel_cosine = cosine(source_mel, cut_mel)
    rms_ratio = cut_rms / source_rms if source_rms > 1e-12 else math.nan
    terminal_envelope_cosine = cosine(source_terminal_env, cut_terminal_env)
    terminal_logmel_cosine = cosine(source_terminal_mel, cut_terminal_mel)
    terminal_rms_ratio = (
        cut_terminal_rms / source_terminal_rms
        if source_terminal_rms > 1e-12
        else math.nan
    )

    return ProbeResult(
        cut=cut,
        duration=cut_audio.duration,
        waveform_cosine=cosine(source_tail, cut_tail),
        envelope_cosine=envelope_cosine,
        logmel_cosine=logmel_cosine,
        rms_ratio=rms_ratio,
        terminal_envelope_cosine=terminal_envelope_cosine,
        terminal_logmel_cosine=terminal_logmel_cosine,
        terminal_rms_ratio=terminal_rms_ratio,
        verdict=classify_tail_similarity(
            logmel_cosine,
            envelope_cosine,
            rms_ratio,
            terminal_logmel_cosine,
            terminal_envelope_cosine,
            terminal_rms_ratio,
        ),
    )
