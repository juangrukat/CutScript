"""
MLX Whisper decoder backend.

Produces segment-level transcription using Apple Silicon's MLX framework.
Word-level alignment is then applied by the caller (WhisperX wav2vec2
forced alignment) so timestamp precision matches the standard WhisperX path.

This module imports lazily so the rest of the app works on machines without
mlx-whisper installed. Call is_available() to probe before routing to MLX.
"""

from __future__ import annotations

import logging
import platform
from typing import Optional

logger = logging.getLogger(__name__)


# Model names accepted by the UI mapped to HuggingFace repos hosting MLX weights.
MLX_REPOS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def is_available() -> tuple[bool, str]:
    """
    Probe whether the MLX backend can run on this machine.

    Returns (available, reason). Reason is empty when available.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False, "MLX requires Apple Silicon (macOS arm64)"
    try:
        import mlx_whisper  # noqa: F401
    except ImportError:
        return False, "mlx-whisper not installed (pip install mlx-whisper)"
    return True, ""


def decode(
    audio_path: str,
    model_name: str,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    verbatim: bool = False,
    progress_cb=None,
) -> tuple[list[dict], str]:
    """
    Run MLX Whisper on the given audio file and return segment-level output.

    Returns:
        (segments_for_align, detected_language)
        segments_for_align: list of {"start", "end", "text"} dicts ready for
        whisperx.align() to turn into word-level timestamps.
    """
    import mlx_whisper

    repo = MLX_REPOS.get(model_name)
    if repo is None:
        raise ValueError(
            f"MLX backend does not support model '{model_name}'. "
            f"Supported: {sorted(MLX_REPOS)}"
        )

    if progress_cb:
        progress_cb(10, "Transcribing (MLX)...")

    # MLX Whisper has no streaming/generator API — the decode runs to completion
    # in one call. We report a coarse midpoint so users see progress motion even
    # though we can't know real decode position.
    if progress_cb:
        progress_cb(30, "Transcribing (MLX)...")

    opts: dict = {
        "path_or_hf_repo": repo,
        "verbose": None,
        # WhisperX alignment will derive word timestamps, so the cross-attention
        # word timestamps MLX computes would be redundant work.
        "word_timestamps": False,
        "temperature": 0.0,
        # Matches _transcribe_whisperx: disabling previous-text conditioning is
        # the documented fix for chunk-boundary repetition loops.
        "condition_on_previous_text": False,
        # Same thresholds as WhisperX path so behaviour is consistent across
        # backends (including verbatim relaxations).
        "no_speech_threshold": 0.6 if verbatim else 0.8,
        "compression_ratio_threshold": 2.4 if verbatim else 1.8,
    }
    if language:
        opts["language"] = language
    if initial_prompt:
        opts["initial_prompt"] = initial_prompt

    logger.info(f"MLX transcribe: {audio_path} with {repo}")
    result = mlx_whisper.transcribe(audio_path, **opts)

    if progress_cb:
        progress_cb(70, "Transcribing (MLX)...")

    detected_language = result.get("language", language or "en")

    segments_for_align = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        segments_for_align.append({
            "start": float(seg.get("start", 0.0)),
            "end": float(seg.get("end", 0.0)),
            "text": text,
        })

    return segments_for_align, detected_language
