"""
WhisperX-based transcription service with word-level alignment.
Falls back to standard Whisper if WhisperX is not available.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

import torch

# Cache key is derived from the key transcription parameters so it automatically
# invalidates whenever settings change — no manual version bumping needed.
_TRANSCRIPTION_SETTINGS = {
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.8,
    "compression_ratio_threshold": 1.8,
    "beam_size": 5,
    "temperature": 0.0,
}
_CACHE_OP = "transcribe_wx_" + hashlib.md5(
    json.dumps(_TRANSCRIPTION_SETTINGS, sort_keys=True).encode()
).hexdigest()[:8]

from utils.gpu_utils import get_optimal_device, configure_gpu
from utils.audio_processing import extract_audio, preprocess_audio_for_transcription
from utils.cache import load_from_cache, save_to_cache

logger = logging.getLogger(__name__)

_model_cache: dict = {}

try:
    import whisperx
    WHISPERX_AVAILABLE = True
except ImportError:
    WHISPERX_AVAILABLE = False
    import whisper

try:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
except Exception:
    HF_TOKEN = ""


def _get_device(use_gpu: bool = True) -> torch.device:
    if use_gpu:
        return get_optimal_device()
    return torch.device("cpu")


def _load_model(model_name: str, device: torch.device):
    cache_key = f"{model_name}_{device}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    # faster-whisper (used by whisperx for all model sizes including large-v3/distil-*)
    # does NOT support MPS -- only CUDA and CPU. Fall back to CPU float16 for MPS.
    actual_device = device
    if device.type == "mps":
        actual_device = torch.device("cpu")
        logger.info("MPS not supported by faster-whisper, falling back to CPU int8")
    compute_type = "float16" if actual_device.type == "cuda" else "int8"

    logger.info(f"Loading model: {model_name} on {actual_device}")
    if WHISPERX_AVAILABLE:
        model = whisperx.load_model(
            model_name,
            device=str(actual_device),
            compute_type=compute_type,
        )
    else:
        model = whisper.load_model(model_name, device=actual_device)

    _model_cache[cache_key] = model
    return model


def transcribe_audio(
    file_path: str,
    model_name: str = "base",
    use_gpu: bool = True,
    use_cache: bool = True,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    progress_cb=None,
) -> dict:
    """
    Transcribe audio/video file and return word-level timestamps.

    Returns:
        dict with keys: words, segments, language
    """
    file_path = Path(file_path)

    if use_cache:
        cached = load_from_cache(file_path, model_name, _CACHE_OP)
        if cached:
            logger.info("Using cached transcription")
            return cached

    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if file_path.suffix.lower() in video_extensions:
        audio_path = extract_audio(file_path)
    else:
        audio_path = file_path

    # Preprocess: trim leading silence and normalize loudness before Whisper sees
    # the audio. Leading silence is a documented cause of hallucination/drift.
    preprocessed_path = preprocess_audio_for_transcription(Path(audio_path))

    device = _get_device(use_gpu)
    model = _load_model(model_name, device)

    # Determine the actual device for downstream ops (MPS not supported by faster-whisper)
    actual_device = device
    if device.type == "mps":
        actual_device = torch.device("cpu")

    logger.info(f"Transcribing: {file_path}")

    if WHISPERX_AVAILABLE:
        result = _transcribe_whisperx(model, str(preprocessed_path), actual_device, language, initial_prompt, progress_cb=progress_cb)
    else:
        if progress_cb:
            progress_cb(10, "Transcribing...")
        result = _transcribe_standard(model, str(preprocessed_path), language, initial_prompt)
        if progress_cb:
            progress_cb(95, "Finalizing...")

    if use_cache:
        save_to_cache(file_path, result, model_name, _CACHE_OP)

    return result


def _deduplicate_segments(segments: list) -> list:
    """
    Remove consecutively repeated or near-duplicate segments.

    faster-whisper produces these at chunk boundaries when a phrase straddles a
    30-second window, or in noisy/silent sections where the decoder loops. Two
    segments are considered duplicates if their normalized texts are identical or
    one is ≥80% of the other and is fully contained within it.
    """
    import re

    def _norm(text: str) -> str:
        return re.sub(r"[^\w\s]", "", text.lower()).strip()

    out = []
    recent: list[str] = []  # normalized texts of the last few kept segments
    WINDOW = 4

    for seg in segments:
        norm = _norm(seg.get("text", ""))
        if not norm:
            continue
        is_dup = False
        for prev in recent[-WINDOW:]:
            if norm == prev:
                is_dup = True
                break
            # Near-duplicate: shorter text is ≥80% of longer and fully contained in it
            shorter, longer = (norm, prev) if len(norm) <= len(prev) else (prev, norm)
            if len(shorter) >= 15 and shorter in longer and len(shorter) / len(longer) >= 0.8:
                is_dup = True
                break
        if not is_dup:
            out.append(seg)
            recent.append(norm)

    if len(out) < len(segments):
        logger.info(f"Deduplication removed {len(segments) - len(out)} repeated segments")

    return out


def _transcribe_whisperx(model, audio_path: str, device: torch.device, language: Optional[str], initial_prompt: Optional[str] = None, progress_cb=None) -> dict:
    def _p(pct: int, status: str):
        if progress_cb:
            progress_cb(pct, status)

    audio = whisperx.load_audio(audio_path)
    _p(10, "Transcribing...")

    transcribe_opts = {
        "vad_filter": False,
        "chunk_length": 30,
        # beam_size=5 + temperature=0.0 gives deterministic beam decoding without
        # the fallback temperature ramp.
        "beam_size": 5,
        "temperature": 0.0,
        # Disable previous-text conditioning: feeding prior chunk text into the next
        # decoder pass is a documented cause of repetition loops at chunk boundaries.
        "condition_on_previous_text": False,
        # Suppress segments where the model has low confidence there is speech —
        # default 0.6 lets noise/silence through and triggers hallucinations.
        "no_speech_threshold": 0.8,
        # Reject output whose compression ratio signals repeated or looping text.
        # Default 2.4 is too permissive; 1.8 catches most repetition artifacts early.
        "compression_ratio_threshold": 1.8,
    }
    if language:
        transcribe_opts["language"] = language
    if initial_prompt:
        transcribe_opts["initial_prompt"] = initial_prompt

    # WhisperX's FasterWhisperPipeline wrapper doesn't forward vad_filter/chunk_length.
    # Call the underlying faster-whisper model directly to disable VAD filtering,
    # which incorrectly drops lyrics in music tracks (instrumentals confuse Silero VAD).
    # NOTE: model.model.transcribe() returns (segments_generator, info) — the generator
    # is lazy, so we consume it manually to emit real per-chunk progress.
    raw_result = model.model.transcribe(audio, **transcribe_opts)
    segments_gen = raw_result[0]
    info = raw_result[1]
    detected_language = info.language
    total_duration = max(info.duration, 0.001)

    # Consume the segment generator; each yielded segment advances the real decode position.
    # Progress range 10–70% maps to the transcription pass.
    segments_for_align = []
    for seg in segments_gen:
        segments_for_align.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })
        pct = 10 + int(min(seg.end, total_duration) / total_duration * 60)
        _p(min(pct, 70), "Transcribing...")

    segments_for_align = _deduplicate_segments(segments_for_align)

    _p(72, "Aligning words...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=str(device),
    )
    _p(80, "Aligning words...")
    aligned = whisperx.align(
        segments_for_align,
        align_model,
        align_metadata,
        audio,
        str(device),
        return_char_alignments=False,
    )
    _p(95, "Finalizing...")

    aligned_segments = _deduplicate_segments(aligned.get("segments", []))

    words = []
    for seg in aligned_segments:
        for w in seg.get("words", []):
            words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", 0), 3),
                "end": round(w.get("end", 0), 3),
                "confidence": round(w.get("score", 0), 3),
            })

    segments = []
    for i, seg in enumerate(aligned_segments):
        seg_words = []
        for w in seg.get("words", []):
            seg_words.append({
                "word": w.get("word", ""),
                "start": round(w.get("start", 0), 3),
                "end": round(w.get("end", 0), 3),
                "confidence": round(w.get("score", 0), 3),
            })
        segments.append({
            "id": i,
            "start": round(seg.get("start", 0), 3),
            "end": round(seg.get("end", 0), 3),
            "text": seg.get("text", "").strip(),
            "words": seg_words,
        })

    return {
        "words": words,
        "segments": segments,
        "language": detected_language,
    }


def _transcribe_standard(model, audio_path: str, language: Optional[str], initial_prompt: Optional[str] = None) -> dict:
    """Fallback: standard Whisper (segment-level only, synthesized word timestamps)."""
    opts = {"beam_size": 5, "temperature": 0}
    if language:
        opts["language"] = language
    if initial_prompt:
        opts["initial_prompt"] = initial_prompt

    result = model.transcribe(audio_path, **opts)
    detected_language = result.get("language", "en")

    words = []
    segments = []

    for i, seg in enumerate(result.get("segments", [])):
        text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0)
        seg_end = seg.get("end", 0)
        seg_words_text = text.split()
        duration = seg_end - seg_start

        seg_words = []
        for j, w_text in enumerate(seg_words_text):
            w_start = seg_start + (j / max(len(seg_words_text), 1)) * duration
            w_end = seg_start + ((j + 1) / max(len(seg_words_text), 1)) * duration
            word_obj = {
                "word": w_text,
                "start": round(w_start, 3),
                "end": round(w_end, 3),
                "confidence": 0.5,
            }
            words.append(word_obj)
            seg_words.append(word_obj)

        segments.append({
            "id": i,
            "start": round(seg_start, 3),
            "end": round(seg_end, 3),
            "text": text,
            "words": seg_words,
        })

    return {
        "words": words,
        "segments": segments,
        "language": detected_language,
    }
