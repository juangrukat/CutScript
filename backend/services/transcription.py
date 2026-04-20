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




def _make_cache_op(backend: str, beam_size: int, vad_filter: bool, vad_min_silence_ms: int, verbatim: bool) -> str:
    """
    Build a cache-op string that encodes all variable transcription settings.
    Different combinations produce different hashes so cached results are never
    reused across setting changes. `backend` is part of the key so MLX and
    WhisperX decodes never collide in cache.
    """
    settings = {
        "backend": backend,
        "condition_on_previous_text": False,
        "temperature": 0.0,
        "beam_size": beam_size,
        "vad_filter": vad_filter,
        "vad_min_silence_ms": vad_min_silence_ms if vad_filter else 0,
        "verbatim": verbatim,
        # In verbatim mode, thresholds are relaxed to avoid rejecting real repetitions.
        # With VAD, both thresholds are disabled — VAD owns segmentation decisions.
        "no_speech_threshold": None if vad_filter else (0.6 if verbatim else 0.8),
        "compression_ratio_threshold": None if vad_filter else (2.4 if verbatim else 1.8),
    }
    return "transcribe_wx_" + hashlib.md5(
        json.dumps(settings, sort_keys=True).encode()
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

# Maps faster-whisper model names to their HuggingFace repo IDs.
# Mirrors the _MODELS dict in faster-whisper so we can resolve local cache paths.
_FASTER_WHISPER_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "base": "Systran/faster-whisper-base",
    "base.en": "Systran/faster-whisper-base.en",
    "small": "Systran/faster-whisper-small",
    "small.en": "Systran/faster-whisper-small.en",
    "medium": "Systran/faster-whisper-medium",
    "medium.en": "Systran/faster-whisper-medium.en",
    "large-v1": "Systran/faster-whisper-large-v1",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
    "large": "Systran/faster-whisper-large-v3",
    "distil-large-v2": "Systran/faster-distil-whisper-large-v2",
    "distil-medium.en": "Systran/faster-distil-whisper-medium.en",
    "distil-small.en": "Systran/faster-distil-whisper-small.en",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
}


def _resolve_model_path(model_name: str) -> str:
    """
    Return a local filesystem path for the model if it is already in the
    HuggingFace Hub cache, otherwise return the model name unchanged so the
    caller can trigger a normal download.

    Passing a local path to faster-whisper / WhisperX skips all HuggingFace
    network calls (revision checks, HEAD requests, xet-read-token fetches)
    which otherwise fire on every load even when the model is fully cached.
    """
    repo_id = _FASTER_WHISPER_REPOS.get(model_name)
    if repo_id is None:
        return model_name
    try:
        from huggingface_hub import snapshot_download
        local_path = snapshot_download(repo_id, local_files_only=True)
        logger.info(f"Using cached model at {local_path}")
        return local_path
    except Exception:
        # Not cached yet — fall through to normal (downloading) load.
        return model_name


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

    # Resolve to a local path when the model is already cached so that
    # faster-whisper/WhisperX never makes network requests on load.
    model_id = _resolve_model_path(model_name)

    logger.info(f"Loading model: {model_name} on {actual_device}")
    if WHISPERX_AVAILABLE:
        model = whisperx.load_model(
            model_id,
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
    beam_size: int = 5,
    vad_filter: bool = False,
    vad_min_silence_ms: int = 500,
    verbatim: bool = False,
    backend: str = "whisperx",
    progress_cb=None,
) -> dict:
    """
    Transcribe audio/video file and return word-level timestamps.

    Returns:
        dict with keys: words, segments, language
    """
    file_path = Path(file_path)
    cache_op = _make_cache_op(backend, beam_size, vad_filter, vad_min_silence_ms, verbatim)

    if use_cache:
        cached = load_from_cache(file_path, model_name, cache_op)
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
    # preprocess_audio_for_transcription returns (path, trim_offset_seconds) so
    # we can shift all timestamps back onto the original timeline afterwards.
    preprocessed_path, trim_offset = preprocess_audio_for_transcription(Path(audio_path))

    device = _get_device(use_gpu)

    # Determine the actual device for downstream ops (MPS not supported by faster-whisper)
    actual_device = device
    if device.type == "mps":
        actual_device = torch.device("cpu")

    logger.info(f"Transcribing ({backend}): {file_path}")

    if backend == "mlx":
        result = _transcribe_mlx_with_align(
            str(preprocessed_path), model_name, actual_device,
            language=language, initial_prompt=initial_prompt,
            verbatim=verbatim, progress_cb=progress_cb,
        )
    elif WHISPERX_AVAILABLE:
        model = _load_model(model_name, device)
        result = _transcribe_whisperx(
            model, str(preprocessed_path), actual_device, language, initial_prompt,
            beam_size=beam_size, vad_filter=vad_filter, vad_min_silence_ms=vad_min_silence_ms,
            verbatim=verbatim, progress_cb=progress_cb,
        )
    else:
        model = _load_model(model_name, device)
        if progress_cb:
            progress_cb(10, "Transcribing...")
        result = _transcribe_standard(model, str(preprocessed_path), language, initial_prompt)
        if progress_cb:
            progress_cb(95, "Finalizing...")

    # Shift all timestamps forward by the amount of leading silence that was
    # stripped during preprocessing, so they align with the original video timeline.
    if trim_offset > 0:
        result = _apply_timestamp_offset(result, trim_offset)

    # Drop any words/segments whose timestamps exceed the source audio duration.
    # These are hallucinations — Whisper sometimes generates text beyond the end
    # of the file, especially when initial_prompt text leaks into the output.
    try:
        import soundfile as sf
        with sf.SoundFile(str(preprocessed_path)) as f:
            audio_duration = len(f) / f.samplerate
        result = _clip_to_duration(result, audio_duration + trim_offset)
    except Exception:
        pass

    if use_cache:
        save_to_cache(file_path, result, model_name, cache_op)

    return result


def _clip_to_duration(result: dict, max_time: float) -> dict:
    """Remove words and segments whose start timestamp exceeds the audio duration."""
    original_word_count = len(result.get("words", []))
    result["words"] = [w for w in result.get("words", []) if w.get("start", 0) <= max_time]
    result["segments"] = [
        {**seg, "words": [w for w in seg.get("words", []) if w.get("start", 0) <= max_time]}
        for seg in result.get("segments", [])
        if seg.get("start", 0) <= max_time
    ]
    clipped = original_word_count - len(result["words"])
    if clipped > 0:
        logger.warning(f"Clipped {clipped} hallucinated words beyond audio duration ({max_time:.2f}s)")
    return result


def _apply_timestamp_offset(result: dict, offset: float) -> dict:
    """Add offset (seconds) to every word and segment timestamp in-place."""
    for word in result.get("words", []):
        word["start"] = round(word["start"] + offset, 3)
        word["end"] = round(word["end"] + offset, 3)
    for seg in result.get("segments", []):
        seg["start"] = round(seg["start"] + offset, 3)
        seg["end"] = round(seg["end"] + offset, 3)
        for word in seg.get("words", []):
            word["start"] = round(word["start"] + offset, 3)
            word["end"] = round(word["end"] + offset, 3)
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


def _transcribe_whisperx(
    model,
    audio_path: str,
    device: torch.device,
    language: Optional[str],
    initial_prompt: Optional[str] = None,
    beam_size: int = 5,
    vad_filter: bool = False,
    vad_min_silence_ms: int = 500,
    verbatim: bool = False,
    progress_cb=None,
) -> dict:
    def _p(pct: int, status: str):
        if progress_cb:
            progress_cb(pct, status)

    audio = whisperx.load_audio(audio_path)
    _p(10, "Transcribing...")

    transcribe_opts: dict = {
        "chunk_length": 30,
        "beam_size": beam_size,
        "temperature": 0.0,
        # Disable previous-text conditioning: feeding prior chunk text into the next
        # decoder pass is a documented cause of repetition loops at chunk boundaries.
        "condition_on_previous_text": False,
    }

    if vad_filter:
        # When VAD handles segmentation, the noise-suppression thresholds can
        # conflict with it — disable them and let VAD own the silence decisions.
        transcribe_opts["vad_filter"] = True
        transcribe_opts["vad_parameters"] = {"min_silence_duration_ms": vad_min_silence_ms}
    else:
        transcribe_opts["vad_filter"] = False
        if verbatim:
            # Verbatim mode: relax thresholds so that real repeated speech and
            # borderline speech regions are not filtered out.
            # no_speech_threshold 0.8 → 0.6: less likely to skip uncertain regions.
            # compression_ratio_threshold 1.8 → 2.4 (the faster-whisper default):
            #   avoids rejecting genuine repetitions as hallucinated looping.
            transcribe_opts["no_speech_threshold"] = 0.6
            transcribe_opts["compression_ratio_threshold"] = 2.4
        else:
            # Suppress segments where the model has low confidence there is speech.
            transcribe_opts["no_speech_threshold"] = 0.8
            # Reject output whose compression ratio signals repeated or looping text.
            transcribe_opts["compression_ratio_threshold"] = 1.8

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

    # In verbatim mode, skip deduplication — the filter removes consecutive
    # identical segments to suppress hallucination loops, but in verbatim mode
    # a speaker genuinely repeating a phrase looks identical to a loop artifact.
    if not verbatim:
        segments_for_align = _deduplicate_segments(segments_for_align)

    return _align_and_pack(
        segments_for_align, audio, detected_language,
        device=device, verbatim=verbatim, progress_cb=progress_cb,
    )


def _align_and_pack(
    segments_for_align: list,
    audio,
    detected_language: str,
    device: torch.device,
    verbatim: bool,
    progress_cb=None,
) -> dict:
    """
    Run WhisperX wav2vec2 forced alignment on segment-level text and pack the
    result into the final {words, segments, language} shape.

    Shared between the WhisperX and MLX decode paths so word-timestamp quality
    is identical regardless of which backend produced the segment text.
    """
    def _p(pct: int, status: str):
        if progress_cb:
            progress_cb(pct, status)

    _p(72, "Aligning words...")
    try:
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
        aligned_segments = (
            aligned.get("segments", []) if verbatim
            else _deduplicate_segments(aligned.get("segments", []))
        )
    except Exception as align_err:
        logger.warning(
            f"Word alignment failed for language '{detected_language}', "
            f"falling back to segment-level timestamps: {align_err}"
        )
        _p(95, "Finalizing...")
        aligned_segments = segments_for_align

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


def _transcribe_mlx_with_align(
    audio_path: str,
    model_name: str,
    device: torch.device,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    verbatim: bool = False,
    progress_cb=None,
) -> dict:
    """
    MLX decode -> WhisperX wav2vec2 alignment.

    MLX handles the Whisper decode (fast on Apple Silicon); WhisperX owns the
    word-boundary derivation so timestamps stay as precise as the standard path.
    Requires WhisperX to be installed for the alignment step.
    """
    if not WHISPERX_AVAILABLE:
        raise RuntimeError(
            "MLX backend requires WhisperX for word alignment. Install whisperx."
        )

    from services import transcription_mlx

    segments_for_align, detected_language = transcription_mlx.decode(
        audio_path=audio_path,
        model_name=model_name,
        language=language,
        initial_prompt=initial_prompt,
        verbatim=verbatim,
        progress_cb=progress_cb,
    )

    # In non-verbatim mode the WhisperX path dedupes before alignment; mirror
    # that so MLX output can't surface chunk-boundary repetition loops.
    if not verbatim:
        segments_for_align = _deduplicate_segments(segments_for_align)

    audio = whisperx.load_audio(audio_path)
    return _align_and_pack(
        segments_for_align, audio, detected_language,
        device=device, verbatim=verbatim, progress_cb=progress_cb,
    )


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
