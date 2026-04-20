"""Export endpoint for video cutting and rendering."""

import logging
import tempfile
import os
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.video_editor import (
    export_stream_copy,
    export_reencode,
    export_reencode_with_subs,
    get_video_info,
)
from services.audio_cleaner import clean_audio
from services.caption_generator import generate_srt, generate_ass, save_captions
from services.boundary_refiner import BoundaryRefiner
from services.audio_analyzer import load_acoustic_map, analyze_file, AcousticMap
from utils.audio_processing import extract_audio

logger = logging.getLogger(__name__)
router = APIRouter()


class SegmentModel(BaseModel):
    start: float
    end: float


class ExportWordModel(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 0.0


class ExportRequest(BaseModel):
    input_path: str
    output_path: str
    keep_segments: List[SegmentModel]
    mode: str = "fast"
    resolution: str = "1080p"
    format: str = "mp4"
    enhanceAudio: bool = False
    captions: str = "none"
    words: Optional[List[ExportWordModel]] = None
    deleted_indices: Optional[List[int]] = None
    force_refine: bool = False


def _segment_covers_source(
    segment: dict, input_path: str, tolerance: float = 0.050
) -> bool:
    """
    Return True when a single keep segment effectively covers the whole source.

    Fast export can safely stream-copy that case.  A single non-full segment,
    however, is still a cut: if it came from word timestamps, bypassing
    AcousticMap refinement can trim the coda of the final kept word.
    """
    info = get_video_info(input_path)
    duration = float(info.get("duration") or 0.0)
    if duration <= 0:
        return False
    return segment["start"] <= tolerance and segment["end"] >= duration - tolerance


def _gap_has_speech(y, sr: float, t0: float, t1: float, threshold: float) -> bool:
    """
    Detect whether the GAP between t0 and t1 contains deleted speech.

    Checks the interior of the gap rather than its edges.  Word acoustic tails
    and onsets bleed ~30-80ms past WhisperX timestamps, so an edge check always
    fires even over pure silence.  Sampling the middle avoids that contamination:
    a gap filled with deleted words shows energy throughout its interior; a gap
    that is a natural pause shows near-silence in its interior.
    """
    import librosa
    import numpy as np

    gap = t1 - t0
    if gap <= 0:
        return False
    # Skip edge margins proportional to gap size (minimum 40ms each side).
    # This clears word acoustic tails without missing genuine speech in the gap.
    margin = max(0.04, gap * 0.10)
    mid_s, mid_e = t0 + margin, t1 - margin
    if mid_e <= mid_s:
        # Gap too short for margin — fall back to whole-gap check
        mid_s, mid_e = t0, t1
    s = max(0, int(mid_s * sr))
    e = min(len(y), int(mid_e * sr))
    if e - s < 256:
        return False
    rms = librosa.feature.rms(y=y[s:e], frame_length=512, hop_length=128)[0]
    return bool(np.mean(rms > threshold) > 0.15)


def _snap_zc(y, sr: float, t: float, search_ms: float = 40.0) -> float:
    """Snap a timestamp to the nearest zero crossing within ±search_ms."""
    import librosa
    import numpy as np

    idx = int(t * sr)
    half = int(search_ms / 1000.0 * sr)
    start = max(0, idx - half)
    end = min(len(y), idx + half)
    zc_idx = np.where(librosa.zero_crossings(y[start:end]))[0] + start
    if len(zc_idx) == 0:
        return t
    return float(zc_idx[np.argmin(np.abs(zc_idx - idx))]) / sr


def _sample_has_speech(
    y, sr: float, t: float, threshold: float, window_ms: float = 15.0
) -> bool:
    """Check whether a point in time is inside active speech (±window_ms)."""
    import librosa
    import numpy as np

    half = int(window_ms / 1000.0 * sr)
    s = max(0, int(t * sr) - half)
    e = min(len(y), int(t * sr) + half)
    if e - s < 64:
        return False
    rms = librosa.feature.rms(y=y[s:e], frame_length=256, hop_length=64)[0]
    return bool(np.mean(rms > threshold) > 0.50)


_WORD_MARGIN_S = 0.028  # 28ms — typical WhisperX forced-alignment imprecision


def _advance_past_silence(
    y, sr: float, t_start: float, t_cap: float, speech_threshold: float
) -> float:
    """
    If t_start is in a silent zone, advance to the first sustained speech within
    [t_start, t_cap] and return 20ms before its acoustic onset.

    This handles the case where the frontend places the segment start at the end
    of the last deleted word, leaving a gap of near-silence before the first kept
    word.  _find_onset_before only searches ±150ms around seg["start"] and cannot
    bridge a 0.5-2s gap, so this pass covers the rest.

    Algorithm:
      1. If already inside speech (50ms check), return unchanged.
      2. Scan RMS forward with a coarse frame to find the first 3 consecutive
         frames above speech_threshold — i.e., genuine sustained speech, not a
         noise spike.
      3. Find the acoustic onset in a 200ms window just before that speech onset
         (to capture consonant attacks that precede the voiced nucleus).
      4. Return onset − 20ms, clamped to t_start.

    Returns t_start unchanged if already at speech or if no speech is found.
    """
    import librosa
    import numpy as np

    # Quick check: already inside speech?
    if _sample_has_speech(y, sr, t_start + 0.025, speech_threshold, window_ms=25.0):
        return t_start

    # Scan forward for the first sustained speech: 3 consecutive coarse RMS
    # frames above threshold.  hop=256 ≈ 5.8ms/frame; 3 frames ≈ 17ms minimum.
    hop = 256
    s = int(t_start * sr)
    e = int(min(len(y) / sr, t_cap) * sr)
    if e - s < hop * 3:
        return t_start

    rms = librosa.feature.rms(y=y[s:e], frame_length=1024, hop_length=hop)[0]
    rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + t_start

    above = rms > speech_threshold
    speech_t = None
    for i in range(len(above) - 2):
        if above[i] and above[i + 1] and above[i + 2]:
            speech_t = float(rms_t[i])
            break

    if speech_t is None or speech_t - t_start < 0.250:
        # No speech found, or gap is within normal onset pre-roll (< 250ms) —
        # don't trim: _find_onset_before already placed the start correctly, or
        # the gap is a voiceless consonant cluster ("Sp", "St") preceding speech.
        return t_start

    # Find the acoustic onset just before speech_t to capture the consonant attack.
    search_start = max(t_start, speech_t - 0.15)
    search_end = min(len(y) / sr, speech_t + 0.05)
    s2, e2 = int(search_start * sr), int(search_end * sr)
    if e2 - s2 < 512:
        return max(t_start, speech_t - 0.020)

    y2 = y[s2:e2]
    env = librosa.onset.onset_strength(y=y2, sr=sr)
    frames2 = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        pre_max=2,
        post_max=2,
        pre_avg=2,
        post_avg=5,
        delta=0.03,
        wait=1,
    )
    if len(frames2) == 0:
        return max(t_start, speech_t - 0.020)

    onsets_abs = librosa.frames_to_time(frames2, sr=sr) + search_start
    candidates = onsets_abs[onsets_abs <= speech_t + 0.010]
    if len(candidates) == 0:
        return max(t_start, speech_t - 0.020)

    onset_t = float(candidates[-1])
    return max(t_start, onset_t - 0.020)


def _find_onset_before(
    y, sr: float, t_whisper: float, prev_end: float
) -> Optional[float]:
    """
    Find the acoustic onset of the word whose WhisperX timestamp is t_whisper.

    Searches a narrow window [max(prev_end, t_whisper-150ms), t_whisper+50ms] so
    that deleted-speech onsets earlier in the gap are excluded.  Returns the last
    detected onset at or just after t_whisper (≤+10ms tolerance), or None.

    Why onset detection instead of the fixed bias?  Voiceless consonant clusters
    ("Sp", "St", "Str", "Sk") are broadband and unvoiced — their energy starts
    40-60ms before the voiced nucleus that WhisperX aligns to.  A fixed 28ms bias
    misses them when the bias target lands inside the preceding deleted-word tail
    and gets rejected by _sample_has_speech.  Onset detection finds the actual
    transient regardless of voice quality.
    """
    import librosa
    import numpy as np

    search_start = max(prev_end, t_whisper - 0.15)
    search_end = min(len(y) / sr, t_whisper + 0.05)
    s, e = int(search_start * sr), int(search_end * sr)
    if e - s < 512:
        return None

    y_win = y[s:e]
    env = librosa.onset.onset_strength(y=y_win, sr=sr)
    frames = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        pre_max=2,
        post_max=2,
        pre_avg=2,
        post_avg=5,
        delta=0.03,
        wait=1,
    )
    if len(frames) == 0:
        return None

    onsets_abs = librosa.frames_to_time(frames, sr=sr) + search_start
    # Accept the last onset at or up to 10ms after t_whisper (WhisperX can be slightly early)
    candidates = onsets_abs[onsets_abs <= t_whisper + 0.010]
    if len(candidates) == 0:
        return None
    return float(candidates[-1])


def _find_word_end(
    y, sr: float, t_whisper: float, next_start: float, speech_threshold: float
) -> Optional[float]:
    """
    Find where the kept word's energy naturally ends after t_whisper.

    Two-phase search in [t_whisper, min(next_start-5ms, t_whisper+150ms)]:

    Phase 1 — word-relative power threshold.
    Computes the peak frame-level RMS² from the word's voiced body (1s lookback)
    and sets a threshold at -25dB below that peak.  This anchors the detector to
    the specific word's loudness rather than an absolute noise floor, making it
    language- and microphone-agnostic.

    At -25dB: typical voiced speech peak 0.07 RMS → peak frame power 0.0049 →
    threshold 0.0049 × 10^(−2.5) ≈ 1.5×10⁻⁵ → amplitude threshold ≈ 0.0039.
    Unvoiced fricatives ("sh", "s", "f", "th") sit at 0.005-0.015 RMS — above
    this threshold.  Room noise sits at 0.001-0.002 — well below it.

    Phase 2 — onset of next content: if energy never drops (deleted speech fills
    gap immediately), find the first onset in the window and cut 20ms before it.

    Returns None when both phases find nothing; caller applies fixed-bias fallback.
    """
    import librosa
    import numpy as np

    cap = min(len(y) / sr, min(next_start - 0.005, t_whisper + 0.150))
    s = int(t_whisper * sr)
    e = int(cap * sr)
    if e - s < 256:
        return None

    # Phase 1: word-relative frame-power threshold
    hop = 64
    s_ref = max(0, int((t_whisper - 1.0) * sr))
    y_ref = y[s_ref : max(s_ref + 256, int(t_whisper * sr))]
    if len(y_ref) >= 256:
        rms_ref = librosa.feature.rms(y=y_ref, frame_length=256, hop_length=hop)[0]
        word_peak_power = float(np.max(rms_ref**2))  # peak frame RMS²
        if word_peak_power > 1e-10:
            # −25 dB below word's voiced peak: captures fricatives, clears room noise
            threshold_power = word_peak_power * (10 ** (-25.0 / 10.0))

            y_fwd = y[s:e]
            rms_fwd = librosa.feature.rms(y=y_fwd, frame_length=256, hop_length=hop)[0]
            rms_t = (
                librosa.frames_to_time(np.arange(len(rms_fwd)), sr=sr, hop_length=hop)
                + t_whisper
            )

            below = rms_fwd**2 < threshold_power
            for i in range(len(below) - 2):
                if below[i] and below[i + 1] and below[i + 2]:
                    decay_t = float(rms_t[i])
                    if decay_t - t_whisper > 0.003:
                        return decay_t  # word tail captured
                    return None  # WhisperX already at decay → use fallback

    # Phase 2: energy never drops — deleted speech fills the gap immediately.
    # Find first onset of following content and cut 20ms before it.
    y_fwd2 = y[s:e]
    if len(y_fwd2) >= 512:
        env = librosa.onset.onset_strength(y=y_fwd2, sr=sr)
        frames = librosa.onset.onset_detect(
            onset_envelope=env,
            sr=sr,
            pre_max=2,
            post_max=2,
            pre_avg=2,
            post_avg=5,
            delta=0.03,
            wait=1,
        )
        if len(frames) > 0:
            onsets_abs = librosa.frames_to_time(frames, sr=sr) + t_whisper
            next_onset = float(onsets_abs[0])
            result = min(cap, next_onset - 0.020)
            if result > t_whisper:
                return result

    return None


def _refine_from_map(segments: list, wav_path: str, acoustic_map: AcousticMap) -> list:
    """
    Refine segment boundaries using the pre-computed AcousticMap.

    Each segment endpoint is matched to the word whose WhisperX start/end is
    closest to it (within 100ms), and the refiner uses that word's `as_` /
    `ae` — already computed at ingest time with fricative-band awareness,
    phoneme-class decay policies, and intraword-dip detection.

    Falls back to the legacy refinement for any boundary that doesn't match
    a word (e.g. the 1.5s pad on the first/last segment added by the frontend).
    """
    import librosa
    import numpy as np

    y, sr = librosa.load(wav_path, sr=None, mono=True)
    audio_dur = len(y) / sr
    refined = []

    TOL = 0.100
    # EOF guardrail: if the last segment's refined_end lands within this window
    # of audio_dur AND there is still audible tail energy between refined_end
    # and EOF, extend to audio_dur. Normal zero-crossing snapping is bypassed
    # for that endpoint so it cannot pull the boundary backward into the tail.
    # This handles the "freedom." case where Whisper's `we` is ~20ms from EOF,
    # the nasal tail runs to file end, and both `ae` and `_snap_zc` can leave
    # the last few frames of audible content on the floor.
    EOF_WINDOW_S = 0.500

    # Precompute a coarse speech-threshold for the EOF tail check
    _rms_coarse = librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0]
    _speech_thr = (float(np.percentile(_rms_coarse, 10)) + 1e-8) * 3.0

    for i, seg in enumerate(segments):
        refined_start = max(0.0, min(seg["start"], audio_dur))
        refined_end = max(0.0, min(seg["end"], audio_dur))

        start_word = acoustic_map.find_word_by_start(seg["start"], tolerance=TOL)
        end_word = acoustic_map.find_word_by_end(seg["end"], tolerance=TOL)

        prev_cap = segments[i - 1]["end"] if i > 0 else 0.0
        is_last_seg = i == len(segments) - 1
        next_cap = segments[i + 1]["start"] if not is_last_seg else float("inf")

        if start_word is not None:
            refined_start = max(prev_cap, start_word.as_)
        if end_word is not None:
            refined_end = min(
                next_cap - 0.005 if np.isfinite(next_cap) else end_word.ae, end_word.ae
            )

        # Clamp to audio bounds so ffmpeg atrim can never produce an empty segment
        refined_start = _snap_zc(y, sr, refined_start)

        if is_last_seg:
            # For last segment, skip zero-crossing snap to preserve content near EOF
            # EOF guardrail for the final segment only: if we're already within
            # EOF_WINDOW_S of audio end and audible energy exists in the remaining
            # tail, clamp to audio_dur. Otherwise keep as is.
            if (audio_dur - refined_end) <= EOF_WINDOW_S:
                tail_s = int(refined_end * sr)
                tail_e = int(audio_dur * sr)
                tail_rms = (
                    float(np.sqrt(np.mean(y[tail_s:tail_e] ** 2) + 1e-12))
                    if tail_e > tail_s
                    else 0.0
                )
                if tail_rms > _speech_thr:
                    refined_end = audio_dur
                    logger.info(
                        f"EOF guardrail extended last segment to audio_dur "
                        f"(tail rms {tail_rms:.5f} > thr {_speech_thr:.5f})"
                    )
        else:
            refined_end = _snap_zc(y, sr, refined_end)

        refined_start = max(0.0, min(refined_start, audio_dur))
        refined_end = max(0.0, min(refined_end, audio_dur))
        refined.append({"start": refined_start, "end": refined_end})

    for i in range(len(refined) - 1):
        if refined[i]["end"] > refined[i + 1]["start"]:
            mid = (refined[i]["end"] + refined[i + 1]["start"]) / 2
            refined[i]["end"] = mid
            refined[i + 1]["start"] = mid

    # Drop any degenerate (zero/negative duration) segments — an empty atrim
    # can silently truncate the concat output.
    MIN_SEG = 0.010
    kept = [s for s in refined if s["end"] - s["start"] >= MIN_SEG]
    if len(kept) != len(refined):
        logger.warning(
            f"Dropped {len(refined) - len(kept)} degenerate segment(s) during refinement"
        )

    logger.info(f"Boundary refinement via AcousticMap applied to {len(kept)} segments")
    return kept


def _refine_segments(segments: list, wav_path: str, export_mode: str) -> list:
    """
    Legacy refiner — kept as a fallback for when no AcousticMap is available.
    See _refine_from_map for the primary path.
    """
    try:
        import librosa
        import numpy as np

        y, sr = librosa.load(wav_path, sr=None, mono=True)

        global_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
        noise_floor = float(np.percentile(global_rms, 10)) + 1e-8
        speech_threshold = noise_floor * 6.0

        boundary_mode = "tight" if export_mode == "fast" else "natural"
        refiner = BoundaryRefiner()
        refined = []

        for i, seg in enumerate(segments):
            prev_end = segments[i - 1]["end"] if i > 0 else 0.0
            next_start = (
                segments[i + 1]["start"] if i < len(segments) - 1 else seg["end"] + 2.0
            )

            # Check the interior of each gap.  Edge-adjacent word energy is
            # excluded by the margin in _gap_has_speech, so only actual deleted
            # speech (or a full silence) is reported here.
            speech_in_end_gap = _gap_has_speech(
                y, sr, seg["end"], next_start, speech_threshold
            )
            speech_in_start_gap = _gap_has_speech(
                y, sr, prev_end, seg["start"], speech_threshold
            )

            # Onset-guided boundary for word-level cuts; fixed bias as fallback.
            refined_end = seg["end"]
            refined_start = seg["start"]

            if speech_in_end_gap:
                word_end = _find_word_end(
                    y, sr, seg["end"], next_start, speech_threshold
                )
                if word_end is not None:
                    refined_end = _snap_zc(y, sr, word_end)
                else:
                    # Fallback: fixed outward bias + ZC snap
                    gap_after = next_start - seg["end"]
                    end_bias = min(_WORD_MARGIN_S, gap_after * 0.35)
                    bias_target = seg["end"] + end_bias
                    if _sample_has_speech(y, sr, bias_target, speech_threshold):
                        refined_end = _snap_zc(y, sr, seg["end"])
                    else:
                        refined_end = _snap_zc(y, sr, bias_target)

            if speech_in_start_gap:
                onset = _find_onset_before(y, sr, seg["start"], prev_end)
                if onset is not None:
                    # Back up 20ms before the onset to include the attack transient.
                    # Capped at 80ms before WhisperX to prevent runaway detection;
                    # clamped forward so we never start inside the previous segment.
                    pre_onset = max(
                        prev_end + 0.005, onset - 0.020, seg["start"] - 0.080
                    )
                    refined_start = _snap_zc(y, sr, pre_onset)
                else:
                    # Fallback: fixed outward bias + ZC snap (original behavior)
                    gap_before = seg["start"] - prev_end
                    start_bias = min(_WORD_MARGIN_S, gap_before * 0.35)
                    bias_target = seg["start"] - start_bias
                    if _sample_has_speech(y, sr, bias_target, speech_threshold):
                        refined_start = _snap_zc(y, sr, seg["start"])
                    else:
                        refined_start = _snap_zc(y, sr, bias_target)

            # For silence-bordered sides, use BoundaryRefiner
            if not speech_in_end_gap or not speech_in_start_gap:
                start_window_pre = (
                    0.0
                    if speech_in_start_gap
                    else min(0.5, max(0.0, seg["start"] - prev_end))
                )
                end_window_post = (
                    0.0
                    if speech_in_end_gap
                    else min(0.7, max(0.0, next_start - seg["end"]))
                )

                result = refiner.refine_boundaries(
                    y=y,
                    sr=sr,
                    approx_start=seg["start"],
                    approx_end=seg["end"],
                    mode=boundary_mode,
                    start_window_pre=start_window_pre,
                    end_window_post=end_window_post,
                )
                if not speech_in_start_gap:
                    refined_start = result["refined_start"]
                if not speech_in_end_gap:
                    refined_end = result["refined_end"]
                if result["confidence_flags"]:
                    logger.debug(
                        f"Segment {seg['start']:.3f}-{seg['end']:.3f} flags: {result['confidence_flags']}"
                    )

            # Trim any leading silence the above logic didn't reach.
            # Handles the case where seg["start"] is placed at the end of the
            # last deleted word rather than at the kept word itself — leaving
            # up to 1-2s of near-silence that _find_onset_before can't bridge.
            refined_start = _advance_past_silence(
                y,
                sr,
                refined_start,
                min(seg["end"], refined_start + 2.0),
                speech_threshold,
            )

            refined.append({"start": refined_start, "end": refined_end})

        audio_dur = len(y) / sr
        for seg in refined:
            seg["start"] = max(0.0, min(seg["start"], audio_dur))
            seg["end"] = max(0.0, min(seg["end"], audio_dur))

        for i in range(len(refined) - 1):
            if refined[i]["end"] > refined[i + 1]["start"]:
                mid = (refined[i]["end"] + refined[i + 1]["start"]) / 2
                refined[i]["end"] = mid
                refined[i + 1]["start"] = mid

        MIN_SEG = 0.010
        kept = [s for s in refined if s["end"] - s["start"] >= MIN_SEG]
        if len(kept) != len(refined):
            logger.warning(
                f"Dropped {len(refined) - len(kept)} degenerate segment(s) during refinement"
            )

        logger.info(
            f"Boundary refinement applied ({boundary_mode} mode) to {len(kept)} segments"
        )
        return kept
    except Exception as e:
        logger.warning(f"Boundary refinement failed, using original timestamps: {e}")
        return segments


def _mux_audio(video_path: str, audio_path: str, output_path: str) -> str:
    """Replace video's audio track with cleaned audio using FFmpeg."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio mux failed: {result.stderr[-300:]}")
    return output_path


@router.post("/export")
async def export_video(req: ExportRequest):
    try:
        segments = [{"start": s.start, "end": s.end} for s in req.keep_segments]

        if not segments:
            raise HTTPException(status_code=400, detail="No segments to export")

        words_dicts = [w.model_dump() for w in req.words] if req.words else []
        deleted_set = set(req.deleted_indices or [])

        needs_reencode_for_subs = req.captions == "burn-in"
        single_segment_full_span = (
            len(segments) == 1 and _segment_covers_source(segments[0], req.input_path)
        )
        needs_boundary_refinement = bool(
            req.force_refine or (words_dicts and not single_segment_full_span)
        )

        use_stream_copy = (
            req.mode == "fast"
            and len(segments) == 1
            and not needs_reencode_for_subs
            and not needs_boundary_refinement
        )

        if req.mode == "fast" and len(segments) == 1 and needs_boundary_refinement:
            logger.info(
                "Fast single-segment export is using refined re-encode "
                "because the segment is word-aligned and does not cover the full source"
            )

        # Generate ASS file for burn-in
        ass_path = None
        if req.captions == "burn-in" and words_dicts:
            ass_content = generate_ass(words_dicts, deleted_set)
            tmp = tempfile.NamedTemporaryFile(
                suffix=".ass", delete=False, mode="w", encoding="utf-8"
            )
            tmp.write(ass_content)
            tmp.close()
            ass_path = tmp.name

        # Extract PCM WAV for sample-accurate audio cuts.
        # The original MP4's AAC audio has 1024-sample (~23ms) frame boundaries
        # which cause audible bleed at cut points. PCM WAV has no such constraints.
        audio_wav_path = None
        if not use_stream_copy:
            try:
                audio_wav_path = str(extract_audio(Path(req.input_path)))
                logger.info(f"Extracted WAV for sample-accurate cuts: {audio_wav_path}")
            except Exception as e:
                logger.warning(f"WAV extraction failed, falling back to AAC cuts: {e}")

        # Refine segment boundaries to natural splice points before FFmpeg cuts.
        # Reuses the already-extracted WAV — no extra I/O.
        if audio_wav_path:
            # Prefer the pre-computed AcousticMap.  If it's missing (e.g. the
            # ingest-time analysis was skipped or the cache was cleared), build
            # it on the fly so export still gets the good refinement.
            acoustic_map = load_acoustic_map(req.input_path)
            if acoustic_map is None and req.words is not None:
                try:
                    acoustic_map = analyze_file(
                        req.input_path,
                        [w.model_dump() for w in req.words],
                    )
                except Exception as e:
                    logger.warning(f"On-demand acoustic analysis failed: {e}")
                    acoustic_map = None

            if acoustic_map is not None:
                try:
                    segments = _refine_from_map(segments, audio_wav_path, acoustic_map)
                except Exception as e:
                    logger.warning(f"AcousticMap refinement failed, falling back: {e}")
                    segments = _refine_segments(segments, audio_wav_path, req.mode)
            else:
                segments = _refine_segments(segments, audio_wav_path, req.mode)

        try:
            if use_stream_copy:
                output = export_stream_copy(req.input_path, req.output_path, segments)
            elif ass_path:
                output = export_reencode_with_subs(
                    req.input_path,
                    req.output_path,
                    segments,
                    ass_path,
                    resolution=req.resolution,
                    format_hint=req.format,
                    audio_wav_path=audio_wav_path,
                )
            else:
                output = export_reencode(
                    req.input_path,
                    req.output_path,
                    segments,
                    resolution=req.resolution,
                    format_hint=req.format,
                    audio_wav_path=audio_wav_path,
                )
        finally:
            if ass_path and os.path.exists(ass_path):
                os.unlink(ass_path)
            if audio_wav_path and os.path.exists(audio_wav_path):
                try:
                    os.unlink(audio_wav_path)
                except OSError:
                    pass

        # Audio enhancement: clean, then mux back into the exported video
        if req.enhanceAudio:
            try:
                tmp_dir = tempfile.mkdtemp(prefix="cutscript_audio_")
                cleaned_audio = os.path.join(tmp_dir, "cleaned.wav")
                clean_audio(output, cleaned_audio)

                muxed_path = output + ".muxed.mp4"
                _mux_audio(output, cleaned_audio, muxed_path)

                os.replace(muxed_path, output)
                logger.info(f"Audio enhanced and muxed into {output}")

                # Cleanup
                try:
                    os.remove(cleaned_audio)
                    os.rmdir(tmp_dir)
                except OSError:
                    pass
            except Exception as e:
                logger.warning(f"Audio enhancement failed (non-fatal): {e}")

        # Sidecar SRT: generate and save alongside video
        srt_path = None
        if req.captions == "sidecar" and words_dicts:
            srt_content = generate_srt(words_dicts, deleted_set)
            srt_path = req.output_path.rsplit(".", 1)[0] + ".srt"
            save_captions(srt_content, srt_path)
            logger.info(f"Sidecar SRT saved to {srt_path}")

        result = {"status": "ok", "output_path": output}
        if srt_path:
            result["srt_path"] = srt_path
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        logger.error(f"Export failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Export error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
