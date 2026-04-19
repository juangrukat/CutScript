"""Test the legacy _refine_segments with the same 7-segment scenario."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np
import librosa

from services.boundary_refiner import BoundaryRefiner  # noqa: E402


# Copy of the legacy refinement logic (inlined to avoid FastAPI import)
_WORD_MARGIN_S = 0.028


def _gap_has_speech(y, sr, t0, t1, threshold):
    gap = t1 - t0
    if gap <= 0:
        return False
    margin = max(0.04, gap * 0.10)
    mid_s, mid_e = t0 + margin, t1 - margin
    if mid_e <= mid_s:
        mid_s, mid_e = t0, t1
    s = max(0, int(mid_s * sr))
    e = min(len(y), int(mid_e * sr))
    if e - s < 256:
        return False
    rms = librosa.feature.rms(y=y[s:e], frame_length=512, hop_length=128)[0]
    return bool(np.mean(rms > threshold) > 0.15)


def _snap_zc(y, sr, t, search_ms=40.0):
    idx = int(t * sr)
    half = int(search_ms / 1000.0 * sr)
    s = max(0, idx - half)
    e = min(len(y), idx + half)
    zc_idx = np.where(librosa.zero_crossings(y[s:e]))[0] + s
    if len(zc_idx) == 0:
        return t
    return float(zc_idx[np.argmin(np.abs(zc_idx - idx))]) / sr


def _sample_has_speech(y, sr, t, threshold, window_ms=15.0):
    half = int(window_ms / 1000.0 * sr)
    s = max(0, int(t * sr) - half)
    e = min(len(y), int(t * sr) + half)
    if e - s < 64:
        return False
    rms = librosa.feature.rms(y=y[s:e], frame_length=256, hop_length=64)[0]
    return bool(np.mean(rms > threshold) > 0.50)


def _find_onset_before(y, sr, t_whisper, prev_end):
    search_start = max(prev_end, t_whisper - 0.15)
    search_end = min(len(y) / sr, t_whisper + 0.05)
    s, e = int(search_start * sr), int(search_end * sr)
    if e - s < 512:
        return None
    y_win = y[s:e]
    env = librosa.onset.onset_strength(y=y_win, sr=sr)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr,
        pre_max=2, post_max=2, pre_avg=2, post_avg=5,
        delta=0.03, wait=1,
    )
    if len(frames) == 0:
        return None
    onsets_abs = librosa.frames_to_time(frames, sr=sr) + search_start
    candidates = onsets_abs[onsets_abs <= t_whisper + 0.010]
    if len(candidates) == 0:
        return None
    return float(candidates[-1])


def _find_word_end(y, sr, t_whisper, next_start, speech_threshold):
    cap = min(len(y) / sr, min(next_start - 0.005, t_whisper + 0.150))
    s = int(t_whisper * sr)
    e = int(cap * sr)
    if e - s < 256:
        return None
    hop = 64
    s_ref = max(0, int((t_whisper - 1.0) * sr))
    y_ref = y[s_ref:max(s_ref + 256, int(t_whisper * sr))]
    if len(y_ref) >= 256:
        rms_ref = librosa.feature.rms(y=y_ref, frame_length=256, hop_length=hop)[0]
        word_peak_power = float(np.max(rms_ref ** 2))
        if word_peak_power > 1e-10:
            threshold_power = word_peak_power * (10 ** (-25.0 / 10.0))
            y_fwd = y[s:e]
            rms_fwd = librosa.feature.rms(y=y_fwd, frame_length=256, hop_length=hop)[0]
            rms_t = librosa.frames_to_time(np.arange(len(rms_fwd)), sr=sr, hop_length=hop) + t_whisper
            below = rms_fwd ** 2 < threshold_power
            for i in range(len(below) - 2):
                if below[i] and below[i + 1] and below[i + 2]:
                    decay_t = float(rms_t[i])
                    if decay_t - t_whisper > 0.003:
                        return decay_t
                    return None
    y_fwd2 = y[s:e]
    if len(y_fwd2) >= 512:
        env = librosa.onset.onset_strength(y=y_fwd2, sr=sr)
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr,
            pre_max=2, post_max=2, pre_avg=2, post_avg=5,
            delta=0.03, wait=1,
        )
        if len(frames) > 0:
            onsets_abs = librosa.frames_to_time(frames, sr=sr) + t_whisper
            next_onset = float(onsets_abs[0])
            result = min(cap, next_onset - 0.020)
            if result > t_whisper:
                return result
    return None


def _advance_past_silence(y, sr, t_start, t_cap, speech_threshold):
    if _sample_has_speech(y, sr, t_start + 0.025, speech_threshold, window_ms=25.0):
        return t_start
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
        return t_start
    return max(t_start, speech_t - 0.020)


def legacy_refine(segments, wav_path, export_mode="reencode"):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    global_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=512)[0]
    noise_floor = float(np.percentile(global_rms, 10)) + 1e-8
    speech_threshold = noise_floor * 6.0
    boundary_mode = "tight" if export_mode == "fast" else "natural"
    refiner = BoundaryRefiner()
    refined = []
    for i, seg in enumerate(segments):
        prev_end = segments[i - 1]["end"] if i > 0 else 0.0
        next_start = segments[i + 1]["start"] if i < len(segments) - 1 else seg["end"] + 2.0
        speech_in_end_gap = _gap_has_speech(y, sr, seg["end"], next_start, speech_threshold)
        speech_in_start_gap = _gap_has_speech(y, sr, prev_end, seg["start"], speech_threshold)
        refined_end = seg["end"]
        refined_start = seg["start"]
        if speech_in_end_gap:
            word_end = _find_word_end(y, sr, seg["end"], next_start, speech_threshold)
            if word_end is not None:
                refined_end = _snap_zc(y, sr, word_end)
            else:
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
                pre_onset = max(prev_end + 0.005, onset - 0.020, seg["start"] - 0.080)
                refined_start = _snap_zc(y, sr, pre_onset)
            else:
                gap_before = seg["start"] - prev_end
                start_bias = min(_WORD_MARGIN_S, gap_before * 0.35)
                bias_target = seg["start"] - start_bias
                if _sample_has_speech(y, sr, bias_target, speech_threshold):
                    refined_start = _snap_zc(y, sr, seg["start"])
                else:
                    refined_start = _snap_zc(y, sr, bias_target)
        if not speech_in_end_gap or not speech_in_start_gap:
            start_window_pre = 0.0 if speech_in_start_gap else min(0.5, max(0.0, seg["start"] - prev_end))
            end_window_post = 0.0 if speech_in_end_gap else min(0.7, max(0.0, next_start - seg["end"]))
            result = refiner.refine_boundaries(
                y=y, sr=sr,
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
            print(f"  seg {i}: BoundaryRefiner result start={result['refined_start']:.3f} end={result['refined_end']:.3f} flags={result['confidence_flags']}")
        refined_start = _advance_past_silence(
            y, sr, refined_start, min(seg["end"], refined_start + 2.0), speech_threshold
        )
        refined.append({"start": refined_start, "end": refined_end})
    for i in range(len(refined) - 1):
        if refined[i]["end"] > refined[i + 1]["start"]:
            mid = (refined[i]["end"] + refined[i + 1]["start"]) / 2
            refined[i]["end"] = mid
            refined[i + 1]["start"] = mid
    return refined


VIDEO = Path("/Users/kat/cutscript/b_vid/b.mp4")
TX = Path.home() / ".obs_transcriber_cache" / "c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json"
KEPT = [(4, 6), (18, 19), (31, 33), (46, 46), (50, 51), (65, 68), (76, 79)]


def main():
    words = json.load(open(TX))["data"]["words"]
    duration = 49.451667
    segments = [{"start": words[s]["start"], "end": words[e]["end"]} for s, e in KEPT]
    segments[0]["start"] = max(0.0, segments[0]["start"] - 1.5)
    segments[-1]["end"] = min(duration, segments[-1]["end"] + 1.5)

    print("Input segments:")
    for i, s in enumerate(segments):
        print(f"  {i}: {s['start']:7.3f} -> {s['end']:7.3f}")

    wav = Path(tempfile.mkdtemp(prefix="phase5_leg_")) / "b.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(VIDEO), "-ac", "1", "-ar", "48000", str(wav)],
        check=True, capture_output=True,
    )

    print("\nRunning LEGACY _refine_segments...")
    refined = legacy_refine(segments, str(wav))
    print("\nLegacy refined segments:")
    for i, s in enumerate(refined):
        dur = s["end"] - s["start"]
        print(f"  {i}: {s['start']:7.3f} -> {s['end']:7.3f}  (dur {dur:.3f}s)")

    last = refined[-1]
    print(f"\nLast segment duration: {last['end'] - last['start']:.3f}s")


if __name__ == "__main__":
    main()
