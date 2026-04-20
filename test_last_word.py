import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

from services.transcription import transcribe_audio
from services.audio_analyzer import analyze_file, load_acoustic_map
import librosa
import numpy as np

VIDEO_PATH = Path("/Users/kat/cutscript/tests/video/c8.mp4")


def _snap_zc(y, sr, t, search_ms=40.0):
    idx = int(t * sr)
    half = int(search_ms / 1000.0 * sr)
    s = max(0, idx - half)
    e = min(len(y), idx + half)
    zc_idx = np.where(librosa.zero_crossings(y[s:e]))[0] + s
    if len(zc_idx) == 0:
        return t
    return float(zc_idx[np.argmin(np.abs(zc_idx - idx))]) / sr


def _refine_last_segment(wav_path, acoustic_map):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    audio_dur = len(y) / sr
    TOL = 0.100
    EOF_WINDOW_S = 0.100

    # Assume last segment covers the last word
    seg = {
        "start": acoustic_map.words[-2].ws if len(acoustic_map.words) > 1 else 0,
        "end": audio_dur,
    }  # approx

    refined_start = max(0.0, min(seg["start"], audio_dur))
    refined_end = max(0.0, min(seg["end"], audio_dur))

    start_word = acoustic_map.find_word_by_start(seg["start"], tolerance=TOL)
    end_word = acoustic_map.find_word_by_end(seg["end"], tolerance=TOL)

    prev_cap = 0.0  # first segment
    is_last_seg = True
    next_cap = float("inf")

    if start_word is not None:
        refined_start = max(prev_cap, start_word.as_)
    if end_word is not None:
        refined_end = min(
            next_cap - 0.005 if np.isfinite(next_cap) else end_word.ae, end_word.ae
        )

    print(f"Before snap: refined_end = {refined_end:.3f}")

    refined_start = _snap_zc(y, sr, refined_start)

    if is_last_seg:
        # For last segment, skip zero-crossing snap to preserve content near EOF
        print(f"After snap: refined_end = {refined_end:.3f} (skipped for last segment)")

        # EOF guardrail
        if (audio_dur - refined_end) <= EOF_WINDOW_S:
            tail_s = int(refined_end * sr)
            tail_e = int(audio_dur * sr)
            tail_rms = (
                float(np.sqrt(np.mean(y[tail_s:tail_e] ** 2) + 1e-12))
                if tail_e > tail_s
                else 0.0
            )
            speech_thr = (
                float(
                    np.percentile(
                        librosa.feature.rms(y=y, frame_length=1024, hop_length=256)[0],
                        10,
                    )
                )
                + 1e-8
            ) * 3.0
            print(f"Tail rms: {tail_rms:.5f}, speech_thr: {speech_thr:.5f}")
            if tail_rms > speech_thr:
                refined_end = audio_dur
                print("EOF guardrail extended to audio_dur")
            else:
                print("EOF guardrail did not extend")
    else:
        refined_end = _snap_zc(y, sr, refined_end)
        print(f"After snap: refined_end = {refined_end:.3f}")

    print(f"Final refined_end: {refined_end:.3f}")
    print(f"Audio dur: {audio_dur:.3f}, diff: {audio_dur - refined_end:.3f}")

    return refined_end


def main():
    print(f"Processing {VIDEO_PATH}")

    # Load the new transcription
    import json

    cache_path = "/Users/kat/.obs_transcriber_cache/dc4f936d5bd98d03e7379a88ecd88be3_large-v3_transcribe_wx_3bf4d1b0.json"
    with open(cache_path) as f:
        data = json.load(f)
    words = data["data"]["words"]
    print(f"Loaded {len(words)} words from cache, last: {words[-1]['word']}")
    acoustic_map = analyze_file(str(VIDEO_PATH), words, force=True)

    print(f"AcousticMap has {len(acoustic_map.words)} words")

    # Check last word
    last_word = acoustic_map.words[-1]
    print(
        f"Last word: '{last_word.text}' ws={last_word.ws:.3f} we={last_word.we:.3f} ae={last_word.ae:.3f}"
    )
    print(f"Audio duration: {acoustic_map.duration:.3f}")
    print(f"Tail after ae: {acoustic_map.duration - last_word.ae:.3f}s")

    # Simulate export refinement for last segment
    from utils.audio_processing import extract_audio

    wav_path = extract_audio(VIDEO_PATH)
    try:
        _refine_last_segment(str(wav_path), acoustic_map)
    finally:
        import os

        if os.path.exists(wav_path):
            os.unlink(wav_path)


if __name__ == "__main__":
    main()
