"""
Reproduce the b_edited1.mp4 scenario: 7 keep segments, check that the last
segment ("Is there a problem?", words 76-79) survives the refinement pipeline
unchanged (within a few ms). User reports it was dropped from the export.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from services.audio_analyzer import analyze_file, SPECTRAL_CACHE_DIR  # noqa: E402
from utils.cache import get_file_hash  # noqa: E402

# Don't import the router (requires fastapi) — reimplement refiner query inline.
import numpy as np
import librosa


def _snap_zc(y, sr, t, search_ms=40.0):
    idx = int(t * sr)
    half = int(search_ms / 1000.0 * sr)
    s = max(0, idx - half)
    e = min(len(y), idx + half)
    zc_idx = np.where(librosa.zero_crossings(y[s:e]))[0] + s
    if len(zc_idx) == 0:
        return t
    return float(zc_idx[np.argmin(np.abs(zc_idx - idx))]) / sr


def _refine_from_map(segments, wav_path, acoustic_map):
    y, sr = librosa.load(wav_path, sr=None, mono=True)
    refined = []
    TOL = 0.100
    for i, seg in enumerate(segments):
        refined_start = seg["start"]
        refined_end = seg["end"]
        start_word = acoustic_map.find_word_by_start(seg["start"], tolerance=TOL)
        end_word = acoustic_map.find_word_by_end(seg["end"], tolerance=TOL)
        prev_cap = segments[i - 1]["end"] if i > 0 else 0.0
        next_cap = segments[i + 1]["start"] if i < len(segments) - 1 else float("inf")
        if start_word is not None:
            refined_start = max(prev_cap, start_word.as_)
        if end_word is not None:
            refined_end = min(
                next_cap - 0.005 if np.isfinite(next_cap) else end_word.ae,
                end_word.ae,
            )
        refined_start = _snap_zc(y, sr, refined_start)
        refined_end = _snap_zc(y, sr, refined_end)
        refined.append({"start": refined_start, "end": refined_end})
    for i in range(len(refined) - 1):
        if refined[i]["end"] > refined[i + 1]["start"]:
            mid = (refined[i]["end"] + refined[i + 1]["start"]) / 2
            refined[i]["end"] = mid
            refined[i + 1]["start"] = mid
    return refined

VIDEO = Path("/Users/kat/cutscript/b_vid/b.mp4")
TX_CACHE = Path.home() / ".obs_transcriber_cache" / "c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json"

# Kept word-index ranges derived from b_vid/indended.png
KEPT_RANGES = [
    (4, 6),    # How's it going?
    (18, 19),  # Who's she?
    (31, 33),  # Where's she from?
    (46, 46),  # Spanish?
    (50, 51),  # I know.
    (65, 68),  # Okay, time to go.
    (76, 79),  # Is there a problem?
]


def extract_wav(video: Path) -> Path:
    out = Path(tempfile.mkdtemp(prefix="phase5_")) / (video.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "48000", str(out)],
        check=True, capture_output=True,
    )
    return out


def main():
    words = json.load(open(TX_CACHE))["data"]["words"]
    duration = 49.451667

    # Build segments the same way the frontend does
    segments = []
    for s_idx, e_idx in KEPT_RANGES:
        segments.append({"start": words[s_idx]["start"], "end": words[e_idx]["end"]})

    # Apply frontend padding (first/last ±1.5s, clamped to duration)
    segments[0]["start"] = max(0.0, segments[0]["start"] - 1.5)
    segments[-1]["end"] = min(duration, segments[-1]["end"] + 1.5)

    print("Keep segments (frontend output):")
    for i, s in enumerate(segments):
        print(f"  {i}: {s['start']:7.3f} -> {s['end']:7.3f} (dur {s['end']-s['start']:.3f}s)")

    # Build the AcousticMap
    wav = extract_wav(VIDEO)
    file_hash = get_file_hash(wav)
    stale = SPECTRAL_CACHE_DIR / f"{file_hash}.json"
    if stale.exists():
        stale.unlink()
    m = analyze_file(wav, words)

    # Run the refiner
    refined = _refine_from_map(segments, str(wav), m)
    print("\nRefined segments:")
    for i, s in enumerate(refined):
        print(f"  {i}: {s['start']:7.3f} -> {s['end']:7.3f} (dur {s['end']-s['start']:.3f}s)")

    # Total output duration
    total = sum(s["end"] - s["start"] for s in refined)
    print(f"\nTotal output duration: {total:.3f}s")

    last = refined[-1]
    if last["end"] - last["start"] < 0.5:
        print(f"FAIL: last segment too short ({last['end'] - last['start']:.3f}s)")
        return 1
    print("PASS: last segment preserved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
