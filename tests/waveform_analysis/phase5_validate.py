"""
Phase 5 validation: build AcousticMap for b.mp4 and check that the
"Spanish?" word's acoustic_end extends far enough past WhisperX's endpoint
to preserve the /ʃ/ fricative tail that Phase 0 measured at ~310ms.
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

VIDEO = Path("/Users/kat/Desktop/b.mp4")


def extract_wav(video: Path) -> Path:
    out = Path(tempfile.mkdtemp(prefix="phase5_")) / (video.stem + ".wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(video), "-ac", "1", "-ar", "48000", str(out)],
        check=True, capture_output=True,
    )
    return out

# Use the cached transcript from the existing Whisper cache
CACHE_FILE = Path.home() / ".obs_transcriber_cache" / "c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json"


def main():
    if not VIDEO.exists():
        print(f"FAIL: {VIDEO} not found")
        return 1
    if not CACHE_FILE.exists():
        print(f"FAIL: transcript cache {CACHE_FILE} not found")
        return 1

    with open(CACHE_FILE) as f:
        payload = json.load(f)
    tx = payload["data"]
    words = tx["words"]
    print(f"Loaded {len(words)} words from transcript cache")

    # Extract WAV with ffmpeg so we bypass the moviepy dependency in extract_audio.
    wav_path = extract_wav(VIDEO)
    print(f"Extracted WAV: {wav_path}")

    # Clear any stale spectral cache for this file so we rebuild fresh.
    # Note: cache key hashes the WAV path here, not the MP4 — that's OK for this test.
    file_hash = get_file_hash(wav_path)
    stale = SPECTRAL_CACHE_DIR / f"{file_hash}.json"
    if stale.exists():
        stale.unlink()
        print(f"Removed stale spectral cache: {stale.name}")

    print("Building AcousticMap...")
    m = analyze_file(wav_path, words)
    print(f"Map built: {len(m.words)} words, sr={m.sr}, "
          f"noise_floor={m.noise_floor_rms:.5f}, fric_floor={m.fricative_noise_floor:.6f}")

    # Find all Spanish words (case-insensitive, includes punctuation)
    spans = [w for w in m.words if w.text.lower().strip(".,?!") == "spanish"]
    print(f"\nFound {len(spans)} Spanish word(s):")
    for w in spans:
        ext_end = w.ae - w.we
        ext_start = w.ws - w.as_
        print(f"  i={w.i:3d} '{w.text}' ws={w.ws:.3f} we={w.we:.3f} "
              f"as={w.as_:.3f} ae={w.ae:.3f} "
              f"(+{ext_start * 1000:.0f}ms onset, +{ext_end * 1000:.0f}ms coda) "
              f"onset={w.onset} coda={w.coda} "
              f"peak_rms={w.peak_rms:.4f} peak_fric={w.peak_fric:.4f} "
              f"dips={len(w.dips)}")

    # The kept word in b_edited2.mp4 is "Spanish?" at ws=28.739
    target = next((w for w in spans if abs(w.ws - 28.739) < 0.05), None)
    if target is None:
        print("\nFAIL: could not locate 'Spanish?' (ws≈28.739)")
        return 1

    coda_extension_ms = (target.ae - target.we) * 1000.0
    print(f"\n'Spanish?' coda extension: {coda_extension_ms:.0f}ms")

    # Phase 0 measured the actual /ʃ/ tail extending ~310ms past we (to ~29.430s).
    # Accept anything >= 150ms as "the fricative is being captured now."
    if coda_extension_ms >= 150.0:
        print(f"PASS: fricative tail is captured (>=150ms extension).")
        return 0
    else:
        print(f"FAIL: coda extension {coda_extension_ms:.0f}ms < 150ms — "
              "Spanish will still be truncated.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
