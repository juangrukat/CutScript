"""
Acoustic pre-analysis ("AcousticMap") for a source video.

Runs once per file after transcription, caches per-word spectral fingerprints
to ~/.cutscript_spectral_cache/<file_hash>.json, and exposes a query API that
the export refiner calls to place cut boundaries.

The map stores, for every WhisperX word:
  - ws / we: whisperX word start/end (original)
  - as / ae: acoustic start/end — extended to include onset transients (voiceless
             consonant clusters like "Sp", "St") and coda tails (fricatives like
             "sh", "s"; nasals like "n"/"m"). This is the fix for the "Spanish"
             truncation issue: the /ʃ/ fricative in "Spanish?" extends ~310ms past
             WhisperX's endpoint, and 'ae' captures that.
  - onset / coda: phoneme class of the first/last phone, derived from spelling.
             Used by the refiner to choose a decay threshold appropriate to the
             phoneme type (fricatives decay slowly in the 2–8kHz band; stops are
             sharp).
  - peak_rms / peak_fric: word-peak broadband and 2–8kHz band RMS. The 2–8kHz
             peak is what's used to detect fricative tails — a /ʃ/ fricative has
             low broadband energy but high band energy, so a broadband decay
             detector mis-fires on it.
  - dips: intra-word RMS dips more than 10dB below the word peak. These mark
             nasals and internal pauses, which an onset detector would otherwise
             confuse for a word boundary (the false "new word" detection at
             29.155s inside Spanish's /n/ was caused by exactly this).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from utils.cache import get_file_hash

logger = logging.getLogger(__name__)

SPECTRAL_CACHE_DIR = Path.home() / ".cutscript_spectral_cache"
# v3: Additional EOF fix — when Whisper's `we` for the last word lands within
# ~20ms of audio EOF, the tail window is too short (≤3 frames at hop=256,
# sr=48000) to run the decay analysis, so the `if len(la_idx) > 3` guard
# short-circuits and `ae` stays at its initial value. v3 initializes the
# last word's `ae` to `lookahead_end` (the EOF cap) instead of `we`, so short
# tail windows still extend to EOF. Without this, v2 caches show `ae == we`
# for words like "freedom." and the final 20ms of nasal tail gets clipped.
_MAP_VERSION = 3

# Short-horizon window for all detection work. Matches _find_word_end's old cap,
# but extended to 500ms for coda search because a /ʃ/ fricative can trail ~300ms
# past the WhisperX endpoint.
_CODA_SEARCH_MS = 500.0
_ONSET_LOOKBACK_MS = 200.0

# Phoneme-class onset/coda inference from spelling.  Rough but sufficient:
# we only need to know "is this phone sustained & noisy (fricative) vs sharp
# (stop) vs resonant (vowel/approximant/nasal)" to pick the right decay policy.
_FRICATIVE_LETTERS = {"s", "f", "v", "z", "h"}
_STOP_LETTERS = {"p", "b", "t", "d", "k", "g"}
_NASAL_LETTERS = {"m", "n"}
_APPROX_LETTERS = {"w", "y", "l", "r"}
_VOWEL_LETTERS = {"a", "e", "i", "o", "u"}

_FRICATIVE_DIGRAPHS = {"sh", "ch", "th", "ph", "gh", "zh"}


def _classify_onset(text: str) -> str:
    """Phoneme-class of the first phone of a word, from spelling."""
    t = "".join(c for c in text.lower() if c.isalpha())
    if not t:
        return "vowel"
    if t[:2] in _FRICATIVE_DIGRAPHS:
        return "fricative"
    # "sp", "st", "sk", "sc" — the /s/ dominates the onset transient
    if len(t) >= 2 and t[0] == "s" and t[1] in {"p", "t", "k", "c", "l", "m", "n", "w"}:
        return "fricative"
    c = t[0]
    if c in _FRICATIVE_LETTERS:
        return "fricative"
    if c in _STOP_LETTERS:
        return "stop"
    if c in _NASAL_LETTERS:
        return "nasal"
    if c in _APPROX_LETTERS:
        return "approximant"
    return "vowel"


def _classify_coda(text: str) -> str:
    """Phoneme-class of the last phone of a word, from spelling."""
    t = "".join(c for c in text.lower() if c.isalpha())
    if not t:
        return "vowel"
    if t[-2:] in _FRICATIVE_DIGRAPHS:
        return "fricative"
    if t.endswith("ng"):
        return "nasal"
    c = t[-1]
    if c in _FRICATIVE_LETTERS:
        return "fricative"
    if c in _STOP_LETTERS:
        return "stop"
    if c in _NASAL_LETTERS:
        return "nasal"
    if c in _APPROX_LETTERS:
        return "approximant"
    return "vowel"


@dataclass
class WordFingerprint:
    i: int
    text: str
    ws: float
    we: float
    as_: float   # acoustic start (stored as "as" in JSON — "as" is a Python keyword)
    ae: float    # acoustic end
    onset: str
    coda: str
    peak_rms: float
    peak_fric: float
    dips: list   # list of [t_sec, db_below_peak]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["as"] = d.pop("as_")
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "WordFingerprint":
        return cls(
            i=d["i"], text=d["text"],
            ws=d["ws"], we=d["we"],
            as_=d.get("as", d["ws"]), ae=d.get("ae", d["we"]),
            onset=d.get("onset", "vowel"), coda=d.get("coda", "vowel"),
            peak_rms=d.get("peak_rms", 0.0), peak_fric=d.get("peak_fric", 0.0),
            dips=d.get("dips", []),
        )


@dataclass
class AcousticMap:
    version: int
    file_hash: str
    duration: float
    sr: int
    noise_floor_rms: float
    speech_threshold: float
    fricative_noise_floor: float
    words: list  # list[WordFingerprint]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "file_hash": self.file_hash,
            "duration": self.duration,
            "sr": self.sr,
            "noise_floor_rms": self.noise_floor_rms,
            "speech_threshold": self.speech_threshold,
            "fricative_noise_floor": self.fricative_noise_floor,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "AcousticMap":
        return cls(
            version=d.get("version", 1),
            file_hash=d["file_hash"],
            duration=d["duration"],
            sr=d["sr"],
            noise_floor_rms=d["noise_floor_rms"],
            speech_threshold=d["speech_threshold"],
            fricative_noise_floor=d.get("fricative_noise_floor", d["noise_floor_rms"]),
            words=[WordFingerprint.from_dict(w) for w in d["words"]],
        )

    def find_word_by_end(self, t: float, tolerance: float = 0.05) -> Optional[WordFingerprint]:
        """Return the word whose whisper end (we) is closest to t, within tolerance."""
        best = None
        best_d = tolerance
        for w in self.words:
            d = abs(w.we - t)
            if d < best_d:
                best = w
                best_d = d
        return best

    def find_word_by_start(self, t: float, tolerance: float = 0.05) -> Optional[WordFingerprint]:
        """Return the word whose whisper start (ws) is closest to t, within tolerance."""
        best = None
        best_d = tolerance
        for w in self.words:
            d = abs(w.ws - t)
            if d < best_d:
                best = w
                best_d = d
        return best


def _spectral_cache_path(file_hash: str) -> Path:
    SPECTRAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return SPECTRAL_CACHE_DIR / f"{file_hash}.json"


def load_acoustic_map(file_path: str | os.PathLike) -> Optional[AcousticMap]:
    """Load a cached AcousticMap for the given source file, or None if absent/stale."""
    file_hash = get_file_hash(Path(file_path))
    if not file_hash:
        return None
    path = _spectral_cache_path(file_hash)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("version") != _MAP_VERSION:
            return None
        return AcousticMap.from_dict(data)
    except Exception as e:
        logger.warning(f"Failed to load acoustic map {path}: {e}")
        return None


def _save_acoustic_map(m: AcousticMap) -> Path:
    path = _spectral_cache_path(m.file_hash)
    with open(path, "w") as f:
        json.dump(m.to_dict(), f, separators=(",", ":"))
    return path


def _fricative_band_rms(y, sr: int, frame_length: int = 1024, hop_length: int = 256):
    """RMS of the 2–8 kHz band — the band where fricatives (/ʃ/, /s/, /f/) sit."""
    import numpy as np
    import librosa

    stft = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    band_mask = (freqs >= 2000) & (freqs <= 8000)
    band_power = np.abs(stft[band_mask, :]) ** 2
    band_rms = np.sqrt(np.mean(band_power, axis=0) + 1e-12)
    return band_rms


def _analyze_word(
    y, sr: int, word: dict, next_ws: float, prev_we: float,
    speech_threshold: float, fric_threshold_floor: float,
    rms_full, rms_full_t, band_rms, band_rms_t,
    hop_length: int,
    is_last: bool = False,
) -> WordFingerprint:
    """Build a fingerprint for a single word by probing the audio around its WhisperX span."""
    import numpy as np

    ws = float(word["start"])
    we = float(word["end"])
    text = word.get("word", "").strip()

    onset = _classify_onset(text)
    coda = _classify_coda(text)

    # Word peak RMS & peak fricative-band energy — used as reference for decay
    def _slice_at(arr_t, t0, t1):
        mask = (arr_t >= t0) & (arr_t <= t1)
        return np.where(mask)[0]

    interior_idx = _slice_at(rms_full_t, ws, we)
    peak_rms = float(np.max(rms_full[interior_idx])) if len(interior_idx) > 0 else 0.0

    band_interior_idx = _slice_at(band_rms_t, ws, we)
    peak_fric = float(np.max(band_rms[band_interior_idx])) if len(band_interior_idx) > 0 else 0.0

    # ---- acoustic start (as_): look back up to 200ms for the onset transient.
    as_ = ws
    if onset in ("fricative", "stop"):
        lookback_start = max(prev_we, ws - _ONSET_LOOKBACK_MS / 1000.0)
        lb_idx = _slice_at(rms_full_t, lookback_start, ws)
        if len(lb_idx) > 0:
            # Onset threshold: halfway between noise floor and word peak, in RMS space.
            # This picks up the beginning of energy rise without being fooled by
            # the preceding word's decay tail.
            onset_threshold = max(speech_threshold, peak_rms * 0.1)
            rising = np.where(rms_full[lb_idx] > onset_threshold)[0]
            if len(rising) > 0:
                as_ = float(rms_full_t[lb_idx[rising[0]]])
                # 20ms pre-roll to include the attack transient
                as_ = max(lookback_start, as_ - 0.020)

    # ---- acoustic end (ae): look forward up to 500ms for the real decay point.
    # This is the critical fix for the "Spanish" bug.  Coda policy:
    #   - fricative coda: watch the 2–8kHz band, not broadband. Fricatives like
    #       /ʃ/, /s/, /f/ drop below broadband speech threshold but linger in the
    #       fricative band — and they are *audibly present* as long as that band
    #       is loud. We cut when the band drops below max(peak_fric*0.1, 2×floor).
    #   - nasal coda: broadband decay, but allow more time (nasals have slow tails).
    #   - stop coda: broadband decay; stops have sharp offsets, so default window.
    #   - vowel / approximant: broadband decay at -25dB below peak_rms.
    # For a mid-stream word the coda search cap is the next word's start minus
    # a 5ms guard so `ae` never crosses into the next word's onset. For the
    # LAST word of the audio there is no next word — the 5ms guard would just
    # leave a hardcoded hole at the end of the video, so we let the cap run
    # all the way to the audio end (EOF is its own natural boundary).
    if is_last:
        lookahead_end = min(
            len(y) / sr,
            we + _CODA_SEARCH_MS / 1000.0,
        )
    else:
        lookahead_end = min(
            len(y) / sr,
            we + _CODA_SEARCH_MS / 1000.0,
            next_ws - 0.005,
        )

    # For the last word, initialize ae at the EOF cap rather than `we`. If the
    # tail window [we, lookahead_end] is too short for decay analysis (fewer
    # than ~4 frames — happens when Whisper's `we` lands within ~20ms of audio
    # EOF, as with "freedom." at the end of the video), the decay branch below
    # is skipped entirely, so this initial value is what ends up in the cache.
    # Without this, the final word's `ae` collapses to `we` and the export
    # refiner truncates any remaining nasal/coda tail audio.
    ae = lookahead_end if is_last else we
    la_idx = _slice_at(rms_full_t, we, lookahead_end)
    la_band_idx = _slice_at(band_rms_t, we, lookahead_end)

    if len(la_idx) > 3 and peak_rms > 1e-6:
        if coda == "fricative" and len(la_band_idx) > 3 and peak_fric > 1e-6:
            fric_thr = max(peak_fric * 0.10, fric_threshold_floor * 2.0)
            bb_thr = max(speech_threshold * 0.7, peak_rms * 0.05)
            # First run of 3 consecutive frames where BOTH fall below threshold.
            # Using "both" prevents a noisy broadband spike (e.g., a click) from
            # prematurely ending the word while the fricative is still audible.
            below_band = band_rms[la_band_idx] < fric_thr
            below_bb = rms_full[la_idx] < bb_thr
            # Align the two arrays by time (they use the same hop)
            L = min(len(below_band), len(below_bb))
            found = None
            for k in range(L - 2):
                if below_band[k] and below_band[k + 1] and below_band[k + 2] \
                   and below_bb[k] and below_bb[k + 1] and below_bb[k + 2]:
                    found = k
                    break
            if found is not None:
                ae = float(band_rms_t[la_band_idx[found]])
            else:
                ae = float(lookahead_end)  # fricative still loud — take the cap
        else:
            # vowel / approximant / nasal / stop: -25dB broadband decay
            db_target = -25.0 if coda in ("vowel", "approximant") else -20.0
            # for nasals give a bit more slack because nasal murmur decays gently
            if coda == "nasal":
                db_target = -18.0
            thr_power = (peak_rms ** 2) * (10 ** (db_target / 10.0))
            pow_la = rms_full[la_idx] ** 2
            below = pow_la < thr_power
            found = None
            for k in range(len(below) - 2):
                if below[k] and below[k + 1] and below[k + 2]:
                    found = k
                    break
            if found is not None:
                ae = float(rms_full_t[la_idx[found]])
            elif is_last:
                # Last word + no decay found within the search window: the tail
                # is running into EOF (e.g. the speaker's last syllable fades
                # into silence after the cap, or the file truly ends mid-word).
                # Extend to the cap so the final frames of the audio are kept —
                # leaving ae=we here is what made end-of-video cuts fall short.
                ae = float(lookahead_end)
            # Mid-stream: if the decay never drops, another word likely starts
            # immediately. Keep ae = we so we don't swallow the next word's onset.

    # Clamp: ae must be ≥ we (don't shrink the word), and ≤ lookahead_end.
    ae = max(we, min(ae, lookahead_end))
    as_ = min(ws, max(as_, prev_we))

    # ---- intraword dips: 10dB below peak, inside [as_, ae]
    dip_idx = _slice_at(rms_full_t, as_, ae)
    dips = []
    if len(dip_idx) > 0 and peak_rms > 1e-6:
        word_rms = rms_full[dip_idx]
        word_t = rms_full_t[dip_idx]
        peak_power = peak_rms ** 2
        dip_threshold = peak_power * (10 ** (-10.0 / 10.0))
        local_powers = word_rms ** 2
        # Find local minima below the dip threshold
        for k in range(1, len(local_powers) - 1):
            if (local_powers[k] < dip_threshold and
                local_powers[k] <= local_powers[k - 1] and
                local_powers[k] <= local_powers[k + 1]):
                db = 10.0 * float(np.log10(max(local_powers[k], 1e-12) / peak_power))
                dips.append([round(float(word_t[k]), 4), round(db, 2)])

    return WordFingerprint(
        i=int(word.get("_index", 0)),
        text=text,
        ws=round(ws, 4), we=round(we, 4),
        as_=round(as_, 4), ae=round(ae, 4),
        onset=onset, coda=coda,
        peak_rms=round(peak_rms, 6),
        peak_fric=round(peak_fric, 6),
        dips=dips,
    )


def analyze_file(
    file_path: str | os.PathLike,
    words: list,
    force: bool = False,
    progress_cb=None,
) -> AcousticMap:
    """
    Build (or load cached) AcousticMap for `file_path` given Whisper `words`.

    The cache key is the file hash (path+size+mtime, same scheme as the
    transcript cache).  Pass force=True to rebuild.

    This is intentionally synchronous and CPU-bound — it runs after
    transcription as part of the ingest pipeline.  For a 5-minute video with
    ~400 words, analysis takes 1–3 seconds on M-series Macs.
    """
    import librosa
    import numpy as np

    file_path = Path(file_path)
    file_hash = get_file_hash(file_path)
    if not file_hash:
        raise FileNotFoundError(f"Cannot hash {file_path}")

    if not force:
        cached = load_acoustic_map(file_path)
        if cached is not None:
            logger.info(f"Using cached acoustic map for {file_path.name}")
            return cached

    def _p(pct: int, status: str):
        if progress_cb:
            progress_cb(pct, status)

    _p(5, "Extracting audio for analysis...")

    # We want sample-accurate PCM for analysis — extract to WAV once.
    # (The export path does this too, but they don't share the file because
    # analysis runs at ingest, long before export.)
    video_ext = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    if file_path.suffix.lower() in video_ext:
        from utils.audio_processing import extract_audio
        wav_path = extract_audio(file_path)
    else:
        wav_path = file_path

    _p(15, "Analyzing audio...")
    y, sr = librosa.load(str(wav_path), sr=None, mono=True)
    duration = len(y) / sr

    # Global noise floors
    hop_length = 256
    frame_length = 1024
    rms_full = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_full_t = librosa.frames_to_time(np.arange(len(rms_full)), sr=sr, hop_length=hop_length)

    noise_floor_rms = float(np.percentile(rms_full, 10)) + 1e-8
    speech_threshold = noise_floor_rms * 6.0

    _p(40, "Computing fricative band...")
    band_rms = _fricative_band_rms(y, sr, frame_length=frame_length, hop_length=hop_length)
    band_rms_t = librosa.frames_to_time(np.arange(len(band_rms)), sr=sr, hop_length=hop_length)
    fricative_noise_floor = float(np.percentile(band_rms, 10)) + 1e-10

    _p(55, "Fingerprinting words...")
    fingerprints: list[WordFingerprint] = []
    N = len(words)
    for i, w in enumerate(words):
        w_with_idx = {**w, "_index": i}
        prev_we = float(words[i - 1]["end"]) if i > 0 else 0.0
        next_ws = float(words[i + 1]["start"]) if i < N - 1 else duration
        fp = _analyze_word(
            y, sr, w_with_idx,
            next_ws=next_ws, prev_we=prev_we,
            speech_threshold=speech_threshold,
            fric_threshold_floor=fricative_noise_floor,
            rms_full=rms_full, rms_full_t=rms_full_t,
            band_rms=band_rms, band_rms_t=band_rms_t,
            hop_length=hop_length,
            is_last=(i == N - 1),
        )
        fingerprints.append(fp)
        if N > 0 and i % max(1, N // 10) == 0:
            _p(55 + int(i / max(N, 1) * 40), "Fingerprinting words...")

    m = AcousticMap(
        version=_MAP_VERSION,
        file_hash=file_hash,
        duration=round(duration, 4),
        sr=int(sr),
        noise_floor_rms=round(noise_floor_rms, 6),
        speech_threshold=round(speech_threshold, 6),
        fricative_noise_floor=round(fricative_noise_floor, 8),
        words=fingerprints,
    )

    path = _save_acoustic_map(m)
    logger.info(f"AcousticMap saved: {path} ({N} words, {duration:.1f}s)")
    _p(100, "Audio analysis complete")

    # Cleanup the temp WAV if we extracted it
    try:
        if wav_path != file_path and os.path.exists(wav_path):
            os.unlink(wav_path)
    except OSError:
        pass

    return m


def get_spectral_cache_size() -> tuple[int, int]:
    """Return (total_bytes, file_count) for the spectral cache directory."""
    if not SPECTRAL_CACHE_DIR.exists():
        return 0, 0
    total = 0
    n = 0
    for p in SPECTRAL_CACHE_DIR.glob("*.json"):
        try:
            total += p.stat().st_size
            n += 1
        except OSError:
            pass
    return total, n


def clear_spectral_cache() -> int:
    """Remove all spectral cache files. Returns number of files deleted."""
    if not SPECTRAL_CACHE_DIR.exists():
        return 0
    n = 0
    for p in SPECTRAL_CACHE_DIR.glob("*.json"):
        try:
            p.unlink()
            n += 1
        except OSError as e:
            logger.warning(f"Failed to delete {p}: {e}")
    return n
