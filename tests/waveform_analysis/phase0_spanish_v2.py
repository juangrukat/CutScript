"""
Phase 0 v2: Correct analysis — the kept word is 'Spanish?' (word 46, 28.739-29.120),
not 'Spanish.' (word 41).  The previous kept/deleted segments around it are:

  KEPT:    word 31-33 'Where's she from?'  (21.341-22.544)
  DELETED: 34-45 'I'm not sure, but I think she's Spanish. From Madrid, I think.'
  KEPT:    46 'Spanish?'                    (28.739-29.120)
  DELETED: 47-53 'Wow, that's awesome. I know. Hi, YouTube.'
  KEPT:    54-55 'I'm Daniela.'              (34.245-35.206)

So the segment containing Spanish? is the single-word segment 28.739-29.120,
with prev_end=22.544 and next_start=34.245.

Goal:
  1. Confirm _find_word_end Phase 2 truncates the /ʃ/ fricative
  2. Measure how far the fricative actually extends in the original
  3. Measure what the cut preserves vs the true fricative end
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORIG_MP4 = Path("/Users/kat/Desktop/b.mp4")
CUT_MP4  = Path("/Users/kat/Desktop/b_edited2.mp4")
ORIG_JSON = Path("/Users/kat/.obs_transcriber_cache/c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json")
OUT_DIR = Path(__file__).parent


def extract_wav(mp4: Path) -> Path:
    wav = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", "44100",
         "-sample_fmt", "s16", str(wav)],
        check=True, capture_output=True,
    )
    return wav


orig_wav = extract_wav(ORIG_MP4)
cut_wav  = extract_wav(CUT_MP4)
y_orig, sr = librosa.load(str(orig_wav), sr=None, mono=True)
y_cut, sr_c = librosa.load(str(cut_wav), sr=None, mono=True)
orig_wav.unlink(); cut_wav.unlink()
assert sr == sr_c

# ── The kept word: Spanish? at 28.739-29.120 ────────────────────────────────
kept_start = 28.739
kept_end   = 29.120
prev_keep_end   = 22.544   # end of 'from?'
next_keep_start = 34.245   # start of 'I'm'

print(f"Kept segment (Spanish?): {kept_start:.3f}-{kept_end:.3f} (duration {kept_end-kept_start:.3f}s)")
print(f"prev_end={prev_keep_end:.3f}, next_start={next_keep_start:.3f}")
print(f"Gap before (deleted speech): {kept_start-prev_keep_end:.3f}s")
print(f"Gap after  (deleted speech): {next_keep_start-kept_end:.3f}s")

# ── Measure actual fricative end in original ────────────────────────────────
# Scan 150ms at a time past WhisperX endpoint to find where high-frequency
# (fricative) energy actually decays to noise floor.

# High-pass energy: approximate fricative-band energy via spectral rolloff +
# RMS in a 2-8kHz band
print("\n=== Spectral analysis past WhisperX end ===")
win_start = kept_end - 0.05
win_end   = kept_end + 0.50
s = int(win_start * sr); e = int(win_end * sr)
y_win = y_orig[s:e]

# STFT for high-freq energy
n_fft = 1024
hop = 128
S = np.abs(librosa.stft(y_win, n_fft=n_fft, hop_length=hop))
freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
# Band: 2kHz-8kHz (fricative range)
band_mask = (freqs >= 2000) & (freqs <= 8000)
band_energy = np.sqrt(np.mean(S[band_mask, :] ** 2, axis=0))
times = librosa.frames_to_time(np.arange(len(band_energy)), sr=sr, hop_length=hop) + win_start

# Full RMS for comparison
rms_full = librosa.feature.rms(y=y_win, frame_length=256, hop_length=hop)[0]

print(f"  time         full_rms     2-8kHz band")
for i in range(0, len(times), max(1, len(times)//40)):
    marker = " <-- WhisperX end" if abs(times[i] - kept_end) < 0.005 else ""
    print(f"  {times[i]:.4f}    {rms_full[i]:.5f}      {band_energy[i]:.5f}{marker}")

# Find where band energy drops below some threshold (e.g., noise floor)
# Look at 100-200ms before the word (silence before) for baseline
sil_s = int((kept_start - 0.3) * sr)
sil_e = int((kept_start - 0.1) * sr)
S_sil = np.abs(librosa.stft(y_orig[sil_s:sil_e], n_fft=n_fft, hop_length=hop))
sil_band = np.sqrt(np.mean(S_sil[band_mask, :] ** 2, axis=0))
fricative_noise_floor = float(np.percentile(sil_band, 75))
print(f"\n  Fricative-band noise floor (silence before word, 75th pct): {fricative_noise_floor:.5f}")

# Find first frame where band_energy drops below 2× noise floor for 3 consecutive frames
fric_end_t = None
thr = 2.0 * fricative_noise_floor
for i in range(len(band_energy) - 2):
    if (band_energy[i] < thr and band_energy[i+1] < thr and band_energy[i+2] < thr
        and times[i] > kept_end - 0.02):
        fric_end_t = float(times[i])
        break
print(f"  Fricative decays to 2× noise floor at t={fric_end_t:.3f}s "
      f"(+{(fric_end_t-kept_end)*1000:.0f}ms past WhisperX end)" if fric_end_t else
      "  Fricative did not decay within 500ms window")

# ── Simulate _find_word_end for Spanish? with correct params ────────────────
print("\n=== _find_word_end simulation (Spanish?) ===")
global_rms = librosa.feature.rms(y=y_orig, frame_length=1024, hop_length=512)[0]
noise_floor = float(np.percentile(global_rms, 10)) + 1e-8
speech_threshold = noise_floor * 6.0
print(f"  global speech_threshold = {speech_threshold:.5f}")

# _gap_has_speech for end gap
t0, t1 = kept_end, next_keep_start
margin = max(0.04, (t1 - t0) * 0.10)
mid_s, mid_e = t0 + margin, t1 - margin
rms_gap = librosa.feature.rms(
    y=y_orig[int(mid_s*sr):int(mid_e*sr)], frame_length=512, hop_length=128)[0]
gap_speech_frac = float(np.mean(rms_gap > speech_threshold))
print(f"  _gap_has_speech(end gap): {gap_speech_frac*100:.1f}% above thr → "
      f"{'TRUE (speech-guided branch)' if gap_speech_frac > 0.15 else 'FALSE (silence-guided)'}")

# _find_word_end
seg_end = kept_end
cap = min(len(y_orig)/sr, min(next_keep_start - 0.005, seg_end + 0.150))
hop2 = 64
s_ref = max(0, int((seg_end - 1.0)*sr))
e_ref = int(seg_end * sr)
y_ref = y_orig[s_ref:e_ref]
rms_ref = librosa.feature.rms(y=y_ref, frame_length=256, hop_length=hop2)[0]
peak_pwr = float(np.max(rms_ref**2))
thr_pwr = peak_pwr * (10**(-25.0/10.0))
print(f"  word_peak_power={peak_pwr:.2e}, thr_pwr={thr_pwr:.2e} "
      f"(thr_rms={np.sqrt(thr_pwr):.5f})")

y_fwd = y_orig[int(seg_end*sr):int(cap*sr)]
rms_fwd = librosa.feature.rms(y=y_fwd, frame_length=256, hop_length=hop2)[0]
rms_t_fwd = librosa.frames_to_time(np.arange(len(rms_fwd)), sr=sr, hop_length=hop2) + seg_end
below = rms_fwd**2 < thr_pwr
print(f"  Phase 1: {np.sum(below)}/{len(below)} frames below threshold in 150ms window")
decay_found = None
for i in range(len(below)-2):
    if below[i] and below[i+1] and below[i+2]:
        decay_found = float(rms_t_fwd[i])
        break
if decay_found:
    print(f"  Phase 1: decay at {decay_found:.4f}s (+{(decay_found-seg_end)*1000:.1f}ms)")
else:
    print("  Phase 1: NO DECAY → falls through to Phase 2")

    # Phase 2
    env = librosa.onset.onset_strength(y=y_fwd, sr=sr)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr,
        pre_max=2, post_max=2, pre_avg=2, post_avg=5, delta=0.03, wait=1)
    if len(frames) > 0:
        onsets = librosa.frames_to_time(frames, sr=sr) + seg_end
        print(f"  Phase 2 onsets detected at: {onsets[:5]}")
        first_onset = float(onsets[0])
        phase2_result = min(cap, first_onset - 0.020)
        print(f"  Phase 2 result: {phase2_result:.4f}s (+{(phase2_result-seg_end)*1000:.1f}ms)")
        print(f"  >>> This is where the cut truncates Spanish?")
        if fric_end_t and fric_end_t > phase2_result:
            lost = fric_end_t - phase2_result
            print(f"  >>> LOST: {lost*1000:.0f}ms of fricative tail "
                  f"(truncated at {phase2_result:.3f} but fricative extends to {fric_end_t:.3f})")

# ── Plot: Original Spanish? with fricative extent, and cut rendering ────────
print("\nPlotting…")
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=False)
fig.suptitle("Phase 0 v2: 'Spanish?' fricative truncation analysis", fontsize=12)

# Top: Original waveform around Spanish?
pad = 0.6
t0o = kept_start - pad
t1o = kept_end + pad
y_o = y_orig[int(t0o*sr):int(t1o*sr)]
tx_o = np.linspace(t0o, t1o, len(y_o))
axes[0].plot(tx_o, y_o, color="steelblue", lw=0.4)
axes[0].axvline(kept_start, color="green", ls="--", label=f"WhisperX start {kept_start:.3f}")
axes[0].axvline(kept_end, color="red", ls="--", label=f"WhisperX end {kept_end:.3f}")
if decay_found is None and len(frames) > 0:
    axes[0].axvline(phase2_result, color="purple", ls="--",
                    label=f"Phase 2 cut {phase2_result:.3f}")
if fric_end_t:
    axes[0].axvline(fric_end_t, color="orange", ls="--",
                    label=f"Actual /ʃ/ end {fric_end_t:.3f}")
axes[0].set_title("ORIGINAL — Spanish? waveform (±0.6s around WhisperX bounds)")
axes[0].legend(fontsize=8)

# Middle: Full-band RMS vs fricative-band energy (original)
t_rms_win_s = int((kept_end - 0.1) * sr)
t_rms_win_e = int((kept_end + 0.5) * sr)
y_rw = y_orig[t_rms_win_s:t_rms_win_e]
t_full = np.linspace(kept_end - 0.1, kept_end + 0.5, len(y_rw))
rms_full_w = librosa.feature.rms(y=y_rw, frame_length=256, hop_length=64)[0]
rms_full_t = librosa.frames_to_time(np.arange(len(rms_full_w)), sr=sr, hop_length=64) + (kept_end - 0.1)
S_w = np.abs(librosa.stft(y_rw, n_fft=n_fft, hop_length=64))
band_e_w = np.sqrt(np.mean(S_w[band_mask, :] ** 2, axis=0))
band_t_w = librosa.frames_to_time(np.arange(len(band_e_w)), sr=sr, hop_length=64) + (kept_end - 0.1)

axes[1].plot(rms_full_t, rms_full_w, color="steelblue", lw=1.2, label="Full-band RMS")
axes[1].plot(band_t_w, band_e_w, color="crimson", lw=1.2, label="2-8kHz band energy")
axes[1].axhline(fricative_noise_floor, color="gray", ls=":", label=f"noise floor ({fricative_noise_floor:.4f})")
axes[1].axhline(2*fricative_noise_floor, color="orange", ls=":", label="2× noise floor")
axes[1].axvline(kept_end, color="red", ls="--", label="WhisperX end")
if decay_found is None and len(frames) > 0:
    axes[1].axvline(phase2_result, color="purple", ls="--", label=f"Phase 2 cut {phase2_result:.3f}")
if fric_end_t:
    axes[1].axvline(fric_end_t, color="orange", ls="--")
axes[1].set_title("ORIGINAL — energy around word end (red = fricative band)")
axes[1].set_ylabel("Energy")
axes[1].legend(fontsize=8, loc="upper right")

# Bottom: Mel spectrogram showing the sh/ʃ fricative pattern
mel = librosa.feature.melspectrogram(y=y_rw, sr=sr, n_mels=80, fmax=8000)
mel_db = librosa.power_to_db(mel, ref=np.max)
im = axes[2].imshow(mel_db, origin="lower", aspect="auto",
                    extent=[kept_end - 0.1, kept_end + 0.5, 0, 8000], cmap="magma")
axes[2].axvline(kept_end, color="red", ls="--", lw=2)
if decay_found is None and len(frames) > 0:
    axes[2].axvline(phase2_result, color="magenta", ls="--", lw=2)
if fric_end_t:
    axes[2].axvline(fric_end_t, color="cyan", ls="--", lw=2)
axes[2].set_title("ORIGINAL — Mel spectrogram (red=WhisperX end, magenta=Phase 2 cut, cyan=actual /ʃ/ end)")
axes[2].set_ylabel("Hz")
axes[2].set_xlabel("Original time (s)")

plt.tight_layout()
out = OUT_DIR / "phase0_v2_spanish_question.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")
