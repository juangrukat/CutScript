"""
Phase 0 diagnosis: why is 'Spanish' audibly cut in b_edited2.mp4?

Goals:
  1. Determine which original 'Spanish' was kept (25.431 or 28.739).
  2. Compare the preserved audio in the cut to the original source region
     sample-by-sample — is the fricative tail actually truncated?
  3. Simulate _refine_segments on the plausible keep segment to see which
     helper chose the (wrong) endpoint.
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
CUT_JSON  = Path("/Users/kat/.obs_transcriber_cache/0f64a266702f1e5374ea062ef8f1cef9_large-v3_transcribe_wx_e4f7217e.json")
OUT_DIR = Path(__file__).parent


def extract_wav(mp4: Path) -> Path:
    wav = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", "44100",
         "-sample_fmt", "s16", str(wav)],
        check=True, capture_output=True,
    )
    return wav


def load_words(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
    return [w for seg in data["data"]["segments"] for w in seg.get("words", [])]


# ── Load everything ─────────────────────────────────────────────────────────
orig_words = load_words(ORIG_JSON)
cut_words  = load_words(CUT_JSON)

orig_wav = extract_wav(ORIG_MP4)
cut_wav  = extract_wav(CUT_MP4)
y_orig, sr = librosa.load(str(orig_wav), sr=None, mono=True)
y_cut,  sr_c = librosa.load(str(cut_wav), sr=None, mono=True)
orig_wav.unlink()
cut_wav.unlink()
assert sr == sr_c, f"sr mismatch: orig {sr} vs cut {sr_c}"
print(f"Loaded  original: {len(y_orig)/sr:.3f}s   cut: {len(y_cut)/sr:.3f}s   @ {sr}Hz")

# ── Identify which Spanish was kept via cross-correlation ──────────────────
cut_sp = [w for w in cut_words if "spanish" in w["word"].lower()][0]
orig_sps = [w for w in orig_words if "spanish" in w["word"].lower()]

# Take 300ms window centered at the cut spanish's midpoint
cut_mid = (cut_sp["start"] + cut_sp["end"]) / 2
cut_window = y_cut[int((cut_mid - 0.15) * sr): int((cut_mid + 0.15) * sr)]

best_score = -1.0
best_orig = None
for osp in orig_sps:
    orig_mid = (osp["start"] + osp["end"]) / 2
    orig_window = y_orig[int((orig_mid - 0.15) * sr): int((orig_mid + 0.15) * sr)]
    # Normalize for cross-correlation
    a = (cut_window - cut_window.mean()) / (cut_window.std() + 1e-9)
    b = (orig_window - orig_window.mean()) / (orig_window.std() + 1e-9)
    L = min(len(a), len(b))
    score = float(np.correlate(a[:L], b[:L], mode="valid")[0]) / L
    print(f"  Match vs original '{osp['word']}' at {osp['start']:.3f}: score={score:.3f}")
    if score > best_score:
        best_score = score
        best_orig = osp

print(f"\n>>> Cut Spanish best matches original '{best_orig['word']}' at {best_orig['start']:.3f}-{best_orig['end']:.3f}")

# ── Find the cut-to-original time offset via the match ─────────────────────
# Sweep: look for the exact offset that maximizes correlation over a 1s window
orig_mid = (best_orig["start"] + best_orig["end"]) / 2
search_lo = orig_mid - 0.6
search_hi = orig_mid + 0.6
orig_search = y_orig[int(search_lo * sr): int(search_hi * sr)]
cut_sp_audio = y_cut[int((cut_sp["start"] - 0.1) * sr): int((cut_sp["end"] + 0.4) * sr)]
# Normalize
o = (orig_search - orig_search.mean()) / (orig_search.std() + 1e-9)
c = (cut_sp_audio - cut_sp_audio.mean()) / (cut_sp_audio.std() + 1e-9)
if len(o) >= len(c):
    xcorr = np.correlate(o, c, mode="valid")
    best_k = int(np.argmax(xcorr))
    orig_start_of_cut_window = search_lo + best_k / sr
    # So: cut time (cut_sp["start"] - 0.1) corresponds to orig_start_of_cut_window
    offset = orig_start_of_cut_window - (cut_sp["start"] - 0.1)
    print(f">>> Cut → original time offset: {offset:.4f}s "
          f"(cut t=0 corresponds to orig t={offset:.4f})")
    print(f">>> Best xcorr value: {xcorr[best_k]:.1f}")

    # Fraction of match quality
    print(f">>> Cut Spanish at cut-time {cut_sp['start']:.3f}-{cut_sp['end']:.3f} "
          f"maps to original {cut_sp['start']+offset:.3f}-{cut_sp['end']+offset:.3f}")
    print(f">>> Original WhisperX Spanish: {best_orig['start']:.3f}-{best_orig['end']:.3f}")
else:
    print("!!! search window too small for xcorr")

# ── Examine what's after cut Spanish in both audios ─────────────────────────
print("\n--- Audio after cut Spanish end ---")

def rms(y, sr, hop=64, frame=256):
    r = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=hop)
    return t, r

# In cut: look 100ms before Spanish end to 200ms after
cut_end_t = cut_sp["end"]
cut_audio_win = y_cut[int((cut_end_t - 0.1) * sr): int((cut_end_t + 0.2) * sr)]
t_cw, r_cw = rms(cut_audio_win, sr)
t_cw += (cut_end_t - 0.1)
print(f"  CUT around Spanish end ({cut_end_t:.3f}s):")
for i in range(0, len(t_cw), max(1, len(t_cw) // 20)):
    marker = ""
    if abs(t_cw[i] - cut_end_t) < 0.005: marker = " <- cut Spanish end"
    print(f"    t={t_cw[i]:.4f}   rms={r_cw[i]:.5f}{marker}")

# In original around best_orig end
orig_end_t = best_orig["end"]
orig_audio_win = y_orig[int((orig_end_t - 0.1) * sr): int((orig_end_t + 0.6) * sr)]
t_ow, r_ow = rms(orig_audio_win, sr)
t_ow += (orig_end_t - 0.1)
print(f"\n  ORIGINAL around '{best_orig['word']}' end ({orig_end_t:.3f}s):")
for i in range(0, len(t_ow), max(1, len(t_ow) // 25)):
    marker = ""
    if abs(t_ow[i] - orig_end_t) < 0.005: marker = " <- orig WhisperX end"
    print(f"    t={t_ow[i]:.4f}   rms={r_ow[i]:.5f}{marker}")

# ── Simulate _refine_segments on the inferred keep segment ──────────────────
print("\n--- Simulating _refine_segments for segment ending at Spanish ---")

# Determine next_start: user's cut shows "i" immediately after → user deleted content after spanish
# But what if Spanish. (word 41) was kept and "From" (42) was deleted?
# Then the keep segment's end would be 26.032, and the next keep segment would start somewhere later.
# For the simulation, we test with next_start = the WhisperX start of the first deleted word.

# We try both hypotheses: Spanish. with next_start = "From" at 26.574
# and Spanish? with next_start = "Wow" at 29.961
for hypoth in orig_sps:
    # find next word after this spanish
    idx = next(i for i, w in enumerate(orig_words) if w is hypoth)
    if idx + 1 >= len(orig_words):
        continue
    next_w = orig_words[idx + 1]
    print(f"\n  Hypothesis: kept '{hypoth['word']}' ({hypoth['start']:.3f}-{hypoth['end']:.3f}), "
          f"deleted '{next_w['word']}' at {next_w['start']:.3f}")

    # compute speech threshold as production does
    global_rms = librosa.feature.rms(y=y_orig, frame_length=1024, hop_length=512)[0]
    noise_floor = float(np.percentile(global_rms, 10)) + 1e-8
    speech_threshold = noise_floor * 6.0
    print(f"    speech_threshold={speech_threshold:.6f}")

    seg_end = hypoth["end"]
    next_start = next_w["start"]
    gap = next_start - seg_end
    print(f"    gap between seg_end and next_start: {gap*1000:.1f}ms")

    # _gap_has_speech check
    t0, t1 = seg_end, next_start
    margin = max(0.04, (t1 - t0) * 0.10)
    mid_s, mid_e = t0 + margin, t1 - margin
    if mid_e <= mid_s: mid_s, mid_e = t0, t1
    s = int(mid_s * sr); e = int(mid_e * sr)
    r_gap = librosa.feature.rms(y=y_orig[s:e], frame_length=512, hop_length=128)[0]
    gap_has_sp = float(np.mean(r_gap > speech_threshold)) > 0.15
    print(f"    gap has speech (deleted words): {gap_has_sp} "
          f"({np.mean(r_gap > speech_threshold)*100:.1f}% frames above thr)")

    # _find_word_end simulation
    cap = min(len(y_orig)/sr, min(next_start - 0.005, seg_end + 0.150))
    hop = 64
    s_ref = max(0, int((seg_end - 1.0)*sr))
    e_ref = int(seg_end * sr)
    s_fwd = int(seg_end * sr)
    e_fwd = int(cap * sr)
    if e_ref > s_ref + 256 and e_fwd > s_fwd + 256:
        y_ref = y_orig[s_ref:e_ref]
        y_fwd = y_orig[s_fwd:e_fwd]
        rms_ref = librosa.feature.rms(y=y_ref, frame_length=256, hop_length=hop)[0]
        peak_pwr = float(np.max(rms_ref**2))
        thr_pwr = peak_pwr * (10**(-25.0/10.0))
        rms_fwd = librosa.feature.rms(y=y_fwd, frame_length=256, hop_length=hop)[0]
        rms_t_fwd = librosa.frames_to_time(np.arange(len(rms_fwd)), sr=sr, hop_length=hop) + seg_end
        below = rms_fwd**2 < thr_pwr
        print(f"    _find_word_end: cap={cap:.3f}, word_peak_power={peak_pwr:.2e}, "
              f"thr_pwr={thr_pwr:.2e} (thr RMS={np.sqrt(thr_pwr):.5f})")
        print(f"    window size {cap-seg_end:.3f}s, {len(rms_fwd)} frames, "
              f"{np.sum(below)} below threshold")
        decay_found = None
        for i in range(len(below)-2):
            if below[i] and below[i+1] and below[i+2]:
                decay_t = float(rms_t_fwd[i])
                decay_found = decay_t
                print(f"    Phase 1 decay found at {decay_t:.4f}s (+{(decay_t-seg_end)*1000:.1f}ms)")
                break
        if decay_found is None:
            print(f"    Phase 1: no decay found in {cap-seg_end:.3f}s window")
            # Phase 2 tried: first onset after
            env = librosa.onset.onset_strength(y=y_fwd, sr=sr)
            frames = librosa.onset.onset_detect(
                onset_envelope=env, sr=sr,
                pre_max=2, post_max=2, pre_avg=2, post_avg=5, delta=0.03, wait=1)
            if len(frames) > 0:
                onsets = librosa.frames_to_time(frames, sr=sr) + seg_end
                print(f"    Phase 2 onsets at: {onsets[:3]}")
                print(f"    Phase 2 would cut at: {min(cap, float(onsets[0])-0.020):.4f}")

# ── Plot: cut vs original around Spanish ───────────────────────────────────
fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharey="row")
fig.suptitle("Phase 0: Spanish truncation diagnosis", fontsize=12)

# Cut
pad = 0.5
t_cs = max(0, cut_sp["start"] - pad)
t_ce = min(len(y_cut)/sr, cut_sp["end"] + pad)
y_c = y_cut[int(t_cs*sr):int(t_ce*sr)]
tx_c = np.linspace(t_cs, t_ce, len(y_c))
axes[0,0].plot(tx_c, y_c, color="darkorange", lw=0.4)
axes[0,0].axvline(cut_sp["start"], color="green", ls="--", label=f"WhisperX start {cut_sp['start']:.3f}")
axes[0,0].axvline(cut_sp["end"],   color="red",   ls="--", label=f"WhisperX end {cut_sp['end']:.3f}")
axes[0,0].set_title("CUT waveform around Spanish")
axes[0,0].legend(fontsize=7)

t_cr, r_cr = rms(y_c, sr)
t_cr += t_cs
axes[1,0].plot(t_cr, r_cr, color="darkorange", lw=1)
axes[1,0].axvline(cut_sp["start"], color="green", ls="--")
axes[1,0].axvline(cut_sp["end"],   color="red",   ls="--")
axes[1,0].set_title("CUT RMS")

# Mel
mel_c = librosa.feature.melspectrogram(y=y_c, sr=sr, n_mels=80, fmax=8000)
mel_c_db = librosa.power_to_db(mel_c, ref=np.max)
im1 = axes[2,0].imshow(mel_c_db, origin="lower", aspect="auto",
                       extent=[t_cs, t_ce, 0, 8000], cmap="magma")
axes[2,0].axvline(cut_sp["start"], color="lime", ls="--", lw=2)
axes[2,0].axvline(cut_sp["end"],   color="red",   ls="--", lw=2)
axes[2,0].set_title("CUT Mel spectrogram")
axes[2,0].set_ylabel("Hz")

# Original (best match)
t_os = max(0, best_orig["start"] - pad)
t_oe = min(len(y_orig)/sr, best_orig["end"] + pad)
y_o = y_orig[int(t_os*sr):int(t_oe*sr)]
tx_o = np.linspace(t_os, t_oe, len(y_o))
axes[0,1].plot(tx_o, y_o, color="steelblue", lw=0.4)
axes[0,1].axvline(best_orig["start"], color="green", ls="--", label=f"WhisperX start {best_orig['start']:.3f}")
axes[0,1].axvline(best_orig["end"],   color="red",   ls="--", label=f"WhisperX end {best_orig['end']:.3f}")
axes[0,1].set_title(f"ORIGINAL waveform around '{best_orig['word']}'")
axes[0,1].legend(fontsize=7)

t_or, r_or = rms(y_o, sr)
t_or += t_os
axes[1,1].plot(t_or, r_or, color="steelblue", lw=1)
axes[1,1].axvline(best_orig["start"], color="green", ls="--")
axes[1,1].axvline(best_orig["end"],   color="red",   ls="--")
axes[1,1].set_title("ORIGINAL RMS")

mel_o = librosa.feature.melspectrogram(y=y_o, sr=sr, n_mels=80, fmax=8000)
mel_o_db = librosa.power_to_db(mel_o, ref=np.max)
im2 = axes[2,1].imshow(mel_o_db, origin="lower", aspect="auto",
                       extent=[t_os, t_oe, 0, 8000], cmap="magma")
axes[2,1].axvline(best_orig["start"], color="lime", ls="--", lw=2)
axes[2,1].axvline(best_orig["end"],   color="red",   ls="--", lw=2)
axes[2,1].set_title(f"ORIGINAL Mel spectrogram")

plt.tight_layout()
out = OUT_DIR / "phase0_spanish_comparison.png"
plt.savefig(out, dpi=140, bbox_inches="tight")
print(f"\nSaved: {out}")
