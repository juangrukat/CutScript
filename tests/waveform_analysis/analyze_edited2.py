"""
Diagnose two problems in b_edited2.mp4:
  1. Hissing at start (~0-1.2s): likely over-extended "thanks" sss tail
  2. "Spanish" cut to ~0.204s: likely safety-clamp truncation

Generates PNG plots and prints key numeric data.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────────
ORIG_MP4   = Path("/Users/kat/Desktop/b.mp4")
CUT2_MP4   = Path("/Users/kat/Desktop/b_edited2.mp4")

ORIG_JSON  = Path("/Users/kat/.obs_transcriber_cache/c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json")
CUT2_JSON  = Path("/Users/kat/.obs_transcriber_cache/627e06f577194c6bc55e0e192329aa98_large-v3_transcribe_wx_e4f7217e.json")

OUT_DIR    = Path(__file__).parent


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


def find_word(words, text):
    text_lo = text.lower()
    return [w for w in words if text_lo in w["word"].lower()]


def rms_envelope(y, sr, hop=128, frame=512):
    r = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    t = librosa.frames_to_time(np.arange(len(r)), sr=sr, hop_length=hop)
    return t, r


# ── Load ────────────────────────────────────────────────────────────────────
print("Loading transcripts…")
orig_words = load_words(ORIG_JSON)
cut2_words = load_words(CUT2_JSON)

orig_thanks = find_word(orig_words, "thanks")
cut2_thanks = find_word(cut2_words,  "thanks")
orig_spanish = find_word(orig_words, "Spanish")
cut2_spanish = find_word(cut2_words,  "Spanish")

print(f"\nOriginal 'thanks':  {[(w['word'], round(w['start'],3), round(w['end'],3)) for w in orig_thanks]}")
print(f"Cut2    'thanks':   {[(w['word'], round(w['start'],3), round(w['end'],3)) for w in cut2_thanks]}")
print(f"\nOriginal 'Spanish': {[(w['word'], round(w['start'],3), round(w['end'],3)) for w in orig_spanish]}")
print(f"Cut2    'Spanish':  {[(w['word'], round(w['start'],3), round(w['end'],3)) for w in cut2_spanish]}")

print("\nExtracting audio…")
orig_wav = extract_wav(ORIG_MP4)
cut2_wav = extract_wav(CUT2_MP4)

print("Loading audio…")
y_orig, sr_orig = librosa.load(str(orig_wav), sr=None, mono=True)
y_cut2, sr_cut2 = librosa.load(str(cut2_wav), sr=None, mono=True)
orig_wav.unlink()
cut2_wav.unlink()

print(f"Original: {len(y_orig)/sr_orig:.2f}s @ {sr_orig}Hz")
print(f"Cut2:     {len(y_cut2)/sr_cut2:.2f}s  @ {sr_cut2}Hz")

# ── Compute global RMS stats for cut2 ───────────────────────────────────────
print("\n--- Cut2 global stats ---")
global_rms = librosa.feature.rms(y=y_cut2, frame_length=1024, hop_length=512)[0]
noise_floor = float(np.percentile(global_rms, 10)) + 1e-8
speech_threshold = noise_floor * 6.0
print(f"  Noise floor (10th pct): {noise_floor:.6f}")
print(f"  Speech threshold (6x):  {speech_threshold:.6f}")

# ── PROBLEM 1: Hissing at start ──────────────────────────────────────────────
print("\n--- Problem 1: Hissing at start of cut2 ---")
# Check first 2s of cut2
first2s = y_cut2[:int(2.0 * sr_cut2)]
t_rms, r_rms = rms_envelope(first2s, sr_cut2, hop=128, frame=512)
above_thresh = r_rms > speech_threshold
print(f"  First 2s: {np.mean(above_thresh)*100:.1f}% of frames above speech_threshold")
print(f"  Max RMS in first 2s: {r_rms.max():.5f}")
print(f"  First frame above threshold: {t_rms[np.where(above_thresh)[0][0] if np.any(above_thresh) else -1]:.3f}s")
print(f"  Last frame above threshold in first 2s: {t_rms[np.where(above_thresh)[0][-1] if np.any(above_thresh) else -1]:.3f}s")

# Find where the hissing ends
t_hiss_end = None
for i in range(len(r_rms)-3, 0, -1):
    if r_rms[i] > speech_threshold:
        t_hiss_end = float(t_rms[i])
        break

# Also check what's in original around "thanks" end
print("\n  Original 'thanks' instances:")
for tw in orig_thanks:
    # Check 200ms window after thanks end
    tw_end = tw["end"]
    s = int(tw_end * sr_orig)
    e = min(len(y_orig), int((tw_end + 0.250) * sr_orig))
    y_after = y_orig[s:e]
    t_a, r_a = rms_envelope(y_after, sr_orig, hop=128, frame=256)
    print(f"    {tw['word']!r} {tw['start']:.3f}-{tw['end']:.3f}:  "
          f"max RMS in 250ms after = {r_a.max():.5f}  "
          f"frames above speech_thr = {np.mean(r_a > speech_threshold)*100:.1f}%")


# ── PROBLEM 2: Spanish cut in half ──────────────────────────────────────────
print("\n--- Problem 2: Spanish in cut2 ---")
print(f"  Cut2 full duration: {len(y_cut2)/sr_cut2:.3f}s")
print(f"  Cut2 Spanish words: {cut2_spanish}")

# Check the words AROUND spanish in cut2
if cut2_spanish:
    sp2 = cut2_spanish[0]
    sp2_idx = next(i for i, w in enumerate(cut2_words) if abs(w["start"] - sp2["start"]) < 0.01)
    prev_w2 = cut2_words[sp2_idx - 1] if sp2_idx > 0 else None
    next_w2 = cut2_words[sp2_idx + 1] if sp2_idx < len(cut2_words) - 1 else None
    print(f"  Word before Spanish in cut2: {prev_w2}")
    print(f"  Spanish in cut2:             {sp2}")
    print(f"  Word after  Spanish in cut2: {next_w2}")

    # How much audio exists at the "Spanish" timestamp?
    sp_s = int(sp2["start"] * sr_cut2)
    sp_e = min(len(y_cut2), int((sp2["start"] + 0.700) * sr_cut2))
    y_sp2_win = y_cut2[sp_s:sp_e]
    t_sp, r_sp = rms_envelope(y_sp2_win, sr_cut2, hop=128, frame=256)
    peak_sp_frame = int(np.argmax(r_sp))
    print(f"  Peak RMS at {float(t_sp[peak_sp_frame]) + sp2['start']:.3f}s: {r_sp[peak_sp_frame]:.5f}")
    print(f"  RMS at cut2 Spanish end ({sp2['end']:.3f}s): "
          f"{r_sp[min(len(r_sp)-1, int((sp2['end']-sp2['start'])*sr_cut2/128))]:.5f}")

    # Is there audio after sp2["end"] in the cut?
    after_s = int(sp2["end"] * sr_cut2)
    after_e = min(len(y_cut2), after_s + int(0.300 * sr_cut2))
    y_after_sp = y_cut2[after_s:after_e]
    t_after, r_after = rms_envelope(y_after_sp, sr_cut2, hop=128, frame=256)
    print(f"  Max RMS in 300ms after cut2-Spanish-end: {r_after.max():.5f}")

    # Compare with original Spanish
    print(f"\n  Original Spanish entries: {orig_spanish}")
    for sp_o in orig_spanish:
        print(f"    Duration: {sp_o['end']-sp_o['start']:.3f}s  ({sp_o['start']:.3f}-{sp_o['end']:.3f})")

    # Cut2 Spanish duration
    print(f"  Cut2 Spanish duration: {sp2['end'] - sp2['start']:.3f}s")


# ── FIGURES ─────────────────────────────────────────────────────────────────

# Figure 1: Hissing zone — compare cut2 first 2s vs original "thanks" regions
print("\nGenerating Figure 1: hissing zone…")
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=False)
fig.suptitle("Problem 1: Hissing at start of b_edited2.mp4", fontsize=12)

# Row 0: Cut2 first 2s waveform
t0 = np.linspace(0, min(2.0, len(y_cut2)/sr_cut2),
                 min(int(2.0*sr_cut2), len(y_cut2)))
axes[0].plot(t0, y_cut2[:len(t0)], color="darkorange", lw=0.4, alpha=0.8)
axes[0].axhline(0, color="black", lw=0.3)
axes[0].set_ylabel("Amplitude")
axes[0].set_title("Cut2 — first 2s waveform")
axes[0].set_xlim(0, 2.0)

# Row 1: Cut2 first 2s RMS + threshold
t_rms, r_rms = rms_envelope(y_cut2[:int(2.0*sr_cut2)], sr_cut2)
axes[1].plot(t_rms, r_rms, color="darkorange", lw=1.2)
axes[1].axhline(speech_threshold, color="red", lw=1, linestyle="--", label=f"speech_thr={speech_threshold:.5f}")
axes[1].fill_between(t_rms, 0, r_rms, where=(r_rms > speech_threshold), alpha=0.3, color="red", label="above thr")
axes[1].set_ylabel("RMS")
axes[1].set_title("Cut2 — first 2s RMS energy (red = above speech threshold)")
axes[1].legend(fontsize=8)
axes[1].set_xlim(0, 2.0)

# Row 2: Original thanks (first occurrence) — 500ms after end
for tw in orig_thanks[:1]:
    pad = 0.1
    t_start = max(0, tw["start"] - pad)
    t_end = min(len(y_orig)/sr_orig, tw["end"] + 0.5)
    s = int(t_start * sr_orig)
    e = int(t_end * sr_orig)
    y_tw = y_orig[s:e]
    t_tw = np.linspace(t_start, t_end, len(y_tw))
    axes[2].plot(t_tw, y_tw, color="steelblue", lw=0.4, alpha=0.8)
    axes[2].axvline(tw["end"], color="red", lw=1.5, linestyle="--", label=f"WhisperX end {tw['end']:.3f}s")
    axes[2].set_title(f"Original — 'thanks' (first: {tw['start']:.3f}-{tw['end']:.3f}s) + 500ms after")
    axes[2].set_ylabel("Amplitude")
    axes[2].legend(fontsize=8)

# Row 3: Original thanks (first) RMS
for tw in orig_thanks[:1]:
    t_start = max(0, tw["start"] - 0.1)
    t_end = min(len(y_orig)/sr_orig, tw["end"] + 0.5)
    s = int(t_start * sr_orig)
    e = int(t_end * sr_orig)
    y_tw = y_orig[s:e]
    t_r, r_r = rms_envelope(y_tw, sr_orig, hop=128, frame=256)
    t_r += t_start
    axes[3].plot(t_r, r_r, color="steelblue", lw=1.2)
    axes[3].axvline(tw["end"], color="red", lw=1.5, linestyle="--", label=f"WhisperX end {tw['end']:.3f}s")
    axes[3].axhline(speech_threshold, color="red", lw=1, linestyle=":", label="speech_thr")
    # find where RMS drops below threshold after thanks end
    in_region = t_r > tw["end"]
    if np.any(in_region):
        r_after_end = r_r[in_region]
        t_after_end = t_r[in_region]
        below = r_after_end < speech_threshold
        if np.any(below):
            natural_end = float(t_after_end[np.where(below)[0][0]])
            axes[3].axvline(natural_end, color="purple", lw=1.5, linestyle="--",
                           label=f"1st below thr {natural_end:.3f}s")
    axes[3].set_title(f"Original — 'thanks' RMS (500ms window)")
    axes[3].set_ylabel("RMS")
    axes[3].set_xlabel("Time (s)")
    axes[3].legend(fontsize=8)

plt.tight_layout()
out1 = OUT_DIR / "edited2_problem1_hissing.png"
plt.savefig(out1, dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved: {out1}")


# Figure 2: Spanish comparison — cut2 vs original
print("\nGenerating Figure 2: Spanish comparison…")
if cut2_spanish and orig_spanish:
    sp2 = cut2_spanish[0]
    sp_o = orig_spanish[0]
    pad = 0.5

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(
        f"Problem 2: 'Spanish' — Original ({sp_o['start']:.3f}-{sp_o['end']:.3f}, "
        f"{sp_o['end']-sp_o['start']:.3f}s) vs "
        f"Cut2 ({sp2['start']:.3f}-{sp2['end']:.3f}, {sp2['end']-sp2['start']:.3f}s)",
        fontsize=11
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.3)

    for col, (y, sr, sp, label, color) in enumerate([
        (y_orig, sr_orig, sp_o, "ORIGINAL b.mp4", "steelblue"),
        (y_cut2, sr_cut2, sp2, "CUT2 b_edited2.mp4", "darkorange"),
    ]):
        t_start = max(0, sp["start"] - pad)
        t_end   = min(len(y)/sr, sp["end"] + pad)
        s = int(t_start * sr)
        e = int(t_end * sr)
        y_w = y[s:e]
        t_w = np.linspace(t_start, t_end, len(y_w))

        # Waveform
        ax_w = fig.add_subplot(gs[0, col])
        ax_w.plot(t_w, y_w, color=color, lw=0.4, alpha=0.8)
        ax_w.axvline(sp["start"], color="green", lw=1.5, linestyle="--", label="start")
        ax_w.axvline(sp["end"],   color="red",   lw=1.5, linestyle="--", label="end")
        ax_w.axvspan(sp["start"], sp["end"], alpha=0.12, color="yellow")
        ax_w.set_title(f"{label} — waveform")
        ax_w.set_ylabel("Amplitude")
        ax_w.legend(fontsize=7)

        # RMS
        ax_r = fig.add_subplot(gs[1, col])
        t_r, r_r = rms_envelope(y_w, sr, hop=64, frame=256)
        t_r += t_start
        ax_r.plot(t_r, r_r, color=color, lw=1)
        ax_r.axvline(sp["start"], color="green", lw=1.5, linestyle="--")
        ax_r.axvline(sp["end"],   color="red",   lw=1.5, linestyle="--")
        ax_r.axvspan(sp["start"], sp["end"], alpha=0.12, color="yellow")
        ax_r.axhline(speech_threshold, color="gray", lw=1, linestyle=":", label=f"speech_thr={speech_threshold:.5f}")

        # compute word-relative threshold (-25dB)
        s_ref = max(0, int((sp["start"] - 0.5) * sr))
        e_ref = int(sp["start"] * sr)
        if e_ref > s_ref + 256:
            rms_ref = librosa.feature.rms(y=y[s_ref:e_ref], frame_length=256, hop_length=64)[0]
            peak_pwr = float(np.max(rms_ref**2))
            if peak_pwr > 1e-10:
                thr_pwr = peak_pwr * (10**(-25.0/10.0))
                thr_rms = float(np.sqrt(thr_pwr))
                ax_r.axhline(thr_rms, color="purple", lw=1, linestyle="--", label=f"-25dB thr={thr_rms:.5f}")

        ax_r.set_title(f"{label} — RMS")
        ax_r.set_ylabel("RMS")
        ax_r.legend(fontsize=7)

        # Mel spectrogram
        ax_m = fig.add_subplot(gs[2, col])
        mel = librosa.feature.melspectrogram(y=y_w, sr=sr, n_mels=80, fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        img = librosa.display.specshow(mel_db, sr=sr, hop_length=512,
                                       x_axis="time", y_axis="mel", fmax=8000,
                                       ax=ax_m, cmap="magma")
        # Manually offset x ticks
        xticks = ax_m.get_xticks()
        ax_m.set_xticks(xticks)
        ax_m.set_xticklabels([f"{x + t_start:.2f}" for x in xticks])
        ax_m.axvline(sp["start"] - t_start, color="lime", lw=2, linestyle="--", label="start")
        ax_m.axvline(sp["end"]   - t_start, color="red",  lw=2, linestyle="--", label="end")
        ax_m.set_title(f"{label} — Mel spectrogram")
        ax_m.legend(fontsize=7)
        fig.colorbar(img, ax=ax_m, format="%+2.0f dB")

    out2 = OUT_DIR / "edited2_problem2_spanish.png"
    plt.savefig(out2, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")


# Figure 3: Context around cut2 Spanish — show 3s window to see neighboring segments
print("\nGenerating Figure 3: cut2 context around Spanish…")
if cut2_spanish:
    sp2 = cut2_spanish[0]
    ctx_pad = 1.2
    t_start = max(0, sp2["start"] - ctx_pad)
    t_end   = min(len(y_cut2)/sr_cut2, sp2["end"] + ctx_pad)
    s = int(t_start * sr_cut2)
    e = int(t_end * sr_cut2)
    y_ctx = y_cut2[s:e]
    t_ctx = np.linspace(t_start, t_end, len(y_ctx))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True)
    fig.suptitle(
        f"Cut2 context around 'Spanish' ({sp2['start']:.3f}-{sp2['end']:.3f}s, "
        f"{sp2['end']-sp2['start']:.3f}s)\n"
        f"Words in window: {[w['word'] for w in cut2_words if t_start - 0.1 < w['start'] < t_end + 0.1]}",
        fontsize=10,
    )

    ax1.plot(t_ctx, y_ctx, color="darkorange", lw=0.4, alpha=0.8)
    ax1.axvline(sp2["start"], color="green", lw=2, linestyle="--", label=f"Spanish start {sp2['start']:.3f}s")
    ax1.axvline(sp2["end"],   color="red",   lw=2, linestyle="--", label=f"Spanish end {sp2['end']:.3f}s")
    ax1.axvspan(sp2["start"], sp2["end"], alpha=0.15, color="yellow", label="Spanish span")

    # Mark all words in window
    for w in cut2_words:
        if t_start - 0.1 < w["start"] < t_end + 0.1 and "spanish" not in w["word"].lower():
            ax1.axvspan(w["start"], w["end"], alpha=0.1, color="cyan")
            ax1.text((w["start"]+w["end"])/2, ax1.get_ylim()[0] if ax1.get_ylim()[0] != ax1.get_ylim()[1] else -0.1,
                     w["word"], fontsize=6, ha="center", color="navy", rotation=90)

    ax1.set_ylabel("Amplitude")
    ax1.set_title("Waveform (yellow=Spanish, cyan=other words)")
    ax1.legend(fontsize=8)

    t_rms_ctx, r_rms_ctx = rms_envelope(y_ctx, sr_cut2, hop=128, frame=256)
    t_rms_ctx += t_start
    ax2.plot(t_rms_ctx, r_rms_ctx, color="darkorange", lw=1.2)
    ax2.axhline(speech_threshold, color="red", lw=1, linestyle="--", label=f"speech_thr={speech_threshold:.5f}")
    ax2.axvline(sp2["start"], color="green", lw=2, linestyle="--")
    ax2.axvline(sp2["end"],   color="red",   lw=2, linestyle="--")
    ax2.axvspan(sp2["start"], sp2["end"], alpha=0.15, color="yellow")
    ax2.set_ylabel("RMS")
    ax2.set_xlabel("Time (s)")
    ax2.set_title("RMS energy (red dash = speech threshold)")
    ax2.legend(fontsize=8)

    out3 = OUT_DIR / "edited2_problem2_context.png"
    plt.savefig(out3, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

# ── Detailed per-word boundary check for the segment before Spanish ─────────
print("\n--- Words near Spanish in cut2 ---")
if cut2_spanish:
    sp2 = cut2_spanish[0]
    sp2_idx = next(i for i, w in enumerate(cut2_words) if abs(w["start"] - sp2["start"]) < 0.01)
    window_words = cut2_words[max(0, sp2_idx-5):sp2_idx+3]
    for w in window_words:
        dur = w["end"] - w["start"]
        marker = " <-- SPANISH" if "spanish" in w["word"].lower() else ""
        print(f"  {w['word']!r:20s} {w['start']:.3f}-{w['end']:.3f}  ({dur:.3f}s){marker}")

print("\n--- Cut2 _find_word_end simulation for segment before Spanish ---")
if cut2_spanish:
    sp2 = cut2_spanish[0]
    sp2_idx = next(i for i, w in enumerate(cut2_words) if abs(w["start"] - sp2["start"]) < 0.01)
    if sp2_idx > 0:
        prev_w = cut2_words[sp2_idx - 1]
        print(f"  Previous word: {prev_w['word']!r} end={prev_w['end']:.3f}s")
        print(f"  Spanish start: {sp2['start']:.3f}s")
        print(f"  Gap: {sp2['start']-prev_w['end']:.3f}s")

        # Simulate what _find_word_end does for prev_w end
        t_wend = prev_w["end"]
        next_st = sp2["start"]
        cap = min(len(y_cut2)/sr_cut2, min(next_st - 0.005, t_wend + 0.150))
        hop = 64
        s_ref = max(0, int((t_wend - 1.0)*sr_cut2))
        e_ref = int(t_wend * sr_cut2)
        s_fwd = int(t_wend * sr_cut2)
        e_fwd = int(cap * sr_cut2)
        print(f"  _find_word_end search cap: {cap:.3f}s (window: {cap-t_wend:.3f}s)")
        if e_ref > s_ref + 256 and e_fwd > s_fwd + 256:
            y_ref = y_cut2[s_ref:e_ref]
            y_fwd = y_cut2[s_fwd:e_fwd]
            rms_ref = librosa.feature.rms(y=y_ref, frame_length=256, hop_length=hop)[0]
            peak_pwr = float(np.max(rms_ref**2))
            thr_pwr = peak_pwr * (10**(-25.0/10.0))
            rms_fwd = librosa.feature.rms(y=y_fwd, frame_length=256, hop_length=hop)[0]
            rms_t_fwd = librosa.frames_to_time(np.arange(len(rms_fwd)), sr=sr_cut2, hop_length=hop) + t_wend
            below = rms_fwd**2 < thr_pwr
            print(f"  word_peak_power={peak_pwr:.2e}, threshold_power={thr_pwr:.2e} "
                  f"(thr RMS={np.sqrt(thr_pwr):.5f})")
            print(f"  Frames below threshold: {np.sum(below)}/{len(below)} "
                  f"({np.mean(below)*100:.1f}%)")
            decay_found = None
            for i in range(len(below)-2):
                if below[i] and below[i+1] and below[i+2]:
                    decay_t = float(rms_t_fwd[i])
                    decay_found = decay_t
                    print(f"  Phase 1 fires at {decay_t:.3f}s (+{decay_t-t_wend:.3f}s from WhisperX end)")
                    break
            if decay_found is None:
                print("  Phase 1: no decay found (energy stays elevated)")

print("\nDone — open PNGs in tests/waveform_analysis/")
