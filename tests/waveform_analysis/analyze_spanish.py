"""
Waveform analysis for the 'Spanish' word cut quality issue.

Compares the original b.mp4 and cut b_edited.mp4 around the word 'Spanish'
using librosa onset detection, RMS energy, and zero-crossing analysis.

Saves PNG plots to the same directory as this script.
"""

import json
import subprocess
import tempfile
import os
from pathlib import Path

import numpy as np
import librosa
import librosa.display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── paths ──────────────────────────────────────────────────────────────────
ORIG_MP4  = Path("/Users/kat/Desktop/b.mp4")
CUT_MP4   = Path("/Users/kat/Desktop/b_edited.mp4")

ORIG_JSON = Path("/Users/kat/.obs_transcriber_cache/c0c2d225492c5da001c949aeab0034f5_large-v3_transcribe_wx_e4f7217e.json")
CUT_JSON  = Path("/Users/kat/.obs_transcriber_cache/05894515124e45360b6e4238dcd00e5b_large-v3_transcribe_wx_e4f7217e.json")

OUT_DIR   = Path(__file__).parent


# ── helpers ────────────────────────────────────────────────────────────────
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
    inner = data["data"]
    return [w for seg in inner["segments"] for w in seg.get("words", [])]


def find_word(words, text):
    text_lo = text.lower()
    return [w for w in words if text_lo in w["word"].lower()]


def onset_times(y, sr, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.02):
    """Return onset times in seconds using librosa onset detection."""
    env = librosa.onset.onset_strength(y=y, sr=sr)
    frames = librosa.onset.onset_detect(
        onset_envelope=env, sr=sr,
        pre_max=pre_max, post_max=post_max,
        pre_avg=pre_avg, post_avg=post_avg,
        delta=delta, wait=1,
    )
    return librosa.frames_to_time(frames, sr=sr)


def plot_word_region(ax_wave, ax_rms, ax_onset, y, sr, t_start, t_end,
                     label, color, whisper_start=None, whisper_end=None,
                     context_pad=0.4):
    """
    Plot waveform, RMS, and onset strength for a region around a word.
    t_start/t_end: the WhisperX word boundaries (or expected boundaries).
    context_pad: seconds of context either side.
    """
    s = max(0, int((t_start - context_pad) * sr))
    e = min(len(y), int((t_end + context_pad) * sr))
    y_clip = y[s:e]
    t_offset = s / sr
    times = np.linspace(t_offset, t_offset + len(y_clip) / sr, len(y_clip))

    # ── waveform
    ax_wave.plot(times, y_clip, color=color, lw=0.5, alpha=0.8)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(f"{label} — waveform")
    if whisper_start is not None:
        ax_wave.axvline(whisper_start, color="green",  lw=1.5, linestyle="--", label="WhisperX start")
    if whisper_end is not None:
        ax_wave.axvline(whisper_end,   color="red",    lw=1.5, linestyle="--", label="WhisperX end")
    ax_wave.axvspan(t_start, t_end, alpha=0.12, color="yellow", label="WhisperX word span")
    ax_wave.legend(fontsize=7, loc="upper right")

    # ── RMS energy
    hop = 128
    rms = librosa.feature.rms(y=y_clip, frame_length=512, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop) + t_offset
    ax_rms.plot(rms_times, rms, color=color, lw=1)
    ax_rms.set_ylabel("RMS energy")
    ax_rms.set_title("RMS energy")
    if whisper_start is not None:
        ax_rms.axvline(whisper_start, color="green",  lw=1.5, linestyle="--")
    if whisper_end is not None:
        ax_rms.axvline(whisper_end,   color="red",    lw=1.5, linestyle="--")
    ax_rms.axvspan(t_start, t_end, alpha=0.12, color="yellow")

    # ── onset strength
    env = librosa.onset.onset_strength(y=y_clip, sr=sr)
    env_times = librosa.frames_to_time(np.arange(len(env)), sr=sr) + t_offset
    onsets = onset_times(y_clip, sr)
    onsets_abs = onsets + t_offset

    ax_onset.plot(env_times, env, color=color, lw=1, label="onset strength")
    for ot in onsets_abs:
        ax_onset.axvline(ot, color="purple", lw=1, alpha=0.7)
    ax_onset.set_ylabel("Onset strength")
    ax_onset.set_title("Onset strength (purple = detected onsets)")
    ax_onset.set_xlabel("Time (s)")
    if whisper_start is not None:
        ax_onset.axvline(whisper_start, color="green",  lw=1.5, linestyle="--")
    if whisper_end is not None:
        ax_onset.axvline(whisper_end,   color="red",    lw=1.5, linestyle="--")
    ax_onset.axvspan(t_start, t_end, alpha=0.12, color="yellow")

    return onsets_abs


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("Loading transcriptions…")
    orig_words = load_words(ORIG_JSON)
    cut_words  = load_words(CUT_JSON)

    orig_spanish = find_word(orig_words, "Spanish")
    cut_spanish  = find_word(cut_words,  "Spanish")
    print(f"Original 'Spanish' entries: {orig_spanish}")
    print(f"Cut      'Spanish' entries: {cut_spanish}")

    print("Extracting audio from videos…")
    orig_wav = extract_wav(ORIG_MP4)
    cut_wav  = extract_wav(CUT_MP4)

    print("Loading audio…")
    y_orig, sr_orig = librosa.load(str(orig_wav), sr=None, mono=True)
    y_cut,  sr_cut  = librosa.load(str(cut_wav),  sr=None, mono=True)
    orig_wav.unlink()
    cut_wav.unlink()

    print(f"Original: {len(y_orig)/sr_orig:.2f}s @ {sr_orig}Hz")
    print(f"Cut:      {len(y_cut)/sr_cut:.2f}s  @ {sr_cut}Hz")

    # ── Figure 1: side-by-side for every 'Spanish' hit ─────────────────────
    for idx, sp_orig in enumerate(orig_spanish):
        fig = plt.figure(figsize=(18, 10))
        fig.suptitle(
            f"'Spanish' word analysis — occurrence {idx+1}\n"
            f"Original: {sp_orig['start']:.3f}s – {sp_orig['end']:.3f}s  |  "
            f"Cut: {cut_spanish[idx]['start']:.3f}s – {cut_spanish[idx]['end']:.3f}s"
            if idx < len(cut_spanish) else
            f"'Spanish' word analysis — occurrence {idx+1}\n"
            f"Original: {sp_orig['start']:.3f}s – {sp_orig['end']:.3f}s  (not found in cut)",
            fontsize=11,
        )
        gs = gridspec.GridSpec(3, 2, hspace=0.5, wspace=0.3)

        # left column = original
        ax_ow = fig.add_subplot(gs[0, 0])
        ax_or = fig.add_subplot(gs[1, 0], sharex=ax_ow)
        ax_oo = fig.add_subplot(gs[2, 0], sharex=ax_ow)

        onsets_orig = plot_word_region(
            ax_ow, ax_or, ax_oo,
            y_orig, sr_orig,
            t_start=sp_orig["start"], t_end=sp_orig["end"],
            label=f"ORIGINAL b.mp4",
            color="steelblue",
            whisper_start=sp_orig["start"], whisper_end=sp_orig["end"],
        )

        # right column = cut (if exists)
        if idx < len(cut_spanish):
            sp_cut = cut_spanish[idx]
            ax_cw = fig.add_subplot(gs[0, 1])
            ax_cr = fig.add_subplot(gs[1, 1], sharex=ax_cw)
            ax_co = fig.add_subplot(gs[2, 1], sharex=ax_cw)
            onsets_cut = plot_word_region(
                ax_cw, ax_cr, ax_co,
                y_cut, sr_cut,
                t_start=sp_cut["start"], t_end=sp_cut["end"],
                label=f"CUT b_edited.mp4",
                color="darkorange",
                whisper_start=sp_cut["start"], whisper_end=sp_cut["end"],
            )
            print(f"  Cut onsets near 'Spanish': {[f'{t:.3f}' for t in onsets_cut]}")

        print(f"  Orig onsets near 'Spanish': {[f'{t:.3f}' for t in onsets_orig]}")

        out = OUT_DIR / f"spanish_word_{idx+1}.png"
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    # ── Figure 2: full-context boundary view (original only) ───────────────
    # Show the 3s window around "Spanish" in the original to see deleted context
    # and how the boundary looks acoustically.
    if orig_spanish:
        sp = orig_spanish[0]
        # find the words immediately before and after in original
        all_w = orig_words
        sp_idx = next(i for i, w in enumerate(all_w) if abs(w["start"] - sp["start"]) < 0.01)
        prev_w = all_w[sp_idx - 1] if sp_idx > 0 else None
        next_w = all_w[sp_idx + 1] if sp_idx < len(all_w) - 1 else None

        fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
        fig.suptitle(
            "'Spanish' splice context — ORIGINAL b.mp4\n"
            f"Word before: {prev_w['word']!r} ({prev_w['start']:.3f}-{prev_w['end']:.3f})  "
            f"| Spanish: ({sp['start']:.3f}-{sp['end']:.3f})  "
            f"| Word after: {next_w['word']!r} ({next_w['start']:.3f}-{next_w['end']:.3f})",
            fontsize=10,
        )

        pad = 0.6
        t0 = max(0, sp["start"] - pad)
        t1 = min(len(y_orig) / sr_orig, sp["end"] + pad)
        s, e = int(t0 * sr_orig), int(t1 * sr_orig)
        y_win = y_orig[s:e]
        times = np.linspace(t0, t1, len(y_win))

        axes[0].plot(times, y_win, color="steelblue", lw=0.5)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_title("Waveform")

        hop = 128
        rms = librosa.feature.rms(y=y_win, frame_length=512, hop_length=hop)[0]
        rms_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr_orig, hop_length=hop) + t0
        axes[1].plot(rms_t, rms, color="steelblue", lw=1)
        axes[1].set_ylabel("RMS")
        axes[1].set_title("RMS energy")

        env = librosa.onset.onset_strength(y=y_win, sr=sr_orig)
        env_t = librosa.frames_to_time(np.arange(len(env)), sr=sr_orig) + t0
        onsets_win = onset_times(y_win, sr_orig) + t0
        axes[2].plot(env_t, env, color="steelblue", lw=1)
        for ot in onsets_win:
            axes[2].axvline(ot, color="purple", lw=1, alpha=0.7)
        axes[2].set_ylabel("Onset strength")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_title("Onset strength (purple = detected onsets)")

        for ax in axes:
            if prev_w:
                ax.axvspan(prev_w["start"], prev_w["end"], alpha=0.2, color="red",   label=f"DELETED: {prev_w['word']}")
            ax.axvspan(sp["start"],     sp["end"],     alpha=0.2, color="green", label="KEPT: Spanish")
            if next_w:
                ax.axvspan(next_w["start"], next_w["end"], alpha=0.2, color="cyan",  label=f"KEPT: {next_w['word']}")
            ax.axvline(sp["start"], color="green", lw=2, linestyle="--")

        axes[0].legend(fontsize=7, loc="upper right")

        out2 = OUT_DIR / "spanish_splice_context_original.png"
        plt.savefig(out2, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out2}")

    # ── Figure 3: spectrogram comparison ────────────────────────────────────
    if orig_spanish and cut_spanish:
        sp_o = orig_spanish[0]
        sp_c = cut_spanish[0]
        pad = 0.3

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle("Mel spectrogram — 'Spanish' word  (left=original, right=cut)", fontsize=11)

        for ax, y, sr, sp, title, color in [
            (ax1, y_orig, sr_orig, sp_o, "ORIGINAL", "Blues"),
            (ax2, y_cut,  sr_cut,  sp_c, "CUT",       "Oranges"),
        ]:
            s = max(0, int((sp["start"] - pad) * sr))
            e = min(len(y), int((sp["end"] + pad) * sr))
            y_clip = y[s:e]
            mel = librosa.feature.melspectrogram(y=y_clip, sr=sr, n_mels=128, fmax=8000)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            t_offset = s / sr

            img = librosa.display.specshow(
                mel_db, sr=sr, hop_length=512,
                x_axis="time", y_axis="mel", fmax=8000,
                ax=ax, cmap=color,
            )
            # Shift x-axis tick labels by t_offset so they show absolute time
            xticks = ax.get_xticks()
            ax.set_xticklabels([f"{t + t_offset:.2f}" for t in xticks])
            ax.axvline(sp["start"] - t_offset, color="lime", lw=2, linestyle="--", label="WhisperX start")
            ax.axvline(sp["end"]   - t_offset, color="red",  lw=2, linestyle="--", label="WhisperX end")
            ax.set_title(f"{title}  ({sp['start']:.3f}s – {sp['end']:.3f}s)")
            ax.legend(fontsize=8)
            fig.colorbar(img, ax=ax, format="%+2.0f dB")

        out3 = OUT_DIR / "spanish_spectrogram.png"
        plt.savefig(out3, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out3}")

    print("\nDone. Open the PNGs in tests/waveform_analysis/")


if __name__ == "__main__":
    main()
