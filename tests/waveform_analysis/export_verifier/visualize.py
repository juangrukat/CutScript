"""Visual diagnostics for the experimental export verifier."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .decode import decode_audio
from .features import logmel, rms_envelope
from .models import MediaAudio, ProbeResult, VerifierConfig
from .windows import tail


def render_tail_visual(
    source_audio: MediaAudio,
    cut_path: Path,
    result: ProbeResult,
    config: VerifierConfig,
    output_path: Path,
) -> Path:
    """Render a source-vs-cut EOF tail diagnostic PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cut_audio = decode_audio(cut_path, config.sr)
    source_tail = tail(source_audio.samples, source_audio.sr, config.tail_window_s)
    cut_tail = tail(cut_audio.samples, cut_audio.sr, config.tail_window_s)

    source_env = rms_envelope(source_tail, source_audio.sr)
    cut_env = rms_envelope(cut_tail, cut_audio.sr)
    source_mel = logmel(source_tail, source_audio.sr)
    cut_mel = logmel(cut_tail, cut_audio.sr)

    t_audio = np.arange(len(source_tail)) / source_audio.sr
    t_env = np.arange(len(source_env)) * 0.010

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    fig.suptitle(
        (
            f"{cut_path.name} vs source EOF tail | verdict={result.verdict} | "
            f"env={result.envelope_cosine:.3f} logmel={result.logmel_cosine:.3f} "
            f"rms={result.rms_ratio:.3f} | terminal env={result.terminal_envelope_cosine:.3f} "
            f"terminal rms={result.terminal_rms_ratio:.3f}"
        ),
        fontsize=13,
    )

    ax = axes[0, 0]
    ax.plot(t_audio, source_tail, color="#1f77b4", linewidth=0.8, label="source EOF")
    ax.plot(t_audio, cut_tail, color="#d62728", linewidth=0.8, alpha=0.75, label="cut EOF")
    ax.set_title("Waveform Tail Overlay")
    ax.set_xlabel("seconds into EOF window")
    ax.set_ylabel("amplitude")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(t_env, source_env, color="#1f77b4", linewidth=1.4, label="source EOF")
    ax.plot(t_env, cut_env, color="#d62728", linewidth=1.4, label="cut EOF")
    ax.set_title("RMS Envelope Tail Overlay")
    ax.set_xlabel("seconds into EOF window")
    ax.set_ylabel("RMS")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.25)

    vmin = float(min(np.min(source_mel), np.min(cut_mel)))
    vmax = float(max(np.max(source_mel), np.max(cut_mel)))

    ax = axes[1, 0]
    im = ax.imshow(
        source_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Source EOF Log-Mel")
    ax.set_xlabel("frames")
    ax.set_ylabel("mel bins")

    ax = axes[1, 1]
    ax.imshow(
        cut_mel,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title("Cut EOF Log-Mel")
    ax.set_xlabel("frames")
    ax.set_ylabel("mel bins")

    fig.colorbar(im, ax=axes[1, :], shrink=0.75, label="dB")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
