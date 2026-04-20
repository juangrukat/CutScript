"""Report rendering for the experimental export verifier."""

from __future__ import annotations

from collections.abc import Sequence

from .models import MediaAudio, ProbeResult, VerifierConfig


def render_table(
    source_audio: MediaAudio,
    config: VerifierConfig,
    results: Sequence[ProbeResult],
) -> str:
    lines = [
        f"source: {source_audio.path} duration={source_audio.duration:.3f}s",
        f"tail window: {config.tail_window_s:.3f}s, sr={source_audio.sr}",
        f"terminal window: {config.terminal_window_s:.3f}s",
        "",
        (
            f"{'cut':42} {'dur':>8} {'wave':>8} {'env':>8} "
            f"{'logmel':>8} {'rms':>8} {'t_env':>8} {'t_rms':>8} verdict"
        ),
        "-" * 114,
    ]

    for result in results:
        lines.append(
            f"{str(result.cut):42} "
            f"{result.duration:8.3f} "
            f"{result.waveform_cosine:8.3f} "
            f"{result.envelope_cosine:8.3f} "
            f"{result.logmel_cosine:8.3f} "
            f"{result.rms_ratio:8.3f} "
            f"{result.terminal_envelope_cosine:8.3f} "
            f"{result.terminal_rms_ratio:8.3f} "
            f"{result.verdict}"
        )

    return "\n".join(lines)
