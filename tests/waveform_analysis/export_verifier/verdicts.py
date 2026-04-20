"""Verdict thresholds for the experimental export verifier."""

from __future__ import annotations


def classify_tail_similarity(
    logmel_cosine: float,
    envelope_cosine: float,
    rms_ratio: float,
    terminal_logmel_cosine: float,
    terminal_envelope_cosine: float,
    terminal_rms_ratio: float,
) -> str:
    """Classify using the current EOF probe thresholds.

    The contextual window can legitimately differ when a user deletes material
    close to EOF. The terminal window protects the final coda check from that
    false positive while still catching loudnorm-style tail weakening.
    """
    terminal_ok = (
        terminal_logmel_cosine >= 0.95
        and terminal_envelope_cosine >= 0.90
        and 0.90 <= terminal_rms_ratio <= 1.10
    )
    if (
        logmel_cosine >= 0.88
        and envelope_cosine >= 0.88
        and 0.85 <= rms_ratio <= 1.15
        and terminal_ok
    ):
        return "ok"
    if terminal_ok:
        return "review"
    if (
        logmel_cosine >= 0.75
        and envelope_cosine >= 0.70
        and 0.65 <= rms_ratio <= 1.35
    ):
        return "review"
    return "suspicious"
