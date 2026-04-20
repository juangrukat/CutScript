"""Smoke coverage for the experimental export-tail verifier probe."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("librosa")
pytest.importorskip("numpy")
pytest.importorskip("soundfile")

PROBE_DIR = Path(__file__).parent
ROOT = PROBE_DIR.parent.parent
sys.path.insert(0, str(PROBE_DIR))

from export_verifier import VerifierConfig, probe_cut, probe_source  # noqa: E402

SOURCE = ROOT / "tests/video/c8.mp4"
CUTS = {
    "c8_edited2.mp4": ROOT / "tests/video/c8_edited2.mp4",
    "c8_edited2_fixed.mp4": ROOT / "tests/video/c8_edited2_fixed.mp4",
    "c8_edited3.mp4": ROOT / "tests/video/c8_edited3.mp4",
    "c8_edited3_fixed.mp4": ROOT / "tests/video/c8_edited3_fixed.mp4",
}
CONTEXT_MISMATCH_CUT = ROOT / "tests/video/c8_editedz.mp4"


def _require_fixtures() -> None:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required for media fixture smoke test")
    missing = [path for path in [SOURCE, *CUTS.values()] if not path.exists()]
    if missing:
        pytest.skip(f"missing media fixtures: {missing}")


def test_c8_tail_probe_classification_shape() -> None:
    _require_fixtures()
    config = VerifierConfig()
    source_audio = probe_source(SOURCE, config)

    verdicts = {
        name: probe_cut(source_audio, path, config).verdict
        for name, path in CUTS.items()
    }

    assert verdicts["c8_edited2.mp4"] in {"review", "suspicious"}
    assert verdicts["c8_edited2_fixed.mp4"] == "ok"
    assert verdicts["c8_edited3.mp4"] in {"review", "suspicious"}
    assert verdicts["c8_edited3_fixed.mp4"] == "ok"


def test_c8_tail_probe_context_mismatch_still_preserves_terminal_tail() -> None:
    _require_fixtures()
    if not CONTEXT_MISMATCH_CUT.exists():
        pytest.skip("missing c8 context-mismatch fixture")

    config = VerifierConfig()
    source_audio = probe_source(SOURCE, config)
    result = probe_cut(source_audio, CONTEXT_MISMATCH_CUT, config)

    assert result.verdict == "review"
    assert result.terminal_envelope_cosine >= 0.95
    assert 0.90 <= result.terminal_rms_ratio <= 1.10
