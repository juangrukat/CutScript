"""
Standalone export-tail verifier probe.

This is intentionally not wired into CutScript's export path. It answers one
question: does the end of an exported cut acoustically resemble the end of the
original source file?

Usage:
  /Users/kat/.cutscript-venv/bin/python tests/waveform_analysis/export_tail_probe.py
  /Users/kat/.cutscript-venv/bin/python tests/waveform_analysis/export_tail_probe.py \
    --source tests/video/c8.mp4 tests/video/c8_edited3.mp4
"""

from __future__ import annotations

from export_verifier.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
