"""CLI for the experimental export-tail verifier probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from .models import VerifierConfig
from .report import render_table
from .scoring import probe_cut, probe_source
from .visualize import render_tail_visual

DEFAULT_SOURCE = Path("tests/video/c8.mp4")
DEFAULT_CUTS = [
    Path("tests/video/c8_edited2.mp4"),
    Path("tests/video/c8_edited2_fixed.mp4"),
    Path("tests/video/c8_edited3.mp4"),
    Path("tests/video/c8_edited3_fixed.mp4"),
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("cuts", nargs="*", type=Path, default=DEFAULT_CUTS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--window", type=float, default=1.2)
    parser.add_argument("--terminal-window", type=float, default=0.6)
    parser.add_argument(
        "--visual-dir",
        type=Path,
        help="Optional directory for source-vs-cut EOF tail diagnostic PNGs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = VerifierConfig(
        sr=args.sr,
        tail_window_s=args.window,
        terminal_window_s=args.terminal_window,
    )

    source_audio = probe_source(args.source, config)
    results = [probe_cut(source_audio, cut, config) for cut in args.cuts]

    print(render_table(source_audio, config, results))
    if args.visual_dir:
        for cut, result in zip(args.cuts, results):
            output_path = args.visual_dir / f"{cut.stem}_tail_probe.png"
            render_tail_visual(source_audio, cut, result, config, output_path)
            print(f"visual: {output_path}")
    return 0
