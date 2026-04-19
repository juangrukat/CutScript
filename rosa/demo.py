#!/usr/bin/env python3
"""
Demo script for the BoundaryRefiner.

Usage:
    python demo.py audio_file.wav start_time end_time [mode]

Example:
    python demo.py sample_audio.wav 1.5 3.2 natural
"""

import sys
import librosa
from boundary_refiner import BoundaryRefiner


def main():
    if len(sys.argv) < 4:
        print("Usage: python demo.py audio_file.wav start_time end_time [mode]")
        print("Modes: tight, natural, aggressive (default: natural)")
        sys.exit(1)

    audio_file = sys.argv[1]
    start_time = float(sys.argv[2])
    end_time = float(sys.argv[3])
    mode = sys.argv[4] if len(sys.argv) > 4 else "natural"

    # Load audio
    print(f"Loading audio: {audio_file}")
    y, sr = librosa.load(audio_file, sr=None)

    # Initialize refiner
    refiner = BoundaryRefiner()

    # Refine boundaries
    print(f"Refining boundaries for mode: {mode}")
    result = refiner.refine_boundaries(y, sr, start_time, end_time, mode)

    print("\nResults:")
    print(".3f")
    print(".3f")
    print(".1f")
    print(".1f")
    print(f"Confidence flags: {result['confidence_flags']}")


if __name__ == "__main__":
    main()
