#!/usr/bin/env python3
"""
Demo script to create audio clips showing boundary refinement results.
"""

import os
import subprocess
import librosa
from boundary_refiner import BoundaryRefiner


def extract_audio_segment(input_file, output_file, start_time, duration):
    """Extract audio segment using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-ss",
        str(start_time),
        "-t",
        str(duration),
        "-c",
        "copy",
        output_file,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def create_demo_clips(audio_path, mode="natural"):
    """Create demo audio clips showing original vs refined boundaries."""
    # Load audio to get duration
    y, sr = librosa.load(audio_path)

    # Original boundaries from SRT (first sentence)
    original_start = 0.001
    original_end = 1.120
    original_duration = original_end - original_start

    # Initialize refiner and get refined boundaries
    refiner = BoundaryRefiner()
    result = refiner.refine_boundaries(y, sr, original_start, original_end, mode)

    refined_start = result["refined_start"]
    refined_end = result["refined_end"]
    refined_duration = refined_end - refined_start

    print(
        f"Original segment: {original_start:.3f}s - {original_end:.3f}s ({original_duration:.3f}s)"
    )
    print(
        f"Refined segment: {refined_start:.3f}s - {refined_end:.3f}s ({refined_duration:.3f}s)"
    )
    print(f"Mode: {mode}")

    # Create output directory
    output_dir = "demo_clips"
    os.makedirs(output_dir, exist_ok=True)

    # Extract original segment
    original_file = os.path.join(output_dir, f"original_{mode}.mp4")
    extract_audio_segment(audio_path, original_file, original_start, original_duration)
    print(f"Created: {original_file}")

    # Extract refined segment
    refined_file = os.path.join(output_dir, f"refined_{mode}.mp4")
    extract_audio_segment(audio_path, refined_file, refined_start, refined_duration)
    print(f"Created: {refined_file}")

    # Create a combined clip with padding for comparison
    # Add 0.5s silence before each clip
    combined_file = os.path.join(output_dir, f"comparison_{mode}.mp4")

    # Create silence
    silence_file = os.path.join(output_dir, "silence.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=r=22050:cl=mono",
            "-t",
            "0.5",
            "-c:a",
            "aac",
            silence_file,
        ],
        check=True,
        capture_output=True,
    )

    # Concatenate: silence + original + silence + refined
    concat_file = os.path.join(output_dir, "concat_list.txt")
    with open(concat_file, "w") as f:
        f.write(f"file '{silence_file}'\n")
        f.write(f"file '{original_file}'\n")
        f.write(f"file '{silence_file}'\n")
        f.write(f"file '{refined_file}'\n")

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-c",
            "copy",
            combined_file,
        ],
        check=True,
        capture_output=True,
    )

    print(f"Created comparison: {combined_file}")
    print("Listen to the clips to hear the difference:")
    print("- Original: starts immediately at transcript boundary")
    print("- Refined: has natural padding and better cut points")

    # Cleanup temp files
    os.remove(silence_file)
    os.remove(concat_file)


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "natural"
    create_demo_clips("../sound/1.mp4", mode)
