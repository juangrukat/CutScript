#!/usr/bin/env python3
"""
Test script for the BoundaryRefiner using the provided audio and transcript files.
"""

import sys
import os
import re
import json
import librosa
from boundary_refiner import BoundaryRefiner


def parse_srt_time(time_str):
    """Parse SRT timestamp to seconds."""
    # Format: 00:00:00,001
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if not match:
        raise ValueError(f"Invalid SRT time format: {time_str}")
    hours, minutes, seconds, milliseconds = map(int, match.groups())
    return hours * 3600 + minutes * 60 + seconds + milliseconds / 1000


def parse_srt(file_path):
    """Parse SRT file and return list of (start, end, text) tuples."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into entries
    entries = []
    blocks = content.strip().split("\n\n")

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # Skip index line, parse timestamp line
        timestamp_line = lines[1]
        if "-->" not in timestamp_line:
            continue

        start_str, end_str = timestamp_line.split("-->")
        start_time = parse_srt_time(start_str.strip())
        end_time = parse_srt_time(end_str.strip())

        # Join remaining lines as text
        text = " ".join(lines[2:]).strip()

        entries.append((start_time, end_time, text))

    return entries


def parse_json(file_path):
    """Parse JSON file and return list of (start, end, text) tuples."""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = []
    for line in data.get("lines", []):
        start_time = parse_srt_time(line["startTime"])
        end_time = parse_srt_time(line["endTime"])
        text = line["text"]
        entries.append((start_time, end_time, text))

    return entries


def test_boundary_refiner(audio_path, transcript_path, format="srt", mode="natural"):
    """Test the boundary refiner on the first sentence."""
    # Load audio
    print(f"Loading audio: {audio_path}")
    y, sr = librosa.load(audio_path)
    print(".2f")

    # Parse transcript
    print(f"Parsing transcript: {transcript_path}")
    if format == "srt":
        entries = parse_srt(transcript_path)
    elif format == "json":
        entries = parse_json(transcript_path)
    else:
        raise ValueError("Format must be 'srt' or 'json'")

    print(f"Found {len(entries)} entries")

    if not entries:
        print("No entries found in transcript")
        return

    # Test on first sentence
    start_time, end_time, text = entries[0]
    print(f"\nTesting on first sentence:")
    print(f"Original text: '{text}'")
    print(".3f")

    # Initialize refiner
    refiner = BoundaryRefiner()

    # Refine boundaries
    print(f"\nRefining boundaries with mode: {mode}")
    result = refiner.refine_boundaries(y, sr, start_time, end_time, mode)

    print("\nResults:")
    print(".3f")
    print(".3f")
    print(".1f")
    print(".1f")
    print(f"Confidence flags: {result['confidence_flags']}")

    # Show adjustment
    start_adjust = result["refined_start"] - start_time
    end_adjust = result["refined_end"] - end_time
    print("\nAdjustments:")
    print(".3f")
    print(".3f")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test.py <audio_file> [transcript_file] [format] [mode]")
        print("  audio_file: path to audio file")
        print(
            "  transcript_file: path to srt or json file (optional, will look for 1.srt or 1.json)"
        )
        print("  format: 'srt' or 'json' (default: srt)")
        print("  mode: 'tight', 'natural', or 'aggressive' (default: natural)")
        sys.exit(1)

    audio_file = sys.argv[1]
    transcript_file = sys.argv[2] if len(sys.argv) > 2 else None
    format_type = sys.argv[3] if len(sys.argv) > 3 else "srt"
    mode = sys.argv[4] if len(sys.argv) > 4 else "natural"

    # If no transcript file specified, look for default ones
    if transcript_file is None:
        audio_dir = os.path.dirname(audio_file)
        if format_type == "srt":
            transcript_file = os.path.join(audio_dir, "1.srt")
        elif format_type == "json":
            transcript_file = os.path.join(audio_dir, "1.json")
        else:
            print("Please specify transcript file or use srt/json format")
            sys.exit(1)

    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        sys.exit(1)

    if not os.path.exists(transcript_file):
        print(f"Transcript file not found: {transcript_file}")
        sys.exit(1)

    test_boundary_refiner(audio_file, transcript_file, format_type, mode)


if __name__ == "__main__":
    main()
