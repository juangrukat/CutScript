# Librosa Sentence Boundary Refiner

A librosa-based tool that refines approximate sentence timestamps into editorially better cut points for audio editing.

## Overview

This tool uses librosa's audio analysis features to improve sentence boundary detection around approximate timestamps (e.g., from speech-to-text systems like CutScript). It employs RMS energy envelopes, onset strength detection, and zero-crossing snapping to find natural cut points.

## Features

- **Three refinement modes**: Tight, Natural, and Aggressive silence trim
- **Multi-cue detection**: Combines RMS energy, onset strength, and zero-crossing analysis
- **Adaptive thresholds**: Local noise floor estimation for robust performance
- **Fade suggestions**: Appropriate fade durations for each mode
- **Confidence flags**: Indicators for boundary adjustments and potential issues

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### As a library

```python
import librosa
from boundary_refiner import BoundaryRefiner

# Load audio
y, sr = librosa.load('audio.wav')

# Initialize refiner
refiner = BoundaryRefiner()

# Refine boundaries
result = refiner.refine_boundaries(
    y=y,
    sr=sr,
    approx_start=1.5,  # seconds
    approx_end=3.2,    # seconds
    mode='natural'
)

print(f"Refined start: {result['refined_start']:.3f}s")
print(f"Refined end: {result['refined_end']:.3f}s")
print(f"Fade in: {result['fade_in_duration']:.3f}s")
print(f"Fade out: {result['fade_out_duration']:.3f}s")
```

### Command-line demo

```bash
python demo.py audio.wav start_time end_time [mode]
```

Example:
```bash
python demo.py sample.wav 1.5 3.2 natural
```

## Modes

- **Tight**: Minimal retained silence, fast pacing. Best for snappy edits.
- **Natural**: Preserves breaths and pre-vocalization. Best for conversational content.
- **Aggressive**: Maximum silence reduction while avoiding choppy cuts.

## Algorithm Details

The refiner works by:

1. **Start boundary**: Searches for sustained energy rise, optionally using onset cues for natural mode
2. **End boundary**: Finds sustained energy decay with appropriate post-roll
3. **Zero-crossing snap**: Adjusts cut points to reduce clicks
4. **Fade suggestions**: Provides mode-appropriate fade durations

## Parameters

- `frame_length`: FFT frame length (default: 1024)
- `hop_length`: Hop length for features (default: 256)
- `smoothing_window`: RMS smoothing window (default: 5)

## Dependencies

- librosa: Audio analysis library
- numpy: Numerical computing