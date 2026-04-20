# CutScript

An open-source, local-first, Descript-style text-based audio and video editor powered by AI. Edit by editing text — delete a word from the transcript and it disappears from the video. Seamless cuts via acoustic-map–guided boundary refinement.

<img width="1034" height="661" alt="image" src="https://github.com/user-attachments/assets/b1ed9505-792e-42ca-bb73-85458d0f02a5" />

## What you can do with it

- **Transcribe any video or audio file** with word-level timestamps (WhisperX, multilingual). On Apple Silicon, switch to the MLX backend for ~2-3× faster decodes — word boundaries still come from WhisperX's wav2vec2 alignment, so cut quality is unchanged.
- **Edit by selecting text** — select words, hit Delete, and those spans are cut from the video on export.
- **Cut seamlessly.** CutScript doesn't just slice at WhisperX timestamps. At ingest, it builds an AcousticMap — per-word fingerprints covering broadband RMS, 2-8 kHz fricative energy, onset/coda phoneme classes, and dip profiles. At export, those fingerprints drive boundary extension so the fricative tail of "Spanish" or the aspirated release of a final "t" stays intact. Zero-crossing snap avoids clicks at splice points.
- **Detect filler words in any language.** The AI infers the transcript language and applies that language's filler conventions (Spanish "este", French "euh", Japanese "eto", etc). Add custom phrases in any language via the UI.
- **Generate social-ready clips.** Point it at a long video, pick one or more target durations (15/30/60/90s), and get back as many candidate clips as the material supports — each with a confidence score, a reason, and a predictable filename (`source_30s_1.mp4`). Clips can be quick-exported, loaded back into the editor for tweaking, or batch-exported to a folder you choose.
- **Focus modes.** Reshape long-form content with one click: *Remove repetition*, *Tighten pace*, *Keep key points*, *Q&A only*, or *Focus on topic* (free-text). Each mode returns a reviewable plan — you approve cuts individually or in bulk before they touch the edit.
- **Burn captions.** Word-level SRT/VTT/ASS generation; burn-in supported on export.
- **Clean up audio.** Studio Sound (DeepFilterNet) noise reduction on export.
- **Work offline.** Ollama support means the whole pipeline can run locally with no cloud dependency.
- **Bring your own model.** OpenAI (structured outputs via `json_schema`), Anthropic (structured outputs via forced tool-use), and Ollama (JSON-schema `format`). API keys are stored in the OS keychain.

## Architecture

- **Electron + React** desktop app with Tailwind CSS
- **FastAPI** Python backend (spawned as a child process)
- **WhisperX** for word-level transcription with alignment (default)
- **MLX Whisper** optional decoder on Apple Silicon — decodes with MLX, aligns with WhisperX's wav2vec2 for identical timestamp precision
- **librosa** for audio analysis (AcousticMap ingest-time fingerprinting, RMS-energy boundary refinement, zero-crossing snap)
- **FFmpeg** for video processing (stream-copy and re-encode, concat + loudnorm)
- **Ollama / OpenAI / Anthropic** for AI features, all called with **structured output** (JSON-Schema-constrained) and validated through Pydantic models before they touch the edit.

## Quick Start

### Prerequisites

- Node.js 18+
- Python 3.10+
- FFmpeg (in PATH)
- (Optional) Ollama for local AI features

### Install

```bash
# Root dependencies (Electron, concurrently)
npm install

# Frontend dependencies (React, Tailwind, Zustand)
cd frontend && npm install && cd ..

# Backend dependencies — install into the same Python the backend runs with.
# Check package.json → dev:backend to see which Python binary is used,
# then use the matching pip. Example for a project venv at ~/.cutscript-venv:
/Users/kat/.cutscript-venv/bin/pip install -r requirements.txt

# Optional: MLX Whisper backend (Apple Silicon only)
# IMPORTANT: must use the same pip as above — plain `pip install mlx-whisper`
# will often install into a different Python and the backend won't see it.
/Users/kat/.cutscript-venv/bin/pip install mlx-whisper
```

The MLX backend is auto-detected at startup: if `mlx-whisper` is importable on an arm64 Mac, the "Backend" dropdown on the open-file screen lights up a second option. Timestamps still route through WhisperX alignment, so there is no change to the seamless-cut pipeline.

> **Troubleshooting — MLX option stays greyed out after `pip install mlx-whisper`:**
> The backend runs under the specific Python listed in `package.json → dev:backend` (e.g. `/Users/kat/.cutscript-venv/bin/python`). Running plain `pip install` targets whatever Python `pip` resolves to in your shell, which is often a *different* interpreter (pyenv, system, etc.). Install with the venv's own pip as shown above, then restart the app.

> **Troubleshooting — MLX transcripts clip word tails or the end of the video:**
> Two independent clips existed and are both fixed. (1) MLX emits segment-end timestamps quantized to 20ms Whisper tokens that tend to land at the last phoneme, not at the end of the word's acoustic decay; the transcription layer now pads each MLX segment end forward (capped at the next segment's start, and the last segment to the full audio duration) so WhisperX's forced alignment can place word boundaries on the true decay point. (2) The AcousticMap's coda-search cap used a 5ms guard to avoid crossing the next word's start — but the final word has no next word, so the guard carved a hole at EOF and, when the decay threshold was never met inside the search window, the last word's `ae` collapsed back to its phoneme peak. The last word now extends through the search cap to the audio end. If you transcribed or analyzed a file **before** these fixes landed, clear the spectral cache (Settings → Clear spectral cache) and re-transcribe — otherwise the cached AcousticMap still reflects the old clipped boundaries.

### Run (Development)

```bash
# Backend + frontend + electron
npm run dev
```

Or separately:

```bash
# Terminal 1: Backend
cd backend && python -m uvicorn main:app --reload --port 8642

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Electron
npx electron .
```

## How the edit pipeline works

```
open video → WhisperX transcribe → /analyze builds AcousticMap (cached)
                                       │
                                       ▼
edit in UI (delete / filler / focus / clip) → getKeepSegments()  →  /export
                                       │
                                       ▼
   extract PCM WAV → _refine_from_map (AcousticMap-guided)  → ffmpeg trim+concat+loudnorm
```

Every AI feature converges on the **same truth**: word-index ranges in the transcript. The AI produces a plan, the validator clamps it to valid ranges, the editor store turns it into `DeletedRange` objects, and export runs through the normal refinement pipeline. That's why clips, focus cuts, and manual edits all sound equally seamless.

## Project Structure

```
cutscript/
├── electron/          # Electron main + IPC bridge
├── frontend/          # React + Vite + Tailwind
│   └── src/
│       ├── components/  # VideoPlayer, TranscriptEditor, AIPanel, ExportDialog
│       ├── store/       # Zustand: editorStore, aiStore
│       └── types/       # TypeScript interfaces
├── backend/           # FastAPI
│   ├── routers/       # transcribe, analyze, export, ai, captions, audio, cache
│   ├── services/      # transcription, audio_analyzer, video_editor,
│   │                  # ai_provider (structured output), ai_validator (Pydantic),
│   │                  # boundary_refiner, caption_generator, audio_cleaner
│   └── utils/         # gpu, cache, audio_processing
└── tests/waveform_analysis/   # phase0-phase5 diagnostic scripts
```

## Features

| Feature | Status |
|---------|--------|
| Word-level transcription (WhisperX) | Done |
| MLX Whisper decode backend (Apple Silicon, WhisperX-aligned) | Done |
| Text-based video editing | Done |
| Undo/redo | Done |
| Clear-all cuts (restore transcript) | Done |
| Waveform timeline | Done |
| FFmpeg stream-copy export | Done |
| FFmpeg re-encode (up to 4K) | Done |
| Seamless audio cuts (boundary fades + loudness normalization) | Done |
| AcousticMap ingest-time analysis (fricative/stop/nasal/vowel fingerprints) | Done |
| Phoneme-class-aware coda decay (preserves fricative tails like /ʃ/ in "Spanish") | Done |
| Interior gap speech detection (word-level vs silence-level cut routing) | Done |
| Cache management UI (transcripts + spectral maps, per-type clear) | Done |
| AI filler word detection (multilingual, principle-based) | Done |
| AI clip candidates (multi-duration, batch export, custom save folder) | Done |
| AI focus modes (redundancy / tighten / key-points / Q&A / topic) | Done |
| Structured-output AI (OpenAI json_schema, Anthropic tool-use, Ollama format) | Done |
| Ollama + OpenAI + Claude | Done |
| Word-level captions (SRT/VTT/ASS) | Done |
| Caption burn-in on export | Done |
| Studio Sound (DeepFilterNet) | Done |
| Real-time transcription progress (SSE streaming) | Done |
| Transcription audio preprocessing (silence trim + loudnorm) | Done |
| Vocabulary prompt for domain-specific transcription accuracy | Done |
| Keyboard shortcuts (J/K/L) | Done |
| Speaker diarization | Done |
| Virtualized transcript (react-virtuoso) | Done |
| Encrypted API key storage (OS keychain) | Done |
| Project save/load (.cutscript) | Done |
| AI background removal | Planned |

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Play / Pause |
| J / K / L | Reverse / Pause / Forward |
| ← / → | Seek ±5 seconds |
| Delete | Delete selected words |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+S | Save project |
| Ctrl+E | Export |
| ? | Shortcut cheatsheet |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Health check |
| POST | /transcribe | Transcribe video (WhisperX or MLX backend) |
| POST | /transcribe/stream | Transcribe with real-time SSE progress |
| GET | /transcribe/backends | Report which transcription backends are available on this machine |
| POST | /analyze | Build AcousticMap (per-word spectral fingerprints) |
| GET | /cache/sizes | Report transcript + spectral cache sizes |
| POST | /cache/clear/{kind} | Clear `transcripts` or `spectral` cache |
| POST | /export | Export edited video (stream copy or re-encode) |
| POST | /ai/filler-removal | Detect filler words via LLM (multilingual, structured output) |
| POST | /ai/create-clip | Return ClipPlan (multiple durations, confidence-scored) |
| POST | /ai/focus | Return FocusPlan (redundancy / tighten / key_points / qa_extract / topic) |
| GET | /ai/ollama-models | List local Ollama models |
| POST | /captions | Generate SRT/VTT/ASS captions |
| POST | /audio/clean | Noise reduction (DeepFilterNet) |
| GET | /audio/capabilities | Check audio processing availability |

## License

MIT License — see [LICENSE](LICENSE) for details.
