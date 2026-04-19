# CutScript

An open-source, local-first, Descript-like text-based audio and video editor powered by AI. Edit audio/video by editing text — delete a word from the transcript and it's cut from the audio/video.

<img width="1034" height="661" alt="image" src="https://github.com/user-attachments/assets/b1ed9505-792e-42ca-bb73-85458d0f02a5" />


## Architecture

- **Electron + React** desktop app with Tailwind CSS
- **FastAPI** Python backend (spawned as child process)
- **WhisperX** for word-level transcription with alignment
- **librosa** for audio boundary refinement (natural splice-point detection, interior gap speech detection, zero-crossing snap)
- **FFmpeg** for video processing (stream-copy and re-encode)
- **Ollama / OpenAI / Claude** for AI features (filler removal, clip creation)

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

# Backend dependencies (run from repo root)
pip install -r requirements.txt
```

### Run (Development)

```bash
# Start all three (backend + frontend + electron)
npm run dev
```

Or run them separately:

```bash
# Terminal 1: Backend
cd backend && python -m uvicorn main:app --reload --port 8642

# Terminal 2: Frontend
cd frontend && npm run dev

# Terminal 3: Electron
npx electron .
```

## Project Structure

```
cutscript/
├── electron/          # Electron main process
│   ├── main.js        # App entry, spawns Python backend
│   ├── preload.js     # Secure IPC bridge
│   └── python-bridge.js
├── frontend/          # React + Vite + Tailwind
│   └── src/
│       ├── components/  # VideoPlayer, TranscriptEditor, etc.
│       ├── store/       # Zustand state (editorStore, aiStore)
│       ├── hooks/       # useVideoSync, useKeyboardShortcuts
│       └── types/       # TypeScript interfaces
├── backend/           # FastAPI Python backend
│   ├── main.py
│   ├── routers/       # API endpoints
│   ├── services/      # Core logic (transcription, editing, AI)
│   └── utils/         # GPU, cache, audio helpers
└── shared/            # Project schema
```

## Features

| Feature | Status |
|---------|--------|
| Word-level transcription (WhisperX) | Done |
| Text-based video editing | Done |
| Undo/redo | Done |
| Waveform timeline | Done |
| FFmpeg stream-copy export | Done |
| FFmpeg re-encode (up to 4K) | Done |
| Seamless audio cuts (boundary fades + loudness normalization) | Done |
| Audio splice-point refinement (RMS energy + zero-crossing snap) | Done |
| AcousticMap ingest-time analysis (per-word fricative/stop/nasal/vowel fingerprints) | Done |
| Phoneme-class-aware coda decay (preserves fricative tails like /ʃ/ in "Spanish") | Done |
| Interior gap speech detection (word-level vs silence-level cut routing) | Done |
| Bias guard (prevents ZC snap landing inside deleted words) | Done |
| Cache management UI (transcripts + spectral maps, per-type clear) | Done |
| AI filler word removal | Done |
| AI clip creation (Shorts) | Done |
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
| Encrypted API key storage | Done |
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
| POST | /transcribe | Transcribe video with WhisperX |
| POST | /transcribe/stream | Transcribe with real-time SSE progress |
| POST | /analyze | Build AcousticMap (per-word spectral fingerprints) |
| GET | /cache/sizes | Report transcript + spectral cache sizes |
| POST | /cache/clear/{kind} | Clear `transcripts` or `spectral` cache |
| POST | /export | Export edited video (stream copy or re-encode) |
| POST | /ai/filler-removal | Detect filler words via LLM |
| POST | /ai/create-clip | AI-suggested clips for shorts |
| GET | /ai/ollama-models | List local Ollama models |
| POST | /captions | Generate SRT/VTT/ASS captions |
| POST | /audio/clean | Noise reduction (DeepFilterNet) |
| GET | /audio/capabilities | Check audio processing availability |

## License

MIT License — see [LICENSE](LICENSE) for details.
