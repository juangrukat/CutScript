# CutScript — AI Session Context

Onboarding notes for a new AI session. Describes the architecture, the edit pipeline, and the latest AI-layer rework.

---

## What CutScript Is

A local-first, Descript-style text-based video editor. The user transcribes a video with WhisperX (word-level timestamps), edits the transcript (delete, filler removal, focus modes, clip extraction), and exports a seamless cut video. Stack: Electron + React frontend, FastAPI Python backend, WhisperX + librosa + FFmpeg. LLM features via Ollama / OpenAI / Anthropic. Optional MLX Whisper decode backend on Apple Silicon (still word-aligned by WhisperX).

---

## The One Truth: Word-Index Ranges

Every edit — manual selection, filler removal, focus plan, clip extraction — converges on **word-index ranges in the transcript**. These become `DeletedRange` objects in the editor store. `getKeepSegments()` turns them into time ranges, `/export` feeds them to `_refine_from_map` (AcousticMap-guided), ffmpeg concat+loudnorm produces the final cut.

This is why everything sounds equally seamless — clips, focus cuts, fillers, manual edits all traverse the same refinement pipeline.

---

## Transcription Backends

Two decoder paths converge on the same aligner:

- **WhisperX** (default) — faster-whisper decode + WhisperX wav2vec2 alignment.
- **MLX Whisper** (optional, Apple Silicon) — `mlx-whisper` decode + WhisperX wav2vec2 alignment. Repos come from `mlx-community/whisper-*` (see `transcription_mlx.MLX_REPOS`).

`backend/services/transcription.py` dispatches on `backend` (`"whisperx" | "mlx"`). Both paths funnel into the shared `_align_and_pack()` helper that runs `whisperx.align()` and produces the final `{words, segments, language}`. **Word-timestamp precision is identical across backends** — only the Whisper decode itself differs.

`backend/services/transcription_mlx.py` is lazily imported: `is_available()` returns `(False, "...")` when MLX isn't runnable (non-arm64 mac or `mlx-whisper` missing). The frontend calls `GET /transcribe/backends` at app load and disables the MLX option accordingly; if the user's saved backend isn't available, it falls back to WhisperX.

The cache key (`_make_cache_op`) includes `backend` so MLX and WhisperX transcripts never collide in `~/.obs_transcriber_cache/`. `distil-large-v3` exists only under WhisperX; `large-v3-turbo` exists only under MLX. The router enforces this per-backend via `ensure_model_matches_backend()`.

---

## The Cut Pipeline at a Glance

```
open video → WhisperX or MLX transcribe → /analyze builds AcousticMap (cached)
                                       │
                                       ▼
edit in UI (manual / filler / focus / clip) → getKeepSegments()  →  /export
                                       │
                                       ▼
  extract PCM WAV → _refine_from_map (AcousticMap-guided)  → ffmpeg trim+concat+loudnorm
                   (or _refine_segments fallback if map missing)
```

Two disk caches, both cleanable from the Settings panel:

- `~/.obs_transcriber_cache/<hash>_<model>_transcribe_wx_*.json` — WhisperX results
- `~/.cutscript_spectral_cache/<hash>.json` — AcousticMap fingerprints

File hash = md5(path + size + mtime).

---

## AcousticMap (refinement primary)

Built at ingest time by `POST /analyze` and consumed at export time by `_refine_from_map`. Lives in `backend/services/audio_analyzer.py`.

**Per-word `WordFingerprint`:** WhisperX start/end (`ws`/`we`), **acoustic** start/end (`as_`/`ae` — these are the refined boundaries), onset/coda phoneme classes (fricative/stop/nasal/vowel/approximant), `peak_rms`, `peak_fric` (2-8 kHz), intraword `dips`.

**Coda-specific decay policies** in `_refine_from_map` control how far `ae` extends past `we`: fricative tails get band+broadband both-below; nasals/stops/vowels use tiered dB thresholds (-18/-20/-25). Clamps to audio duration; drops any segment <10 ms.

Legacy `_refine_segments` kept only as a fallback (same safety clamps).

---

## AI Layer — current architecture

Three features, one pipeline:

| Feature | Endpoint | Output |
|---|---|---|
| Filler detection (multilingual) | `POST /ai/filler-removal` | `FillerReport` |
| Clip candidates (multi-duration) | `POST /ai/create-clip` | `ClipPlan` |
| Focus modes (redundancy/tighten/key_points/qa_extract/topic) | `POST /ai/focus` | `FocusPlan` |

### Structured output

Every call uses **native constrained decoding**:

- **OpenAI**: `response_format={"type": "json_schema", "json_schema": {..., "strict": True}}`. The helper `_strictify_schema` in `ai_provider.py` transforms Pydantic schemas by adding `additionalProperties: false`, marking every field required, stripping `default` and `title`.
- **Anthropic**: forced tool-use. A virtual tool is invented with `input_schema = <Pydantic schema>`, `tool_choice` forces the tool, we read `tool_use.input`.
- **Ollama**: `format: <schema>` (Ollama 0.5+). Falls back to brace-extract if validation fails.

Every response goes through `response_model.model_validate(raw)` (Pydantic). The old "find the first `{`" parser is only used as a last resort if JSON decoding fails outright.

### The shared validator

`backend/services/ai_validator.py` owns all the Pydantic models and per-feature validators:

- `validate_filler_report` — clamps indices to [0, word_count), dedupes, filters by confidence, flags `needs_review` if >40% of transcript is flagged
- `validate_clip_plan` — clamps word indices, **rebuilds startTime/endTime from the transcript's ground truth** (so the exported clip always aligns with real word boundaries), drops clips under 1 s or with <2 words, sorts ascending
- `validate_focus_plan` — merges overlapping/adjacent ranges, hard-rejects if >80% of transcript would be deleted, flags `needs_review` above 50%

This is the "seamless guarantee" layer — whatever the model returns, the rest of the pipeline sees clean, clamped, word-indexed data.

### Frontend filtering of already-deleted words

**Important behaviour**: when the user has already deleted ranges before invoking an AI feature, `AIPanel` sends only the **kept** words to the backend (original indices preserved via `keptWordsPayload`). Without this, the AI would re-flag words inside already-deleted spans — "Apply All" would add duplicate ranges and produce no visible change.

This applies to filler, clips, and focus alike. If the user wants to analyze the original full transcript, they clear deletions first via the transcript header's "Clear all" button.

### Clip exports

- Filename format: `{source_basename}_{target_duration}s_{bucket_index}.mp4` (e.g. `a_30s_1.mp4`). Index is per-duration so multiple 30 s clips don't collide with each other.
- Save location: defaults to the source video's folder; user can override via a persisted picker (`dialog:openDirectory` IPC in Electron).
- At export, `handleExportClip` intersects the clip's time range with the editor's current `getKeepSegments()` so any in-editor edits (fillers, focus cuts) are honoured, then calls `/export` with the full `words` payload + `deleted_indices`. That's what pipes the clip through the "regular route" — same AcousticMap refinement as a main-dialog export.
- "Edit" button stages a clip into the editor (deletes everything outside the clip range) so the user can tweak before exporting through the main dialog.

### Focus modes — UX

Five preset buttons: Remove repetition, Tighten pace, Keep key points, Q&A only, Focus on topic. Only *topic* requires free-text input. Results appear as reviewable cuts with confidence pills; each can be applied individually via a check-mark or all at once. Applying converts `FocusDeletion` → `deleteWordRange()` → `DeletedRange` → standard export path.

---

## Recent Changes (current session)

### AI layer rewrite

- **New**: `backend/services/ai_validator.py` — Pydantic models + validators that clamp AI output to the transcript's truth (word indices, valid ranges, confidence thresholds, deletion caps).
- **Rewritten**: `backend/services/ai_provider.py` — every call uses structured output. Prompts restructured into Role/Task/Rules/Output sections. New `focus_transcript()` function. Clip prompt now asks for as many candidates as the material supports (~15 total) across requested durations. Filler prompt is principle-based and multilingual (AI infers language from transcript, no plumbing required).
- **Updated**: `backend/routers/ai.py` — added `/ai/focus`, added `target_durations: List[int]` to `/ai/create-clip`.

### Frontend AI panel

- Three tabs: Filler / Clips / Focus.
- Clip duration chips (15/30/60/90, multi-select).
- Clip save-location picker (persisted per machine via aiStore).
- Per-duration clip filename (`{source}_{dur}s_{n}.mp4`), previewed in each clip card.
- Focus mode cards with inline apply / dismiss.
- Confidence pills and warning banners on every result.
- AI requests filter out already-deleted words so behaviour matches user expectations.

### Editor

- `editorStore.clearAllDeletions()` — restores every cut.
- `TranscriptEditor` header gained a "Clear all" button (only visible when cuts exist, confirmation dialog before firing).

### Electron

- New IPC: `dialog:openDirectory` (used by the clip save-location picker).

### Transcription: MLX backend option

- **New**: `backend/services/transcription_mlx.py` — Apple Silicon decode via `mlx-whisper`. Returns segment-level text that the shared `_align_and_pack()` helper feeds into WhisperX's wav2vec2 forced alignment. Timestamps are as precise as the WhisperX path.
- **New endpoint**: `GET /transcribe/backends` — probes which backends the machine can run (platform + import check) so the UI can disable unavailable options.
- **App.tsx**: backend selector on the open-file screen. Queries backends at load, auto-falls-back if the saved backend isn't available, and filters the model list per backend (`large-v3-turbo` is MLX-only; `distil-large-v3` is WhisperX-only).

---

## Legacy / fallback refiner

Kept in `_refine_segments` for when no AcousticMap is available. Decision tree:

```
Is there speech in the INTERIOR of the gap between segments?
├── YES (deleted words) → onset/decay guided
│   START: _find_onset_before → backs up 20 ms before detected onset (captures "Sp", "St")
│         → fallback: 28 ms fixed bias + ZC snap
│   END:   _find_word_end → scans RMS for natural decay at −25 dB below word peak
│         → Phase 2: onset of next content if energy stays elevated
│         → fallback: 28 ms fixed bias + ZC snap
└── NO (natural pause/breath) → BoundaryRefiner
    └── Energy onset/decay detection with constrained search window

After all start refinement:
  → _advance_past_silence: trims leading silence if gap > 250 ms before first speech
```

Speech threshold: 10th-percentile RMS × 6, adaptive per file.

---

## Known Issues / Design Notes

- **On-demand analyze** at export time can add 2-5 s for a long video if the spectral cache was cleared between transcribe and export. Non-fatal — falls back to legacy on any error.
- **AcousticMap hash collision** isn't protected — if two files somehow map to the same md5(path+size+mtime), the cached map would be stale. Unlikely.
- **WhisperX re-transcription drift**: when re-transcribing a cut file to verify, word timestamps drift 0.5-1 s from actual positions. This is a WhisperX forced-alignment artifact on short concatenated audio, not an export bug.
- **Clip filename collisions** across create-clip runs: if the user runs create-clip twice and exports clips with the same duration in both runs, the second run's `_1.mp4` overwrites the first. Clear the results or pick a different save folder between runs.

---

## Project Structure

```
cutscript/
├── electron/
│   ├── main.js                          # IPC: dialog:openDirectory, etc.
│   └── preload.js
├── frontend/src/
│   ├── App.tsx                          # open-video modal + /analyze kickoff
│   ├── components/
│   │   ├── AIPanel.tsx                  # Filler / Clips / Focus tabs
│   │   ├── TranscriptEditor.tsx         # Clear-all button in header
│   │   ├── ExportDialog.tsx
│   │   ├── SettingsPanel.tsx
│   │   └── VideoPlayer.tsx
│   ├── store/
│   │   ├── editorStore.ts               # clearAllDeletions, getKeepSegments (duration=0 guard)
│   │   └── aiStore.ts                   # providers, clipDurations, clipSaveLocation, focusPlan
│   └── types/project.ts                 # FocusPlan, ClipPlan, FillerWordResult
├── backend/
│   ├── main.py
│   ├── routers/
│   │   ├── ai.py                        # /ai/filler-removal, /ai/create-clip, /ai/focus
│   │   ├── export.py                    # _refine_from_map (primary) + legacy fallback
│   │   ├── analysis.py                  # POST /analyze
│   │   └── cache.py
│   ├── services/
│   │   ├── ai_provider.py               # Structured output per provider + prompts
│   │   ├── ai_validator.py              # Pydantic models + validators (the truth layer)
│   │   ├── audio_analyzer.py            # AcousticMap
│   │   ├── boundary_refiner.py
│   │   ├── transcription.py             # Backend dispatch + shared _align_and_pack()
│   │   ├── transcription_mlx.py         # Lazy MLX decoder (Apple Silicon)
│   │   └── video_editor.py
│   └── utils/
└── tests/waveform_analysis/             # phase0-phase5 diagnostic + validation scripts
```
