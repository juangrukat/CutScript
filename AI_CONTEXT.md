# CutScript — AI Session Context

Onboarding notes for a new AI session. Describes the architecture, the acoustic-analysis pipeline, and the most recent changes.

---

## What CutScript Is

A local-first, Descript-style text-based video editor. The user transcribes a video with WhisperX (word-level timestamps), deletes words from the transcript, and exports a cut video where those words have been removed. Stack: Electron + React frontend, FastAPI Python backend, WhisperX + librosa + FFmpeg.

---

## The Cut Pipeline at a Glance

```
open video → WhisperX transcribe → /analyze builds AcousticMap (cached)
                                       │
                                       ▼
delete words in UI  →  getKeepSegments()  →  /export
                                       │
                                       ▼
  extract PCM WAV → _refine_from_map (AcousticMap-guided)  → ffmpeg trim+concat+loudnorm
                   (or _refine_segments fallback if map missing)
```

Two independent caches on disk, both cleanable from the Settings panel:

- `~/.obs_transcriber_cache/<hash>_<model>_transcribe_wx_*.json` — WhisperX results
- `~/.cutscript_spectral_cache/<hash>.json` — AcousticMap fingerprints

File hash = md5(path + size + mtime).

---

## AcousticMap (the current refinement primary)

Built at ingest time by `POST /analyze` and consumed at export time by `_refine_from_map`. Lives in `backend/services/audio_analyzer.py`.

**Per-word `WordFingerprint`:**
- `ws`, `we` — WhisperX start/end
- `as_`, `ae` — **acoustic** start/end (these are what the refiner uses)
- `onset`, `coda` — phoneme class (`fricative` / `stop` / `nasal` / `vowel` / `approximant`) inferred from spelling
- `peak_rms`, `peak_fric` — broadband and 2–8 kHz fricative-band peaks inside the word
- `dips` — intraword RMS dips (tracks word-internal silences so we don't split on them)

**Coda-specific decay policies** control how far `ae` extends past `we`:
- fricative (`sh`, `s`, `f`, `th`) — band+broadband both-below (captures the /ʃ/ tail in "Spanish")
- nasal (`n`, `m`) — −18 dB decay
- stop (`p`, `t`, `k`) — −20 dB decay
- vowel / approximant — −25 dB decay

**`_refine_from_map` (backend/routers/export.py)**:
- Matches each segment endpoint to a WordFingerprint within 100 ms
- Uses `word.as_` / `word.ae` as the refined boundary (clamped by the neighbouring segment)
- Zero-crossing snap
- **Clamps to audio duration** — ffmpeg cannot see a segment that extends past EOF
- **Drops any segment shorter than 10 ms** — empty atrim silently truncates concat output

The legacy `_refine_segments` is kept as a fallback (used if the map is missing *and* on-demand `analyze_file` fails). Same clamp/drop safety applied.

---

## Recent Changes

### "Is there a problem?" dropped from exports — fixed

Symptom: `b_edited1.mp4` was 7.5 s and ended after "Okay, time to go." Last kept segment (words 76–79) was missing.

Reproduced the full pipeline end-to-end in `tests/waveform_analysis/phase5_last_segment.py` — both refiners produced the correct 11.6 s output with all 7 segments. Pipeline was fine. Bug was in the UI.

**Root cause:** `getKeepSegments` in `frontend/src/store/editorStore.ts` seeded its last-segment padding reducer with `duration`:
```ts
.reduce((min, r) => Math.min(min, r.start), duration);
lastSeg.end = Math.min(nextDeletedStart, lastSeg.end + 1.5);
```
When video metadata hadn't loaded yet, `duration === 0` → `nextDeletedStart = 0` → `lastSeg.end = 0`. The last segment silently became `(46.7, 0)`, which ffmpeg's `atrim` treated as empty and concat dropped it.

**Fix:** seed with `Infinity` and apply a duration cap only when duration is known:
```ts
.reduce((min, r) => Math.min(min, r.start), Infinity);
const durationCap = duration > 0 ? duration : Infinity;
lastSeg.end = Math.min(nextDeletedStart, durationCap, lastSeg.end + 1.5);
```

**Defensive backend guards** added alongside:
- `_refine_from_map` and `_refine_segments` clamp every boundary to `[0, audio_dur]`
- Segments shorter than 10 ms are dropped with a warning before reaching ffmpeg

### AcousticMap pipeline shipped

Full five-phase plan completed. Key files:

- `backend/services/audio_analyzer.py` — new. `AcousticMap`, `WordFingerprint`, `analyze_file`, `_fricative_band_rms`, `_classify_onset/coda`, `_analyze_word`, persistence helpers.
- `backend/routers/analysis.py` — new. `POST /analyze { file_path, words } → { status, file_hash, words, cached }`.
- `backend/routers/cache.py` — new. `GET /cache/sizes`, `POST /cache/clear/{transcripts|spectral}`.
- `backend/routers/export.py` — added `_refine_from_map`. `export_video` now: `load_acoustic_map → analyze_file on demand if missing → _refine_from_map → legacy fallback on error`.
- `frontend/src/App.tsx` — posts to `/analyze` after transcription completes (progress 98% → 100%, non-fatal on failure).
- `frontend/src/components/ExportDialog.tsx` — always sends `words` so the backend can rebuild the map on demand if cache was cleared between transcribe and export.
- `frontend/src/components/SettingsPanel.tsx` — new `CacheManager` component with size + file-count rows for both caches and independent Clear buttons.

Validated against the "Spanish?" truncation bug that originally motivated this work: Phase 0 ground truth measured the /ʃ/ tail extending ~310 ms past WhisperX's `we`. `tests/waveform_analysis/phase5_validate.py` reports `coda extension: 288 ms — PASS`.

---

## Legacy Refiner (fallback only)

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

Helpers (all in `export.py`): `_gap_has_speech`, `_snap_zc`, `_sample_has_speech`, `_find_onset_before`, `_find_word_end`, `_advance_past_silence`.

---

## Known Issues / Design Notes

- **`duration === 0` case** is now handled, but relies on the video element emitting `onLoadedMetadata` before the user exports. The backend clamps provide a second line of defence.
- **AcousticMap hash collision** isn't protected — if two different files somehow map to the same hash (path+size+mtime md5), the cached map would be wrong. Unlikely but possible.
- **On-demand analyze** at export time can add 2–5 s for a long video if the spectral cache was cleared. Non-fatal — falls back to legacy on any error.
- **`_advance_past_silence`'s 250 ms threshold** may need tuning for languages with unusually long voiceless onset clusters.
- **WhisperX re-transcription drift**: when re-transcribing a cut file to verify, word timestamps drift 0.5–1 s from actual positions. This is a WhisperX forced-alignment artifact on short concatenated audio, not an export bug.

---

## Files Changed in Recent Sessions

### "Is there a problem?" fix + README/AI_CONTEXT sync

| File | Change |
|---|---|
| `frontend/src/store/editorStore.ts` | Fixed `getKeepSegments` last-seg padding collapse when `duration === 0` |
| `backend/routers/export.py` | Clamp refined boundaries to audio duration; drop <10 ms segments in both refiners |
| `README.md` | Added AcousticMap rows and `/analyze`, `/cache/*` endpoints |
| `AI_CONTEXT.md` | This rewrite |

### AcousticMap pipeline

| File | Change |
|---|---|
| `backend/services/audio_analyzer.py` | New — AcousticMap, WordFingerprint, analyze_file, persistence |
| `backend/routers/analysis.py` | New — `POST /analyze` |
| `backend/routers/cache.py` | New — size + per-kind clear endpoints |
| `backend/routers/export.py` | Added `_refine_from_map`; on-demand rebuild; legacy fallback |
| `backend/main.py` | Mounted `analysis` and `cache` routers |
| `frontend/src/App.tsx` | Post-transcription `/analyze` call |
| `frontend/src/components/ExportDialog.tsx` | Always send `words` |
| `frontend/src/components/SettingsPanel.tsx` | Added `CacheManager` UI |
| `tests/waveform_analysis/phase5_validate.py` | New — validates /ʃ/ tail preservation |
| `tests/waveform_analysis/phase5_last_segment.py` | New — 7-segment end-to-end reproduction |
| `tests/waveform_analysis/phase5_legacy.py` | New — same scenario against legacy refiner |

### Earlier (hissing + onset clusters)

| File | Change |
|---|---|
| `backend/routers/export.py` | `_advance_past_silence`; `_find_onset_before`; `_find_word_end`; `_gap_has_speech` |
| `backend/services/boundary_refiner.py` | Copied from `rosa/` |
| `backend/services/transcription.py` | Removed hallucination-prone `initial_prompt`; added `_clip_to_duration` |

---

## Project Structure

```
cutscript/
├── electron/
├── frontend/src/
│   ├── App.tsx                          # open-video modal + /analyze kickoff
│   ├── components/
│   │   ├── ExportDialog.tsx             # sends words + deleted_indices
│   │   ├── SettingsPanel.tsx            # CacheManager lives here
│   │   └── TranscriptEditor.tsx
│   ├── store/
│   │   └── editorStore.ts               # getKeepSegments — duration=0 guard here
│   └── hooks/
├── backend/
│   ├── main.py                          # mounts analysis + cache routers
│   ├── routers/
│   │   ├── export.py                    # _refine_from_map (primary) + legacy fallback
│   │   ├── analysis.py                  # POST /analyze
│   │   └── cache.py                     # GET /cache/sizes, POST /cache/clear/{kind}
│   ├── services/
│   │   ├── audio_analyzer.py            # AcousticMap, WordFingerprint, analyze_file
│   │   ├── boundary_refiner.py
│   │   ├── transcription.py
│   │   └── video_editor.py              # ffmpeg trim+concat+loudnorm
│   └── utils/
│       ├── audio_processing.py
│       └── cache.py                     # get_file_hash
├── rosa/
└── tests/
    └── waveform_analysis/               # phase0–phase5 diagnostic + validation scripts
```
