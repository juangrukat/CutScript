# CutScript Current Handoff

This file intentionally replaces the older broad architecture notes. The active work is about silence removal, smoother joins, EOF-safe export, and Studio Sound.

## Current Product Direction

CutScript is moving from word-only transcript cutting toward two kinds of editable removals:

- word cuts: transcript word-index ranges
- silence cuts: timestamp-only gaps derived from word timing

The goal is not just to remove material, but to make joins sound natural after speech or silence is removed.

## Silence Markers And Silence Cuts

The transcript still stores real words only. Silence is derived from gaps between adjacent word timestamps.

Implemented frontend behavior:

- `frontend/src/utils/silence.ts` derives silence ranges from `words`, `duration`, and a `0.5s` threshold.
- `TranscriptEditor` renders each pause inline as a marker like `... 0.7s`.
- Clicking an uncut pause marker cuts that silence.
- Clicking a cut pause marker restores it.
- The transcript header has `Cut N pauses` to remove all detected pauses.
- `DeletedRange.kind` can now be `words` or `silence`.
- Silence cuts use `wordIndices: []`.
- `shared/project-schema.json` accepts optional `kind`.

Export behavior:

- `editorStore.getKeepSegments()` still builds speech ranges from word deletions first.
- It then subtracts silence time ranges.
- The backend receives ordinary keep segments, so the existing export/refinement path remains in use.

Important next product decision:

- Full pause removal can sound rushed.
- Add a "shorten pauses" mode next: reduce pauses over a threshold to a target remainder such as `80-150ms`.
- This should become the default for natural speech pacing; full removal can remain an aggressive option.

## Join Quality And Adaptive Splice Direction

Current renderer behavior:

- Re-encode exports use PCM WAV extraction for sample-accurate audio trimming.
- Internal joins get short `12ms` audio fade-out/fade-in ramps.
- EOF exports are audio-first and skip `loudnorm` when the final range touches source EOF.

Reasoning:

- For word cuts, a true crossfade can smear deleted speech back into the output.
- For silence cuts, there is usually safe room tone on both sides, so an adaptive crossfade is more feasible.
- Adaptive splice mode should be conservative:
  - keep `12ms` micro-fades as baseline
  - allow `20-40ms` equal-power crossfades only when the deleted gap is silence or room tone
  - avoid internal video padding because it creates visible micro-freezes
  - do not use spectral/EQ matching as a default product behavior

Future adaptive splice inputs:

- derived silence ranges from the frontend
- gap energy from backend audio analysis
- local RMS/log-mel discontinuity around the join
- click risk at the exact boundary

## EOF Export And Verification

The important EOF fix is already in production export code:

- final segment that touches source EOF preserves audio to source audio EOF
- video clamps to source video EOF
- final video frame is padded only at EOF so audio can finish cleanly
- EOF exports skip `loudnorm`
- EOF gets only a tiny terminal fade

The standalone verifier remains experimental under `tests/waveform_analysis/export_verifier/`.

Verifier direction:

- compare source audio to rendered output, not Whisper re-transcription
- use aligned RMS envelope and log-mel similarity
- add small-lag alignment before scoring
- add EOF boundary checks over final `150-300ms`
- keep it report-only until validated on more fixtures

## Studio Sound / Enhance Audio

`requirements.txt` includes `deepfilternet>=0.5.0`, so DeepFilterNet is intended to be a standard dependency.

Bug found:

- DeepFilterNet was selected when installed, but its loader could not read MP4/AAC directly.
- Export caught the failure as non-fatal, so enhanced exports could be identical to non-enhanced exports.
- `tests/video/t1_edited.mp4` and `tests/video/t1A_edited.mp4` were byte-for-byte identical before the fix.

Fix implemented:

- `backend/services/audio_cleaner.py` now decodes any media input to a temporary mono WAV before DeepFilterNet.
- DeepFilterNet output is post-processed with:
  - `highpass=f=70`
  - `loudnorm=I=-16:TP=-1.5:LRA=11`
  - output sample rate constrained to `48000`
- FFmpeg fallback also uses denoise plus the same highpass/loudness/peak-control finish.

Measured on `t1_edited.mp4`:

- before enhancement: quiet-frame floor around `-26.9 dBFS`, SNR proxy around `9 dB`
- raw DeepFilter: quiet-frame floor around `-50.9 dBFS`, SNR proxy around `31 dB`, but too quiet and peak-unsafe
- finished Studio Sound chain: quiet-frame floor around `-47.6 dBFS`, SNR proxy around `30 dB`, integrated loudness around `-15.7 LUFS`, true peak `-1.5 dBTP`

Frontend warning:

- Export dialog now warns that Studio Sound changes audio.
- It denoises, converts to mono 48 kHz, re-levels loudness, and limits peaks.

Open caution:

- Studio Sound improves noise floor but can make joins more exposed because pauses/noise beds are cleaner.
- Cleaner joining is still needed even when DeepFilterNet is not used.
- Silence shortening and adaptive splice mixing should be developed independently from noise removal.

## Useful Test Files

- `tests/video/t1.mp4`: original source
- `tests/video/t1_edited.mp4`: silence-cut edit without enhancement
- `tests/video/t1A_edited.mp4`: regenerated enhanced edit for listening tests
- `tests/video/c8.mp4`: EOF-tail regression source
- `tests/video/c8_edited*_fixed.mp4`: EOF fixed examples

## Near-Term Next Steps

1. Add "shorten pauses to X ms" instead of only "cut all pauses".
2. Add backend/export metadata for silence cuts so adaptive splice mode can tell silence joins apart from word joins.
3. Prototype silence-only crossfade handles.
4. Add a verification metric for internal splice smoothness: local RMS gap, log-mel discontinuity, and click risk.
5. Keep Studio Sound optional because it intentionally changes audio character and loudness.
