# Experimental Context: Export Verification

This file is a handoff for future AI sessions. It records an experimental export-verification direction for CutScript. The verifier is currently modular and should not be treated as part of product export behavior unless code explicitly wires it into `/export`.

## Why This Exists

CutScript exports edited video by mapping transcript word-index ranges to source time ranges, refining those boundaries with AcousticMap, and rendering with FFmpeg. Recent EOF bugs showed that a good forward boundary algorithm is not enough. The rendered output can still be wrong because of:

- raw WhisperX word-end timestamps that miss acoustic coda tails
- single-segment fast exports bypassing refinement
- source audio duration being longer than source video duration
- post-processing such as `loudnorm` suppressing low-energy EOF speech tails
- short exported clips decoding differently from the original when re-transcribed

The experimental idea is a closed-loop verifier:

1. The editor chooses source ranges to keep.
2. The exporter renders the output.
3. A verifier compares the rendered output back against the intended source ranges.
4. If the output differs suspiciously, the app can flag it or eventually try bounded correction.

This is the "tune the television" model: we know what the output should preserve, so we compare the rendered result to the original source evidence.

## What Was Fixed Before This Experiment

These fixes are implemented separately from the verifier:

- `/export` accepts `force_refine`.
- `AIPanel.handleExportClip` sends `force_refine: true`.
- `ExportDialog` sends `force_refine: hasCuts`.
- The backend infers refinement when `words` are present and a single segment does not cover the full source.
- A true full-source single-segment fast export can still stream-copy.
- Single-segment re-encode skips `loudnorm`.
- Re-encode also skips `loudnorm` when the final segment reaches source audio EOF. Testing showed `loudnorm` could collapse the c8 EOF word from `freedom` to `free`.

The important lesson: EOF preservation is not only boundary placement. Post-processing can damage the coda after the boundary is correct.

## Current Standalone Probe

There is a standalone probe:

```bash
/Users/kat/.cutscript-venv/bin/python tests/waveform_analysis/export_tail_probe.py
```

It is intentionally not integrated into `/export`.

Current behavior:

- decode source and cuts to mono PCM
- compare the source EOF tail to each cut EOF tail
- compute:
  - normalized waveform cosine
  - RMS envelope cosine
  - log-mel spectrogram cosine
  - RMS ratio
- classify as `ok`, `review`, or `suspicious`

## Probe Results On c8

Default command:

```bash
/Users/kat/.cutscript-venv/bin/python tests/waveform_analysis/export_tail_probe.py
```

With a `1.2s` EOF tail window on `tests/video/c8.mp4`, the probe separated the old clipped exports from the fixed exports:

```text
source: tests/video/c8.mp4 duration=60.075s
tail window: 1.200s, sr=16000

cut                                             dur     wave      env   logmel      rms verdict
------------------------------------------------------------------------------------------------
tests/video/c8_edited2.mp4                   33.899    0.199    0.746    0.914    0.774 review
tests/video/c8_edited2_fixed.mp4             34.005   -0.010    0.915    0.974    0.987 ok
tests/video/c8_edited3.mp4                   39.509    0.129    0.744    0.909    0.785 review
tests/video/c8_edited3_fixed.mp4             39.616   -0.268    0.997    1.000    1.001 ok
```

With a shorter `0.45s` window, old and fixed exports both looked `ok`. That means the useful signal is not isolated to the final phoneme/coda. The verifier needs enough context before EOF, roughly `1.0-1.5s`, to catch continuity and tail-shape differences.

## What We Learned From c8

`tests/video/c8.mp4` ends with the phrase:

```text
Look guys, you don't have no freedom.
```

Important observed behavior:

- The original app transcription says `freedom.` because `transcription.py` tail rescue repairs a full-file decode from `free` to `freedom`.
- A raw WAV cut from the same source range transcribes as `freedom`.
- An MP4 trim without `loudnorm` transcribes as `freedom`.
- The same trim with `loudnorm` transcribes as `free`.
- The old `c8_edited3.mp4` app-transcribed as `you don't have no friends`.
- The regenerated `c8_edited3_fixed.mp4` app-transcribed as `look guys, you don't have no freedom.`

So the verifier should compare source audio to output audio, not rely on Whisper as the primary truth. Whisper is useful as a diagnostic, but short exported clips can decode differently even when the audio is correct.

## Research-Informed Design Shift

The current probe is useful, but it is still too naive because plain cosine assumes the source and output tails are already aligned.

The better model is:

```text
alignment first, similarity second, verdict last
```

Direct sample/frame cosine can be fragile when the export introduces:

- small codec delay
- AAC padding
- tiny trim drift
- fade shape differences
- resampling differences
- a few milliseconds of truncation or added silence

Future versions should find the best small alignment offset before scoring.

## Recommended Next Probe Algorithm

Do not integrate yet. First improve the standalone probe.

### Step 1: Decode

Decode source and output to mono PCM at a fixed sample rate.

For speech verification, `sr=16000` is probably enough. For exact export timing, `sr=48000` may be useful. Keep the probe configurable.

### Step 2: Extract Tail Windows

Use a contextual EOF window:

```text
tail_window = 1.2s initially
```

Also extract a boundary micro-window:

```text
boundary_window = 150-300ms
```

The contextual window catches phrase-level continuity. The boundary window targets EOF coda damage, abrupt fades, and clicks.

### Step 3: Small-Lag Alignment

Before scoring, align the output tail against the source tail within a small lag window:

```text
lag_window = ±100ms or ±150ms
```

Preferred alignment feature:

- RMS envelope, or
- log-mel frame energy

Avoid raw waveform as the primary alignment driver. It is too phase-sensitive.

Possible simple algorithm:

```text
for lag in candidate_lags:
    crop overlapping source/output feature frames
    score envelope cosine or log-mel cosine
choose lag with best score
```

Record the selected lag as a diagnostic:

```text
best_lag_ms
```

Large lag may itself be a `review` signal.

### Step 4: Score Aligned Tails

After alignment, compute:

- aligned RMS envelope cosine
- aligned log-mel cosine
- waveform cosine as a secondary/debug metric
- RMS ratio as a loudness sanity check
- final-boundary envelope score over last `150-300ms`
- final-boundary log-mel score over last `150-300ms`

Treat waveform cosine as diagnostic, not primary. Re-encoding and phase differences can reduce waveform similarity even when perceptual similarity is good.

### Step 5: Verdict

Early placeholder thresholds:

```text
ok:
  aligned logmel >= 0.90
  aligned envelope >= 0.85
  rms ratio between 0.85 and 1.15
  boundary check passes

review:
  aligned logmel >= 0.80
  aligned envelope >= 0.70
  rms ratio between 0.65 and 1.35

suspicious:
  below those thresholds
  or boundary drop/click/discontinuity detected
```

These thresholds are not validated broadly. They must be tested on more examples before product use.

## Boundary-Focused Checks To Add

The verifier should include checks designed for edit/export artifacts, not just global tail similarity.

Possible EOF checks:

- final `150-300ms` RMS envelope shape
- final `150-300ms` log-mel similarity
- final-band energy ratio for fricative-heavy tails
- abrupt drop detector: source still has energy but output collapses
- spectral-flux comparison over the final frames

Possible splice checks for non-EOF boundaries:

- click/pop detector using waveform derivative around splice
- spectral discontinuity detector before/after splice
- start attack preservation for the first kept word after a deletion
- coda preservation for the last kept word before a deletion

## Modular Implementation Plan

Goal: build the verifier as a replaceable standalone subsystem first. It should be useful from the command line, produce structured diagnostics, and remain disconnected from `/export` until the metrics have enough evidence behind them.

Primary rule:

```text
probe first, report second, product behavior last
```

If any stage fails, the fallback should be to keep the previous stage working rather than partially wiring the verifier into export.

### Design Constraints

- Keep `/export` behavior unchanged until a later explicit integration step.
- Keep the current `tests/waveform_analysis/export_tail_probe.py` command working as the compatibility entry point.
- Make every algorithmic layer independently swappable: decode, window extraction, alignment, scoring, verdicts, and reporting.
- Prefer structured return objects over print-only output so tests and future backend integration can inspect the same data.
- Treat verifier failures as `verification_error`, not export failures.
- Fail open in product paths: a bad verifier result can warn, but must not block or mutate an export until an explicit correction phase exists.

### Proposed Standalone Package Shape

Start under `tests/waveform_analysis/export_verifier/` so it is clearly experimental:

```text
tests/waveform_analysis/export_verifier/
  __init__.py
  models.py
  decode.py
  windows.py
  features.py
  alignment.py
  scoring.py
  verdicts.py
  report.py
  cli.py
```

Keep `tests/waveform_analysis/export_tail_probe.py` as a thin wrapper around `export_verifier.cli`. That preserves the current probe command while allowing internals to become modular.

If the verifier later becomes product code, promote the package to:

```text
backend/services/export_verifier/
```

Do not import test modules from backend production code.

### Module Responsibilities

`models.py`

- `VerifierConfig`: sample rate, tail window, boundary window, lag window, feature settings, threshold profile.
- `MediaAudio`: path, PCM array, sample rate, duration.
- `WindowPair`: source/output windows plus their source/output time ranges.
- `AlignmentResult`: best lag, score used for alignment, cropped frame ranges.
- `ScoreSet`: waveform, envelope, log-mel, RMS ratio, boundary scores, optional flags.
- `VerifierResult`: input paths, config, scores, verdict, reasons, diagnostics.

`decode.py`

- Own all FFmpeg and soundfile decoding.
- Decode to mono float PCM at configurable `sr`.
- Return explicit errors that the CLI can render as `verification_error`.
- Keep FFmpeg command construction isolated so product integration can later reuse or replace it.

`windows.py`

- Extract EOF tail windows for current probe behavior.
- Later extract source/output segment windows from segment maps.
- Pad short windows deterministically and record when padding occurred.
- Keep time-range math here instead of spreading it through scoring.

`features.py`

- RMS envelope.
- log-mel spectrogram.
- waveform normalization helpers.
- Later optional MFCC.
- No verdict logic in this module.

`alignment.py`

- Implement small-lag alignment over frame features.
- Use envelope or log-mel energy as the primary alignment feature.
- Return `best_lag_ms`, `best_alignment_score`, and the cropped overlap used by scoring.
- Keep lag direction documented:

```text
positive lag = output appears later than source
negative lag = output appears earlier than source
```

`scoring.py`

- Compute aligned envelope cosine.
- Compute aligned log-mel cosine.
- Compute waveform cosine as diagnostic only.
- Compute RMS ratio.
- Compute boundary-window scores.
- Emit named flags such as `large_lag`, `low_tail_similarity`, `boundary_energy_drop`, or `rms_ratio_out_of_range`.

`verdicts.py`

- Convert scores and flags into `ok`, `review`, `suspicious`, or `verification_error`.
- Hold threshold profiles in one place.
- Start with `c8_eof_v0` thresholds rather than pretending they are universal.

`report.py`

- Render the current table format.
- Render JSON for tests and future backend use.
- Include enough diagnostics to explain a verdict without rerunning the probe.

`cli.py`

- Parse args.
- Decode source once.
- Run one or more cuts.
- Print table by default.
- Support `--json`.
- Support config flags: `--sr`, `--tail-window`, `--boundary-window`, `--lag-window`, `--threshold-profile`.

### Implementation Phases

#### Phase 0: Baseline Lock

Before refactoring, capture the current probe behavior:

```bash
/Users/kat/.cutscript-venv/bin/python tests/waveform_analysis/export_tail_probe.py
```

Add a lightweight smoke test that runs only when the c8 fixture files exist. It should assert the broad classification shape, not exact floating-point values:

```text
c8_edited2.mp4        => review or suspicious
c8_edited2_fixed.mp4  => ok
c8_edited3.mp4        => review or suspicious
c8_edited3_fixed.mp4  => ok
```

Failure fallback: keep the existing script unchanged and only add the smoke test after the baseline command is stable.

#### Phase 1: Extract Modules Without Behavior Change

Move existing functions into `decode.py`, `features.py`, `windows.py`, `scoring.py`, and `verdicts.py`.

Expected outcome:

- Same default command.
- Same table columns.
- Same rough verdicts on c8.
- No alignment yet.

Failure fallback: revert only the wrapper/module split and keep the current single-file probe.

#### Phase 2: Add Structured Results

Introduce dataclasses in `models.py` and make the CLI render from `VerifierResult`.

Expected outcome:

- Table output remains human-readable.
- `--json` produces machine-readable diagnostics.
- Tests can inspect `verdict`, `scores`, and `reasons` without parsing stdout.

Failure fallback: keep dataclasses internally but disable `--json` until the schema stabilizes.

#### Phase 3: Add Small-Lag Alignment

Add `alignment.py` and score aligned windows.

Implementation detail:

```text
1. compute alignment feature frames for source and output
2. scan candidate lags across ±lag_window
3. crop overlapping frames for each lag
4. choose lag with best envelope/log-mel score
5. compute final metrics on the selected overlap
```

Diagnostics to add:

- `best_lag_ms`
- `alignment_score`
- `alignment_feature`
- `overlap_duration_s`

Review signal:

```text
abs(best_lag_ms) > 75ms initially means review
abs(best_lag_ms) > 150ms initially means suspicious
```

Failure fallback: expose `--alignment none|small_lag` and keep `none` available. If alignment makes c8 worse, default back to `none` while retaining the module for experiments.

#### Phase 4: Add Boundary Checks

Add EOF boundary scoring over the final `150-300ms` after alignment.

Start with:

- boundary envelope cosine
- boundary log-mel cosine
- boundary RMS ratio
- source-active/output-collapsed detector

Do not add click/pop splice checks in this phase. EOF preservation is the first target.

Failure fallback: boundary checks contribute `reasons` only, not verdict changes, until they separate known good/bad examples.

#### Phase 5: Expand Fixtures

Add more exports before changing product behavior:

- c8 old bad exports
- c8 fixed exports
- full-source single-segment export
- single-segment non-full export
- multi-segment export with an internal splice
- export that does not touch EOF
- intentionally loudnorm-damaged EOF clip, if easy to generate

Expected outcome:

- EOF verifier catches the known EOF failures.
- Full-source and fixed exports stay `ok`.
- Non-EOF exports are not misclassified by EOF-only assumptions.

Failure fallback: narrow the verifier declaration to `eof_tail_verifier` and do not generalize to splice verification yet.

#### Phase 6: Segment-Map Prototype

Prototype a data contract without wiring it into `/export` responses:

```json
{
  "source_path": "tests/video/c8.mp4",
  "output_path": "tests/video/c8_edited3_fixed.mp4",
  "segments": [
    {
      "source_start": 20.457,
      "source_end": 60.074667,
      "output_start": 0.0,
      "output_end": 39.617667
    }
  ]
}
```

Add CLI support:

```bash
python tests/waveform_analysis/export_tail_probe.py --map path/to/map.json
```

Expected outcome:

- EOF comparison still works.
- Segment-local comparison becomes possible.
- Future splice checks have the source/output timing data they need.

Failure fallback: keep only direct source/output EOF mode and postpone segment maps.

#### Phase 7: Optional Backend Report Hook

Only after standalone validation, add an optional backend service wrapper. It should be disabled by default and must not alter export output.

Possible shape:

```text
POST /export
  render as today
  if experimental verifier flag enabled:
      run verifier
      attach verifier report to response/log
  return output either way
```

Required guardrails:

- environment/config flag required
- timeout required
- verifier errors logged as diagnostics only
- no auto-correction
- no deletion or overwrite of rendered output

Failure fallback: remove the route-level hook and keep the standalone CLI.

#### Phase 8: Bounded Correction Experiment

Do not start this until the report-only verifier is reliable.

First correction candidate:

```text
if final segment touches source EOF and verifier reports missing EOF tail:
    extend final refined segment to source audio duration
    skip EOF loudnorm/fade
    re-render once
    verify once more
```

Rules:

- one retry maximum per correction type
- write correction reason into report
- preserve the original failed report for comparison
- never loop on metrics

Failure fallback: disable correction and keep warning/report behavior.

### Test Strategy

Use three layers of tests.

Unit tests:

- cosine handles empty/silent arrays without crashing
- tail window padding is deterministic
- lag alignment recovers synthetic offsets
- RMS ratio and boundary-drop flags behave on synthetic signals

Fixture smoke tests:

- c8 known-bad exports classify as `review` or `suspicious`
- c8 fixed exports classify as `ok`
- tests skip cleanly if media fixtures are missing

CLI tests:

- default command exits `0`
- `--json` is valid JSON
- decode errors exit nonzero with a clear message

Do not assert exact metric values unless the decoder, librosa version, sample rate, and fixture files are pinned. Prefer threshold bands and verdict categories.

### Risk Register

Risk: alignment overfits to c8.

Mitigation: keep threshold profile named `c8_eof_v0`, add fixtures before using it broadly.

Risk: verifier disagrees with human perception.

Mitigation: report multiple metrics and reason flags; do not let one metric dominate until validated.

Risk: AAC padding or codec delay creates false positives.

Mitigation: score after small-lag alignment and record overlap duration.

Risk: low-energy speech tails look like silence.

Mitigation: compare source-active vs output-active energy instead of absolute output energy only.

Risk: verifier slows export.

Mitigation: standalone first, optional backend flag later, decode source once, add timeout before any product hook.

Risk: product path starts depending on test code.

Mitigation: keep standalone package experimental; promote to `backend/services/export_verifier/` before backend import.

### Immediate Next Implementation Task

The next concrete implementation should be Phase 0 and Phase 1 only:

1. Add fixture-aware smoke coverage for the current probe.
2. Split the script into modules without changing behavior.
3. Keep `export_tail_probe.py` as the public command.
4. Re-run the c8 probe and compare verdict categories against the current baseline.

Do not add alignment, boundary checks, JSON schema, or backend hooks in the first implementation pass. Those become easier and safer after the module split proves it did not change behavior.

## DTW And MFCC: Later, Not First

Dynamic Time Warping is relevant because it compares sequences with timing mismatch. It may be useful for:

- RMS envelope sequences
- log-mel frames
- MFCC sequences

MFCC comparison is also relevant because it summarizes spectral shape compactly and can be paired with cosine, Euclidean distance, or DTW.

But do not jump to DTW first. The recommended order is:

1. small lag alignment
2. aligned envelope/log-mel scoring
3. boundary-focused checks
4. MFCC or DTW only if simple alignment misses real cases

Reason: export errors are usually small shifts, truncation, codec delay, fades, or dynamics changes. Small-lag alignment is cheaper and easier to debug than DTW.

## Possible Future Export Integration

If this becomes part of export, the backend should return source/output segment maps:

```json
{
  "refined_segments": [
    { "start": 20.457, "end": 60.074667 }
  ],
  "output_segments": [
    { "start": 0.0, "end": 39.617667 }
  ]
}
```

The verifier can then compare:

```text
source[refined_start:refined_end]
vs
output[output_start:output_end]
```

For EOF-only testing, comparing source EOF tail to output EOF tail is enough. For general export verification, segment maps are required.

## Potential Auto-Correction Loop

Do not implement yet, but the eventual structure could be:

```text
render export
verify source-vs-output
if ok:
    return output
if missing EOF tail:
    extend final segment to source audio duration
    disable EOF loudnorm/final fade
    re-render once
if start attack clipped:
    move segment start earlier by 20-50ms, within safe cap
    re-render once
if still bad:
    return output with warning/report
```

Corrections must be bounded. The verifier should not silently chase metrics forever.

## Promoted Export Behavior

The best-performing export fix is now promoted into the main backend export path in `backend/services/video_editor.py`.

Implemented production behavior:

- re-encode export builds separate video/audio trim ranges
- internal split ranges keep the same nominal video/audio timestamps for sync
- final EOF range is audio-first:
  - video clamps to source video EOF
  - audio extends to source audio EOF
  - final video frame is padded with `tpad=stop_mode=clone`
  - padding includes two frame durations of safety so video outlasts audio in players
- EOF exports skip `loudnorm`
- EOF exports get only a tiny `12ms` audio fade-out to smooth the transition into AAC/end padding
- normal internal cut boundaries keep the existing `12ms` audio fades
- A/V duration differences at EOF are logged for diagnostics

Important distinction:

```text
EOF video padding is safe because there is no next video segment.
Internal video padding is not used because it would introduce micro-freezes before later segments.
```

The standalone verifier remains experimental. It is useful for diagnostics and visual reports, but it is not part of product export behavior.

## Research Directions Mentioned By User

The user found references supporting this direction. The key ideas:

- cosine similarity is common for normalized feature vectors
- mel/log-mel features are standard for time-frequency matching
- waveform and mel-spectrogram features are complementary
- direct cosine has temporal-sensitivity limits
- DTW is relevant for timing mismatch
- MFCC and spectral summaries are common in robust audio comparison
- boundary/discontinuity metrics matter for edit artifacts

Links provided by user:

- https://ieeexplore.ieee.org/document/10848704/
- https://researchers.mq.edu.au/files/460573993/446065851.pdf
- https://dcase.community/documents/challenge2021/technical_reports/DCASE2021_Fedorishin_97_t1.pdf
- https://pmc.ncbi.nlm.nih.gov/articles/PMC3745477/
- https://www.boxentriq.com/steganography/audio-spectrogram
- https://archives.ismir.net/ismir2015/paper/000094.pdf
- https://ieeexplore.ieee.org/document/9214302/
- https://www.irjet.net/archives/V12/i12/IRJET-V12I1269.pdf
- https://www.sciencedirect.com/science/article/abs/pii/S0957417422019819
- https://towardsdatascience.com/calculating-audio-song-similarity-using-siamese-neural-networks-62730e8f3e3d/
- https://www.testdevlab.com/blog/audio-comparison-using-mfcc-and-dtw
- https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/el.2013.3554

Some references were partially checked in-session:

- The DCASE report supports waveform/mel-spectrogram complementarity, though in acoustic scene classification rather than export verification.
- The ISMIR DTW paper supports alignment-aware comparison for timing variation.
- The TestDevLab MFCC/DTW article is practical rather than primary research, but maps well to the proposed probe.
- The PMC link was not accessible in-session due to a reCAPTCHA page.

Do not overclaim that these sources directly validate CutScript thresholds. They support the architecture, not final pass/fail numbers.
