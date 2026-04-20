"""
Pydantic models and validators for AI-produced edit plans.

All AI features converge on the same "truth": word-index ranges in the
transcript. Converting to DeletedRange in the editor store and letting
getKeepSegments() feed /export is what gives us seamless cuts with
AcousticMap refinement. These validators clamp/dedupe/sanitise whatever
the model returns so the rest of the pipeline never sees malformed data.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filler detection
# ---------------------------------------------------------------------------


class FillerWord(BaseModel):
    index: int
    word: str
    reason: str
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class FillerReport(BaseModel):
    language: str = "auto"
    fillerWords: List[FillerWord] = Field(default_factory=list)
    wordIndices: List[int] = Field(default_factory=list)
    needs_review: bool = False
    warnings: List[str] = Field(default_factory=list)


def validate_filler_report(report: FillerReport, word_count: int, min_confidence: float = 0.5) -> FillerReport:
    """Clamp indices, rebuild wordIndices from fillerWords, cap by confidence."""
    valid: List[FillerWord] = []
    seen: set[int] = set()
    dropped = 0
    for fw in report.fillerWords:
        if fw.index < 0 or fw.index >= word_count:
            dropped += 1
            continue
        if fw.index in seen:
            continue
        if fw.confidence < min_confidence:
            dropped += 1
            continue
        seen.add(fw.index)
        valid.append(fw)
    valid.sort(key=lambda f: f.index)

    warnings = list(report.warnings)
    if dropped:
        warnings.append(f"Dropped {dropped} filler suggestions (out of range or low confidence).")

    # Safety cap: if the model wants to delete >40% of the transcript,
    # flag for review instead of silently applying.
    pct = len(valid) / max(1, word_count)
    needs_review = report.needs_review or pct > 0.40
    if pct > 0.40:
        warnings.append(f"AI flagged {pct:.0%} of words as filler — review recommended.")

    return FillerReport(
        language=report.language,
        fillerWords=valid,
        wordIndices=[fw.index for fw in valid],
        needs_review=needs_review,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Clip suggestions
# ---------------------------------------------------------------------------


class ClipSuggestion(BaseModel):
    title: str
    startWordIndex: int
    endWordIndex: int
    startTime: float
    endTime: float
    reason: str
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    target_duration: int = 60


class ClipPlan(BaseModel):
    clips: List[ClipSuggestion] = Field(default_factory=list)
    rationale: str = ""
    needs_review: bool = False
    warnings: List[str] = Field(default_factory=list)


def validate_clip_plan(
    plan: ClipPlan,
    words: List[dict],
    audio_duration: Optional[float] = None,
    min_confidence: float = 0.4,
) -> ClipPlan:
    """
    Normalise clip times against the transcript's ground truth.

    The AI is allowed to be loose — we rebuild startTime/endTime from the
    word indices so the exported clip always aligns with real word
    boundaries. AcousticMap refinement at export time handles the fine-grain
    boundary extension (word tails, onset clusters, etc).
    """
    if not words:
        return ClipPlan(clips=[], rationale=plan.rationale, needs_review=True,
                        warnings=[*plan.warnings, "No words available — cannot build clip plan."])

    word_count = len(words)
    valid: List[ClipSuggestion] = []
    warnings = list(plan.warnings)

    for c in plan.clips:
        s = max(0, min(word_count - 1, c.startWordIndex))
        e = max(0, min(word_count - 1, c.endWordIndex))
        if e < s:
            s, e = e, s
        if e - s < 2:  # single-word clip is almost always wrong
            warnings.append(f"Dropped clip '{c.title}': word range too small ({s}..{e}).")
            continue

        start_time = float(words[s].get("start", 0.0) or 0.0)
        end_time = float(words[e].get("end", 0.0) or 0.0)
        if audio_duration is not None:
            end_time = min(end_time, audio_duration)
        if end_time - start_time < 1.0:
            warnings.append(f"Dropped clip '{c.title}': duration too short ({end_time - start_time:.1f}s).")
            continue
        if c.confidence < min_confidence:
            warnings.append(f"Dropped clip '{c.title}': confidence {c.confidence:.2f} below threshold.")
            continue

        valid.append(ClipSuggestion(
            title=c.title.strip() or "Untitled clip",
            startWordIndex=s,
            endWordIndex=e,
            startTime=start_time,
            endTime=end_time,
            reason=c.reason,
            confidence=c.confidence,
            target_duration=c.target_duration,
        ))

    valid.sort(key=lambda x: x.startTime)
    needs_review = plan.needs_review or not valid
    return ClipPlan(clips=valid, rationale=plan.rationale, needs_review=needs_review, warnings=warnings)


# ---------------------------------------------------------------------------
# Focus modes
# ---------------------------------------------------------------------------


class FocusDeletion(BaseModel):
    startIndex: int
    endIndex: int
    reason: str
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)


class FocusPlan(BaseModel):
    mode: str
    deletions: List[FocusDeletion] = Field(default_factory=list)
    summary: str = ""
    needs_review: bool = False
    warnings: List[str] = Field(default_factory=list)


def validate_focus_plan(
    plan: FocusPlan,
    word_count: int,
    min_confidence: float = 0.45,
    max_deletion_pct: float = 0.80,
) -> FocusPlan:
    """
    Clamp ranges, merge overlaps, reject if too aggressive.

    Focus modes often propose deleting large chunks — we accept that, but
    hard-cap at 80% deleted. Above that the user is probably looking for a
    summary, not an edit.
    """
    if word_count <= 0:
        return FocusPlan(mode=plan.mode, deletions=[], summary=plan.summary, needs_review=True,
                         warnings=[*plan.warnings, "Empty transcript."])

    clamped: List[FocusDeletion] = []
    warnings = list(plan.warnings)
    dropped = 0

    for d in plan.deletions:
        s = max(0, min(word_count - 1, d.startIndex))
        e = max(0, min(word_count - 1, d.endIndex))
        if e < s:
            s, e = e, s
        if d.confidence < min_confidence:
            dropped += 1
            continue
        clamped.append(FocusDeletion(startIndex=s, endIndex=e, reason=d.reason, confidence=d.confidence))

    # Merge overlapping / adjacent ranges.
    clamped.sort(key=lambda x: x.startIndex)
    merged: List[FocusDeletion] = []
    for d in clamped:
        if merged and d.startIndex <= merged[-1].endIndex + 1:
            prev = merged[-1]
            new_end = max(prev.endIndex, d.endIndex)
            reason = prev.reason if len(prev.reason) >= len(d.reason) else d.reason
            confidence = max(prev.confidence, d.confidence)
            merged[-1] = FocusDeletion(
                startIndex=prev.startIndex, endIndex=new_end,
                reason=reason, confidence=confidence,
            )
        else:
            merged.append(d)

    deleted_words = sum(d.endIndex - d.startIndex + 1 for d in merged)
    pct = deleted_words / word_count

    needs_review = plan.needs_review or pct > 0.50
    if pct > max_deletion_pct:
        warnings.append(
            f"Focus plan would delete {pct:.0%} of the transcript — exceeds {max_deletion_pct:.0%} cap. "
            "Returning empty plan; try a less aggressive mode or narrower topic."
        )
        return FocusPlan(mode=plan.mode, deletions=[], summary=plan.summary,
                         needs_review=True, warnings=warnings)
    if pct > 0.50:
        warnings.append(f"Focus plan deletes {pct:.0%} of transcript — please review.")
    if dropped:
        warnings.append(f"Dropped {dropped} low-confidence suggestions.")

    return FocusPlan(mode=plan.mode, deletions=merged, summary=plan.summary,
                     needs_review=needs_review, warnings=warnings)
