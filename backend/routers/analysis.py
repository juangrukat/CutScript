"""Endpoint to build the acoustic map for a file after transcription."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services.audio_analyzer import analyze_file, load_acoustic_map

logger = logging.getLogger(__name__)
router = APIRouter()


class AnalyzeWord(BaseModel):
    word: str
    start: float
    end: float
    confidence: float = 0.0


class AnalyzeRequest(BaseModel):
    file_path: str
    words: List[AnalyzeWord]
    force: bool = False


class AnalyzeResponse(BaseModel):
    status: str
    file_hash: str
    words: int
    cached: bool


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Build (or load cached) AcousticMap for `file_path` using the supplied
    Whisper word list. Called from the frontend immediately after a
    successful transcription so the spectral fingerprints are ready by
    the time the user hits Export.
    """
    try:
        was_cached = load_acoustic_map(req.file_path) is not None and not req.force
        m = analyze_file(
            req.file_path,
            [w.model_dump() for w in req.words],
            force=req.force,
        )
        return AnalyzeResponse(
            status="ok",
            file_hash=m.file_hash,
            words=len(m.words),
            cached=was_cached,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Acoustic analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
