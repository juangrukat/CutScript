"""AI feature endpoints: filler word detection, clip creation, focus modes, Ollama model listing."""

import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from services.ai_provider import (
    AIProvider,
    create_clip_suggestion,
    detect_filler_words,
    focus_transcript,
)

logger = logging.getLogger(__name__)
router = APIRouter()


ALLOWED_FOCUS_MODES = {"redundancy", "tighten", "topic", "qa_extract", "key_points"}


class WordInfo(BaseModel):
    index: int
    word: str
    start: Optional[float] = None
    end: Optional[float] = None


class FillerRequest(BaseModel):
    transcript: str
    words: List[WordInfo]
    provider: str = "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    custom_filler_words: Optional[str] = None


class ClipRequest(BaseModel):
    transcript: str
    words: List[WordInfo]
    provider: str = "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    target_duration: int = 60
    target_durations: Optional[List[int]] = None


class FocusRequest(BaseModel):
    transcript: str
    words: List[WordInfo]
    mode: str = Field(..., description="One of: redundancy, tighten, topic, qa_extract, key_points")
    topic: Optional[str] = None
    provider: str = "ollama"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@router.post("/ai/filler-removal")
async def filler_removal(req: FillerRequest):
    try:
        words_dicts = [w.model_dump() for w in req.words]
        return detect_filler_words(
            transcript=req.transcript,
            words=words_dicts,
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
            custom_filler_words=req.custom_filler_words,
        )
    except Exception as e:
        logger.error(f"Filler detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/create-clip")
async def create_clip(req: ClipRequest):
    try:
        words_dicts = [w.model_dump() for w in req.words]
        return create_clip_suggestion(
            transcript=req.transcript,
            words=words_dicts,
            target_duration=req.target_duration,
            target_durations=req.target_durations,
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
    except Exception as e:
        logger.error(f"Clip creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ai/focus")
async def focus(req: FocusRequest):
    if req.mode not in ALLOWED_FOCUS_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{req.mode}'. Allowed: {sorted(ALLOWED_FOCUS_MODES)}",
        )
    try:
        words_dicts = [w.model_dump() for w in req.words]
        return focus_transcript(
            transcript=req.transcript,
            words=words_dicts,
            mode=req.mode,
            topic=req.topic,
            provider=req.provider,
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
        )
    except Exception as e:
        logger.error(f"Focus plan failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ai/ollama-models")
async def ollama_models(base_url: str = "http://localhost:11434"):
    models = AIProvider.list_ollama_models(base_url)
    return {"models": models}
