"""Cache management endpoints — transcripts and spectral maps."""

import logging

from fastapi import APIRouter
from pydantic import BaseModel

from utils.cache import clear_cache as clear_transcript_cache, get_cache_size as get_transcript_cache_size
from services.audio_analyzer import clear_spectral_cache, get_spectral_cache_size

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/cache")


class CacheSizes(BaseModel):
    transcripts_bytes: int
    transcripts_files: int
    spectral_bytes: int
    spectral_files: int


class ClearResult(BaseModel):
    status: str
    deleted: int


@router.get("/sizes", response_model=CacheSizes)
async def sizes():
    t_size, t_count = get_transcript_cache_size()
    s_size, s_count = get_spectral_cache_size()
    return CacheSizes(
        transcripts_bytes=t_size,
        transcripts_files=t_count,
        spectral_bytes=s_size,
        spectral_files=s_count,
    )


@router.post("/clear/transcripts", response_model=ClearResult)
async def clear_transcripts():
    n = clear_transcript_cache()
    logger.info(f"Cleared {n} transcript cache files")
    return ClearResult(status="ok", deleted=n)


@router.post("/clear/spectral", response_model=ClearResult)
async def clear_spectral():
    n = clear_spectral_cache()
    logger.info(f"Cleared {n} spectral cache files")
    return ClearResult(status="ok", deleted=n)
