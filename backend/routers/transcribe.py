"""Transcription endpoint using WhisperX."""

import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from services.transcription import transcribe_audio
from services.diarization import diarize_and_label

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3", "distil-large-v3"}


class TranscribeRequest(BaseModel):
    file_path: str
    model: str = "base"

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model '{v}'. Allowed: {sorted(ALLOWED_MODELS)}")
        return v
    language: Optional[str] = None
    use_gpu: bool = True
    use_cache: bool = True
    diarize: bool = False
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    initial_prompt: Optional[str] = None


@router.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    try:
        result = transcribe_audio(
            file_path=req.file_path,
            model_name=req.model,
            use_gpu=req.use_gpu,
            use_cache=req.use_cache,
            language=req.language,
            initial_prompt=req.initial_prompt or None,
        )

        if req.diarize and req.hf_token:
            result = diarize_and_label(
                transcription_result=result,
                audio_path=req.file_path,
                hf_token=req.hf_token,
                num_speakers=req.num_speakers,
                use_gpu=req.use_gpu,
            )

        return result

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {req.file_path}")
    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/transcribe/stream")
async def transcribe_stream(req: TranscribeRequest):
    """
    Transcribe with real-time progress via Server-Sent Events.

    Streams JSON events:
      {"type": "progress", "value": 0-100, "status": "..."}
      {"type": "done",     "result": {...}}
      {"type": "error",    "message": "..."}

    Progress values come directly from faster-whisper's lazy segment generator,
    so 10–70% reflects actual decode position in the audio, not an estimate.
    """
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            def on_progress(pct: int, status: str):
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "progress", "value": pct, "status": status}),
                    loop,
                )

            result = transcribe_audio(
                file_path=req.file_path,
                model_name=req.model,
                use_gpu=req.use_gpu,
                use_cache=req.use_cache,
                language=req.language,
                initial_prompt=req.initial_prompt or None,
                progress_cb=on_progress,
            )

            if req.diarize and req.hf_token:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "progress", "value": 97, "status": "Diarizing speakers..."}),
                    loop,
                )
                result = diarize_and_label(
                    transcription_result=result,
                    audio_path=req.file_path,
                    hf_token=req.hf_token,
                    num_speakers=req.num_speakers,
                    use_gpu=req.use_gpu,
                )

            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "done", "result": result}),
                loop,
            )
        except Exception as e:
            logger.error(f"Streaming transcription failed: {e}", exc_info=True)
            asyncio.run_coroutine_threadsafe(
                queue.put({"type": "error", "message": str(e)}),
                loop,
            )

    threading.Thread(target=run, daemon=True).start()

    async def generate():
        while True:
            event = await queue.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event["type"] in ("done", "error"):
                break

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
