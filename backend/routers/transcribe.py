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
from services import transcription_mlx

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_BACKENDS = {"whisperx", "mlx"}
ALLOWED_MODELS_WHISPERX = {"tiny", "base", "small", "medium", "large-v3", "distil-large-v3"}
ALLOWED_MODELS_MLX = set(transcription_mlx.MLX_REPOS.keys())
ALLOWED_MODELS = ALLOWED_MODELS_WHISPERX | ALLOWED_MODELS_MLX
ALLOWED_BEAM_SIZES = {1, 3, 5, 8, 10}


class TranscribeRequest(BaseModel):
    file_path: str
    model: str = "base"
    language: Optional[str] = None
    use_gpu: bool = True
    use_cache: bool = True
    diarize: bool = False
    hf_token: Optional[str] = None
    num_speakers: Optional[int] = None
    initial_prompt: Optional[str] = None
    beam_size: int = 5
    vad_filter: bool = False
    vad_min_silence_ms: int = 500
    verbatim: bool = False
    backend: str = "whisperx"

    @field_validator("backend")
    @classmethod
    def validate_backend(cls, v: str) -> str:
        if v not in ALLOWED_BACKENDS:
            raise ValueError(f"Unknown backend '{v}'. Allowed: {sorted(ALLOWED_BACKENDS)}")
        return v

    @field_validator("model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        if v not in ALLOWED_MODELS:
            raise ValueError(f"Unknown model '{v}'. Allowed: {sorted(ALLOWED_MODELS)}")
        return v

    @field_validator("beam_size")
    @classmethod
    def validate_beam_size(cls, v: int) -> int:
        if v not in ALLOWED_BEAM_SIZES:
            raise ValueError(f"beam_size must be one of {sorted(ALLOWED_BEAM_SIZES)}")
        return v

    @field_validator("vad_min_silence_ms")
    @classmethod
    def validate_vad_min_silence_ms(cls, v: int) -> int:
        if not (100 <= v <= 2000):
            raise ValueError("vad_min_silence_ms must be between 100 and 2000")
        return v

    def ensure_model_matches_backend(self) -> None:
        allowed = ALLOWED_MODELS_MLX if self.backend == "mlx" else ALLOWED_MODELS_WHISPERX
        if self.model not in allowed:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{self.model}' is not supported by the '{self.backend}' "
                    f"backend. Allowed for {self.backend}: {sorted(allowed)}"
                ),
            )


@router.get("/transcribe/backends")
async def list_backends():
    """Return which transcription backends are runnable on this machine."""
    mlx_ok, mlx_reason = transcription_mlx.is_available()
    return {
        "backends": [
            {
                "id": "whisperx",
                "label": "WhisperX (faster-whisper)",
                "available": True,
                "models": sorted(ALLOWED_MODELS_WHISPERX),
                "reason": "",
            },
            {
                "id": "mlx",
                "label": "MLX Whisper (Apple Silicon)",
                "available": mlx_ok,
                "models": sorted(ALLOWED_MODELS_MLX),
                "reason": mlx_reason,
            },
        ]
    }


@router.post("/transcribe")
async def transcribe(req: TranscribeRequest):
    req.ensure_model_matches_backend()
    try:
        result = transcribe_audio(
            file_path=req.file_path,
            model_name=req.model,
            use_gpu=req.use_gpu,
            use_cache=req.use_cache,
            language=req.language,
            initial_prompt=req.initial_prompt or None,
            beam_size=req.beam_size,
            vad_filter=req.vad_filter,
            vad_min_silence_ms=req.vad_min_silence_ms,
            verbatim=req.verbatim,
            backend=req.backend,
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
    req.ensure_model_matches_backend()
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
                beam_size=req.beam_size,
                vad_filter=req.vad_filter,
                vad_min_silence_ms=req.vad_min_silence_ms,
                verbatim=req.verbatim,
                backend=req.backend,
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
