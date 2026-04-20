"""
Audio noise reduction using DeepFilterNet.
Falls back to a basic FFmpeg noise filter if DeepFilterNet is not installed.
"""

import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from df.enhance import enhance, init_df, load_audio, save_audio
    DEEPFILTER_AVAILABLE = True
except ImportError:
    DEEPFILTER_AVAILABLE = False


_df_model = None
_df_state = None


def _init_deepfilter():
    global _df_model, _df_state
    if _df_model is None:
        logger.info("Initializing DeepFilterNet model")
        _df_model, _df_state, _ = init_df()
    return _df_model, _df_state


def clean_audio(
    input_path: str,
    output_path: str = "",
) -> str:
    """
    Apply noise reduction to an audio file.

    If DeepFilterNet is available, uses it for high-quality results.
    Otherwise falls back to FFmpeg's anlmdn filter.

    Returns: path to the cleaned audio file.
    """
    input_path = Path(input_path)
    if not output_path:
        output_path = str(input_path.with_stem(input_path.stem + "_clean"))

    if DEEPFILTER_AVAILABLE:
        return _clean_with_deepfilter(str(input_path), output_path)
    else:
        return _clean_with_ffmpeg(str(input_path), output_path)


def _clean_with_deepfilter(input_path: str, output_path: str) -> str:
    model, state = _init_deepfilter()
    tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_wav.close()
    tmp_enhanced = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_enhanced.close()
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            input_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(state.sr()),
            tmp_wav.name,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg decode for DeepFilterNet failed: {result.stderr[-300:]}")

        audio, _ = load_audio(tmp_wav.name, sr=state.sr())
        enhanced = enhance(model, state, audio)
        save_audio(tmp_enhanced.name, enhanced, sr=state.sr())
        _postprocess_studio_audio(tmp_enhanced.name, output_path)
        logger.info(f"DeepFilterNet cleaned audio saved to {output_path}")
        return output_path
    finally:
        for tmp in (tmp_wav.name, tmp_enhanced.name):
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass


def _postprocess_studio_audio(input_path: str, output_path: str) -> None:
    """
    Finish denoised speech for export.

    DeepFilterNet removes background noise but can return quieter audio with
    inter-sample peaks near/over full scale. This pass keeps the result usable
    as a "studio sound" export: clear low rumble, normalize perceived loudness,
    and constrain true peak before AAC muxing.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-af",
        "highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar",
        "48000",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg studio post-processing failed: {result.stderr[-300:]}")


def _clean_with_ffmpeg(input_path: str, output_path: str) -> str:
    """Fallback: basic noise reduction using FFmpeg's anlmdn filter."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "anlmdn=s=7:p=0.002:r=0.002:m=15,highpass=f=70,loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "48000",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio cleaning failed: {result.stderr[-300:]}")
    logger.info(f"FFmpeg cleaned audio saved to {output_path}")
    return output_path


def is_deepfilter_available() -> bool:
    return DEEPFILTER_AVAILABLE
