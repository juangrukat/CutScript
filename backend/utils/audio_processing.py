from pathlib import Path
import subprocess
import tempfile
import os
import logging

try:
    from moviepy import AudioFileClip
except ImportError:
    from moviepy.editor import AudioFileClip

logger = logging.getLogger(__name__)

_temp_audio_files = []


def extract_audio(video_path: Path):
    """Extract audio from a video file into a temp directory for automatic cleanup."""
    try:
        audio = AudioFileClip(str(video_path))
        temp_dir = tempfile.mkdtemp(prefix="videotranscriber_")
        audio_path = Path(temp_dir) / f"{video_path.stem}_audio.wav"
        try:
            audio.write_audiofile(str(audio_path), logger=None)
        except TypeError:
            # moviepy 1.x uses verbose parameter; moviepy 2.x removed it
            audio.write_audiofile(str(audio_path), verbose=False, logger=None)
        audio.close()
        _temp_audio_files.append(str(audio_path))
        return audio_path
    except Exception as e:
        raise RuntimeError(f"Audio extraction failed: {e}")


def cleanup_temp_audio():
    """Remove all temporary audio files created during processing."""
    cleaned = 0
    for fpath in _temp_audio_files:
        try:
            if os.path.exists(fpath):
                os.remove(fpath)
                parent = os.path.dirname(fpath)
                if os.path.isdir(parent) and not os.listdir(parent):
                    os.rmdir(parent)
                cleaned += 1
        except Exception as e:
            logger.warning(f"Could not remove temp file {fpath}: {e}")
    _temp_audio_files.clear()
    return cleaned


def _get_duration(audio_path: Path) -> float:
    """Return the duration of an audio/video file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def preprocess_audio_for_transcription(audio_path: Path) -> tuple[Path, float]:
    """
    Whisper-optimized preprocessing: trim leading silence and normalize loudness.

    Long leading silences are a documented cause of Whisper inaccuracy (the model
    can drift into hallucination or wrong-language output before speech begins).
    Loudnorm ensures the signal is in a range Whisper handles well regardless of
    recording level.

    Returns (preprocessed_path, trim_offset_seconds) so callers can shift
    WhisperX timestamps back to match the original timeline.  silenceremove
    physically removes audio frames, so all timestamps produced by Whisper are
    relative to the trimmed file — without the offset they will be systematically
    early by however many seconds of silence were stripped.

    Falls back to the raw audio path (offset 0.0) if FFmpeg preprocessing fails
    so transcription is never blocked by a preprocessing error.
    """
    output_path = audio_path.parent / f"{audio_path.stem}_prep.wav"

    original_duration = _get_duration(audio_path)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(audio_path),
        "-af", (
            "silenceremove=start_periods=1:start_threshold=-50dB:start_silence=0.1,"
            "loudnorm=I=-16:LRA=11:TP=-1.5"
        ),
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Transcription audio preprocessing failed, using raw audio: {result.stderr[-200:]}")
        return audio_path, 0.0

    preprocessed_duration = _get_duration(output_path)
    # silenceremove with start_periods=1 removes exactly one leading silence
    # block, so the duration difference is the timestamp offset to add back.
    trim_offset = round(max(0.0, original_duration - preprocessed_duration), 3)
    if trim_offset > 0:
        logger.info(f"Leading silence trimmed: {trim_offset:.3f}s — will shift timestamps by this amount")

    return output_path, trim_offset


def get_video_duration(video_path: Path):
    """Get duration of a video/audio file in seconds."""
    try:
        clip = AudioFileClip(str(video_path))
        duration = clip.duration
        clip.close()
        return duration
    except Exception:
        return None
