"""
FFmpeg-based video cutting engine.
Uses stream copy for fast, lossless cuts and falls back to re-encode when needed.
"""

import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _find_ffmpeg() -> str:
    """Locate ffmpeg binary."""
    for cmd in ["ffmpeg", "ffmpeg.exe"]:
        try:
            subprocess.run([cmd, "-version"], capture_output=True, check=True)
            return cmd
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
    raise RuntimeError("FFmpeg not found. Install it or add it to PATH.")


def export_stream_copy(
    input_path: str,
    output_path: str,
    keep_segments: List[dict],
) -> str:
    """
    Export video using FFmpeg concat demuxer with stream copy.
    ~100x faster than re-encoding. No quality loss.

    Args:
        input_path: source video file
        output_path: destination file
        keep_segments: list of {"start": float, "end": float} to keep

    Returns:
        output_path on success
    """
    ffmpeg = _find_ffmpeg()
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not keep_segments:
        raise ValueError("No segments to export")

    temp_dir = tempfile.mkdtemp(prefix="aive_export_")

    try:
        segment_files = []
        for i, seg in enumerate(keep_segments):
            seg_file = os.path.join(temp_dir, f"seg_{i:04d}.ts")
            cmd = [
                ffmpeg, "-y",
                "-ss", str(seg["start"]),
                "-to", str(seg["end"]),
                "-i", input_path,
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                "-f", "mpegts",
                seg_file,
            ]
            logger.info(f"Extracting segment {i}: {seg['start']:.2f}s - {seg['end']:.2f}s")
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Stream copy segment {i} failed, will try re-encode: {result.stderr[-200:]}")
                return export_reencode(input_path, output_path, keep_segments)
            segment_files.append(seg_file)

        concat_str = "|".join(segment_files)
        cmd = [
            ffmpeg, "-y",
            "-i", f"concat:{concat_str}",
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        logger.info(f"Concatenating {len(segment_files)} segments -> {output_path}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Concat failed, falling back to re-encode: {result.stderr[-200:]}")
            return export_reencode(input_path, output_path, keep_segments)

        return output_path

    finally:
        for f in os.listdir(temp_dir):
            try:
                os.remove(os.path.join(temp_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(temp_dir)
        except OSError:
            pass


def export_reencode(
    input_path: str,
    output_path: str,
    keep_segments: List[dict],
    resolution: str = "1080p",
    format_hint: str = "mp4",
    audio_wav_path: str = None,
) -> str:
    """
    Export video with full re-encode. Slower but supports resolution changes,
    format conversion, and avoids stream-copy edge cases.

    When audio_wav_path is provided, uses that PCM WAV for audio trimming instead
    of the original video's compressed audio. This gives sample-accurate cuts
    with no codec frame boundary bleed (e.g. AAC 1024-sample frames).
    """
    ffmpeg = _find_ffmpeg()
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    if not keep_segments:
        raise ValueError("No segments to export")

    # Probe source specs — only downscale if target is smaller, never upscale.
    src = get_video_info(input_path)
    src_height = src.get("height", 0)
    src_sample_rate = src.get("audio_sample_rate", 0) or 48000
    src_audio_bitrate = src.get("audio_bitrate", 0)
    # Use source audio bitrate clamped to a reasonable range; fall back to 192k.
    audio_br = f"{min(max(src_audio_bitrate // 1000, 64), 320)}k" if src_audio_bitrate > 0 else "192k"

    target_height_map = {"720p": 720, "1080p": 1080, "4k": 2160}
    target_height = target_height_map.get(resolution, 0)
    # Only apply scale when the source is taller than the target (downscale only).
    scale = f"scale=-2:{target_height}" if (target_height and src_height > target_height) else ""

    audio_input = "[0:a]" if audio_wav_path is None else "[1:a]"
    ffmpeg_inputs = ["-i", input_path]
    if audio_wav_path:
        ffmpeg_inputs += ["-i", str(Path(audio_wav_path).resolve())]

    # Per-segment audio: 12 ms equal-power fade at every cut boundary eliminates
    # click/pop from waveform discontinuities. afade does not alter segment duration
    # so A/V sync is preserved. Loudness normalization (EBU R128) is applied once
    # after concat — doing it per-segment with speechnorm caused audible pumping
    # artifacts due to the aggressive expansion coefficient.
    FADE_DUR = 0.012

    filter_parts = []
    n = len(keep_segments)
    for i, seg in enumerate(keep_segments):
        seg_dur = seg["end"] - seg["start"]
        audio_steps = []
        if i > 0:
            audio_steps.append(f"afade=t=in:d={FADE_DUR}:curve=esin")
        if i < n - 1:
            fade_start = max(0.0, seg_dur - FADE_DUR)
            audio_steps.append(f"afade=t=out:st={fade_start:.6f}:d={FADE_DUR}:curve=esin")
        audio_filter = ("," + ",".join(audio_steps)) if audio_steps else ""

        filter_parts.append(
            f"[0:v]trim=start={seg['start']}:end={seg['end']},setpts=PTS-STARTPTS[v{i}];"
            f"{audio_input}atrim=start={seg['start']}:end={seg['end']},asetpts=PTS-STARTPTS{audio_filter}[a{i}];"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa_raw];")
    # EBU R128 loudness normalization on the full concat — consistent output level
    # without per-segment pumping artifacts.
    filter_parts.append("[outa_raw]loudnorm=I=-16:TP=-1.5:LRA=11[outa]")

    filter_complex = "".join(filter_parts)

    if scale:
        filter_complex += f";[outv]{scale}[outv_scaled]"
        video_map = "[outv_scaled]"
    else:
        video_map = "[outv]"

    codec_args = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", audio_br, "-ar", str(src_sample_rate)]
    if format_hint == "webm":
        codec_args = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                      "-c:a", "libopus", "-b:a", audio_br, "-ar", str(src_sample_rate)]

    cmd = [
        ffmpeg, "-y",
        *ffmpeg_inputs,
        "-filter_complex", filter_complex,
        "-map", video_map,
        "-map", "[outa]",
        *codec_args,
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info(f"Re-encoding {n} segments -> {output_path} (src {src_height}p, out {scale or 'native'})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg re-encode failed: {result.stderr[-500:]}")

    return output_path


def export_reencode_with_subs(
    input_path: str,
    output_path: str,
    keep_segments: List[dict],
    subtitle_path: str,
    resolution: str = "1080p",
    format_hint: str = "mp4",
    audio_wav_path: str = None,
) -> str:
    """
    Export video with re-encode and burn-in subtitles (ASS format).
    Applies trim+concat first, then overlays the subtitle file.

    When audio_wav_path is provided, uses that PCM WAV for audio trimming
    instead of the original video's compressed audio for sample-accurate cuts.
    """
    ffmpeg = _find_ffmpeg()
    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())
    subtitle_path = str(Path(subtitle_path).resolve())

    if not keep_segments:
        raise ValueError("No segments to export")

    src = get_video_info(input_path)
    src_height = src.get("height", 0)
    src_sample_rate = src.get("audio_sample_rate", 0) or 48000
    src_audio_bitrate = src.get("audio_bitrate", 0)
    audio_br = f"{min(max(src_audio_bitrate // 1000, 64), 320)}k" if src_audio_bitrate > 0 else "192k"

    target_height_map = {"720p": 720, "1080p": 1080, "4k": 2160}
    target_height = target_height_map.get(resolution, 0)
    scale = f"scale=-2:{target_height}" if (target_height and src_height > target_height) else ""

    audio_input = "[0:a]" if audio_wav_path is None else "[1:a]"
    ffmpeg_inputs = ["-i", input_path]
    if audio_wav_path:
        ffmpeg_inputs += ["-i", str(Path(audio_wav_path).resolve())]

    FADE_DUR = 0.012  # 12 ms equal-power boundary fade — eliminates click artifacts

    filter_parts = []
    n = len(keep_segments)
    for i, seg in enumerate(keep_segments):
        seg_dur = seg["end"] - seg["start"]
        audio_steps = []
        if i > 0:
            audio_steps.append(f"afade=t=in:d={FADE_DUR}:curve=esin")
        if i < n - 1:
            fade_start = max(0.0, seg_dur - FADE_DUR)
            audio_steps.append(f"afade=t=out:st={fade_start:.6f}:d={FADE_DUR}:curve=esin")
        audio_filter = ("," + ",".join(audio_steps)) if audio_steps else ""

        filter_parts.append(
            f"[0:v]trim=start={seg['start']}:end={seg['end']},setpts=PTS-STARTPTS[v{i}];"
            f"{audio_input}atrim=start={seg['start']}:end={seg['end']},asetpts=PTS-STARTPTS{audio_filter}[a{i}];"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa_raw]")
    filter_parts.append("[outa_raw]loudnorm=I=-16:TP=-1.5:LRA=11[outa]")

    filter_complex = "".join(filter_parts)

    # Escape path for FFmpeg subtitle filter (Windows backslashes need escaping)
    escaped_sub = subtitle_path.replace("\\", "/").replace(":", "\\:")

    if scale:
        filter_complex += f";[outv]{scale},ass='{escaped_sub}'[outv_final]"
    else:
        filter_complex += f";[outv]ass='{escaped_sub}'[outv_final]"
    video_map = "[outv_final]"

    codec_args = ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", audio_br, "-ar", str(src_sample_rate)]
    if format_hint == "webm":
        codec_args = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0",
                      "-c:a", "libopus", "-b:a", audio_br, "-ar", str(src_sample_rate)]

    cmd = [
        ffmpeg, "-y",
        *ffmpeg_inputs,
        "-filter_complex", filter_complex,
        "-map", video_map,
        "-map", "[outa]",
        *codec_args,
        "-movflags", "+faststart",
        output_path,
    ]

    logger.info(f"Re-encoding {n} segments with subtitles -> {output_path} (src {src_height}p, out {scale or 'native'})")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg re-encode with subs failed: {result.stderr[-500:]}")

    return output_path


def get_video_info(input_path: str) -> dict:
    """Get basic video metadata using ffprobe."""
    ffmpeg = _find_ffmpeg()
    ffprobe = ffmpeg.replace("ffmpeg", "ffprobe")

    cmd = [
        ffprobe, "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(input_path),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        import json
        data = json.loads(result.stdout)
        fmt = data.get("format", {})
        video_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
        audio_stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "audio"), {})

        return {
            "duration": float(fmt.get("duration", 0)),
            "size": int(fmt.get("size", 0)),
            "format": fmt.get("format_name", ""),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", ""),
            "fps": eval(video_stream.get("r_frame_rate", "0/1")) if "/" in video_stream.get("r_frame_rate", "") else 0,
            "audio_codec": audio_stream.get("codec_name", ""),
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
            "audio_channels": int(audio_stream.get("channels", 2) or 2),
            "audio_bitrate": int(audio_stream.get("bit_rate", 0) or 0),
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}
