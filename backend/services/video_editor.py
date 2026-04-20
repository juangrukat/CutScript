"""
FFmpeg-based video cutting engine.
Uses stream copy for fast, lossless cuts and falls back to re-encode when needed.
"""

import logging
import subprocess
import tempfile
import os
from fractions import Fraction
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

BOUNDARY_FADE_DUR = 0.012
EOF_FADE_DUR = 0.012


def _parse_fps(value: str) -> float:
    """Parse ffprobe frame-rate strings like '24000/1001' safely."""
    try:
        if not value:
            return 0.0
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _source_timing(src: dict) -> tuple[float, float, float]:
    """Return source video duration, audio duration, and frame duration."""
    fmt_duration = float(src.get("duration") or 0.0)
    video_duration = float(src.get("video_duration") or 0.0) or fmt_duration
    audio_duration = float(src.get("audio_duration") or 0.0) or fmt_duration
    fps = float(src.get("fps") or 0.0)
    frame_duration = 1.0 / fps if fps > 0 else 0.0
    return video_duration, audio_duration, frame_duration


def _touches_source_eof(
    end: float,
    video_duration: float,
    audio_duration: float,
    frame_duration: float,
) -> bool:
    """
    Treat EOF as an audio-first boundary.

    Video lands on a frame grid, but speech tails live on the audio sample grid.
    A segment ending within roughly one video frame of video EOF should preserve
    audio to audio EOF when the source audio runs slightly longer than video.
    """
    eof_tolerance = max(0.100, frame_duration * 1.5)
    return (
        (audio_duration > 0 and end >= audio_duration - eof_tolerance)
        or (video_duration > 0 and end >= video_duration - eof_tolerance)
    )


def _av_trim_ranges(
    keep_segments: List[dict],
    src_video_duration: float,
    src_audio_duration: float,
    frame_duration: float,
) -> list[dict]:
    """
    Build separate video/audio trim ranges.

    Internal cuts keep the same nominal timestamps for A/V sync. Final EOF cuts
    clamp video to video EOF while allowing audio to reach audio EOF, preserving
    low-energy speech codas that may extend beyond the last video frame.
    """
    ranges = []
    for i, seg in enumerate(keep_segments):
        start = max(0.0, float(seg["start"]))
        end = max(start, float(seg["end"]))
        is_last = i == len(keep_segments) - 1
        touches_eof = is_last and _touches_source_eof(
            end,
            src_video_duration,
            src_audio_duration,
            frame_duration,
        )

        video_start = min(start, src_video_duration) if src_video_duration > 0 else start
        audio_start = min(start, src_audio_duration) if src_audio_duration > 0 else start

        if touches_eof:
            video_end = src_video_duration if src_video_duration > 0 else end
            audio_end = src_audio_duration if src_audio_duration > 0 else end
        else:
            video_end = min(end, src_video_duration) if src_video_duration > 0 else end
            audio_end = min(end, src_audio_duration) if src_audio_duration > 0 else end

        video_end = max(video_start, video_end)
        audio_end = max(audio_start, audio_end)
        video_pad = 0.0
        if touches_eof:
            video_pad = max(
                0.0,
                (audio_end - audio_start) - (video_end - video_start) + (2 * frame_duration),
            )
        ranges.append(
            {
                "video_start": video_start,
                "video_end": video_end,
                "audio_start": audio_start,
                "audio_end": audio_end,
                "video_pad": video_pad,
                "touches_eof": touches_eof,
            }
        )

    return ranges


def _log_av_trim_ranges(ranges: list[dict], frame_duration: float) -> None:
    """Log meaningful A/V trim-duration differences for export diagnostics."""
    tolerance = max(frame_duration, 0.020)
    for i, av in enumerate(ranges):
        video_dur = av["video_end"] - av["video_start"]
        audio_dur = av["audio_end"] - av["audio_start"]
        delta = audio_dur - video_dur
        if av["touches_eof"] or abs(delta) > tolerance:
            logger.info(
                "A/V trim segment %d: video %.6f-%.6f (%.3fs), "
                "audio %.6f-%.6f (%.3fs), delta=%+.3fs%s",
                i,
                av["video_start"],
                av["video_end"],
                video_dur,
                av["audio_start"],
                av["audio_end"],
                audio_dur,
                delta,
                (
                    f" EOF video_pad={av['video_pad']:.3f}s"
                    if av["touches_eof"] and av.get("video_pad", 0.0) > 0
                    else " EOF"
                    if av["touches_eof"]
                    else ""
                ),
            )


def _build_trim_concat_filters(
    audio_input: str,
    keep_segments: List[dict],
    src_video_duration: float,
    src_audio_duration: float,
    frame_duration: float,
) -> tuple[str, bool]:
    """Build the shared A/V trim + concat filter graph."""
    filter_parts = []
    av_ranges = _av_trim_ranges(
        keep_segments,
        src_video_duration,
        src_audio_duration,
        frame_duration,
    )
    _log_av_trim_ranges(av_ranges, frame_duration)

    n = len(av_ranges)
    final_touches_source_eof = bool(av_ranges and av_ranges[-1]["touches_eof"])

    for i, av in enumerate(av_ranges):
        audio_dur = av["audio_end"] - av["audio_start"]
        video_filter = (
            f"[0:v]trim=start={av['video_start']:.6f}:end={av['video_end']:.6f},"
            "setpts=PTS-STARTPTS"
        )
        if av.get("video_pad", 0.0) > 0:
            video_filter += f",tpad=stop_mode=clone:stop_duration={av['video_pad']:.6f}"

        audio_steps = []
        if i > 0:
            audio_steps.append(f"afade=t=in:d={BOUNDARY_FADE_DUR}:curve=esin")
        if i < n - 1:
            fade_start = max(0.0, audio_dur - BOUNDARY_FADE_DUR)
            audio_steps.append(
                f"afade=t=out:st={fade_start:.6f}:d={BOUNDARY_FADE_DUR}:curve=esin"
            )
        elif av["touches_eof"] and audio_dur > EOF_FADE_DUR:
            fade_start = max(0.0, audio_dur - EOF_FADE_DUR)
            audio_steps.append(
                f"afade=t=out:st={fade_start:.6f}:d={EOF_FADE_DUR}:curve=esin"
            )
        audio_filter = ("," + ",".join(audio_steps)) if audio_steps else ""

        filter_parts.append(
            f"{video_filter}[v{i}];"
            f"{audio_input}atrim=start={av['audio_start']:.6f}:end={av['audio_end']:.6f},"
            f"asetpts=PTS-STARTPTS{audio_filter}[a{i}];"
        )

    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    filter_parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa_raw];")
    if n == 1 or final_touches_source_eof:
        filter_parts.append("[outa_raw]anull[outa]")
    else:
        # EBU R128 loudness normalization on the full concat — consistent output
        # level without per-segment pumping artifacts.
        filter_parts.append("[outa_raw]loudnorm=I=-16:TP=-1.5:LRA=11[outa]")

    return "".join(filter_parts), final_touches_source_eof


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
    src_video_duration, src_audio_duration, frame_duration = _source_timing(src)
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

    # Per-segment audio fades eliminate click/pop at edit boundaries. EOF
    # handling is audio-first: preserve the source audio tail, hold the final
    # video frame long enough to carry it, then apply only a tiny terminal fade
    # into encoder padding.
    filter_complex, _ = _build_trim_concat_filters(
        audio_input,
        keep_segments,
        src_video_duration,
        src_audio_duration,
        frame_duration,
    )

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

    logger.info(
        f"Re-encoding {len(keep_segments)} segments -> {output_path} "
        f"(src {src_height}p, out {scale or 'native'})"
    )
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
    src_video_duration, src_audio_duration, frame_duration = _source_timing(src)
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

    filter_complex, _ = _build_trim_concat_filters(
        audio_input,
        keep_segments,
        src_video_duration,
        src_audio_duration,
        frame_duration,
    )

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

    logger.info(
        f"Re-encoding {len(keep_segments)} segments with subtitles -> {output_path} "
        f"(src {src_height}p, out {scale or 'native'})"
    )
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
            "video_duration": float(video_stream.get("duration", 0) or 0),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "codec": video_stream.get("codec_name", ""),
            "fps": _parse_fps(video_stream.get("avg_frame_rate", "") or video_stream.get("r_frame_rate", "")),
            "audio_codec": audio_stream.get("codec_name", ""),
            "audio_duration": float(audio_stream.get("duration", 0) or 0),
            "audio_sample_rate": int(audio_stream.get("sample_rate", 0) or 0),
            "audio_channels": int(audio_stream.get("channels", 2) or 2),
            "audio_bitrate": int(audio_stream.get("bit_rate", 0) or 0),
        }
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        return {}
