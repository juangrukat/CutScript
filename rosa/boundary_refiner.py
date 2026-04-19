import numpy as np
import librosa
from typing import Tuple, Dict, Optional


class BoundaryRefiner:
    """
    Librosa-based sentence boundary refiner that adjusts approximate timestamps
    into editorially better cut points using RMS energy, onset strength, and zero-crossing snap.
    """

    def __init__(
        self, frame_length: int = 1024, hop_length: int = 256, smoothing_window: int = 5
    ):
        """
        Initialize the boundary refiner.

        Args:
            frame_length: FFT frame length for feature computation
            hop_length: Hop length for feature computation
            smoothing_window: Window size for smoothing RMS
        """
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.smoothing_window = smoothing_window

    def refine_boundaries(
        self,
        y: np.ndarray,
        sr: float,
        approx_start: float,
        approx_end: float,
        mode: str = "natural",
        start_window_pre: float = 0.5,
        start_window_post: float = 0.3,
        end_window_pre: float = 0.3,
        end_window_post: float = 0.7,
    ) -> Dict:
        """
        Refine sentence boundaries using librosa features.

        Args:
            y: Audio waveform
            sr: Sample rate
            approx_start: Approximate start time in seconds
            approx_end: Approximate end time in seconds
            mode: 'tight', 'natural', or 'aggressive'
            start_window_pre: Seconds before approx_start to search
            start_window_post: Seconds after approx_start to search
            end_window_pre: Seconds before approx_end to search
            end_window_post: Seconds after approx_end to search

        Returns:
            Dict with refined_start, refined_end, fade_in_duration, fade_out_duration, confidence_flags
        """

        # Convert times to sample indices
        start_idx = int(approx_start * sr)
        end_idx = int(approx_end * sr)

        # Refine start boundary
        refined_start = self._refine_start_boundary(
            y, sr, start_idx, mode, start_window_pre, start_window_post
        )

        # Refine end boundary
        refined_end = self._refine_end_boundary(
            y, sr, end_idx, mode, end_window_pre, end_window_post
        )

        # Suggest fade durations based on mode
        fade_in, fade_out = self._suggest_fades(mode)

        # Generate confidence flags
        flags = self._assess_confidence(
            y, sr, refined_start, refined_end, approx_start, approx_end
        )

        return {
            "refined_start": refined_start / sr,
            "refined_end": refined_end / sr,
            "fade_in_duration": fade_in,
            "fade_out_duration": fade_out,
            "confidence_flags": flags,
        }

    def _refine_start_boundary(
        self,
        y: np.ndarray,
        sr: float,
        approx_start_idx: int,
        mode: str,
        window_pre: float,
        window_post: float,
    ) -> int:
        """
        Refine the start boundary using RMS and onset strength.
        """
        # Define search window
        window_start = max(0, approx_start_idx - int(window_pre * sr))
        window_end = min(len(y), approx_start_idx + int(window_post * sr))

        if window_end <= window_start:
            return approx_start_idx

        y_window = y[window_start:window_end]

        # Compute RMS energy
        rms = librosa.feature.rms(
            y=y_window, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]

        # Smooth RMS
        rms_smooth = self._smooth_signal(rms)

        # Compute onset strength
        onset_env = librosa.onset.onset_strength(
            y=y_window, sr=sr, hop_length=self.hop_length
        )

        # Estimate noise floor from leftmost frames
        noise_frames = min(10, len(rms_smooth) // 4)
        noise_floor = np.mean(rms_smooth[:noise_frames])

        # Mode-specific threshold
        threshold_multiplier = self._get_threshold_multiplier(mode)
        threshold = noise_floor * threshold_multiplier

        # Find sustained rise (consecutive frames above threshold)
        sustained_frames = self._get_sustained_frames(mode)
        rise_idx = self._find_first_sustained_rise(
            rms_smooth, threshold, sustained_frames
        )

        if rise_idx is None:
            # Fallback to approximate start
            return approx_start_idx

        # Check onset strength for earlier cue in natural mode
        if mode == "natural":
            onset_rise_idx = self._find_onset_rise(onset_env, rms_smooth, rise_idx)
            if onset_rise_idx is not None and onset_rise_idx < rise_idx:
                rise_idx = onset_rise_idx

        # Apply pre-roll
        pre_roll_samples = int(self._get_pre_roll(mode) * sr / 1000)  # ms to samples
        refined_start_idx = (
            window_start + (rise_idx * self.hop_length) - pre_roll_samples
        )
        refined_start_idx = max(window_start, refined_start_idx)

        # Snap to zero crossing
        refined_start_idx = self._snap_to_zero_crossing(y, refined_start_idx)

        return refined_start_idx

    def _refine_end_boundary(
        self,
        y: np.ndarray,
        sr: float,
        approx_end_idx: int,
        mode: str,
        window_pre: float,
        window_post: float,
    ) -> int:
        """
        Refine the end boundary using RMS.
        """
        # Define search window
        window_start = max(0, approx_end_idx - int(window_pre * sr))
        window_end = min(len(y), approx_end_idx + int(window_post * sr))

        if window_end <= window_start:
            return approx_end_idx

        y_window = y[window_start:window_end]

        # Compute RMS energy
        rms = librosa.feature.rms(
            y=y_window, frame_length=self.frame_length, hop_length=self.hop_length
        )[0]

        # Smooth RMS
        rms_smooth = self._smooth_signal(rms)

        # Estimate noise floor from rightmost frames
        noise_frames = min(10, len(rms_smooth) // 4)
        noise_floor = np.mean(rms_smooth[-noise_frames:])

        # Mode-specific threshold
        threshold_multiplier = self._get_threshold_multiplier(mode)
        threshold = noise_floor * threshold_multiplier

        # Find sustained fall
        sustained_frames = self._get_sustained_frames(mode)
        fall_idx = self._find_last_sustained_fall(
            rms_smooth, threshold, sustained_frames
        )

        if fall_idx is None:
            # Fallback to approximate end
            return approx_end_idx

        # Apply post-roll
        post_roll_samples = int(self._get_post_roll(mode) * sr / 1000)  # ms to samples
        refined_end_idx = (
            window_start + (fall_idx * self.hop_length) + post_roll_samples
        )
        refined_end_idx = min(window_end, refined_end_idx)

        # Snap to zero crossing
        refined_end_idx = self._snap_to_zero_crossing(y, refined_end_idx)

        return refined_end_idx

    def _smooth_signal(self, signal: np.ndarray) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if len(signal) < self.smoothing_window:
            return signal
        return np.convolve(
            signal, np.ones(self.smoothing_window) / self.smoothing_window, mode="same"
        )

    def _find_first_sustained_rise(
        self, rms: np.ndarray, threshold: float, sustained: int
    ) -> Optional[int]:
        """Find first index where RMS stays above threshold for sustained frames."""
        above = rms > threshold
        for i in range(len(above) - sustained + 1):
            if np.all(above[i : i + sustained]):
                return i
        return None

    def _find_last_sustained_fall(
        self, rms: np.ndarray, threshold: float, sustained: int
    ) -> Optional[int]:
        """Find last index where RMS stays below threshold for sustained frames."""
        below = rms < threshold
        for i in range(len(below) - sustained, -1, -1):
            if np.all(below[i : i + sustained]):
                return i + sustained - 1
        return None

    def _find_onset_rise(
        self, onset_env: np.ndarray, rms: np.ndarray, rms_rise_idx: int
    ) -> Optional[int]:
        """Find earlier onset rise for natural mode."""
        # Look for onset peak before RMS rise
        onset_threshold = np.mean(onset_env) + np.std(onset_env)
        onset_peaks = librosa.util.peak_pick(
            onset_env,
            pre_max=3,
            post_max=3,
            pre_avg=3,
            post_avg=3,
            delta=onset_threshold,
            wait=10,
        )

        # Find the latest onset peak before RMS rise
        valid_peaks = onset_peaks[onset_peaks < rms_rise_idx]
        if len(valid_peaks) > 0:
            return valid_peaks[-1]
        return None

    def _snap_to_zero_crossing(
        self, y: np.ndarray, idx: int, search_window: int = 100
    ) -> int:
        """Snap to nearest zero crossing within search window."""
        start = max(0, idx - search_window)
        end = min(len(y), idx + search_window)

        # Find zero crossings in window
        zero_crossings = librosa.zero_crossings(y[start:end])
        crossing_indices = np.where(zero_crossings)[0] + start

        if len(crossing_indices) == 0:
            return idx

        # Find closest
        distances = np.abs(crossing_indices - idx)
        closest_idx = crossing_indices[np.argmin(distances)]

        return closest_idx

    def _get_threshold_multiplier(self, mode: str) -> float:
        """Get threshold multiplier based on mode."""
        multipliers = {"tight": 1.5, "natural": 1.2, "aggressive": 1.8}
        return multipliers.get(mode, 1.2)

    def _get_sustained_frames(self, mode: str) -> int:
        """Get number of sustained frames required."""
        sustained = {"tight": 3, "natural": 2, "aggressive": 4}
        return sustained.get(mode, 3)

    def _get_pre_roll(self, mode: str) -> int:
        """Get pre-roll in milliseconds."""
        pre_roll = {"tight": 20, "natural": 80, "aggressive": 35}
        return pre_roll.get(mode, 50)

    def _get_post_roll(self, mode: str) -> int:
        """Get post-roll in milliseconds."""
        post_roll = {"tight": 35, "natural": 105, "aggressive": 55}
        return post_roll.get(mode, 70)

    def _suggest_fades(self, mode: str) -> Tuple[float, float]:
        """Suggest fade durations in seconds."""
        fades = {
            "tight": (0.003, 0.005),  # 3ms, 5ms
            "natural": (0.008, 0.012),  # 8ms, 12ms
            "aggressive": (0.005, 0.008),  # 5ms, 8ms
        }
        return fades.get(mode, (0.005, 0.008))

    def _assess_confidence(
        self,
        y: np.ndarray,
        sr: float,
        start_idx: int,
        end_idx: int,
        approx_start: float,
        approx_end: float,
    ) -> list:
        """Assess confidence and generate flags."""
        flags = []

        # Check if boundaries moved significantly
        start_diff = abs(start_idx / sr - approx_start)
        end_diff = abs(end_idx / sr - approx_end)

        if start_diff > 0.2:
            flags.append("start_boundary_adjusted_significantly")
        if end_diff > 0.2:
            flags.append("end_boundary_adjusted_significantly")

        # Check for weak onset or long breath (simple heuristics)
        # This is a placeholder - could be enhanced
        if start_diff < 0.01:
            flags.append("weak_onset_or_no_clear_rise")

        if end_diff < 0.01:
            flags.append("weak_decay_or_no_clear_fall")

        return flags
