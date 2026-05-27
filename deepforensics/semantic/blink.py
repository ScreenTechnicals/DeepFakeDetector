"""
DeepForensics Eye-Blink Module (Module 4B).

Detects blinks via the Eye Aspect Ratio (EAR) and flags anomalous
blink patterns.  Humans blink 12–20 times per minute (μ=16, σ=4).
GANs trained on still images produce faces that either don't blink,
blink too rarely, or blink in an unnaturally regular cadence.

EAR (Eq. 1 from the manuscript):
    EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from deepforensics.config import (
    BLINK_HUMAN_MEAN,
    BLINK_HUMAN_STD,
    BLINK_MAX_FRAMES,
    BLINK_MIN_FRAMES,
    EAR_THRESHOLD,
    LEFT_EYE_LANDMARKS,
    RIGHT_EYE_LANDMARKS,
)
from deepforensics.result import BlinkScores

logger = logging.getLogger(__name__)


class BlinkAnalyzer:
    """
    Analyses eye-blink dynamics from a sequence of MediaPipe 468-point
    landmark arrays.
    """

    def analyze(
        self,
        landmarks_sequence: List[Optional[np.ndarray]],
        fps: int,
    ) -> BlinkScores:
        """
        Analyse blink dynamics.

        Parameters
        ----------
        landmarks_sequence : list of Optional[np.ndarray]
            Per-frame MediaPipe landmarks (468, 3), or None.
        fps : int
            Video sampling rate (frames per second).

        Returns
        -------
        BlinkScores
        """
        if not landmarks_sequence or fps <= 0:
            return self._default_scores()

        # -- 1. Compute per-frame EAR --------------------------------------
        ear_series = self._compute_ear_series(landmarks_sequence)
        valid_count = np.isfinite(ear_series).sum()

        if valid_count < BLINK_MIN_FRAMES:
            return self._default_scores()

        # -- 2. Detect blinks -----------------------------------------------
        blinks = self._detect_blinks(ear_series)

        # -- 3. Compute statistics ------------------------------------------
        total_seconds = len(ear_series) / fps
        total_minutes = total_seconds / 60.0

        if total_minutes < 0.01:
            # Very short clip — can't estimate per-minute rate reliably
            return self._default_scores()

        blink_rate = len(blinks) / total_minutes if total_minutes > 0 else 0.0

        if len(blinks) == 0:
            mean_duration = 0.0
            ibi_std = 0.0
        else:
            durations = [b["duration"] for b in blinks]
            mean_duration = float(np.mean(durations))

            if len(blinks) >= 2:
                starts = [b["start"] for b in blinks]
                # Inter-blink intervals in seconds
                ibis = [
                    (starts[i + 1] - starts[i]) / fps
                    for i in range(len(starts) - 1)
                ]
                ibi_std = float(np.std(ibis))
            else:
                ibi_std = 0.0

        # Z-score against human baseline
        z_score = (blink_rate - BLINK_HUMAN_MEAN) / BLINK_HUMAN_STD

        return BlinkScores(
            blink_rate_per_min=round(blink_rate, 2),
            mean_duration_frames=round(mean_duration, 2),
            ibi_std=round(ibi_std, 4),
            anomaly_z_score=round(z_score, 4),
        )

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _compute_ear(landmarks: np.ndarray, eye_indices: List[int]) -> float:
        """
        Compute the Eye Aspect Ratio for one eye.

        Landmark ordering within the 6-point array:
            p1 (outer), p2 (upper-outer), p3 (upper-inner),
            p4 (inner), p5 (lower-inner), p6 (lower-outer)
        """
        pts = landmarks[eye_indices][:, :2]  # (6, 2)

        # Vertical distances
        v1 = np.linalg.norm(pts[1] - pts[5])  # p2 - p6
        v2 = np.linalg.norm(pts[2] - pts[4])  # p3 - p5

        # Horizontal distance
        h = np.linalg.norm(pts[0] - pts[3])    # p1 - p4

        if h < 1e-6:
            return 0.0

        return (v1 + v2) / (2.0 * h)

    def _compute_ear_series(
        self, landmarks_sequence: List[Optional[np.ndarray]]
    ) -> np.ndarray:
        """Compute average EAR (left + right eye) per frame."""
        ears: List[float] = []

        for lm in landmarks_sequence:
            if lm is None:
                ears.append(np.nan)
                continue

            try:
                ear_right = self._compute_ear(lm, RIGHT_EYE_LANDMARKS)
                ear_left = self._compute_ear(lm, LEFT_EYE_LANDMARKS)
                ears.append((ear_right + ear_left) / 2.0)
            except (IndexError, ValueError):
                ears.append(np.nan)

        return np.array(ears, dtype=np.float64)

    @staticmethod
    def _detect_blinks(ear_series: np.ndarray) -> List[dict]:
        """
        Detect blinks as consecutive runs of EAR < threshold,
        filtering to runs of BLINK_MIN_FRAMES–BLINK_MAX_FRAMES.
        """
        blinks: List[dict] = []
        n = len(ear_series)
        i = 0

        while i < n:
            if np.isfinite(ear_series[i]) and ear_series[i] < EAR_THRESHOLD:
                start = i
                while (
                    i < n
                    and np.isfinite(ear_series[i])
                    and ear_series[i] < EAR_THRESHOLD
                ):
                    i += 1
                duration = i - start

                if BLINK_MIN_FRAMES <= duration <= BLINK_MAX_FRAMES:
                    blinks.append({"start": start, "duration": duration})
            else:
                i += 1

        return blinks

    @staticmethod
    def _default_scores() -> BlinkScores:
        """Return default scores when analysis is not possible."""
        return BlinkScores(
            blink_rate_per_min=0.0,
            mean_duration_frames=0.0,
            ibi_std=0.0,
            anomaly_z_score=round(-BLINK_HUMAN_MEAN / BLINK_HUMAN_STD, 4),
        )
