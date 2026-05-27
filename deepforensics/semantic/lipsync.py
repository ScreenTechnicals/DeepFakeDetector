"""
DeepForensics Lip-Sync Module (Module 4A).

Measures audio-visual temporal coherence by computing a sliding
Pearson correlation between the MFCC energy envelope (from the audio
track) and the Mouth Aspect Ratio series (from MediaPipe landmarks).

On real clips the lip motion tracks the audio energy almost exactly
(Pearson r ≈ 0.86); on fakes the two signals drift and typically
>50 % of windows fall below r = 0.4.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from deepforensics.config import (
    LIPSYNC_MIN_VARIANCE,
    LIPSYNC_R_THRESHOLD,
    LIPSYNC_WINDOW_SECONDS,
    OUTER_LIP_LANDMARKS,
)
from deepforensics.result import LipSyncScores

logger = logging.getLogger(__name__)


class LipSyncAnalyzer:
    """
    Computes lip–audio coherence by correlating the MFCC energy
    envelope with the Mouth Aspect Ratio (MAR) time-series.
    """

    def analyze(
        self,
        landmarks_sequence: List[Optional[np.ndarray]],
        energy_envelope: Optional[np.ndarray],
        fps: int,
    ) -> LipSyncScores:
        """
        Analyse lip-sync coherence.

        Parameters
        ----------
        landmarks_sequence : list of Optional[np.ndarray]
            Per-frame MediaPipe 468-point landmarks (468, 3), or None.
        energy_envelope : np.ndarray or None
            Scalar MFCC energy series from ``MFCCExtractor.energy_envelope``.
        fps : int
            Video sampling rate (frames per second).

        Returns
        -------
        LipSyncScores
        """
        if energy_envelope is None or len(energy_envelope) == 0:
            logger.info("No audio available — returning default lip-sync scores.")
            return LipSyncScores(
                mean_correlation=0.0,
                min_correlation=0.0,
                std_correlation=0.0,
                below_threshold_fraction=1.0,
            )

        # -- 1. Compute MAR per frame --------------------------------------
        mar_series = self._compute_mar_series(landmarks_sequence)
        if mar_series is None or len(mar_series) < fps:
            logger.warning("Too few MAR values — returning default scores.")
            return LipSyncScores(
                mean_correlation=0.0,
                min_correlation=0.0,
                std_correlation=0.0,
                below_threshold_fraction=1.0,
            )

        # -- 2. Interpolate to common rate ----------------------------------
        mar_interp, energy_interp = self._align_signals(
            mar_series, energy_envelope, fps
        )

        # -- 3. Sliding Pearson r -------------------------------------------
        window_size = int(LIPSYNC_WINDOW_SECONDS * fps)
        window_size = max(window_size, 3)  # need ≥3 points for correlation
        correlations = self._sliding_pearson(
            mar_interp, energy_interp, window_size
        )

        if len(correlations) == 0:
            return LipSyncScores(
                mean_correlation=0.0,
                min_correlation=0.0,
                std_correlation=0.0,
                below_threshold_fraction=1.0,
            )

        below = np.sum(correlations < LIPSYNC_R_THRESHOLD) / len(correlations)

        return LipSyncScores(
            mean_correlation=float(np.mean(correlations)),
            min_correlation=float(np.min(correlations)),
            std_correlation=float(np.std(correlations)),
            below_threshold_fraction=float(below),
        )

    # -- private helpers ----------------------------------------------------

    @staticmethod
    def _compute_mar_series(
        landmarks_sequence: List[Optional[np.ndarray]],
    ) -> Optional[np.ndarray]:
        """
        Compute Mouth Aspect Ratio per frame.

        MAR = sum(vertical_distances) / (2 × horizontal_distance)

        Uses 12 outer-lip landmarks from the MediaPipe 468-mesh.
        """
        mars: List[float] = []

        for lm in landmarks_sequence:
            if lm is None:
                mars.append(np.nan)
                continue

            try:
                lip_pts = lm[OUTER_LIP_LANDMARKS][:, :2]  # (N, 2) xy only

                # Vertical distances: pair upper and lower lip points
                # Upper indices (relative to OUTER_LIP_LANDMARKS): 2-8
                # Lower indices: 12-18
                upper = lip_pts[2:9]   # 7 points
                lower = lip_pts[12:19]  # 7 points
                n_pairs = min(len(upper), len(lower))
                vert_sum = 0.0
                for j in range(n_pairs):
                    vert_sum += np.linalg.norm(upper[j] - lower[n_pairs - 1 - j])

                # Horizontal: left corner (index 0) to right corner (index 10)
                horiz = np.linalg.norm(lip_pts[0] - lip_pts[10])

                if horiz < 1e-6:
                    mars.append(0.0)
                else:
                    mars.append(vert_sum / (2.0 * horiz))
            except (IndexError, ValueError):
                mars.append(np.nan)

        arr = np.array(mars, dtype=np.float64)
        valid = np.isfinite(arr)
        if valid.sum() < 3:
            return None

        # Linearly interpolate NaN gaps
        indices = np.arange(len(arr))
        arr[~valid] = np.interp(indices[~valid], indices[valid], arr[valid])
        return arr

    @staticmethod
    def _align_signals(
        mar: np.ndarray, energy: np.ndarray, target_rate: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Linearly interpolate both signals to a common length."""
        common_len = min(len(mar), len(energy))
        if common_len < 3:
            common_len = max(len(mar), len(energy))

        x_mar = np.linspace(0, 1, len(mar))
        x_energy = np.linspace(0, 1, len(energy))
        x_common = np.linspace(0, 1, common_len)

        mar_interp = np.interp(x_common, x_mar, mar)
        energy_interp = np.interp(x_common, x_energy, energy)
        return mar_interp, energy_interp

    @staticmethod
    def _sliding_pearson(
        sig_a: np.ndarray, sig_b: np.ndarray, window: int
    ) -> np.ndarray:
        """
        Compute sliding-window Pearson r, skipping windows where
        either signal has near-zero variance.
        """
        n = len(sig_a)
        correlations: List[float] = []

        for start in range(0, n - window + 1):
            end = start + window
            a = sig_a[start:end]
            b = sig_b[start:end]

            var_a = np.var(a)
            var_b = np.var(b)

            if var_a < LIPSYNC_MIN_VARIANCE or var_b < LIPSYNC_MIN_VARIANCE:
                continue  # skip silent / frozen windows

            a_centered = a - np.mean(a)
            b_centered = b - np.mean(b)
            denom = np.sqrt(var_a * var_b) * window
            if denom < 1e-12:
                continue
            r = np.dot(a_centered, b_centered) / (np.std(a) * np.std(b) * window)
            r = np.clip(r, -1.0, 1.0)
            correlations.append(float(r))

        return np.array(correlations, dtype=np.float64)
