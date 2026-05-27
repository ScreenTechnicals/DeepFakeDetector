"""
DeepForensics result types — Pydantic models for the AnalysisResult
object described in Section 3.3 (Module 8) of the manuscript.

Every prediction is accompanied by:
  - A verdict (REAL / FAKE) with calibrated confidence
  - Four interpretable semantic scores
  - A Grad-CAM heatmap with region label
  - An audit block for reproducibility
  - A natural-language summary
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from deepforensics.config import SHA256_PREFIX_LENGTH


# ---------------------------------------------------------------------------
# Semantic score sub-models
# ---------------------------------------------------------------------------

class LipSyncScores(BaseModel):
    """4-d feature vector from the lip-sync module (Module 4A)."""

    mean_correlation: float = Field(
        ..., description="Mean Pearson r across all 1-second windows"
    )
    min_correlation: float = Field(
        ..., description="Minimum Pearson r across windows"
    )
    std_correlation: float = Field(
        ..., description="Standard deviation of windowed correlations"
    )
    below_threshold_fraction: float = Field(
        ...,
        description="Fraction of windows with r < 0.4",
        ge=0.0,
        le=1.0,
    )


class BlinkScores(BaseModel):
    """4-d feature vector from the eye-blink module (Module 4B)."""

    blink_rate_per_min: float = Field(
        ..., description="Detected blinks per minute"
    )
    mean_duration_frames: float = Field(
        ..., description="Mean blink duration in frames (at 5 fps)"
    )
    ibi_std: float = Field(
        ..., description="Standard deviation of inter-blink intervals (seconds)"
    )
    anomaly_z_score: float = Field(
        ...,
        description=(
            "Z-score against human baseline (μ=16, σ=4 blinks/min). "
            "Large positive → too many blinks; large negative → too few."
        ),
    )


class LightingScores(BaseModel):
    """4-d feature vector from the lighting module (Module 4C)."""

    direction_variance: float = Field(
        ...,
        description="Variance of estimated light directions across facial regions",
    )
    max_pairwise_angle_deg: float = Field(
        ...,
        description="Maximum pairwise angle (degrees) between region light vectors",
    )
    mean_intensity_ratio: float = Field(
        ...,
        description="Mean ratio of brightest to dimmest region",
    )
    consistency_score: float = Field(
        ...,
        description="Normalised consistency (0 = inconsistent, 1 = perfectly consistent)",
        ge=0.0,
        le=1.0,
    )


class SemanticScores(BaseModel):
    """Aggregated semantic scores from all three semantic modules."""

    lip_sync: Optional[LipSyncScores] = Field(
        None,
        description="Lip-sync scores (None when audio is unavailable)",
    )
    blink: Optional[BlinkScores] = Field(
        None,
        description="Blink scores (None for single-image input)",
    )
    lighting: Optional[LightingScores] = Field(
        None,
        description="Lighting consistency scores",
    )


# ---------------------------------------------------------------------------
# Grad-CAM result
# ---------------------------------------------------------------------------

class GradCAMResult(BaseModel):
    """Output from the Grad-CAM explainability module (Module 7)."""

    heatmap_path: Optional[str] = Field(
        None, description="Path to the saved heatmap PNG"
    )
    overlay_path: Optional[str] = Field(
        None, description="Path to the heatmap overlaid on the face crop"
    )
    peak_coordinate: Optional[Tuple[int, int]] = Field(
        None, description="(row, col) of peak activation in 224×224 space"
    )
    region_label: Optional[str] = Field(
        None,
        description=(
            "Coarse facial region mapped from peak coordinate: "
            "eyes, nose, mouth, cheeks, boundary, forehead"
        ),
    )
    confidence: Optional[float] = Field(
        None, description="FAKE-class confidence of the top frame(s) used"
    )


# ---------------------------------------------------------------------------
# Audit block
# ---------------------------------------------------------------------------

class AuditBlock(BaseModel):
    """
    Reproducibility record attached to every analysis.

    Contains checkpoint hashes, input-file hash, per-module elapsed times,
    and a UTC timestamp — so that two analyses of the same media that
    produce different verdicts can be traced back to the specific model
    version and code commit that produced each.
    """

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="UTC ISO-8601 timestamp of the analysis",
    )
    input_file_sha256: Optional[str] = Field(
        None,
        description=f"First {SHA256_PREFIX_LENGTH} hex chars of the input file's SHA-256",
    )
    backbone_checkpoint_sha256: Optional[str] = Field(
        None, description="SHA-256 prefix of the backbone weights file"
    )
    fusion_checkpoint_sha256: Optional[str] = Field(
        None, description="SHA-256 prefix of the fusion MLP weights file"
    )
    module_times: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-module wall-clock elapsed time in seconds",
    )
    total_time: float = Field(
        0.0, description="Total pipeline wall-clock time in seconds"
    )


# ---------------------------------------------------------------------------
# Top-level analysis result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    """
    The complete output of a single DeepForensics analysis run.

    This is the Pydantic object described in Section 3.3 (Module 8).
    It is serialised to JSON for the API response and also fed to the
    ReportLab renderer to produce the forensic PDF.
    """

    # Verdict
    verdict: str = Field(
        ..., description="REAL or FAKE", pattern="^(REAL|FAKE)$"
    )
    confidence: float = Field(
        ..., description="Calibrated probability [0, 1]", ge=0.0, le=1.0
    )

    # Evidence
    semantic_scores: SemanticScores = Field(
        ..., description="Scores from the three semantic modules"
    )
    gradcam: GradCAMResult = Field(
        ..., description="Grad-CAM heatmap and region attribution"
    )

    # Summary
    summary: str = Field(
        "",
        description=(
            "Natural-language summary of the evidence, suitable for "
            "display in the UI or inclusion in the PDF report"
        ),
    )
    dominant_region: Optional[str] = Field(
        None,
        description="The facial region most responsible for the FAKE verdict",
    )

    # Visual-only baseline (for comparison / ablation)
    visual_only_confidence: Optional[float] = Field(
        None, description="Confidence from the CNN backbone alone"
    )

    # Audit
    audit: AuditBlock = Field(
        default_factory=AuditBlock,
        description="Reproducibility and traceability record",
    )

    # Input metadata
    input_filename: Optional[str] = None
    input_media_type: Optional[str] = None  # "image" or "video"
    num_frames_processed: Optional[int] = None
    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def compute_file_sha256(path: str | Path, prefix_len: int = SHA256_PREFIX_LENGTH) -> str:
    """Return the first ``prefix_len`` hex characters of a file's SHA-256."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:prefix_len]
