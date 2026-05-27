"""
DeepForensics — Explainable Deepfake Detection Pipeline.

An end-to-end framework that combines an EfficientNet-B4 visual backbone
with three deterministic semantic-consistency modules (lip–audio coherence,
eye-blink dynamics, per-region lighting analysis) and an integrated
Grad-CAM explanation layer.
"""

from deepforensics.result import (
    AnalysisResult,
    AuditBlock,
    GradCAMResult,
    SemanticScores,
)

__all__ = [
    "AnalysisResult",
    "AuditBlock",
    "GradCAMResult",
    "SemanticScores",
]

# Lazy imports to avoid heavy dependencies at import time
def analyze(media_path: str, **kwargs) -> AnalysisResult:
    """Convenience function: run the full pipeline on a media file."""
    from deepforensics.pipeline import DeepForensicsPipeline
    return DeepForensicsPipeline(**kwargs).analyze(media_path)

