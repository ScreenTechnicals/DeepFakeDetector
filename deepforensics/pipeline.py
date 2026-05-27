"""
DeepForensics Pipeline Orchestrator.

End-to-end orchestrator that wires all eight modules together:
  1. Input Media         → MediaIngester
  2. Preprocessing       → FaceDetector + MFCCExtractor
  3. CNN Feature Extraction → EfficientNetB4Backbone
  4A. Lip-Sync           → LipSyncAnalyzer
  4B. Eye-Blink          → BlinkAnalyzer
  4C. Lighting           → LightingAnalyzer
  5. Multi-Modal Fusion  → FusionMLP
  6. Classification      → softmax → verdict
  7. Grad-CAM XAI        → GradCAMExplainer
  8. Output Assembly     → AnalysisResult + PDF
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from deepforensics.backbone import EfficientNetB4Backbone
from deepforensics.config import (
    BACKBONE_EMBEDDING_DIM,
    FRAME_SAMPLE_RATE_FPS,
    PipelineConfig,
)
from deepforensics.fusion import FusionMLP
from deepforensics.gradcam import GradCAMExplainer
from deepforensics.preprocessing import FaceDetector, MediaIngester, MFCCExtractor
from deepforensics.result import (
    AnalysisResult,
    AuditBlock,
    GradCAMResult,
    SemanticScores,
    compute_file_sha256,
)
from deepforensics.semantic import BlinkAnalyzer, LightingAnalyzer, LipSyncAnalyzer

logger = logging.getLogger(__name__)


class DeepForensicsPipeline:
    """
    End-to-end inference pipeline for deepfake detection with
    explainability.

    Usage::

        pipeline = DeepForensicsPipeline()
        result = pipeline.analyze("/path/to/video.mp4")
        print(result.verdict, result.confidence)
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._device = self._resolve_device()

        logger.info("Initialising DeepForensics pipeline on %s", self._device)

        # -- Module instances -----------------------------------------------
        self.ingester = MediaIngester(fps=self.config.frame_rate)
        self.face_detector = FaceDetector()
        self.mfcc_extractor = MFCCExtractor()

        self.lip_sync = LipSyncAnalyzer()
        self.blink = BlinkAnalyzer()
        self.lighting = LightingAnalyzer()

        # -- Neural network modules (lazy-loaded) ---------------------------
        self._backbone: Optional[EfficientNetB4Backbone] = None
        self._fusion: Optional[FusionMLP] = None
        self._gradcam: Optional[GradCAMExplainer] = None

    # -- Public API ---------------------------------------------------------

    def analyze(self, media_path: str) -> AnalysisResult:
        """
        Run the full 8-module pipeline on a media file.

        Parameters
        ----------
        media_path : str
            Path to an image or video file.

        Returns
        -------
        AnalysisResult
            Complete analysis with verdict, semantic scores, Grad-CAM,
            and audit block.
        """
        t_start = time.perf_counter()
        module_times: dict[str, float] = {}

        os.makedirs(self.config.output_dir, exist_ok=True)

        # -- Module 1: Ingest -----------------------------------------------
        t0 = time.perf_counter()
        frames, audio, duration = self.ingester.ingest(media_path)
        module_times["ingest"] = time.perf_counter() - t0

        is_video = len(frames) > 1

        # -- Module 2: Face detection & preprocessing -----------------------
        t0 = time.perf_counter()
        crops, landmarks = self.face_detector.detect_faces(frames)
        face_tensor = self.face_detector.prepare_tensor(crops)
        module_times["preprocessing"] = time.perf_counter() - t0

        # Extract MFCC energy envelope if audio is available
        energy_envelope = None
        if audio is not None:
            t0 = time.perf_counter()
            mfccs = self.mfcc_extractor.extract(audio)
            energy_envelope = self.mfcc_extractor.energy_envelope(mfccs)
            module_times["mfcc"] = time.perf_counter() - t0

        # -- Module 3: CNN feature extraction -------------------------------
        t0 = time.perf_counter()
        backbone = self._get_backbone()
        face_tensor_dev = face_tensor.to(self._device)

        with torch.no_grad():
            visual_features = backbone.extract_features(face_tensor_dev)  # (T, 1792)
            visual_logits = backbone(face_tensor_dev)  # (T, 2)

        # Average across frames
        visual_embedding = visual_features.mean(dim=0, keepdim=True)  # (1, 1792)
        visual_probs = F.softmax(visual_logits, dim=-1).mean(dim=0)
        visual_only_conf = visual_probs[1].item()  # FAKE confidence
        module_times["backbone"] = time.perf_counter() - t0

        # -- Modules 4A, 4B, 4C: Semantic analysis (parallel) ---------------
        t0 = time.perf_counter()
        semantic_scores = self._run_semantic_modules(
            landmarks, energy_envelope, crops, is_video
        )
        module_times["semantic"] = time.perf_counter() - t0

        # -- Module 5 & 6: Fusion + Classification --------------------------
        t0 = time.perf_counter()
        semantic_vector = self._semantic_to_tensor(semantic_scores)
        fusion = self._get_fusion()
        verdict, confidence = fusion.predict(visual_embedding, semantic_vector)
        module_times["fusion"] = time.perf_counter() - t0

        # -- Module 7: Grad-CAM ---------------------------------------------
        t0 = time.perf_counter()
        gradcam_result = self._run_gradcam(face_tensor, crops)
        module_times["gradcam"] = time.perf_counter() - t0

        # -- Module 8: Output assembly --------------------------------------
        total_time = time.perf_counter() - t_start

        # Build audit block
        input_hash = compute_file_sha256(media_path)
        audit = AuditBlock(
            input_file_sha256=input_hash,
            module_times=module_times,
            total_time=round(total_time, 3),
        )

        # Generate summary
        summary = self._generate_summary(
            verdict, confidence, semantic_scores, gradcam_result
        )

        result = AnalysisResult(
            verdict=verdict,
            confidence=round(confidence, 4),
            semantic_scores=semantic_scores,
            gradcam=gradcam_result,
            summary=summary,
            dominant_region=gradcam_result.region_label,
            visual_only_confidence=round(visual_only_conf, 4),
            audit=audit,
            input_filename=Path(media_path).name,
            input_media_type="video" if is_video else "image",
            num_frames_processed=len(frames),
            duration_seconds=round(duration, 2),
        )

        logger.info(
            "Analysis complete: %s (%.1f%% confidence) in %.1fs",
            verdict,
            confidence * 100,
            total_time,
        )
        return result

    # -- Private helpers ----------------------------------------------------

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.device

    def _get_backbone(self) -> EfficientNetB4Backbone:
        if self._backbone is None:
            ckpt = self.config.backbone_checkpoint or None
            self._backbone = EfficientNetB4Backbone(
                checkpoint_path=ckpt
            )
            self._backbone.to(self._device)
            self._backbone.eval()
        return self._backbone

    def _get_fusion(self) -> FusionMLP:
        if self._fusion is None:
            self._fusion = FusionMLP()
            if self.config.fusion_checkpoint:
                state = torch.load(
                    self.config.fusion_checkpoint,
                    map_location=self._device,
                    weights_only=True,
                )
                self._fusion.load_state_dict(state)
            self._fusion.to(self._device)
            self._fusion.eval()
        return self._fusion

    def _run_semantic_modules(
        self,
        landmarks,
        energy_envelope,
        crops,
        is_video: bool,
    ) -> SemanticScores:
        """Run the three semantic modules in parallel."""
        fps = self.config.frame_rate

        lip_sync_scores = None
        blink_scores = None
        lighting_scores = None

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Lip-sync (only for video with audio)
            lip_future = executor.submit(
                self.lip_sync.analyze, landmarks, energy_envelope, fps
            )
            # Blink (only for video)
            blink_future = executor.submit(
                self.blink.analyze, landmarks, fps
            )
            # Lighting (always)
            lighting_future = executor.submit(
                self.lighting.analyze, crops, landmarks
            )

            lip_sync_scores = lip_future.result()
            blink_scores = blink_future.result() if is_video else None
            lighting_scores = lighting_future.result()

        return SemanticScores(
            lip_sync=lip_sync_scores,
            blink=blink_scores,
            lighting=lighting_scores,
        )

    def _semantic_to_tensor(self, scores: SemanticScores) -> torch.Tensor:
        """Convert SemanticScores to a (1, 12) tensor for the fusion MLP."""
        parts = []

        # Lip-sync 4-d
        if scores.lip_sync:
            parts.extend([
                scores.lip_sync.mean_correlation,
                scores.lip_sync.min_correlation,
                scores.lip_sync.std_correlation,
                scores.lip_sync.below_threshold_fraction,
            ])
        else:
            parts.extend([0.0, 0.0, 0.0, 1.0])

        # Blink 4-d
        if scores.blink:
            parts.extend([
                scores.blink.blink_rate_per_min / 30.0,  # normalise
                scores.blink.mean_duration_frames / 10.0,
                scores.blink.ibi_std,
                scores.blink.anomaly_z_score / 5.0,  # normalise
            ])
        else:
            parts.extend([0.0, 0.0, 0.0, -1.0])

        # Lighting 4-d
        if scores.lighting:
            parts.extend([
                scores.lighting.direction_variance,
                scores.lighting.max_pairwise_angle_deg / 180.0,  # normalise
                min(scores.lighting.mean_intensity_ratio / 5.0, 1.0),
                scores.lighting.consistency_score,
            ])
        else:
            parts.extend([0.0, 0.0, 0.0, 1.0])

        tensor = torch.tensor([parts], dtype=torch.float32, device=self._device)
        return tensor

    def _run_gradcam(
        self, face_tensor: torch.Tensor, crops: list[np.ndarray]
    ) -> GradCAMResult:
        """Run Grad-CAM and return the result."""
        try:
            if self._gradcam is None:
                backbone = self._get_backbone()
                self._gradcam = GradCAMExplainer(backbone, device=self._device)

            return self._gradcam.explain(
                face_tensor, crops, self.config.output_dir
            )
        except Exception as exc:
            logger.warning("Grad-CAM failed: %s", exc)
            return GradCAMResult()

    @staticmethod
    def _generate_summary(
        verdict: str,
        confidence: float,
        semantic: SemanticScores,
        gradcam: GradCAMResult,
    ) -> str:
        """Generate a natural-language summary of the evidence."""
        lines = []

        lines.append(
            f"The media is classified as **{verdict}** with "
            f"{confidence*100:.1f}% confidence."
        )

        # Lip-sync evidence
        if semantic.lip_sync and semantic.lip_sync.mean_correlation > 0:
            r = semantic.lip_sync.mean_correlation
            below = semantic.lip_sync.below_threshold_fraction
            if r > 0.7:
                lines.append(
                    f"Lip-audio coherence is strong (r={r:.2f}), "
                    f"consistent with genuine speech."
                )
            elif r > 0.4:
                lines.append(
                    f"Lip-audio coherence is moderate (r={r:.2f}), "
                    f"with {below*100:.0f}% of windows below threshold."
                )
            else:
                lines.append(
                    f"Lip-audio coherence is weak (r={r:.2f}), "
                    f"with {below*100:.0f}% of windows below threshold — "
                    f"suggestive of lip-sync mismatch."
                )

        # Blink evidence
        if semantic.blink:
            z = semantic.blink.anomaly_z_score
            rate = semantic.blink.blink_rate_per_min
            if abs(z) < 1.5:
                lines.append(
                    f"Blink rate ({rate:.0f}/min) is within normal human range."
                )
            elif z < -1.5:
                lines.append(
                    f"Blink rate is abnormally low ({rate:.0f}/min, z={z:.1f}), "
                    f"which is common in GAN-generated faces."
                )
            else:
                lines.append(
                    f"Blink rate is abnormally high ({rate:.0f}/min, z={z:.1f})."
                )

        # Lighting evidence
        if semantic.lighting:
            angle = semantic.lighting.max_pairwise_angle_deg
            if angle < 20:
                lines.append(
                    f"Lighting is consistent across facial regions "
                    f"(max angle {angle:.0f}°)."
                )
            else:
                lines.append(
                    f"Lighting inconsistency detected across facial regions "
                    f"(max angle {angle:.0f}°), suggesting a composite."
                )

        # Grad-CAM region
        if gradcam.region_label:
            lines.append(
                f"The region most influencing the verdict is the "
                f"**{gradcam.region_label}** area."
            )

        return " ".join(lines)
