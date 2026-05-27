"""
DeepForensics configuration — all hyperparameters, thresholds, and
constants from Table 7 of the manuscript in a single location.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Input & Preprocessing
# ---------------------------------------------------------------------------
INPUT_RESOLUTION: Tuple[int, int] = (224, 224)
FRAME_SAMPLE_RATE_FPS: int = 5
AUDIO_SAMPLE_RATE_HZ: int = 16_000

# ImageNet normalisation (applied to every face crop)
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)

# Face-detection fallback threshold: if MediaPipe misses more than this
# fraction of frames, switch to RetinaFace.
MEDIAPIPE_FALLBACK_THRESHOLD: float = 0.30

# ---------------------------------------------------------------------------
# EfficientNet-B4 Backbone
# ---------------------------------------------------------------------------
BACKBONE_MODEL_NAME: str = "efficientnet_b4"
BACKBONE_EMBEDDING_DIM: int = 1792
BACKBONE_NUM_CLASSES: int = 2
BACKBONE_PRETRAINED: bool = True  # ImageNet-21k

# Two-stage training schedule
STAGE1_LR: float = 3e-4       # Head warm-up, backbone frozen
STAGE1_EPOCHS: int = 2
STAGE2_LR: float = 1e-5       # Full fine-tuning, backbone unfrozen
STAGE2_EPOCHS: int = 10
EARLY_STOPPING_PATIENCE: int = 5
WEIGHT_DECAY: float = 1e-4
BATCH_SIZE: int = 16
LABEL_SMOOTHING: float = 0.05

# ---------------------------------------------------------------------------
# Lip-Sync Module (Module 4A)
# ---------------------------------------------------------------------------
MFCC_NUM_COEFFICIENTS: int = 13
MFCC_WINDOW_MS: int = 25
MFCC_HOP_MS: int = 10
LIPSYNC_WINDOW_SECONDS: float = 1.0
LIPSYNC_R_THRESHOLD: float = 0.4
LIPSYNC_MIN_VARIANCE: float = 1e-6  # skip near-zero-variance windows

# MediaPipe 468-mesh outer-lip landmark indices
# Upper lip: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
# Lower lip: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
OUTER_LIP_LANDMARKS: List[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    146, 91, 181, 84, 17, 314, 405, 321, 375,
]

# For MAR computation — vertical pairs and horizontal pair indices
# within OUTER_LIP_LANDMARKS.  Mouth Aspect Ratio uses the standard
# 12-point formulation.
MAR_UPPER_INDICES: List[int] = [2, 3, 4, 5, 6, 7, 8]
MAR_LOWER_INDICES: List[int] = [12, 13, 14, 15, 16, 17, 18]
MAR_LEFT_INDEX: int = 0    # landmark 61
MAR_RIGHT_INDEX: int = 10  # landmark 291

# ---------------------------------------------------------------------------
# Blink Module (Module 4B)
# ---------------------------------------------------------------------------
EAR_THRESHOLD: float = 0.21
BLINK_MIN_FRAMES: int = 2   # at 5 fps
BLINK_MAX_FRAMES: int = 7   # at 5 fps
BLINK_HUMAN_MEAN: float = 16.0   # blinks per minute
BLINK_HUMAN_STD: float = 4.0

# MediaPipe 468-mesh eye landmark indices
# Right eye (from the subject's perspective)
RIGHT_EYE_LANDMARKS: List[int] = [33, 160, 158, 133, 153, 144]
# Left eye
LEFT_EYE_LANDMARKS: List[int] = [362, 385, 387, 263, 373, 380]

# EAR landmark indexing within the 6-point eye array:
# p1=0 (outer), p2=1 (upper-outer), p3=2 (upper-inner),
# p4=3 (inner), p5=4 (lower-inner), p6=5 (lower-outer)

# ---------------------------------------------------------------------------
# Lighting Module (Module 4C)
# ---------------------------------------------------------------------------
# MediaPipe landmark indices for four facial regions
FOREHEAD_LANDMARKS: List[int] = [10, 67, 69, 104, 108, 151, 299, 337, 338]
LEFT_CHEEK_LANDMARKS: List[int] = [36, 50, 101, 116, 117, 118, 119, 123, 205]
RIGHT_CHEEK_LANDMARKS: List[int] = [266, 280, 330, 345, 346, 347, 348, 352, 425]
CHIN_LANDMARKS: List[int] = [152, 148, 176, 149, 150, 377, 400, 378, 379]

# Real faces typically agree within ~15°; composites range 45–90°
LIGHTING_ANGLE_THRESHOLD_DEG: float = 30.0

# ---------------------------------------------------------------------------
# Fusion MLP (Module 5)
# ---------------------------------------------------------------------------
FUSION_INPUT_DIM: int = 1804       # 1792 visual + 4 lip + 4 blink + 4 lighting
FUSION_HIDDEN_1: int = 512
FUSION_HIDDEN_2: int = 128
FUSION_OUTPUT_DIM: int = 2
FUSION_DROPOUT_1: float = 0.3
FUSION_DROPOUT_2: float = 0.2

# ---------------------------------------------------------------------------
# Grad-CAM (Module 7)
# ---------------------------------------------------------------------------
GRADCAM_TOP_K_FRAMES: int = 5
GRADCAM_OVERLAY_ALPHA: float = 0.40

# Coarse facial-region dictionary — maps (row_zone, col_zone) to a label.
# The 224×224 heatmap is divided into a 3×3 grid for coarse localisation.
FACIAL_REGION_MAP: Dict[Tuple[int, int], str] = {
    (0, 0): "forehead",
    (0, 1): "forehead",
    (0, 2): "forehead",
    (1, 0): "left_cheek",
    (1, 1): "nose",
    (1, 2): "right_cheek",
    (2, 0): "boundary",
    (2, 1): "mouth",
    (2, 2): "boundary",
}

# ---------------------------------------------------------------------------
# Output & Audit (Module 8)
# ---------------------------------------------------------------------------
SHA256_PREFIX_LENGTH: int = 12   # how many hex chars of the hash to store

# ---------------------------------------------------------------------------
# Aggregate: default pipeline configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Runtime-overridable configuration for the full DeepForensics pipeline."""

    # Preprocessing
    input_resolution: Tuple[int, int] = INPUT_RESOLUTION
    frame_rate: int = FRAME_SAMPLE_RATE_FPS
    audio_sample_rate: int = AUDIO_SAMPLE_RATE_HZ
    mediapipe_fallback_threshold: float = MEDIAPIPE_FALLBACK_THRESHOLD

    # Backbone
    backbone_model: str = BACKBONE_MODEL_NAME
    backbone_checkpoint: str = ""  # path to fine-tuned weights; empty = ImageNet-pretrained

    # Fusion
    fusion_checkpoint: str = ""  # path to fusion MLP weights

    # Grad-CAM
    gradcam_top_k: int = GRADCAM_TOP_K_FRAMES
    gradcam_alpha: float = GRADCAM_OVERLAY_ALPHA

    # Device
    device: str = "auto"  # "auto", "cuda", "cpu"

    # Output directory for heatmaps and reports
    output_dir: str = "/tmp/deepforensics_output"
