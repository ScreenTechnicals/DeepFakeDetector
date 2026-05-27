"""
DeepForensics Grad-CAM Explainability (Module 7).

Computes the gradient of the FAKE-class logit with respect to the
feature maps of the last convolutional block of EfficientNet-B4 (Eq. 3):

    L_GradCAM_FAKE = ReLU( Σ_k  α_k^FAKE · A_k )

For video inputs, the top-K frames ranked by FAKE confidence are
selected, Grad-CAM is computed individually on each, and the maps are
averaged.  The peak activation coordinate is mapped onto a coarse
facial-region dictionary.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from deepforensics.config import (
    FACIAL_REGION_MAP,
    GRADCAM_OVERLAY_ALPHA,
    GRADCAM_TOP_K_FRAMES,
    INPUT_RESOLUTION,
)
from deepforensics.result import GradCAMResult

logger = logging.getLogger(__name__)

# Index of the FAKE class in the 2-class output
_FAKE_CLASS_IDX = 1


class GradCAMExplainer:
    """
    Produces Grad-CAM heatmaps from the EfficientNet-B4 backbone
    and maps peak activation to a facial region label.
    """

    def __init__(self, backbone, device: str = "cpu"):
        """
        Parameters
        ----------
        backbone : EfficientNetB4Backbone
            The loaded visual backbone (already on ``device``).
        device : str
            "cpu" or "cuda".
        """
        self.backbone = backbone
        self.device = device

        self._gradients: Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks = []

        self._register_hooks()

    # -- Hook management ----------------------------------------------------

    def _register_hooks(self):
        target_layer = self.backbone.get_last_conv_layer()
        if target_layer is None:
            logger.warning("No target layer found — Grad-CAM will be unavailable.")
            return

        self._hooks.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self._hooks.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, input, output):
        self._activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # -- Main API -----------------------------------------------------------

    @torch.enable_grad()
    def explain(
        self,
        face_crops: torch.Tensor,
        original_frames: list[np.ndarray],
        output_dir: str,
    ) -> GradCAMResult:
        """
        Generate a Grad-CAM heatmap for the given face crops.

        Parameters
        ----------
        face_crops : torch.Tensor, shape (N, 3, 224, 224)
            Normalised face-crop tensor.
        original_frames : list of np.ndarray
            Original BGR face crops (224×224) for overlay rendering.
        output_dir : str
            Directory to save heatmap and overlay PNGs.

        Returns
        -------
        GradCAMResult
        """
        if self._activations is None and not self._hooks:
            return GradCAMResult()

        os.makedirs(output_dir, exist_ok=True)

        face_crops = face_crops.to(self.device)
        n_frames = face_crops.shape[0]

        # -- 1. Get per-frame FAKE confidence ------------------------------
        self.backbone.eval()
        with torch.no_grad():
            logits = self.backbone(face_crops)
            probs = F.softmax(logits, dim=-1)
            fake_confs = probs[:, _FAKE_CLASS_IDX]  # (N,)

        # -- 2. Select top-K frames ----------------------------------------
        k = min(GRADCAM_TOP_K_FRAMES, n_frames)
        top_indices = torch.topk(fake_confs, k).indices.cpu().tolist()
        avg_fake_conf = fake_confs[top_indices].mean().item()

        # -- 3. Compute Grad-CAM per selected frame, then average ----------
        heatmaps = []
        for idx in top_indices:
            hmap = self._compute_gradcam_single(face_crops[idx : idx + 1])
            if hmap is not None:
                heatmaps.append(hmap)

        if not heatmaps:
            return GradCAMResult(confidence=avg_fake_conf)

        avg_heatmap = np.mean(heatmaps, axis=0)

        # -- 4. Upsample to 224×224 ----------------------------------------
        heatmap_resized = cv2.resize(
            avg_heatmap, INPUT_RESOLUTION, interpolation=cv2.INTER_LINEAR
        )
        # Normalise to [0, 1]
        hmin, hmax = heatmap_resized.min(), heatmap_resized.max()
        if hmax - hmin > 1e-6:
            heatmap_resized = (heatmap_resized - hmin) / (hmax - hmin)
        else:
            heatmap_resized = np.zeros_like(heatmap_resized)

        # -- 5. Peak coordinate → region label -----------------------------
        peak_yx = np.unravel_index(
            np.argmax(heatmap_resized), heatmap_resized.shape
        )
        peak_row, peak_col = int(peak_yx[0]), int(peak_yx[1])
        region_label = self._map_to_region(peak_row, peak_col)

        # -- 6. Save heatmap PNG -------------------------------------------
        heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        heatmap_path = os.path.join(output_dir, "gradcam_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap_color)

        # -- 7. Save overlay PNG -------------------------------------------
        # Use the first of the top-K frames as the base
        base_idx = top_indices[0]
        if base_idx < len(original_frames):
            base_frame = original_frames[base_idx].copy()
        else:
            base_frame = np.zeros((*INPUT_RESOLUTION, 3), dtype=np.uint8)

        overlay = cv2.addWeighted(
            base_frame,
            1.0 - GRADCAM_OVERLAY_ALPHA,
            heatmap_color,
            GRADCAM_OVERLAY_ALPHA,
            0,
        )
        overlay_path = os.path.join(output_dir, "gradcam_overlay.png")
        cv2.imwrite(overlay_path, overlay)

        return GradCAMResult(
            heatmap_path=heatmap_path,
            overlay_path=overlay_path,
            peak_coordinate=(peak_row, peak_col),
            region_label=region_label,
            confidence=round(avg_fake_conf, 4),
        )

    # -- Internal -----------------------------------------------------------

    def _compute_gradcam_single(
        self, x: torch.Tensor
    ) -> Optional[np.ndarray]:
        """Compute Grad-CAM for a single (1, 3, 224, 224) input."""
        x = x.to(self.device).requires_grad_(True)
        self.backbone.zero_grad()

        # Forward
        logits = self.backbone(x)
        fake_logit = logits[0, _FAKE_CLASS_IDX]

        # Backward
        fake_logit.backward(retain_graph=False)

        if self._gradients is None or self._activations is None:
            return None

        # Global-average-pool gradients per channel → importance weights α_k
        weights = self._gradients.mean(dim=(-2, -1))  # (1, C)

        # Weighted sum of feature maps
        activations = self._activations  # (1, C, H, W)
        cam = (weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1)  # (1, H, W)
        cam = F.relu(cam)  # ReLU

        return cam[0].cpu().numpy()

    @staticmethod
    def _map_to_region(row: int, col: int) -> str:
        """Map a (row, col) peak coordinate to a coarse facial region."""
        # Divide 224×224 into a 3×3 grid
        h, w = INPUT_RESOLUTION
        row_zone = min(2, (row * 3) // h)
        col_zone = min(2, (col * 3) // w)
        return FACIAL_REGION_MAP.get((row_zone, col_zone), "unknown")
