"""
DeepForensics Visual Backbone (Module 3).

EfficientNet-B4 loaded with ImageNet-21k pretrained weights, with
the classifier head replaced by a two-class layer.  Provides the
1792-dimensional embedding that feeds into the fusion MLP and the
convolutional feature maps that Grad-CAM hooks into.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

from deepforensics.config import (
    BACKBONE_EMBEDDING_DIM,
    BACKBONE_MODEL_NAME,
    BACKBONE_NUM_CLASSES,
    BACKBONE_PRETRAINED,
)

logger = logging.getLogger(__name__)


class EfficientNetB4Backbone(nn.Module):
    """
    EfficientNet-B4 wrapper for deepfake detection.

    Two-stage training support:
      Stage 1 — ``freeze_backbone()``  → only the head trains.
      Stage 2 — ``unfreeze_backbone()`` → full fine-tuning.
    """

    def __init__(
        self,
        num_classes: int = BACKBONE_NUM_CLASSES,
        pretrained: bool = BACKBONE_PRETRAINED,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__()

        try:
            import timm
        except ImportError:
            raise ImportError(
                "timm is required for EfficientNet-B4. "
                "Install with: pip install timm"
            )

        self.model = timm.create_model(
            BACKBONE_MODEL_NAME,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.embedding_dim = BACKBONE_EMBEDDING_DIM
        self.num_classes = num_classes

        # Store reference to the last conv block for Grad-CAM
        # In timm's EfficientNet, the last conv block is model.conv_head
        # or the last block in model.blocks
        self._last_conv_layer: Optional[nn.Module] = None
        self._resolve_last_conv()

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def _resolve_last_conv(self):
        """Find the last convolutional layer for Grad-CAM."""
        # timm EfficientNet structure:
        #   model.blocks[-1] → last stage of MBConv blocks
        #   model.conv_head  → final 1×1 conv
        if hasattr(self.model, "conv_head"):
            self._last_conv_layer = self.model.conv_head
        elif hasattr(self.model, "blocks"):
            self._last_conv_layer = self.model.blocks[-1]
        else:
            logger.warning(
                "Could not resolve last conv layer — Grad-CAM may not work."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass → 2-class logits.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor, shape (B, num_classes)
        """
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract the 1792-dimensional embedding from the penultimate layer.

        Parameters
        ----------
        x : torch.Tensor, shape (B, 3, 224, 224)

        Returns
        -------
        torch.Tensor, shape (B, 1792)
        """
        return self.model.forward_features(x).mean(dim=(-2, -1))

    def get_last_conv_layer(self) -> Optional[nn.Module]:
        """Return the last convolutional layer for Grad-CAM hooking."""
        return self._last_conv_layer

    def freeze_backbone(self):
        """Freeze all parameters except the classifier head (Stage 1)."""
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreeze the classifier head
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        logger.info("Backbone frozen — only classifier head is trainable.")

    def unfreeze_backbone(self):
        """Unfreeze all parameters (Stage 2)."""
        for param in self.model.parameters():
            param.requires_grad = True
        logger.info("Backbone unfrozen — all parameters are trainable.")

    def load_checkpoint(self, path: str):
        """Load fine-tuned weights from a checkpoint file."""
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        # Handle DataParallel wrappers
        if any(k.startswith("module.") for k in state_dict):
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
        self.load_state_dict(state_dict, strict=False)
        logger.info("Loaded checkpoint from %s", path)
