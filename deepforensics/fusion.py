"""
DeepForensics Fusion MLP (Module 5) and Classification Head (Module 6).

A deliberately small three-layer MLP that combines the 1792-d visual
embedding with the three 4-d semantic feature vectors (lip-sync, blink,
lighting) into a single calibrated verdict.

Architecture:
    Linear(1804 → 512) → GELU → Dropout(0.3)
    Linear(512  → 128) → GELU → Dropout(0.2)
    Linear(128  →   2) → Softmax (at inference)

Label smoothing (ε = 0.05) keeps confidence outputs honest; ECE is
monitored as a secondary metric during training.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepforensics.config import (
    FUSION_DROPOUT_1,
    FUSION_DROPOUT_2,
    FUSION_HIDDEN_1,
    FUSION_HIDDEN_2,
    FUSION_INPUT_DIM,
    FUSION_OUTPUT_DIM,
    LABEL_SMOOTHING,
)


class FusionMLP(nn.Module):
    """
    Three-layer fusion head that combines visual and semantic features.

    The visual backbone must be **frozen** during fusion training to
    preserve cross-dataset generalisation.
    """

    def __init__(
        self,
        input_dim: int = FUSION_INPUT_DIM,
        hidden1: int = FUSION_HIDDEN_1,
        hidden2: int = FUSION_HIDDEN_2,
        output_dim: int = FUSION_OUTPUT_DIM,
        dropout1: float = FUSION_DROPOUT_1,
        dropout2: float = FUSION_DROPOUT_2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden2, output_dim),
        )

    def forward(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass → 2-class logits (no softmax).

        Parameters
        ----------
        visual_features : torch.Tensor, shape (B, 1792)
            Embeddings from EfficientNet-B4.
        semantic_features : torch.Tensor, shape (B, 12)
            Concatenated 4-d vectors from lip-sync, blink, and lighting.

        Returns
        -------
        torch.Tensor, shape (B, 2)
            Raw logits.  Index 0 = REAL, Index 1 = FAKE.
        """
        x = torch.cat([visual_features, semantic_features], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def predict(
        self,
        visual_features: torch.Tensor,
        semantic_features: torch.Tensor,
    ) -> Tuple[str, float]:
        """
        Convenience method for single-sample inference.

        Returns
        -------
        verdict : str
            "REAL" or "FAKE".
        confidence : float
            Softmax probability of the predicted class.
        """
        self.eval()
        logits = self.forward(visual_features, semantic_features)
        probs = F.softmax(logits, dim=-1)

        fake_prob = probs[0, 1].item()
        if fake_prob >= 0.5:
            return "FAKE", fake_prob
        else:
            return "REAL", 1.0 - fake_prob


def label_smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = LABEL_SMOOTHING,
) -> torch.Tensor:
    """
    Cross-entropy loss with label smoothing (ε = 0.05 by default).

    Label smoothing prevents the model from producing over-confident
    predictions and is the main reason the system can report
    meaningful calibrated probabilities rather than saturated near-one
    values.

    Parameters
    ----------
    logits : torch.Tensor, shape (B, C)
        Raw class logits.
    targets : torch.Tensor, shape (B,)
        Ground-truth class indices.
    epsilon : float
        Label smoothing factor.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    n_classes = logits.size(-1)
    log_probs = F.log_softmax(logits, dim=-1)

    # One-hot with smoothing
    with torch.no_grad():
        smooth_targets = torch.full_like(log_probs, epsilon / (n_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - epsilon)

    loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
    return loss


def expected_calibration_error(
    confidences: torch.Tensor,
    accuracies: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE) over ``n_bins`` reliability
    bins.  This is used as a secondary training metric (Section 4.1).

    Parameters
    ----------
    confidences : torch.Tensor, shape (N,)
        Predicted confidence for the chosen class.
    accuracies : torch.Tensor, shape (N,)
        Binary correctness (1.0 if correct, 0.0 otherwise).
    n_bins : int
        Number of reliability bins.

    Returns
    -------
    float
        ECE value.
    """
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum().item()
        if count == 0:
            continue
        avg_conf = confidences[mask].mean().item()
        avg_acc = accuracies[mask].mean().item()
        ece += (count / total) * abs(avg_acc - avg_conf)

    return ece
