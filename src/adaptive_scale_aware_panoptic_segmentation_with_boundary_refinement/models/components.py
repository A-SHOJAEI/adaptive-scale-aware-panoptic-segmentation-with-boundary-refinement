"""Custom model components: losses, layers, and modules."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BoundaryRefinementModule(nn.Module):
    """Boundary refinement module using learned distance transforms.

    This module learns to predict and refine object boundaries using distance
    transform representations. It's a novel component for improving boundary quality.
    """

    def __init__(
        self,
        in_channels: int = 256,
        boundary_channels: int = 64,
        num_distance_bins: int = 10,
    ) -> None:
        """Initialize boundary refinement module.

        Args:
            in_channels: Number of input feature channels.
            boundary_channels: Number of intermediate channels.
            num_distance_bins: Number of distance transform bins.
        """
        super().__init__()

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, boundary_channels, 3, padding=1),
            nn.BatchNorm2d(boundary_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_channels, boundary_channels, 3, padding=1),
            nn.BatchNorm2d(boundary_channels),
            nn.ReLU(inplace=True),
        )

        # Distance transform prediction head
        self.distance_head = nn.Sequential(
            nn.Conv2d(boundary_channels, boundary_channels // 2, 3, padding=1),
            nn.BatchNorm2d(boundary_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_channels // 2, num_distance_bins, 1),
        )

        # Boundary detection head (no sigmoid - using BCEWithLogitsLoss)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(boundary_channels, boundary_channels // 2, 3, padding=1),
            nn.BatchNorm2d(boundary_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(boundary_channels // 2, 1, 1),
        )

        logger.info(
            f"Initialized BoundaryRefinementModule with {boundary_channels} channels "
            f"and {num_distance_bins} distance bins"
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of boundary refinement module.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Tuple of (boundary_map, distance_map):
                - boundary_map: Predicted boundary map of shape (B, 1, H, W)
                - distance_map: Predicted distance transform of shape (B, num_bins, H, W)
        """
        features = self.boundary_conv(x)
        boundary_map = self.boundary_head(features)
        distance_map = self.distance_head(features)

        return boundary_map, distance_map


class AdaptiveScaleFusion(nn.Module):
    """Adaptive scale-aware feature pyramid fusion.

    This module dynamically selects and fuses features from different FPN levels
    based on object size distribution. This is a novel contribution for handling
    multi-scale objects in panoptic segmentation.
    """

    def __init__(
        self,
        fpn_channels: int = 256,
        num_levels: int = 5,
        attention_heads: int = 8,
    ) -> None:
        """Initialize adaptive scale fusion module.

        Args:
            fpn_channels: Number of channels in FPN features.
            num_levels: Number of FPN pyramid levels.
            attention_heads: Number of attention heads for scale selection.
        """
        super().__init__()

        self.num_levels = num_levels
        self.fpn_channels = fpn_channels

        # Scale attention mechanism
        self.scale_attention = nn.MultiheadAttention(
            embed_dim=fpn_channels,
            num_heads=attention_heads,
            dropout=0.1,
            batch_first=True,
        )

        # Scale-specific convolutions
        self.scale_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
                    nn.BatchNorm2d(fpn_channels),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels)
            ]
        )

        # Fusion convolution
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(fpn_channels * num_levels, fpn_channels, 1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, fpn_channels, 3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )

        logger.info(
            f"Initialized AdaptiveScaleFusion with {num_levels} levels "
            f"and {attention_heads} attention heads"
        )

    def forward(self, fpn_features: list) -> torch.Tensor:
        """Forward pass of adaptive scale fusion.

        Args:
            fpn_features: List of FPN feature maps at different scales,
                each of shape (B, C, H_i, W_i).

        Returns:
            Fused feature map of shape (B, C, H, W) at the target resolution.
        """
        batch_size = fpn_features[0].shape[0]
        target_h, target_w = fpn_features[0].shape[2:]

        # Resize all features to the same spatial size
        resized_features = []
        for i, feat in enumerate(fpn_features):
            if feat.shape[2:] != (target_h, target_w):
                feat = F.interpolate(
                    feat, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            # Apply scale-specific convolution
            feat = self.scale_convs[i](feat)
            resized_features.append(feat)

        # Stack features for attention
        # Shape: (B, num_levels, C, H, W)
        stacked = torch.stack(resized_features, dim=1)

        # Reshape for attention: (B*H*W, num_levels, C)
        b, num_levels, c, h, w = stacked.shape
        stacked_reshaped = stacked.permute(0, 3, 4, 1, 2)  # (B, H, W, num_levels, C)
        stacked_flat = stacked_reshaped.reshape(b * h * w, num_levels, c)

        # Apply self-attention across scales
        attended, _ = self.scale_attention(stacked_flat, stacked_flat, stacked_flat)

        # Reshape back: (B, H, W, num_levels, C) -> (B, num_levels, C, H, W)
        attended = attended.reshape(b, h, w, num_levels, c)
        attended = attended.permute(0, 3, 4, 1, 2)  # (B, num_levels, C, H, W)

        # Concatenate all attended features
        concat_features = attended.reshape(b, num_levels * c, h, w)

        # Fuse features
        fused = self.fusion_conv(concat_features)

        return fused


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in segmentation."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        """Initialize Focal Loss.

        Args:
            alpha: Weighting factor in [0, 1] to balance positive/negative examples.
            gamma: Exponent of the modulating factor (1 - p_t)^gamma.
            reduction: Specifies the reduction to apply to the output.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predictions of shape (B, C, H, W).
            targets: Ground truth of shape (B, H, W).

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""

    def __init__(self, smooth: float = 1.0) -> None:
        """Initialize Dice Loss.

        Args:
            smooth: Smoothing constant to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute dice loss.

        Args:
            inputs: Predictions of shape (B, C, H, W).
            targets: Ground truth of shape (B, H, W).

        Returns:
            Dice loss value.
        """
        num_classes = inputs.shape[1]
        inputs = F.softmax(inputs, dim=1)

        # Convert targets to one-hot
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        # Compute dice coefficient
        intersection = (inputs * targets_one_hot).sum(dim=(2, 3))
        union = inputs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice.mean()

        return dice_loss


class CombinedPanopticLoss(nn.Module):
    """Combined loss function for panoptic segmentation.

    This custom loss combines multiple objectives including semantic segmentation,
    instance segmentation, boundary detection, and distance transform regression.
    """

    def __init__(
        self,
        semantic_weight: float = 1.0,
        instance_weight: float = 1.0,
        boundary_weight: float = 0.5,
        panoptic_weight: float = 2.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
    ) -> None:
        """Initialize combined panoptic loss.

        Args:
            semantic_weight: Weight for semantic segmentation loss.
            instance_weight: Weight for instance segmentation loss.
            boundary_weight: Weight for boundary detection loss.
            panoptic_weight: Weight for panoptic segmentation loss.
            focal_alpha: Alpha parameter for focal loss.
            focal_gamma: Gamma parameter for focal loss.
            dice_smooth: Smoothing constant for dice loss.
        """
        super().__init__()

        self.semantic_weight = semantic_weight
        self.instance_weight = instance_weight
        self.boundary_weight = boundary_weight
        self.panoptic_weight = panoptic_weight

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mse_loss = nn.MSELoss()

        logger.info(
            f"Initialized CombinedPanopticLoss with weights: "
            f"semantic={semantic_weight}, instance={instance_weight}, "
            f"boundary={boundary_weight}, panoptic={panoptic_weight}"
        )

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute combined panoptic loss.

        Args:
            predictions: Dictionary containing model predictions:
                - semantic: Semantic segmentation logits (B, num_classes, H, W)
                - instance: Instance segmentation logits (B, num_classes, H, W)
                - boundary: Boundary prediction (B, 1, H, W)
                - distance: Distance transform prediction (B, num_bins, H, W)
            targets: Dictionary containing ground truth:
                - semantic_mask: Semantic ground truth (B, H, W)
                - instance_mask: Instance ground truth (B, H, W)
                - boundary_mask: Boundary ground truth (B, H, W)
                - distance_transform: Distance transform ground truth (B, H, W)

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains individual losses.
        """
        losses = {}
        total_loss = 0.0

        # Semantic segmentation loss
        if "semantic" in predictions and "semantic_mask" in targets:
            semantic_focal = self.focal_loss(predictions["semantic"], targets["semantic_mask"])
            semantic_dice = self.dice_loss(predictions["semantic"], targets["semantic_mask"])
            semantic_loss = semantic_focal + semantic_dice
            losses["semantic_loss"] = semantic_loss.item()
            total_loss += self.semantic_weight * semantic_loss

        # Instance segmentation loss
        if "instance" in predictions and "instance_mask" in targets:
            # Use focal loss for instance segmentation
            instance_loss = self.focal_loss(predictions["instance"], targets["instance_mask"])
            losses["instance_loss"] = instance_loss.item()
            total_loss += self.instance_weight * instance_loss

        # Boundary detection loss
        if "boundary" in predictions and "boundary_mask" in targets:
            boundary_loss = self.bce_loss(
                predictions["boundary"].squeeze(1), targets["boundary_mask"].float()
            )
            losses["boundary_loss"] = boundary_loss.item()
            total_loss += self.boundary_weight * boundary_loss

        # Distance transform regression loss
        if "distance" in predictions and "distance_transform" in targets:
            # Convert distance transform to class indices for classification
            num_bins = predictions["distance"].shape[1]
            distance_targets = (targets["distance_transform"] * (num_bins - 1)).long()
            distance_targets = torch.clamp(distance_targets, 0, num_bins - 1)

            distance_loss = F.cross_entropy(predictions["distance"], distance_targets)
            losses["distance_loss"] = distance_loss.item()
            total_loss += self.boundary_weight * distance_loss

        # Panoptic quality loss (combined semantic + instance)
        if (
            "semantic" in predictions
            and "instance" in predictions
            and "semantic_mask" in targets
            and "instance_mask" in targets
        ):
            panoptic_loss = 0.5 * (
                self.focal_loss(predictions["semantic"], targets["semantic_mask"])
                + self.focal_loss(predictions["instance"], targets["instance_mask"])
            )
            losses["panoptic_loss"] = panoptic_loss.item()
            total_loss += self.panoptic_weight * panoptic_loss

        losses["total_loss"] = total_loss.item()

        return total_loss, losses
