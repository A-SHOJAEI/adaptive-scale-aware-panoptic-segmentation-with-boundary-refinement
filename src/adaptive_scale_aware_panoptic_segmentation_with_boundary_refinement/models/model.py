"""Main panoptic segmentation model with adaptive scale-aware fusion."""

import logging
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.components import (
    AdaptiveScaleFusion,
    BoundaryRefinementModule,
)

logger = logging.getLogger(__name__)


class FPN(nn.Module):
    """Feature Pyramid Network for multi-scale feature extraction."""

    def __init__(self, in_channels_list: list, out_channels: int = 256) -> None:
        """Initialize FPN.

        Args:
            in_channels_list: List of input channel counts for each backbone level.
            out_channels: Number of output channels for all FPN levels.
        """
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for in_channels in in_channels_list:
            lateral_conv = nn.Conv2d(in_channels, out_channels, 1)
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)

    def forward(self, features: list) -> list:
        """Forward pass through FPN.

        Args:
            features: List of feature maps from backbone.

        Returns:
            List of FPN feature maps at different scales.
        """
        # Build top-down pathway
        laterals = [lateral_conv(features[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        # Start from the top level
        fpn_features = [laterals[-1]]

        for i in range(len(laterals) - 2, -1, -1):
            # Upsample and add
            upsampled = F.interpolate(
                fpn_features[-1], size=laterals[i].shape[2:], mode="nearest"
            )
            fpn_features.append(laterals[i] + upsampled)

        # Reverse to get bottom-up order
        fpn_features = fpn_features[::-1]

        # Apply 3x3 convs
        fpn_features = [self.fpn_convs[i](feat) for i, feat in enumerate(fpn_features)]

        return fpn_features


class AdaptiveScalePanopticSegmentation(nn.Module):
    """Adaptive Scale-Aware Panoptic Segmentation Model.

    This model combines:
    1. ResNet backbone with FPN for multi-scale features
    2. Adaptive scale-aware fusion for dynamic feature selection
    3. Boundary refinement module for improved boundary quality
    4. Separate heads for semantic and instance segmentation
    """

    def __init__(
        self,
        num_classes: int = 133,
        thing_classes: int = 80,
        backbone: str = "resnet50",
        pretrained: bool = True,
        fpn_channels: int = 256,
        num_fpn_levels: int = 5,
        use_boundary_refinement: bool = True,
        use_scale_adaptive_fusion: bool = True,
        boundary_channels: int = 64,
        distance_transform_bins: int = 10,
        scale_attention_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the panoptic segmentation model.

        Args:
            num_classes: Total number of classes (thing + stuff).
            thing_classes: Number of thing classes (with instances).
            backbone: Backbone architecture name.
            pretrained: Whether to use pretrained backbone weights.
            fpn_channels: Number of channels in FPN.
            num_fpn_levels: Number of FPN pyramid levels.
            use_boundary_refinement: Whether to use boundary refinement module.
            use_scale_adaptive_fusion: Whether to use adaptive scale fusion.
            boundary_channels: Number of channels in boundary refinement.
            distance_transform_bins: Number of distance transform bins.
            scale_attention_heads: Number of attention heads for scale fusion.
            dropout: Dropout rate.
        """
        super().__init__()

        self.num_classes = num_classes
        self.thing_classes = thing_classes
        self.use_boundary_refinement = use_boundary_refinement
        self.use_scale_adaptive_fusion = use_scale_adaptive_fusion

        # Initialize backbone
        if backbone == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet50(weights=weights)
            backbone_channels = [256, 512, 1024, 2048]
        elif backbone == "resnet101":
            weights = models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = models.resnet101(weights=weights)
            backbone_channels = [256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Extract backbone layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Feature Pyramid Network
        self.fpn = FPN(backbone_channels, fpn_channels)

        # Adaptive scale fusion module
        if use_scale_adaptive_fusion:
            self.scale_fusion = AdaptiveScaleFusion(
                fpn_channels=fpn_channels,
                num_levels=len(backbone_channels),
                attention_heads=scale_attention_heads,
            )
            fusion_channels = fpn_channels
        else:
            fusion_channels = fpn_channels

        # Boundary refinement module
        if use_boundary_refinement:
            self.boundary_refinement = BoundaryRefinementModule(
                in_channels=fusion_channels,
                boundary_channels=boundary_channels,
                num_distance_bins=distance_transform_bins,
            )

        # Semantic segmentation head
        self.semantic_head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(fusion_channels, fusion_channels // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, num_classes, 1),
        )

        # Instance segmentation head
        self.instance_head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels, 3, padding=1),
            nn.BatchNorm2d(fusion_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(fusion_channels, fusion_channels // 2, 3, padding=1),
            nn.BatchNorm2d(fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 2, thing_classes + 1, 1),  # +1 for background
        )

        logger.info(
            f"Initialized AdaptiveScalePanopticSegmentation with {num_classes} classes, "
            f"backbone={backbone}, boundary_refinement={use_boundary_refinement}, "
            f"scale_fusion={use_scale_adaptive_fusion}"
        )

    def forward_backbone(self, x: torch.Tensor) -> list:
        """Extract multi-scale features from backbone.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            List of feature maps at different scales.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Backbone layers
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through the entire model.

        Args:
            x: Input image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing predictions:
                - semantic: Semantic segmentation logits (B, num_classes, H, W)
                - instance: Instance segmentation logits (B, thing_classes+1, H, W)
                - boundary: Boundary prediction (B, 1, H, W) [if enabled]
                - distance: Distance transform prediction (B, num_bins, H, W) [if enabled]
        """
        input_size = x.shape[2:]

        # Extract backbone features
        backbone_features = self.forward_backbone(x)

        # Build FPN
        fpn_features = self.fpn(backbone_features)

        # Adaptive scale fusion
        if self.use_scale_adaptive_fusion:
            fused_features = self.scale_fusion(fpn_features)
        else:
            # Simple fusion: just use the finest resolution FPN level
            fused_features = fpn_features[0]

        # Boundary refinement
        predictions = {}
        if self.use_boundary_refinement:
            boundary_map, distance_map = self.boundary_refinement(fused_features)
            predictions["boundary"] = boundary_map
            predictions["distance"] = distance_map

            # Refine features using boundary information
            boundary_attention = boundary_map.sigmoid()
            fused_features = fused_features * (1 + boundary_attention)

        # Semantic segmentation
        semantic_logits = self.semantic_head(fused_features)
        semantic_logits = F.interpolate(
            semantic_logits, size=input_size, mode="bilinear", align_corners=False
        )
        predictions["semantic"] = semantic_logits

        # Instance segmentation
        instance_logits = self.instance_head(fused_features)
        instance_logits = F.interpolate(
            instance_logits, size=input_size, mode="bilinear", align_corners=False
        )
        predictions["instance"] = instance_logits

        # Upsample boundary and distance maps if present
        if "boundary" in predictions:
            predictions["boundary"] = F.interpolate(
                predictions["boundary"], size=input_size, mode="bilinear", align_corners=False
            )
        if "distance" in predictions:
            predictions["distance"] = F.interpolate(
                predictions["distance"], size=input_size, mode="bilinear", align_corners=False
            )

        return predictions
