"""Tests for model architecture and components."""

import pytest
import torch

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.components import (
    AdaptiveScaleFusion,
    BoundaryRefinementModule,
    CombinedPanopticLoss,
    DiceLoss,
    FocalLoss,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.model import (
    AdaptiveScalePanopticSegmentation,
    FPN,
)


class TestComponents:
    """Tests for custom model components."""

    def test_boundary_refinement_module(self, batch_size: int) -> None:
        """Test boundary refinement module."""
        module = BoundaryRefinementModule(
            in_channels=256,
            boundary_channels=64,
            num_distance_bins=10,
        )

        x = torch.randn(batch_size, 256, 64, 64)
        boundary_map, distance_map = module(x)

        assert boundary_map.shape == torch.Size([batch_size, 1, 64, 64])
        assert distance_map.shape == torch.Size([batch_size, 10, 64, 64])

    def test_adaptive_scale_fusion(self, batch_size: int) -> None:
        """Test adaptive scale fusion module."""
        module = AdaptiveScaleFusion(
            fpn_channels=256,
            num_levels=4,
            attention_heads=4,
        )

        # Create multi-scale features
        fpn_features = [
            torch.randn(batch_size, 256, 64, 64),
            torch.randn(batch_size, 256, 32, 32),
            torch.randn(batch_size, 256, 16, 16),
            torch.randn(batch_size, 256, 8, 8),
        ]

        fused = module(fpn_features)

        assert fused.shape == torch.Size([batch_size, 256, 64, 64])

    def test_focal_loss(self, batch_size: int, num_classes: int) -> None:
        """Test focal loss."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        inputs = torch.randn(batch_size, num_classes, 32, 32)
        targets = torch.randint(0, num_classes, (batch_size, 32, 32))

        loss = loss_fn(inputs, targets)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0.0

    def test_dice_loss(self, batch_size: int, num_classes: int) -> None:
        """Test dice loss."""
        loss_fn = DiceLoss(smooth=1.0)

        inputs = torch.randn(batch_size, num_classes, 32, 32)
        targets = torch.randint(0, num_classes, (batch_size, 32, 32))

        loss = loss_fn(inputs, targets)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0.0

    def test_combined_panoptic_loss(self, batch_size: int, num_classes: int) -> None:
        """Test combined panoptic loss."""
        loss_fn = CombinedPanopticLoss(
            semantic_weight=1.0,
            instance_weight=1.0,
            boundary_weight=0.5,
            panoptic_weight=2.0,
        )

        predictions = {
            "semantic": torch.randn(batch_size, num_classes, 64, 64),
            "instance": torch.randn(batch_size, 81, 64, 64),
            "boundary": torch.sigmoid(torch.randn(batch_size, 1, 64, 64)),
            "distance": torch.randn(batch_size, 10, 64, 64),
        }

        targets = {
            "semantic_mask": torch.randint(0, num_classes, (batch_size, 64, 64)),
            "instance_mask": torch.randint(0, 81, (batch_size, 64, 64)),
            "boundary_mask": torch.rand(batch_size, 64, 64),
            "distance_transform": torch.rand(batch_size, 64, 64),
        }

        total_loss, loss_dict = loss_fn(predictions, targets)

        assert isinstance(total_loss.item(), float)
        assert total_loss.item() >= 0.0
        assert "total_loss" in loss_dict


class TestFPN:
    """Tests for Feature Pyramid Network."""

    def test_fpn_forward(self, batch_size: int) -> None:
        """Test FPN forward pass."""
        in_channels_list = [256, 512, 1024, 2048]
        fpn = FPN(in_channels_list, out_channels=256)

        # Create multi-scale features
        features = [
            torch.randn(batch_size, 256, 64, 64),
            torch.randn(batch_size, 512, 32, 32),
            torch.randn(batch_size, 1024, 16, 16),
            torch.randn(batch_size, 2048, 8, 8),
        ]

        fpn_features = fpn(features)

        assert len(fpn_features) == 4
        assert fpn_features[0].shape[1] == 256
        assert fpn_features[1].shape[1] == 256
        assert fpn_features[2].shape[1] == 256
        assert fpn_features[3].shape[1] == 256


class TestModel:
    """Tests for main panoptic segmentation model."""

    def test_model_creation(self, num_classes: int, thing_classes: int) -> None:
        """Test model creation."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
            use_boundary_refinement=True,
            use_scale_adaptive_fusion=True,
        )

        assert model is not None
        assert model.num_classes == num_classes
        assert model.thing_classes == thing_classes

    def test_model_forward(
        self, batch_size: int, num_classes: int, thing_classes: int, device: torch.device
    ) -> None:
        """Test model forward pass."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
            fpn_channels=256,
            use_boundary_refinement=True,
            use_scale_adaptive_fusion=True,
        ).to(device)

        x = torch.randn(batch_size, 3, 256, 256).to(device)
        predictions = model(x)

        assert "semantic" in predictions
        assert "instance" in predictions
        assert "boundary" in predictions
        assert "distance" in predictions

        assert predictions["semantic"].shape == torch.Size([batch_size, num_classes, 256, 256])
        assert predictions["instance"].shape == torch.Size([batch_size, thing_classes + 1, 256, 256])

    def test_model_without_novel_components(
        self, batch_size: int, num_classes: int, thing_classes: int
    ) -> None:
        """Test model without boundary refinement and scale fusion (ablation)."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
            use_boundary_refinement=False,
            use_scale_adaptive_fusion=False,
        )

        x = torch.randn(batch_size, 3, 256, 256)
        predictions = model(x)

        assert "semantic" in predictions
        assert "instance" in predictions
        assert "boundary" not in predictions  # Disabled
        assert "distance" not in predictions  # Disabled

    def test_model_parameter_count(self, num_classes: int, thing_classes: int) -> None:
        """Test model has reasonable parameter count."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
        )

        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        assert num_params > 1_000_000  # Should have at least 1M parameters
        assert num_trainable == num_params  # All should be trainable
