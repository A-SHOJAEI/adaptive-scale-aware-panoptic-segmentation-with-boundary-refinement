"""Tests for training components."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.components import (
    CombinedPanopticLoss,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.model import (
    AdaptiveScalePanopticSegmentation,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.training.trainer import (
    PanopticTrainer,
)


class TestTrainer:
    """Tests for training loop."""

    @pytest.fixture
    def dummy_dataloader(self, batch_size: int, input_size: tuple):
        """Create a dummy data loader for testing."""
        h, w = input_size
        num_samples = 10

        images = torch.randn(num_samples, 3, h, w)
        semantic_masks = torch.randint(0, 133, (num_samples, h, w))
        instance_masks = torch.randint(0, 10, (num_samples, h, w))
        boundary_masks = torch.rand(num_samples, h, w)
        distance_transforms = torch.rand(num_samples, h, w)

        # Create a custom dataset that returns dictionaries
        class DictDataset(torch.utils.data.Dataset):
            def __init__(self, images, semantic, instance, boundary, distance):
                self.images = images
                self.semantic = semantic
                self.instance = instance
                self.boundary = boundary
                self.distance = distance

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return {
                    "image": self.images[idx],
                    "semantic_mask": self.semantic[idx],
                    "instance_mask": self.instance[idx],
                    "boundary_mask": self.boundary[idx],
                    "distance_transform": self.distance[idx],
                }

        dataset = DictDataset(
            images, semantic_masks, instance_masks, boundary_masks, distance_transforms
        )

        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def test_trainer_creation(
        self,
        dummy_dataloader,
        config_dict: dict,
        num_classes: int,
        thing_classes: int,
        device: torch.device,
    ) -> None:
        """Test trainer creation."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
        )

        criterion = CombinedPanopticLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        trainer = PanopticTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            config=config_dict,
        )

        assert trainer is not None
        assert trainer.model is not None
        assert trainer.optimizer is not None

    def test_train_epoch(
        self,
        dummy_dataloader,
        config_dict: dict,
        num_classes: int,
        thing_classes: int,
        device: torch.device,
    ) -> None:
        """Test single training epoch."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
            fpn_channels=128,  # Smaller for faster testing
        )

        criterion = CombinedPanopticLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Modify config for faster testing
        config_dict["training"]["mixed_precision"] = False
        config_dict["training"]["gradient_clip_norm"] = 1.0

        trainer = PanopticTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            config=config_dict,
        )

        metrics = trainer.train_epoch()

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0.0

    def test_validate(
        self,
        dummy_dataloader,
        config_dict: dict,
        num_classes: int,
        thing_classes: int,
        device: torch.device,
    ) -> None:
        """Test validation."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
            fpn_channels=128,
        )

        criterion = CombinedPanopticLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        config_dict["training"]["mixed_precision"] = False

        trainer = PanopticTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            config=config_dict,
        )

        metrics = trainer.validate()

        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)
        assert metrics["loss"] >= 0.0

    def test_checkpoint_saving(
        self,
        dummy_dataloader,
        config_dict: dict,
        num_classes: int,
        thing_classes: int,
        device: torch.device,
        tmp_path,
    ) -> None:
        """Test checkpoint saving."""
        model = AdaptiveScalePanopticSegmentation(
            num_classes=num_classes,
            thing_classes=thing_classes,
            backbone="resnet50",
            pretrained=False,
        )

        criterion = CombinedPanopticLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Use temporary directory for checkpoints
        config_dict["system"]["checkpoint_dir"] = str(tmp_path)

        trainer = PanopticTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=dummy_dataloader,
            val_loader=dummy_dataloader,
            device=device,
            config=config_dict,
        )

        # Save checkpoint
        trainer.save_checkpoint("test_checkpoint.pth")

        checkpoint_path = tmp_path / "test_checkpoint.pth"
        assert checkpoint_path.exists()

        # Load checkpoint
        trainer.load_checkpoint(str(checkpoint_path))
        assert trainer.current_epoch == 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_panoptic_metrics(self, batch_size: int, num_classes: int, thing_classes: int) -> None:
        """Test panoptic metrics computation."""
        from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.evaluation.metrics import (
            PanopticMetrics,
        )

        metrics = PanopticMetrics(num_classes=num_classes, thing_classes=thing_classes)

        # Create dummy predictions and targets
        predictions = {
            "semantic": torch.randn(batch_size, num_classes, 64, 64),
            "instance": torch.randn(batch_size, thing_classes + 1, 64, 64),
            "boundary": torch.sigmoid(torch.randn(batch_size, 1, 64, 64)),
        }

        targets = {
            "semantic_mask": torch.randint(0, num_classes, (batch_size, 64, 64)),
            "instance_mask": torch.randint(0, thing_classes + 1, (batch_size, 64, 64)),
            "boundary_mask": torch.rand(batch_size, 64, 64),
        }

        metrics.update(predictions, targets)
        computed_metrics = metrics.compute()

        assert "panoptic_quality" in computed_metrics
        assert "segmentation_quality" in computed_metrics
        assert "recognition_quality" in computed_metrics
        assert "boundary_iou" in computed_metrics

    def test_boundary_iou(self) -> None:
        """Test boundary IoU computation."""
        from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.evaluation.metrics import (
            compute_boundary_iou,
        )

        import numpy as np

        # Perfect match
        pred = np.ones((100, 100), dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        iou = compute_boundary_iou(pred, gt)
        assert iou == 1.0

        # No match
        pred = np.zeros((100, 100), dtype=np.float32)
        gt = np.ones((100, 100), dtype=np.float32)
        iou = compute_boundary_iou(pred, gt)
        assert iou == 0.0

        # Partial match
        pred = np.zeros((100, 100), dtype=np.float32)
        pred[:50, :] = 1.0
        gt = np.zeros((100, 100), dtype=np.float32)
        gt[25:75, :] = 1.0
        iou = compute_boundary_iou(pred, gt)
        assert 0.0 < iou < 1.0
