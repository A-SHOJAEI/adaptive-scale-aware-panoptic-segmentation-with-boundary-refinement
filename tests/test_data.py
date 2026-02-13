"""Tests for data loading and preprocessing."""

import numpy as np
import pytest
import torch

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.loader import (
    COCOPanopticDataset,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.preprocessing import (
    compute_boundary_mask,
    compute_distance_transform,
    create_scale_map,
    get_train_transforms,
    get_val_transforms,
)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_get_train_transforms(self, input_size: tuple) -> None:
        """Test train transforms creation."""
        transform = get_train_transforms(input_size)
        assert transform is not None

        # Test transform on sample data
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 133, (512, 512), dtype=np.int64)
        boundary = np.random.rand(512, 512).astype(np.float32)

        transformed = transform(image=image, mask=mask, boundary=boundary)
        assert "image" in transformed
        assert "mask" in transformed
        assert "boundary" in transformed

    def test_get_val_transforms(self, input_size: tuple) -> None:
        """Test validation transforms creation."""
        transform = get_val_transforms(input_size)
        assert transform is not None

        # Test transform on sample data
        image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        mask = np.random.randint(0, 133, (512, 512), dtype=np.int64)
        boundary = np.random.rand(512, 512).astype(np.float32)

        transformed = transform(image=image, mask=mask, boundary=boundary)
        assert transformed["image"].shape == torch.Size([3, input_size[0], input_size[1]])

    def test_compute_boundary_mask(self) -> None:
        """Test boundary mask computation."""
        semantic_mask = np.zeros((100, 100), dtype=np.int64)
        semantic_mask[20:40, 20:40] = 1
        semantic_mask[60:80, 60:80] = 2

        boundary = compute_boundary_mask(semantic_mask, kernel_size=3)

        assert boundary.shape == semantic_mask.shape
        assert boundary.dtype == np.float32
        assert boundary.min() >= 0.0
        assert boundary.max() <= 1.0
        assert boundary.sum() > 0  # Should have some boundary pixels

    def test_compute_distance_transform(self) -> None:
        """Test distance transform computation."""
        boundary_mask = np.zeros((100, 100), dtype=np.float32)
        boundary_mask[50, :] = 1.0  # Horizontal line

        dist_transform = compute_distance_transform(boundary_mask, max_distance=50.0)

        assert dist_transform.shape == boundary_mask.shape
        assert dist_transform.dtype == np.float32
        assert dist_transform.min() >= 0.0
        assert dist_transform.max() <= 1.0

    def test_create_scale_map(self) -> None:
        """Test scale map creation."""
        instance_mask = np.zeros((200, 200), dtype=np.int64)
        # Create instances of different sizes
        instance_mask[10:30, 10:30] = 1  # Small
        instance_mask[50:100, 50:100] = 2  # Medium
        instance_mask[120:180, 120:180] = 3  # Large

        scale_map = create_scale_map(instance_mask, num_bins=3)

        assert scale_map.shape == instance_mask.shape
        assert scale_map.dtype == np.int64
        assert scale_map.min() >= 0
        assert scale_map.max() < 3


class TestDataLoader:
    """Tests for data loader."""

    def test_dataset_creation(self, num_classes: int, thing_classes: int) -> None:
        """Test dataset creation with synthetic data."""
        dataset = COCOPanopticDataset(
            data_dir="./data/coco",
            ann_file="annotations/panoptic_train2017.json",
            img_dir="train2017",
            transform=None,
            num_classes=num_classes,
            thing_classes=thing_classes,
        )

        assert len(dataset) == 100  # Synthetic data has 100 samples

    def test_dataset_getitem(self, num_classes: int, thing_classes: int) -> None:
        """Test dataset __getitem__ method."""
        dataset = COCOPanopticDataset(
            data_dir="./data/coco",
            ann_file="annotations/panoptic_train2017.json",
            img_dir="train2017",
            transform=None,
            num_classes=num_classes,
            thing_classes=thing_classes,
        )

        sample = dataset[0]

        assert "image" in sample
        assert "semantic_mask" in sample
        assert "instance_mask" in sample
        assert "boundary_mask" in sample
        assert "distance_transform" in sample
        assert "scale_map" in sample

        assert sample["image"].dtype == torch.float32
        assert sample["semantic_mask"].dtype == torch.int64
        assert sample["instance_mask"].dtype == torch.int64

    def test_dataset_with_transform(
        self, input_size: tuple, num_classes: int, thing_classes: int
    ) -> None:
        """Test dataset with transforms."""
        transform = get_val_transforms(input_size)

        dataset = COCOPanopticDataset(
            data_dir="./data/coco",
            ann_file="annotations/panoptic_train2017.json",
            img_dir="train2017",
            transform=transform,
            num_classes=num_classes,
            thing_classes=thing_classes,
        )

        sample = dataset[0]

        h, w = input_size
        assert sample["image"].shape == torch.Size([3, h, w])
        assert sample["semantic_mask"].shape == torch.Size([h, w])
