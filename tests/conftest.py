"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

import pytest
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def device() -> torch.device:
    """Get device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def num_classes() -> int:
    """Default number of classes."""
    return 133


@pytest.fixture
def thing_classes() -> int:
    """Default number of thing classes."""
    return 80


@pytest.fixture
def input_size() -> tuple:
    """Default input size."""
    return (256, 256)


@pytest.fixture
def sample_batch(batch_size: int, input_size: tuple):
    """Create a sample batch of data."""
    h, w = input_size
    return {
        "image": torch.randn(batch_size, 3, h, w),
        "semantic_mask": torch.randint(0, 133, (batch_size, h, w)),
        "instance_mask": torch.randint(0, 10, (batch_size, h, w)),
        "boundary_mask": torch.rand(batch_size, h, w),
        "distance_transform": torch.rand(batch_size, h, w),
        "scale_map": torch.randint(0, 5, (batch_size, h, w)),
    }


@pytest.fixture
def config_dict() -> dict:
    """Sample configuration dictionary."""
    return {
        "data": {
            "num_classes": 133,
            "thing_classes": 80,
            "batch_size": 2,
            "input_size": [256, 256],
        },
        "model": {
            "backbone": "resnet50",
            "pretrained": False,
            "fpn_channels": 256,
            "num_fpn_levels": 4,
            "use_boundary_refinement": True,
            "use_scale_adaptive_fusion": True,
            "boundary_channels": 64,
            "distance_transform_bins": 10,
            "scale_attention_heads": 4,
            "dropout": 0.1,
        },
        "training": {
            "epochs": 2,
            "learning_rate": 0.001,
            "optimizer": "adamw",
        },
        "loss": {
            "semantic_weight": 1.0,
            "instance_weight": 1.0,
            "boundary_weight": 0.5,
            "panoptic_weight": 2.0,
        },
        "system": {
            "seed": 42,
            "device": "cpu",
        },
    }
