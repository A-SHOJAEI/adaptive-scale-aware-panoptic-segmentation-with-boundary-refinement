"""Adaptive Scale-Aware Panoptic Segmentation with Boundary Refinement.

This package implements a novel panoptic segmentation approach that combines
adaptive scale-aware feature fusion with boundary refinement for improved
multi-scale object segmentation.
"""

__version__ = "0.1.0"
__author__ = "Alireza Shojaei"

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.model import (
    AdaptiveScalePanopticSegmentation,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.utils.config import (
    load_config,
)

__all__ = [
    "AdaptiveScalePanopticSegmentation",
    "load_config",
]
