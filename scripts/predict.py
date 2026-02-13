#!/usr/bin/env python
"""Prediction script for adaptive scale-aware panoptic segmentation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
from PIL import Image

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.preprocessing import (
    get_val_transforms,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.model import (
    AdaptiveScalePanopticSegmentation,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.utils.config import (
    get_device,
    load_config,
    setup_logging,
)

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: dict, device: torch.device) -> torch.nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint.
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    data_config = config.get("data", {})
    model_config = config.get("model", {})

    # Create model
    model = AdaptiveScalePanopticSegmentation(
        num_classes=data_config.get("num_classes", 133),
        thing_classes=data_config.get("thing_classes", 80),
        backbone=model_config.get("backbone", "resnet50"),
        pretrained=False,
        fpn_channels=model_config.get("fpn_channels", 256),
        num_fpn_levels=model_config.get("num_fpn_levels", 5),
        use_boundary_refinement=model_config.get("use_boundary_refinement", True),
        use_scale_adaptive_fusion=model_config.get("use_scale_adaptive_fusion", True),
        boundary_channels=model_config.get("boundary_channels", 64),
        distance_transform_bins=model_config.get("distance_transform_bins", 10),
        scale_attention_heads=model_config.get("scale_attention_heads", 8),
        dropout=model_config.get("dropout", 0.1),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    logger.info(f"Loaded model from {checkpoint_path}")
    return model


@torch.no_grad()
def predict_image(
    model: torch.nn.Module,
    image_path: str,
    transform,
    device: torch.device,
) -> dict:
    """Run prediction on a single image.

    Args:
        model: Trained model.
        image_path: Path to input image.
        transform: Image preprocessing transform.
        device: Device to run prediction on.

    Returns:
        Dictionary containing predictions.
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    # Preprocess
    image_np = np.array(image)
    transformed = transform(image=image_np)
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    # Predict
    predictions = model(image_tensor)

    # Post-process predictions
    pred_semantic = predictions["semantic"][0].argmax(dim=0).cpu().numpy()
    pred_instance = predictions["instance"][0].argmax(dim=0).cpu().numpy()

    # Resize to original size
    pred_semantic_resized = np.array(
        Image.fromarray(pred_semantic.astype(np.uint8)).resize(
            original_size, Image.NEAREST
        )
    )
    pred_instance_resized = np.array(
        Image.fromarray(pred_instance.astype(np.uint8)).resize(
            original_size, Image.NEAREST
        )
    )

    results = {
        "semantic": pred_semantic_resized,
        "instance": pred_instance_resized,
        "original_size": original_size,
    }

    if "boundary" in predictions:
        pred_boundary = predictions["boundary"][0, 0].cpu().numpy()
        pred_boundary_resized = np.array(
            Image.fromarray((pred_boundary * 255).astype(np.uint8)).resize(
                original_size, Image.BILINEAR
            )
        ) / 255.0
        results["boundary"] = pred_boundary_resized

    return results


def save_predictions(
    predictions: dict,
    output_path: str,
    original_image_path: str,
) -> None:
    """Save prediction results.

    Args:
        predictions: Dictionary of prediction results.
        output_path: Path to save predictions.
        original_image_path: Path to original image for visualization.
    """
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save semantic segmentation
    semantic_img = Image.fromarray(predictions["semantic"].astype(np.uint8))
    semantic_img.save(output_dir / "semantic_mask.png")

    # Save instance segmentation
    instance_img = Image.fromarray(predictions["instance"].astype(np.uint8))
    instance_img.save(output_dir / "instance_mask.png")

    # Save boundary if available
    if "boundary" in predictions:
        boundary_img = Image.fromarray((predictions["boundary"] * 255).astype(np.uint8))
        boundary_img.save(output_dir / "boundary_mask.png")

    logger.info(f"Saved predictions to {output_dir}")


def main() -> None:
    """Main prediction function."""
    parser = argparse.ArgumentParser(description="Run panoptic segmentation prediction")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions",
        help="Directory to save predictions",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        # Check if image exists
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Input image not found: {args.image}")
            sys.exit(1)

        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Get device
        system_config = config.get("system", {})
        device_str = system_config.get("device", "cuda")
        device = get_device(device_str)

        # Create transform
        data_config = config.get("data", {})
        input_size = tuple(data_config.get("input_size", [512, 512]))
        transform = get_val_transforms(input_size)

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, config, device)

        # Run prediction
        logger.info(f"Running prediction on {args.image}")
        predictions = predict_image(model, str(image_path), transform, device)

        # Get confidence scores
        num_classes = data_config.get("num_classes", 133)
        unique_classes = np.unique(predictions["semantic"])

        logger.info(f"\nPredicted {len(unique_classes)} unique classes:")
        for class_id in unique_classes[:10]:  # Show top 10
            pixel_count = np.sum(predictions["semantic"] == class_id)
            percentage = (pixel_count / predictions["semantic"].size) * 100
            logger.info(f"  Class {class_id}: {percentage:.2f}% of pixels")

        # Save predictions
        save_predictions(predictions, args.output, str(image_path))

        logger.info(f"\nPrediction completed! Results saved to {args.output}")

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
