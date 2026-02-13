#!/usr/bin/env python
"""Evaluation script for adaptive scale-aware panoptic segmentation."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import torch
from tqdm import tqdm

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.loader import (
    get_dataloaders,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.evaluation.analysis import (
    create_results_table,
    save_results,
    visualize_predictions,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.evaluation.metrics import (
    PanopticMetrics,
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
        pretrained=False,  # Don't load pretrained weights
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
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    config: dict,
) -> dict:
    """Evaluate model on dataset.

    Args:
        model: Trained model.
        dataloader: Data loader.
        device: Device to run evaluation on.
        config: Configuration dictionary.

    Returns:
        Dictionary of evaluation metrics.
    """
    data_config = config.get("data", {})
    num_classes = data_config.get("num_classes", 133)
    thing_classes = data_config.get("thing_classes", 80)

    # Initialize metrics
    metrics_tracker = PanopticMetrics(num_classes=num_classes, thing_classes=thing_classes)

    # Evaluate
    logger.info("Running evaluation...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        # Move data to device
        images = batch["image"].to(device)
        targets = {
            "semantic_mask": batch["semantic_mask"].to(device),
            "instance_mask": batch["instance_mask"].to(device),
            "boundary_mask": batch["boundary_mask"].to(device),
            "distance_transform": batch["distance_transform"].to(device),
        }

        # Forward pass
        predictions = model(images)

        # Update metrics
        metrics_tracker.update(predictions, targets)

    # Compute final metrics
    metrics = metrics_tracker.compute()

    return metrics


def main() -> None:
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate panoptic segmentation model")
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
        "--visualize",
        action="store_true",
        help="Generate visualization images",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Get device
        system_config = config.get("system", {})
        device_str = system_config.get("device", "cuda")
        device = get_device(device_str)

        # Create data transforms
        data_config = config.get("data", {})
        input_size = tuple(data_config.get("input_size", [512, 512]))
        train_transform = get_train_transforms(input_size)
        val_transform = get_val_transforms(input_size)

        # Create data loaders
        logger.info("Creating data loaders...")
        train_loader, val_loader = get_dataloaders(config, train_transform, val_transform)

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = load_model(args.checkpoint, config, device)

        # Evaluate on validation set
        logger.info("Evaluating on validation set...")
        val_metrics = evaluate_model(model, val_loader, device, config)

        # Evaluate on training set (subset)
        logger.info("Evaluating on training set (subset)...")
        train_subset_size = min(len(train_loader.dataset), 100)
        train_subset_loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(
                train_loader.dataset, list(range(train_subset_size))
            ),
            batch_size=data_config.get("batch_size", 4),
            shuffle=False,
            num_workers=data_config.get("num_workers", 4),
        )
        train_metrics = evaluate_model(model, train_subset_loader, device, config)

        # Compile results
        results = {
            "validation_metrics": val_metrics,
            "training_metrics": train_metrics,
            "config_file": args.config,
            "checkpoint_file": args.checkpoint,
        }

        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        save_results(results, str(output_dir / "evaluation_results.json"))

        # Save as CSV
        results_df = pd.DataFrame(
            [
                {"split": "validation", **val_metrics},
                {"split": "training", **train_metrics},
            ]
        )
        results_df.to_csv(output_dir / "evaluation_results.csv", index=False)

        # Print results table
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION METRICS")
        logger.info("=" * 60)
        logger.info("\n" + create_results_table(val_metrics))

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING METRICS (subset)")
        logger.info("=" * 60)
        logger.info("\n" + create_results_table(train_metrics))

        # Generate visualizations if requested
        if args.visualize:
            logger.info("Generating visualizations...")
            eval_config = config.get("evaluation", {})
            max_vis = eval_config.get("max_visualizations", 50)

            # Visualize a few batches
            model.eval()
            num_visualized = 0

            for batch in val_loader:
                if num_visualized >= max_vis:
                    break

                images = batch["image"].to(device)
                targets = {
                    "semantic_mask": batch["semantic_mask"].to(device),
                    "instance_mask": batch["instance_mask"].to(device),
                    "boundary_mask": batch["boundary_mask"].to(device),
                    "distance_transform": batch["distance_transform"].to(device),
                }

                predictions = model(images)

                visualize_predictions(
                    images,
                    predictions,
                    targets,
                    str(output_dir / "visualizations"),
                    num_samples=min(5, images.shape[0]),
                )

                num_visualized += images.shape[0]

        logger.info(f"\nEvaluation completed! Results saved to {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
