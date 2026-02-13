#!/usr/bin/env python
"""Training script for adaptive scale-aware panoptic segmentation."""

import argparse
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.optim as optim

from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.loader import (
    get_dataloaders,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.data.preprocessing import (
    get_train_transforms,
    get_val_transforms,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.evaluation.analysis import (
    plot_training_curves,
    save_results,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.components import (
    CombinedPanopticLoss,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.models.model import (
    AdaptiveScalePanopticSegmentation,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.training.trainer import (
    PanopticTrainer,
)
from adaptive_scale_aware_panoptic_segmentation_with_boundary_refinement.utils.config import (
    get_device,
    load_config,
    set_seed,
    setup_logging,
)

logger = logging.getLogger(__name__)


def get_optimizer(model: torch.nn.Module, config: dict) -> optim.Optimizer:
    """Create optimizer from config.

    Args:
        model: Model to optimize.
        config: Configuration dictionary.

    Returns:
        Optimizer instance.
    """
    training_config = config.get("training", {})
    optimizer_name = training_config.get("optimizer", "adamw").lower()
    lr = training_config.get("learning_rate", 0.0001)
    weight_decay = training_config.get("weight_decay", 0.0001)

    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    logger.info(f"Created {optimizer_name} optimizer with lr={lr}")
    return optimizer


def get_scheduler(optimizer: optim.Optimizer, config: dict) -> optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler from config.

    Args:
        optimizer: Optimizer instance.
        config: Configuration dictionary.

    Returns:
        Learning rate scheduler.
    """
    training_config = config.get("training", {})
    scheduler_name = training_config.get("scheduler", "cosine").lower()
    epochs = training_config.get("epochs", 50)
    warmup_epochs = training_config.get("warmup_epochs", 5)
    min_lr = training_config.get("min_lr", 0.000001)

    if scheduler_name == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=min_lr)
    elif scheduler_name == "step":
        step_size = max(epochs // 3, 1)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
    elif scheduler_name == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logger.info(f"Created {scheduler_name} scheduler")
    return scheduler


def main() -> None:
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train adaptive scale-aware panoptic segmentation model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Setup logging
    setup_logging()

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)

        # Set random seed
        system_config = config.get("system", {})
        seed = system_config.get("seed", 42)
        set_seed(seed)

        # Get device
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

        # Create model
        logger.info("Creating model...")
        model_config = config.get("model", {})
        model = AdaptiveScalePanopticSegmentation(
            num_classes=data_config.get("num_classes", 133),
            thing_classes=data_config.get("thing_classes", 80),
            backbone=model_config.get("backbone", "resnet50"),
            pretrained=model_config.get("pretrained", True),
            fpn_channels=model_config.get("fpn_channels", 256),
            num_fpn_levels=model_config.get("num_fpn_levels", 5),
            use_boundary_refinement=model_config.get("use_boundary_refinement", True),
            use_scale_adaptive_fusion=model_config.get("use_scale_adaptive_fusion", True),
            boundary_channels=model_config.get("boundary_channels", 64),
            distance_transform_bins=model_config.get("distance_transform_bins", 10),
            scale_attention_heads=model_config.get("scale_attention_heads", 8),
            dropout=model_config.get("dropout", 0.1),
        )

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"Model parameters: {num_params:,} total, {num_trainable:,} trainable"
        )

        # Create loss function
        logger.info("Creating loss function...")
        loss_config = config.get("loss", {})
        criterion = CombinedPanopticLoss(
            semantic_weight=loss_config.get("semantic_weight", 1.0),
            instance_weight=loss_config.get("instance_weight", 1.0),
            boundary_weight=loss_config.get("boundary_weight", 0.5),
            panoptic_weight=loss_config.get("panoptic_weight", 2.0),
            focal_alpha=loss_config.get("focal_alpha", 0.25),
            focal_gamma=loss_config.get("focal_gamma", 2.0),
            dice_smooth=loss_config.get("dice_smooth", 1.0),
        )

        # Create optimizer and scheduler
        optimizer = get_optimizer(model, config)
        scheduler = get_scheduler(optimizer, config)

        # Create trainer
        logger.info("Creating trainer...")
        trainer = PanopticTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            config=config,
        )

        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)

        # MLflow tracking (optional)
        try:
            import mlflow

            mlflow_uri = system_config.get("mlflow_tracking_uri", "./mlruns")
            experiment_name = system_config.get("experiment_name", "panoptic_segmentation")

            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():
                # Log parameters
                mlflow.log_params(
                    {
                        "backbone": model_config.get("backbone"),
                        "batch_size": data_config.get("batch_size"),
                        "learning_rate": config["training"].get("learning_rate"),
                        "epochs": config["training"].get("epochs"),
                        "optimizer": config["training"].get("optimizer"),
                        "scheduler": config["training"].get("scheduler"),
                    }
                )

                # Train model
                history = trainer.train()

                # Log metrics
                for epoch, (train_loss, val_loss) in enumerate(
                    zip(history["train_loss"], history["val_loss"])
                ):
                    mlflow.log_metrics(
                        {"train_loss": train_loss, "val_loss": val_loss}, step=epoch
                    )

                # Log best model
                best_model_path = trainer.checkpoint_dir / "best_model.pth"
                if best_model_path.exists():
                    mlflow.log_artifact(str(best_model_path))

        except ImportError:
            logger.warning("MLflow not available, skipping experiment tracking")
            # Train without MLflow
            history = trainer.train()
        except Exception as e:
            logger.warning(f"MLflow tracking failed: {e}. Continuing without tracking.")
            # Train without MLflow
            history = trainer.train()

        # Save training history
        results_dir = Path(system_config.get("results_dir", "./results"))
        results_dir.mkdir(parents=True, exist_ok=True)

        save_results(history, str(results_dir / "training_history.json"))
        plot_training_curves(history, str(results_dir / "training_curves.png"))

        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Results saved to {results_dir}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
