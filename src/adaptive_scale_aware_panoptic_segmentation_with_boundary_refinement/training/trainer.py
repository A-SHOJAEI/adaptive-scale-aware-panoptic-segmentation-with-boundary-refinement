"""Training loop with LR scheduling, early stopping, and mixed precision."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class PanopticTrainer:
    """Trainer for panoptic segmentation with advanced training features."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: Dict[str, Any],
    ) -> None:
        """Initialize trainer.

        Args:
            model: Panoptic segmentation model.
            criterion: Loss function.
            optimizer: Optimizer.
            scheduler: Learning rate scheduler.
            train_loader: Training data loader.
            val_loader: Validation data loader.
            device: Device to train on.
            config: Configuration dictionary.
        """
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        # Training settings
        training_config = config.get("training", {})
        self.epochs = training_config.get("epochs", 50)
        self.gradient_clip_norm = training_config.get("gradient_clip_norm", 1.0)
        self.mixed_precision = training_config.get("mixed_precision", True)
        self.early_stopping_patience = training_config.get("early_stopping_patience", 10)
        self.save_every_n_epochs = training_config.get("save_every_n_epochs", 5)

        # System settings
        system_config = config.get("system", {})
        self.log_interval = system_config.get("log_interval", 10)
        self.checkpoint_dir = Path(system_config.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Mixed precision scaler
        self.scaler = GradScaler(device.type) if self.mixed_precision else None

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0
        self.training_history = {"train_loss": [], "val_loss": []}

        logger.info(
            f"Initialized trainer: epochs={self.epochs}, "
            f"mixed_precision={self.mixed_precision}, "
            f"early_stopping_patience={self.early_stopping_patience}"
        )

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of training metrics.
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {}

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch["image"].to(self.device)
            targets = {
                "semantic_mask": batch["semantic_mask"].to(self.device),
                "instance_mask": batch["instance_mask"].to(self.device),
                "boundary_mask": batch["boundary_mask"].to(self.device),
                "distance_transform": batch["distance_transform"].to(self.device),
            }

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass with mixed precision
            if self.mixed_precision:
                with autocast(device_type=self.device.type):
                    predictions = self.model(images)
                    loss, losses = self.criterion(predictions, targets)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(images)
                loss, losses = self.criterion(predictions, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.gradient_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip_norm
                    )

                # Optimizer step
                self.optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

            # Update progress bar
            if (batch_idx + 1) % self.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Compute average losses
        num_batches = len(self.train_loader)
        metrics = {"loss": total_loss / num_batches}
        for key, value in loss_components.items():
            metrics[key] = value / num_batches

        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model.

        Returns:
            Dictionary of validation metrics.
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {}

        pbar = tqdm(self.val_loader, desc="Validation")

        for batch in pbar:
            # Move data to device
            images = batch["image"].to(self.device)
            targets = {
                "semantic_mask": batch["semantic_mask"].to(self.device),
                "instance_mask": batch["instance_mask"].to(self.device),
                "boundary_mask": batch["boundary_mask"].to(self.device),
                "distance_transform": batch["distance_transform"].to(self.device),
            }

            # Forward pass
            if self.mixed_precision:
                with autocast(device_type=self.device.type):
                    predictions = self.model(images)
                    loss, losses = self.criterion(predictions, targets)
            else:
                predictions = self.model(images)
                loss, losses = self.criterion(predictions, targets)

            # Accumulate losses
            total_loss += loss.item()
            for key, value in losses.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value

        # Compute average losses
        num_batches = len(self.val_loader)
        metrics = {"loss": total_loss / num_batches}
        for key, value in loss_components.items():
            metrics[key] = value / num_batches

        return metrics

    def save_checkpoint(self, filename: str, is_best: bool = False) -> None:
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename.
            is_best: Whether this is the best model so far.
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "training_history": self.training_history,
            "config": self.config,
        }

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.training_history = checkpoint["training_history"]

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {self.current_epoch})")

    def train(self) -> Dict[str, list]:
        """Run full training loop.

        Returns:
            Training history dictionary.
        """
        logger.info("Starting training...")

        for epoch in range(self.current_epoch, self.epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_metrics = self.train_epoch()
            self.training_history["train_loss"].append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            self.training_history["val_loss"].append(val_metrics["loss"])

            # Update learning rate
            self.scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            # Save checkpoint
            if (epoch + 1) % self.save_every_n_epochs == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth")

            # Check for improvement
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", is_best=True)
                logger.info(f"New best validation loss: {self.best_val_loss:.4f}")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                logger.info(
                    f"Early stopping triggered after {epoch + 1} epochs "
                    f"({self.early_stopping_patience} epochs without improvement)"
                )
                break

        logger.info("Training completed!")
        return self.training_history
