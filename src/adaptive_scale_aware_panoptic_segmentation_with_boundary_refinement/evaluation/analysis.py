"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image

logger = logging.getLogger(__name__)


def save_results(results: Dict[str, Any], output_path: str) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Dictionary containing evaluation results.
        output_path: Path to save results JSON.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved results to {output_path}")


def plot_training_curves(
    history: Dict[str, List[float]], output_path: str, title: str = "Training Curves"
) -> None:
    """Plot training and validation loss curves.

    Args:
        history: Dictionary containing training history.
        output_path: Path to save the plot.
        title: Plot title.
    """
    plt.figure(figsize=(10, 6))

    if "train_loss" in history:
        plt.plot(history["train_loss"], label="Train Loss", linewidth=2)

    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Val Loss", linewidth=2)

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved training curves to {output_path}")


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Optional[List[str]],
    output_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot confusion matrix heatmap.

    Args:
        confusion_matrix: Confusion matrix array.
        class_names: List of class names.
        output_path: Path to save the plot.
        title: Plot title.
    """
    plt.figure(figsize=(12, 10))

    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype("float") / (
        confusion_matrix.sum(axis=1, keepdims=True) + 1e-10
    )

    # Plot heatmap
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names if class_names else "auto",
        yticklabels=class_names if class_names else "auto",
        cbar_kws={"label": "Normalized Frequency"},
    )

    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved confusion matrix to {output_path}")


def visualize_predictions(
    images: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    output_dir: str,
    num_samples: int = 5,
    colormap: str = "tab20",
) -> None:
    """Visualize panoptic segmentation predictions.

    Args:
        images: Input images tensor (B, 3, H, W).
        predictions: Dictionary of model predictions.
        targets: Dictionary of ground truth.
        output_dir: Directory to save visualizations.
        num_samples: Number of samples to visualize.
        colormap: Matplotlib colormap for segmentation masks.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    batch_size = min(images.shape[0], num_samples)
    cmap = plt.get_cmap(colormap)

    for i in range(batch_size):
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        # Denormalize image
        image = images[i].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title("Input Image")
        axes[0, 0].axis("off")

        # Predicted semantic
        pred_semantic = predictions["semantic"][i].argmax(dim=0).cpu().numpy()
        axes[0, 1].imshow(pred_semantic, cmap=colormap)
        axes[0, 1].set_title("Predicted Semantic")
        axes[0, 1].axis("off")

        # Ground truth semantic
        gt_semantic = targets["semantic_mask"][i].cpu().numpy()
        axes[0, 2].imshow(gt_semantic, cmap=colormap)
        axes[0, 2].set_title("GT Semantic")
        axes[0, 2].axis("off")

        # Semantic overlay
        semantic_colored = cmap(pred_semantic / pred_semantic.max())[:, :, :3]
        overlay = 0.5 * image + 0.5 * semantic_colored
        axes[0, 3].imshow(overlay)
        axes[0, 3].set_title("Semantic Overlay")
        axes[0, 3].axis("off")

        # Predicted instance
        pred_instance = predictions["instance"][i].argmax(dim=0).cpu().numpy()
        axes[1, 0].imshow(pred_instance, cmap=colormap)
        axes[1, 0].set_title("Predicted Instance")
        axes[1, 0].axis("off")

        # Ground truth instance
        gt_instance = targets["instance_mask"][i].cpu().numpy()
        axes[1, 1].imshow(gt_instance, cmap=colormap)
        axes[1, 1].set_title("GT Instance")
        axes[1, 1].axis("off")

        # Predicted boundary
        if "boundary" in predictions:
            pred_boundary = predictions["boundary"][i, 0].cpu().numpy()
            axes[1, 2].imshow(pred_boundary, cmap="gray")
            axes[1, 2].set_title("Predicted Boundary")
            axes[1, 2].axis("off")
        else:
            axes[1, 2].axis("off")

        # Ground truth boundary
        if "boundary_mask" in targets:
            gt_boundary = targets["boundary_mask"][i].cpu().numpy()
            axes[1, 3].imshow(gt_boundary, cmap="gray")
            axes[1, 3].set_title("GT Boundary")
            axes[1, 3].axis("off")
        else:
            axes[1, 3].axis("off")

        plt.tight_layout()
        plt.savefig(output_path / f"prediction_{i:03d}.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"Saved {batch_size} visualization(s) to {output_dir}")


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: str,
    title: str = "Metrics Comparison",
) -> None:
    """Plot comparison of metrics across different models or configurations.

    Args:
        metrics_dict: Dictionary mapping model names to their metrics.
        output_path: Path to save the plot.
        title: Plot title.
    """
    if not metrics_dict:
        logger.warning("No metrics to plot")
        return

    # Extract metric names
    metric_names = list(next(iter(metrics_dict.values())).keys())

    # Prepare data for plotting
    models = list(metrics_dict.keys())
    x = np.arange(len(metric_names))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        values = [metrics_dict[model].get(metric, 0) for metric in metric_names]
        ax.bar(x + i * width, values, width, label=model)

    ax.set_xlabel("Metrics", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metrics comparison to {output_path}")


def create_results_table(metrics: Dict[str, float]) -> str:
    """Create a formatted results table.

    Args:
        metrics: Dictionary of metric names and values.

    Returns:
        Formatted table string.
    """
    header = "| Metric | Score |\n|--------|-------|\n"
    rows = [f"| {metric} | {value:.2f} |" for metric, value in metrics.items()]

    return header + "\n".join(rows)
