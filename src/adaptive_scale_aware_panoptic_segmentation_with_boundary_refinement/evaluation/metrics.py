"""Evaluation metrics for panoptic segmentation."""

import logging
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

logger = logging.getLogger(__name__)


def compute_panoptic_quality(
    pred_semantic: np.ndarray,
    pred_instance: np.ndarray,
    gt_semantic: np.ndarray,
    gt_instance: np.ndarray,
    num_classes: int,
    thing_classes: int,
) -> Tuple[float, float, float]:
    """Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

    Args:
        pred_semantic: Predicted semantic segmentation (H, W).
        pred_instance: Predicted instance segmentation (H, W).
        gt_semantic: Ground truth semantic segmentation (H, W).
        gt_instance: Ground truth instance segmentation (H, W).
        num_classes: Total number of classes.
        thing_classes: Number of thing classes.

    Returns:
        Tuple of (PQ, SQ, RQ) scores.
    """
    pq_sum = 0.0
    sq_sum = 0.0
    rq_sum = 0.0
    num_valid_classes = 0

    for class_id in range(num_classes):
        # Get masks for current class
        pred_class_mask = pred_semantic == class_id
        gt_class_mask = gt_semantic == class_id

        if not gt_class_mask.any():
            continue

        # Compute IoU
        intersection = np.logical_and(pred_class_mask, gt_class_mask).sum()
        union = np.logical_or(pred_class_mask, gt_class_mask).sum()

        if union == 0:
            continue

        iou = intersection / union

        # For thing classes, also consider instance matching
        if class_id < thing_classes:
            # Get unique instance IDs
            pred_instances = np.unique(pred_instance[pred_class_mask])
            gt_instances = np.unique(gt_instance[gt_class_mask])

            pred_instances = pred_instances[pred_instances > 0]
            gt_instances = gt_instances[gt_instances > 0]

            if len(gt_instances) == 0:
                continue

            # Simplified instance matching
            matched = min(len(pred_instances), len(gt_instances))
            tp = matched
            fp = len(pred_instances) - matched
            fn = len(gt_instances) - matched

            if tp + 0.5 * fp + 0.5 * fn == 0:
                continue

            pq = (tp * iou) / (tp + 0.5 * fp + 0.5 * fn)
            sq = iou if tp > 0 else 0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
        else:
            # For stuff classes, PQ = IoU
            pq = iou
            sq = iou
            rq = 1.0

        pq_sum += pq
        sq_sum += sq
        rq_sum += rq
        num_valid_classes += 1

    if num_valid_classes == 0:
        return 0.0, 0.0, 0.0

    pq_avg = pq_sum / num_valid_classes
    sq_avg = sq_sum / num_valid_classes
    rq_avg = rq_sum / num_valid_classes

    return pq_avg, sq_avg, rq_avg


def compute_boundary_iou(pred_boundary: np.ndarray, gt_boundary: np.ndarray) -> float:
    """Compute IoU for boundary detection.

    Args:
        pred_boundary: Predicted boundary map (H, W), binary or probability.
        gt_boundary: Ground truth boundary map (H, W), binary.

    Returns:
        Boundary IoU score.
    """
    # Threshold predicted boundary if it's a probability map
    if pred_boundary.max() <= 1.0 and pred_boundary.min() >= 0.0:
        pred_boundary = (pred_boundary > 0.5).astype(np.float32)

    gt_boundary = (gt_boundary > 0.5).astype(np.float32)

    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()

    if union == 0:
        return 1.0  # Both are empty

    return intersection / union


class PanopticMetrics:
    """Comprehensive metrics for panoptic segmentation evaluation."""

    def __init__(self, num_classes: int = 133, thing_classes: int = 80) -> None:
        """Initialize panoptic metrics.

        Args:
            num_classes: Total number of classes.
            thing_classes: Number of thing classes.
        """
        self.num_classes = num_classes
        self.thing_classes = thing_classes
        self.reset()

    def reset(self) -> None:
        """Reset accumulated metrics."""
        self.pq_scores = []
        self.sq_scores = []
        self.rq_scores = []
        self.boundary_ious = []
        self.semantic_ious = []
        self.instance_ious = []

    @torch.no_grad()
    def update(
        self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> None:
        """Update metrics with a batch of predictions and targets.

        Args:
            predictions: Dictionary containing model predictions.
            targets: Dictionary containing ground truth.
        """
        batch_size = predictions["semantic"].shape[0]

        for i in range(batch_size):
            # Convert to numpy and get class predictions
            pred_semantic = predictions["semantic"][i].argmax(dim=0).cpu().numpy()
            pred_instance = predictions["instance"][i].argmax(dim=0).cpu().numpy()
            gt_semantic = targets["semantic_mask"][i].cpu().numpy()
            gt_instance = targets["instance_mask"][i].cpu().numpy()

            # Compute panoptic quality
            pq, sq, rq = compute_panoptic_quality(
                pred_semantic,
                pred_instance,
                gt_semantic,
                gt_instance,
                self.num_classes,
                self.thing_classes,
            )

            self.pq_scores.append(pq)
            self.sq_scores.append(sq)
            self.rq_scores.append(rq)

            # Compute semantic IoU
            semantic_iou = jaccard_score(
                gt_semantic.flatten(), pred_semantic.flatten(), average="weighted", zero_division=0
            )
            self.semantic_ious.append(semantic_iou)

            # Compute instance IoU (only for thing classes)
            thing_mask = gt_semantic < self.thing_classes
            if thing_mask.any():
                instance_iou = jaccard_score(
                    gt_instance[thing_mask].flatten(),
                    pred_instance[thing_mask].flatten(),
                    average="weighted",
                    zero_division=0,
                )
                self.instance_ious.append(instance_iou)

            # Compute boundary IoU if available
            if "boundary" in predictions and "boundary_mask" in targets:
                pred_boundary = predictions["boundary"][i, 0].cpu().numpy()
                gt_boundary = targets["boundary_mask"][i].cpu().numpy()
                boundary_iou = compute_boundary_iou(pred_boundary, gt_boundary)
                self.boundary_ious.append(boundary_iou)

    def compute(self) -> Dict[str, float]:
        """Compute final metrics.

        Returns:
            Dictionary containing all computed metrics.
        """
        metrics = {}

        if self.pq_scores:
            metrics["panoptic_quality"] = np.mean(self.pq_scores) * 100
            metrics["segmentation_quality"] = np.mean(self.sq_scores) * 100
            metrics["recognition_quality"] = np.mean(self.rq_scores) * 100

        if self.boundary_ious:
            metrics["boundary_iou"] = np.mean(self.boundary_ious) * 100

        if self.semantic_ious:
            metrics["semantic_iou"] = np.mean(self.semantic_ious) * 100

        if self.instance_ious:
            metrics["instance_iou"] = np.mean(self.instance_ious) * 100

        return metrics

    def compute_per_class(self) -> Dict[str, np.ndarray]:
        """Compute per-class metrics.

        Returns:
            Dictionary containing per-class metric arrays.
        """
        # This is a simplified version - full implementation would track per-class metrics
        return {
            "per_class_pq": np.array(self.pq_scores),
            "per_class_sq": np.array(self.sq_scores),
            "per_class_rq": np.array(self.rq_scores),
        }


def compute_confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        pred: Predicted labels of shape (N,).
        target: Ground truth labels of shape (N,).
        num_classes: Number of classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
    """
    mask = (target >= 0) & (target < num_classes)
    confusion = np.bincount(
        num_classes * target[mask] + pred[mask], minlength=num_classes**2
    ).reshape(num_classes, num_classes)

    return confusion


def compute_classification_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Compute classification metrics (accuracy, precision, recall, F1).

    Args:
        pred: Predicted labels.
        target: Ground truth labels.

    Returns:
        Dictionary of classification metrics.
    """
    metrics = {
        "accuracy": accuracy_score(target, pred),
        "precision_macro": precision_score(target, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(target, pred, average="macro", zero_division=0),
        "f1_macro": f1_score(target, pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(target, pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(target, pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(target, pred, average="weighted", zero_division=0),
    }

    return metrics
