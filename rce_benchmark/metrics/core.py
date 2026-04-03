"""Core metrics and geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    best_iou: float
    n_dets: int


def iou_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """IoU between two bounding boxes."""
    xi1 = max(a[0], b[0])
    yi1 = max(a[1], b[1])
    xi2 = min(a[2], b[2])
    yi2 = min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    denom = area_a + area_b - inter + 1e-6
    return inter / denom


def greedy_nms(detections: list[tuple], iou_thresh: float) -> list[tuple]:
    """Greedy NMS on benchmark detection tuples."""
    ordered = sorted(detections, key=lambda det: det[5], reverse=True)
    kept: list[tuple] = []
    while ordered:
        best = ordered.pop(0)
        kept.append(best)
        ordered = [
            det
            for det in ordered
            if iou_score(best[:4], det[:4]) < iou_thresh
        ]
    return kept


def classification_metrics(y_true: list[str], y_pred: list[str], positive_label: str) -> dict[str, float]:
    """Binary precision / recall / f1 for classification episodes."""
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p == positive_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != positive_label and p == positive_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == positive_label and p != positive_label)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def eval_detection_metrics(detections: list[tuple], gt_box: tuple[int, int, int, int], iou_thresh: float) -> DetectionMetrics:
    """Detection metrics for a single-GT localization task."""
    if not detections:
        return DetectionMetrics(precision=0.0, recall=0.0, f1=0.0, best_iou=0.0, n_dets=0)
    ious = [iou_score(det[:4], gt_box) for det in detections]
    best_iou = max(ious)
    tp = sum(1 for value in ious if value >= iou_thresh)
    fp = len(detections) - tp
    fn = 0 if tp > 0 else 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        best_iou=best_iou,
        n_dets=len(detections),
    )
