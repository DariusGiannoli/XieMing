"""Common task helpers."""

from __future__ import annotations

import time

import numpy as np

from rce_benchmark.metrics import greedy_nms


def sliding_window_detect(
    image: np.ndarray,
    model,
    win_h: int,
    win_w: int,
    stride: int,
    conf_thresh: float,
    nms_iou: float,
) -> tuple[list[tuple], float, int, np.ndarray, np.ndarray]:
    """Run a shared sliding-window detection loop and aggregate a heatmap."""
    height, width = image.shape[:2]
    detections: list[tuple] = []
    heatmap_sum = np.zeros((height, width), dtype=np.float32)
    heatmap_count = np.zeros((height, width), dtype=np.float32)
    n_windows = 0
    start = time.perf_counter()
    for y in range(0, height - win_h + 1, stride):
        for x in range(0, width - win_w + 1, stride):
            patch = image[y : y + win_h, x : x + win_w]
            conf = float(model.score_patch(patch))
            n_windows += 1
            heatmap_sum[y : y + win_h, x : x + win_w] += conf
            heatmap_count[y : y + win_h, x : x + win_w] += 1.0
            if conf >= conf_thresh:
                detections.append((x, y, x + win_w, y + win_h, "object", conf))
    elapsed_ms = (time.perf_counter() - start) * 1000
    if detections:
        detections = greedy_nms(detections, nms_iou)
    heatmap = np.divide(
        heatmap_sum,
        np.maximum(heatmap_count, 1.0),
        out=np.zeros_like(heatmap_sum),
        where=heatmap_count > 0,
    )
    pred_mask = (heatmap >= conf_thresh).astype(np.uint8) * 255
    return detections, elapsed_ms, n_windows, heatmap, pred_mask
