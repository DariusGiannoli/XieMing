"""
src/localization.py  —  Localization Strategy Library
=====================================================
Five strategies that decide WHERE to evaluate a recognition head.
The head stays the same — only the search method changes.

Strategies
----------
1. Exhaustive Sliding Window  — brute-force grid scan
2. Image Pyramid              — multi-scale resize + sliding window
3. Coarse-to-Fine Search      — two-pass hierarchical refinement
4. Contour Proposals          — edge-driven candidate regions
5. Template Matching          — OpenCV cross-correlation (no head)

Every function returns the same tuple:
    (detections, n_proposals, elapsed_ms, heatmap)
"""

import cv2
import numpy as np
import time


# ===================================================================
#  Shared utilities
# ===================================================================

def nms(dets, iou_thresh):
    """Greedy NMS on list of (x1, y1, x2, y2, label, conf)."""
    dets = sorted(dets, key=lambda d: d[5], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if _iou(best, d) < iou_thresh]
    return keep


def _iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-6)


# ===================================================================
#  1. Exhaustive Sliding Window
# ===================================================================

def exhaustive_sliding_window(image, win_h, win_w, feature_fn, head,
                               stride, conf_thresh, nms_iou):
    """
    Brute-force grid scan.  Evaluates the head at **every** position
    spaced by *stride* pixels.
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    n_proposals = 0
    t0 = time.perf_counter()

    for y in range(0, H - win_h + 1, stride):
        for x in range(0, W - win_w + 1, stride):
            patch = image[y:y + win_h, x:x + win_w]
            feats = feature_fn(patch)
            label, conf = head.predict(feats)
            n_proposals += 1
            if label != "background":
                heatmap[y:y + win_h, x:x + win_w] = np.maximum(
                    heatmap[y:y + win_h, x:x + win_w], conf)
                if conf >= conf_thresh:
                    detections.append((x, y, x + win_w, y + win_h, label, conf))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if detections:
        detections = nms(detections, nms_iou)
    return detections, n_proposals, elapsed_ms, heatmap


# ===================================================================
#  2. Image Pyramid
# ===================================================================

def image_pyramid(image, win_h, win_w, feature_fn, head,
                  stride, conf_thresh, nms_iou,
                  scales=(0.5, 0.75, 1.0, 1.25, 1.5)):
    """
    Resize the image at several scales, run a sliding window at each
    level, and map detections back to original coordinates.
    Finds objects at sizes different from the training crop.
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    n_proposals = 0
    t0 = time.perf_counter()

    for scale in scales:
        sH, sW = int(H * scale), int(W * scale)
        if sH < win_h or sW < win_w:
            continue
        scaled = cv2.resize(image, (sW, sH))

        for y in range(0, sH - win_h + 1, stride):
            for x in range(0, sW - win_w + 1, stride):
                patch = scaled[y:y + win_h, x:x + win_w]
                feats = feature_fn(patch)
                label, conf = head.predict(feats)
                n_proposals += 1
                if label != "background":
                    # Map back to original image coordinates
                    ox  = int(x / scale)
                    oy  = int(y / scale)
                    ox2 = min(int((x + win_w) / scale), W)
                    oy2 = min(int((y + win_h) / scale), H)
                    heatmap[oy:oy2, ox:ox2] = np.maximum(
                        heatmap[oy:oy2, ox:ox2], conf)
                    if conf >= conf_thresh:
                        detections.append((ox, oy, ox2, oy2, label, conf))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if detections:
        detections = nms(detections, nms_iou)
    return detections, n_proposals, elapsed_ms, heatmap


# ===================================================================
#  3. Coarse-to-Fine Search
# ===================================================================

def coarse_to_fine(image, win_h, win_w, feature_fn, head,
                   fine_stride, conf_thresh, nms_iou,
                   coarse_factor=4, refine_radius=2):
    """
    Two-pass hierarchical search.

    Pass 1 — Scan at *coarse_factor × fine_stride* to cheaply identify
             hot regions (using a relaxed threshold of 0.7 × conf_thresh).
    Pass 2 — Re-scan **only** the neighbourhood of each hit at
             *fine_stride*, within *refine_radius* steps in each direction.
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    n_proposals = 0
    t0 = time.perf_counter()

    coarse_stride = fine_stride * coarse_factor

    # --- Pass 1: coarse ---
    hot_spots = []
    for y in range(0, H - win_h + 1, coarse_stride):
        for x in range(0, W - win_w + 1, coarse_stride):
            patch = image[y:y + win_h, x:x + win_w]
            feats = feature_fn(patch)
            label, conf = head.predict(feats)
            n_proposals += 1
            if label != "background" and conf >= conf_thresh * 0.7:
                hot_spots.append((x, y))
                heatmap[y:y + win_h, x:x + win_w] = np.maximum(
                    heatmap[y:y + win_h, x:x + win_w], conf)

    # --- Pass 2: fine around hot spots ---
    visited = set()
    for hx, hy in hot_spots:
        for dy in range(-refine_radius, refine_radius + 1):
            for dx in range(-refine_radius, refine_radius + 1):
                x = hx + dx * fine_stride
                y = hy + dy * fine_stride
                if (x, y) in visited:
                    continue
                if x < 0 or y < 0 or x + win_w > W or y + win_h > H:
                    continue
                visited.add((x, y))
                patch = image[y:y + win_h, x:x + win_w]
                feats = feature_fn(patch)
                label, conf = head.predict(feats)
                n_proposals += 1
                if label != "background":
                    heatmap[y:y + win_h, x:x + win_w] = np.maximum(
                        heatmap[y:y + win_h, x:x + win_w], conf)
                    if conf >= conf_thresh:
                        detections.append((x, y, x + win_w, y + win_h,
                                           label, conf))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if detections:
        detections = nms(detections, nms_iou)
    return detections, n_proposals, elapsed_ms, heatmap


# ===================================================================
#  4. Contour Proposals
# ===================================================================

def contour_proposals(image, win_h, win_w, feature_fn, head,
                      conf_thresh, nms_iou,
                      canny_low=50, canny_high=150,
                      area_tolerance=3.0):
    """
    Generate candidate regions from image structure:
    Canny edges → morphological closing → contour extraction.
    Keep contours whose bounding-box area is within *area_tolerance*×
    of the window area, centre a window on each, and score with the head.

    Returns an extra key ``edge_map`` in the heatmap slot for
    visualisation on the page (the caller can detect this).
    """
    H, W = image.shape[:2]
    heatmap = np.zeros((H, W), dtype=np.float32)
    detections = []
    n_proposals = 0
    t0 = time.perf_counter()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    target_area = win_h * win_w
    min_area = target_area / area_tolerance
    max_area = target_area * area_tolerance

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        # Centre a window on the contour centre
        cx, cy = bx + bw // 2, by + bh // 2
        px = max(0, min(cx - win_w // 2, W - win_w))
        py = max(0, min(cy - win_h // 2, H - win_h))

        patch = image[py:py + win_h, px:px + win_w]
        if patch.shape[0] != win_h or patch.shape[1] != win_w:
            continue

        feats = feature_fn(patch)
        label, conf = head.predict(feats)
        n_proposals += 1

        if label != "background":
            heatmap[py:py + win_h, px:px + win_w] = np.maximum(
                heatmap[py:py + win_h, px:px + win_w], conf)
            if conf >= conf_thresh:
                detections.append((px, py, px + win_w, py + win_h,
                                   label, conf))

    elapsed_ms = (time.perf_counter() - t0) * 1000
    if detections:
        detections = nms(detections, nms_iou)
    return detections, n_proposals, elapsed_ms, heatmap, edges


# ===================================================================
#  5. Template Matching
# ===================================================================

def template_matching(image, template, conf_thresh, nms_iou,
                      method=cv2.TM_CCOEFF_NORMED):
    """
    OpenCV normalised cross-correlation.
    No trained head — pure pixel similarity between *template* and every
    image position.  Extremely fast (optimised C++) but not invariant to
    rotation, scale, or illumination.
    """
    H, W = image.shape[:2]
    th, tw = template.shape[:2]
    t0 = time.perf_counter()

    result = cv2.matchTemplate(image, template, method)

    if method in (cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR_NORMED):
        score_map = np.clip(result, 0, 1).astype(np.float32)
    else:
        lo, hi = result.min(), result.max()
        score_map = ((result - lo) / (hi - lo + 1e-6)).astype(np.float32)

    # Full-size heatmap (resize for visualisation)
    heatmap = cv2.resize(score_map, (W, H), interpolation=cv2.INTER_LINEAR)

    # Extract detections above threshold
    detections = []
    locs = np.where(score_map >= conf_thresh)
    for y, x in zip(*locs):
        detections.append((int(x), int(y), int(x + tw), int(y + th),
                           "object", float(score_map[y, x])))

    n_proposals = score_map.shape[0] * score_map.shape[1]
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if detections:
        detections = nms(detections, nms_iou)
    return detections, n_proposals, elapsed_ms, heatmap


# ===================================================================
#  Registry  —  metadata used by the Streamlit page
# ===================================================================

STRATEGIES = {
    "Exhaustive Sliding Window": {
        "icon": "🔲",
        "fn":   exhaustive_sliding_window,
        "needs_head": True,
        "short": "Brute-force grid scan at every stride position.",
        "detail": (
            "The simplest approach: a fixed-size window slides across the "
            "**entire image** at regular intervals.  At every position the "
            "patch is extracted, features are computed, and the head classifies it.\n\n"
            "**Complexity:** $O\\!\\left(\\frac{W}{s} \\times \\frac{H}{s}\\right)$ "
            "where $s$ = stride.\n\n"
            "**Pro:** Guaranteed to evaluate every location — nothing is missed.\n\n"
            "**Con:** Extremely slow on large images or small strides."
        ),
    },
    "Image Pyramid": {
        "icon": "🔺",
        "fn":   image_pyramid,
        "needs_head": True,
        "short": "Multi-scale resize + sliding window.",
        "detail": (
            "Builds a **Gaussian pyramid** by resizing the image to several "
            "scales (e.g. 50 %, 75 %, 100 %, 125 %, 150 %).  A sliding-window "
            "scan runs at each level and detections are mapped back to original "
            "coordinates.\n\n"
            "**Why:** The training crop has a fixed size.  If the real object "
            "appears larger or smaller in the scene, a single-scale scan will "
            "miss it.  The pyramid handles **scale variation**.\n\n"
            "**Cost:** Multiplies the number of proposals by the number of "
            "scales — slower than single-scale exhaustive."
        ),
    },
    "Coarse-to-Fine": {
        "icon": "🎯",
        "fn":   coarse_to_fine,
        "needs_head": True,
        "short": "Two-pass hierarchical refinement.",
        "detail": (
            "**Pass 1 — Coarse:** Scans the image with a large stride "
            "(coarse\\_factor × fine\\_stride) using a relaxed confidence "
            "threshold (70 % of the target) to cheaply identify *hot regions*.\n\n"
            "**Pass 2 — Fine:** Re-scans **only** the neighbourhood around "
            "each coarse hit at the fine stride, within *refine\\_radius* steps "
            "in each direction.\n\n"
            "**Speedup:** Typically **3–10×** faster than exhaustive when the "
            "object is spatially sparse (i.e. most of the image is background)."
        ),
    },
    "Contour Proposals": {
        "icon": "✏️",
        "fn":   contour_proposals,
        "needs_head": True,
        "short": "Edge-driven candidate regions scored by head.",
        "detail": (
            "Instead of scanning everywhere, this method lets **image "
            "structure** drive the search:\n\n"
            "1. Canny edge detection\n"
            "2. Morphological closing to bridge nearby edges\n"
            "3. External contour extraction\n"
            "4. Filter contours whose area falls within *area\\_tolerance* "
            "of the window area\n"
            "5. Centre a window on each surviving contour and score with "
            "the trained head\n\n"
            "**Proposals evaluated:** Typically 10–100× fewer than exhaustive. "
            "Speed depends on scene complexity (more edges → more proposals)."
        ),
    },
    "Template Matching": {
        "icon": "📋",
        "fn":   template_matching,
        "needs_head": False,
        "short": "OpenCV cross-correlation — no head needed.",
        "detail": (
            "Classical **normalised cross-correlation** (NCC).  Slides the "
            "crop template over the image computing pixel-level similarity "
            "at every position.  No trained head is involved.\n\n"
            "**Speed:** Runs entirely in OpenCV's optimised C++ backend — "
            "orders of magnitude faster than Python-level loops.\n\n"
            "**Limitation:** Not invariant to rotation, scale, or illumination "
            "changes.  Works best when the object appears at the **exact same "
            "size and orientation** as the crop."
        ),
    },
}
