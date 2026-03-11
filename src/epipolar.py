"""
src/epipolar.py — Sparse Epipolar Stereo Matching
==================================================
Given detected objects in the left image:
1.  Compute the fundamental matrix **F** from camera calibration.
2.  Extract ORB keypoints inside each detection bounding-box.
3.  For every keypoint, project the epipolar line onto the right image.
4.  Template-match along that line to find the correspondence.
5.  Compute disparity  d = x_L − x_R
6.  Recover depth     Z = f·B / (d + d_offs)
"""

import cv2
import numpy as np
import time


# ------------------------------------------------------------------
#  Fundamental matrix
# ------------------------------------------------------------------

def fundamental_from_calibration(cam0: np.ndarray, cam1: np.ndarray,
                                  baseline_mm: float) -> np.ndarray:
    """Compute F for a rectified stereo pair.

    E  = [t]_×   with  t = [B, 0, 0]
    F  = K_R^{-T}  E  K_L^{-1}

    For rectified images the result confirms that epipolar
    lines are horizontal.
    """
    tx = np.array([[0.0,  0.0,               0.0],
                   [0.0,  0.0, -baseline_mm],
                   [0.0,  baseline_mm,  0.0]])
    F = np.linalg.inv(cam1).T @ tx @ np.linalg.inv(cam0)
    norm = np.linalg.norm(F)
    if norm > 0:
        F /= norm
    return F


def fundamental_from_scalars(focal: float, cx0: float, cy: float,
                              cx1: float) -> np.ndarray:
    """Build F when only scalar calibration values are available."""
    K_L = np.array([[focal, 0, cx0], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    K_R = np.array([[focal, 0, cx1], [0, focal, cy], [0, 0, 1]], dtype=np.float64)
    tx = np.array([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    F = np.linalg.inv(K_R).T @ tx @ np.linalg.inv(K_L)
    norm = np.linalg.norm(F)
    if norm > 0:
        F /= norm
    return F


# ------------------------------------------------------------------
#  Epipolar line helpers
# ------------------------------------------------------------------

def compute_epipolar_lines(F: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Return Nx3 lines  l = F·p  (ax + by + c = 0) for Nx2 *points*."""
    pts_h = np.hstack([points, np.ones((len(points), 1))])
    lines = (F @ pts_h.T).T
    return lines


# ------------------------------------------------------------------
#  Template matching along epipolar line
# ------------------------------------------------------------------

def _match_along_epipolar(img_left, img_right, pt_left,
                           patch_half: int = 25, ndisp: int = 128):
    """NCC template match of a patch around *pt_left* along the same row.

    Returns ``((matched_x, matched_y), confidence)`` or ``(None, 0.0)``.
    """
    H, W = img_left.shape[:2]
    px, py = int(pt_left[0]), int(pt_left[1])

    y0 = max(0, py - patch_half)
    y1 = min(H, py + patch_half + 1)
    x0 = max(0, px - patch_half)
    x1 = min(W, px + patch_half + 1)

    template = img_left[y0:y1, x0:x1]
    if template.size == 0 or template.shape[0] < 5 or template.shape[1] < 5:
        return None, 0.0

    # Search strip — same rows, up to ndisp pixels to the left
    strip_x0 = max(0, px - ndisp - patch_half)
    strip_x1 = min(W, px + patch_half + 1)
    strip = img_right[y0:y1, strip_x0:strip_x1]
    if strip.shape[1] < template.shape[1] or strip.shape[0] < template.shape[0]:
        return None, 0.0

    tmpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
    strip_gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if len(strip.shape) == 3 else strip

    result = cv2.matchTemplate(strip_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    matched_x = strip_x0 + max_loc[0] + template.shape[1] // 2
    return (matched_x, py), float(max_val)


# ------------------------------------------------------------------
#  Full sparse depth pipeline
# ------------------------------------------------------------------

def sparse_epipolar_depth(img_left, img_right, detections, F,
                           focal, baseline, doffs, ndisp=128,
                           n_keypoints=10, patch_half=25,
                           match_thresh=0.3):
    """Run sparse epipolar matching for every detection.

    *detections* : list of ``(source, x1, y1, x2, y2, label, conf)``

    Returns ``(results, elapsed_ms)``.
    """
    orb = cv2.ORB_create(nfeatures=n_keypoints * 3)
    t0 = time.perf_counter()
    results = []
    H, W = img_left.shape[:2]

    for det in detections:
        source, dx1, dy1, dx2, dy2, label, conf = det
        dx1, dy1, dx2, dy2 = int(dx1), int(dy1), int(dx2), int(dy2)
        bx0, by0 = max(0, min(dx1, W - 1)), max(0, min(dy1, H - 1))
        bx1, by1 = max(0, min(dx2, W)),     max(0, min(dy2, H))

        roi = img_left[by0:by1, bx0:bx1]
        if roi.size == 0:
            continue

        # ---- keypoints --------------------------------------------------
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        kps = orb.detect(gray_roi, None)

        if not kps:
            cx, cy = (bx0 + bx1) // 2, (by0 + by1) // 2
            keypoints_global = [(cx, cy)]
        else:
            kps = sorted(kps, key=lambda k: k.response, reverse=True)[:n_keypoints]
            keypoints_global = [(int(k.pt[0]) + bx0, int(k.pt[1]) + by0) for k in kps]

        # ---- epipolar lines ---------------------------------------------
        pts_arr = np.array(keypoints_global, dtype=np.float64)
        epi_lines = compute_epipolar_lines(F, pts_arr)

        # ---- match -------------------------------------------------------
        matches = []
        matched_pts = []
        for kp, eline in zip(keypoints_global, epi_lines):
            m_pt, m_conf = _match_along_epipolar(
                img_left, img_right, kp,
                patch_half=patch_half, ndisp=ndisp)

            if m_pt is not None and m_conf >= match_thresh:
                disp = kp[0] - m_pt[0]
                depth = (focal * baseline) / (disp + doffs) if (disp + doffs) > 0 else 0.0
                matches.append({
                    "left_pt":  kp,
                    "right_pt": m_pt,
                    "epi_line": eline.tolist(),
                    "disparity": float(disp),
                    "depth_mm":  float(depth),
                    "match_conf": m_conf,
                })
                matched_pts.append(m_pt)
            else:
                matched_pts.append(None)

        valid_depths = [m["depth_mm"] for m in matches if m["depth_mm"] > 0]
        median_depth = float(np.median(valid_depths)) if valid_depths else 0.0

        results.append({
            "source": source, "label": label,
            "box": (dx1, dy1, dx2, dy2), "det_conf": conf,
            "keypoints":     keypoints_global,
            "matched_pts":   matched_pts,
            "epi_lines":     epi_lines,
            "matches":       matches,
            "n_keypoints":   len(keypoints_global),
            "n_matched":     len(matches),
            "median_depth_mm": median_depth,
        })

    elapsed_ms = (time.perf_counter() - t0) * 1000
    return results, elapsed_ms


# ------------------------------------------------------------------
#  Visualisation helpers
# ------------------------------------------------------------------

_COLORS = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 255, 0), (255, 128, 0),
           (0, 128, 255), (128, 0, 255)]


def draw_epipolar_canvas(img_left, img_right, result_entry):
    """Side-by-side image: keypoints (L) ↔ epipolar lines + matches (R)."""
    H, W = img_left.shape[:2]
    canvas = np.zeros((H, W * 2 + 20, 3), dtype=np.uint8)
    canvas[:, :W] = img_left
    canvas[:, W + 20:] = img_right

    # Separator
    canvas[:, W:W + 20] = 40

    dx1, dy1, dx2, dy2 = result_entry["box"]
    cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), (255, 255, 0), 2)

    kps = result_entry["keypoints"]
    mpts = result_entry["matched_pts"]
    elines = result_entry["epi_lines"]

    for i, (kp, m_pt, eline) in enumerate(zip(kps, mpts, elines)):
        color = _COLORS[i % len(_COLORS)]
        kx, ky = int(kp[0]), int(kp[1])

        # keypoint on left
        cv2.circle(canvas, (kx, ky), 5, color, -1)

        # epipolar line on right half
        a, b, c = float(eline[0]), float(eline[1]), float(eline[2])
        if abs(b) > 1e-9:
            ry0 = int(-c / b)
            ry1 = int(-(a * W + c) / b)
            cv2.line(canvas, (W + 20, ry0), (W + 20 + W, ry1), color, 1, cv2.LINE_AA)

        if m_pt is not None:
            mx, my = int(m_pt[0]), int(m_pt[1])
            cv2.circle(canvas, (W + 20 + mx, my), 5, color, -1)
            cv2.circle(canvas, (W + 20 + mx, my), 7, (255, 255, 255), 1)
            # correspondence line
            cv2.line(canvas, (kx, ky), (W + 20 + mx, my), color, 1, cv2.LINE_AA)

    return canvas
