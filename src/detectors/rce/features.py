"""
RCE Feature Modules
====================
Each public function is one self-contained physical feature module.
Signature convention:
    compute_<name>(gray: np.ndarray) -> (vector: np.ndarray, viz: np.ndarray)
        gray  — single-channel uint8 image
        vector — 1-D float array contributed to the DNA vector
        viz    — float32 image in [0, 1] used for visualization

To add a new feature:
    1. Write a compute_<name>(gray) function below following the convention.
    2. Add an entry to REGISTRY at the bottom of this file.
    3. Add the corresponding checkbox in pages/3_Feature_Lab.py.
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Module 0 — Intensity (0th-order statistics)
# ---------------------------------------------------------------------------
def compute_intensity(gray: np.ndarray):
    """10-bin histogram of raw pixel brightness."""
    hist = cv2.calcHist([gray], [0], None, [10], [0, 256]).flatten().astype(np.float32)
    viz = gray.astype(np.float32) / 255.0
    return hist, viz


# ---------------------------------------------------------------------------
# Module 1 — Geometric Edges (1st-order, Sobel)
# ---------------------------------------------------------------------------
def compute_sobel(gray: np.ndarray):
    """10-bin histogram of Sobel gradient magnitudes."""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    hist, _ = np.histogram(mag, bins=10, range=(0, 255))
    viz = (mag / (mag.max() + 1e-5)).astype(np.float32)
    return hist.astype(np.float32), viz


# ---------------------------------------------------------------------------
# Module 2 — Spectral / Texture (FFT)
# ---------------------------------------------------------------------------
def compute_spectral(gray: np.ndarray):
    """10-bin histogram of log-magnitude FFT spectrum."""
    dft_shift = np.fft.fftshift(np.fft.fft2(gray))
    spec = 20 * np.log(np.abs(dft_shift) + 1)
    hist, _ = np.histogram(spec, bins=10, range=(0, spec.max()))
    viz = (spec / (spec.max() + 1e-5)).astype(np.float32)
    return hist.astype(np.float32), viz


# ---------------------------------------------------------------------------
# Registry — defines the order and display labels seen by the UI
# Add new modules here; the Feature Lab page iterates this dict.
# ---------------------------------------------------------------------------
REGISTRY: dict = {
    "intensity": {
        "label": "0-Order (Intensity)",
        "fn": compute_intensity,
        "viz_title": "Intensity Distribution",
    },
    "sobel": {
        "label": "1-Order (Sobel/Edges)",
        "fn": compute_sobel,
        "viz_title": "Geometric Edges (Sobel)",
    },
    "spectral": {
        "label": "Spectral (FFT/Texture)",
        "fn": compute_spectral,
        "viz_title": "Frequency Domain (FFT)",
    },
    # --- ADD NEW MODULES BELOW ---
    # "lbp": {
    #     "label": "LBP (Local Texture)",
    #     "fn": compute_lbp,
    #     "viz_title": "LBP Pattern",
    # },
}
