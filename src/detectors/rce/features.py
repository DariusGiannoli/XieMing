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
# Module 3 — Laplacian (2nd-order, curvature / blobs)
# ---------------------------------------------------------------------------
def compute_laplacian(gray: np.ndarray):
    """10-bin histogram of absolute Laplacian response (blob / corner energy)."""
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    mag = np.abs(lap)
    hist, _ = np.histogram(mag, bins=10, range=(0, 255))
    viz = (mag / (mag.max() + 1e-5)).astype(np.float32)
    return hist.astype(np.float32), viz


# ---------------------------------------------------------------------------
# Module 4 — Gradient Orientation (1st-order direction)
# ---------------------------------------------------------------------------
def compute_grad_orient(gray: np.ndarray):
    """10-bin histogram of gradient orientations (0-360°), weighted by magnitude."""
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx ** 2 + sy ** 2)
    angle = np.degrees(np.arctan2(sy, sx)) % 360
    hist, _ = np.histogram(angle, bins=10, range=(0, 360), weights=mag)
    viz = (angle / 360.0).astype(np.float32)
    return hist.astype(np.float32), viz


# ---------------------------------------------------------------------------
# Module 5 — Gabor (oriented texture / frequency)
# ---------------------------------------------------------------------------
def compute_gabor(gray: np.ndarray):
    """10-bin histogram of mean Gabor filter responses across 4 orientations."""
    ksize = 15
    sigma, lambd, gamma, psi = 4.0, 10.0, 0.5, 0.0
    responses = np.zeros_like(gray, dtype=np.float64)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        kernel = cv2.getGaborKernel(
            (ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_64F
        )
        responses += np.abs(cv2.filter2D(gray, cv2.CV_64F, kernel))
    responses /= 4.0
    hist, _ = np.histogram(responses, bins=10, range=(0, responses.max() + 1e-5))
    viz = (responses / (responses.max() + 1e-5)).astype(np.float32)
    return hist.astype(np.float32), viz


# ---------------------------------------------------------------------------
# Module 6 — LBP (Local Binary Pattern, texture micro-structure)
# ---------------------------------------------------------------------------
def compute_lbp(gray: np.ndarray):
    """10-bin histogram of simplified 8-neighbour LBP codes."""
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    h, w = gray.shape
    lbp = np.zeros((h, w), dtype=np.uint8)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    for bit, (dy, dx) in enumerate(offsets):
        neighbour = padded[1 + dy: 1 + dy + h, 1 + dx: 1 + dx + w]
        lbp |= ((neighbour >= gray).astype(np.uint8) << bit)
    hist = cv2.calcHist([lbp], [0], None, [10], [0, 256]).flatten().astype(np.float32)
    viz = lbp.astype(np.float32) / 255.0
    return hist, viz


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
    "laplacian": {
        "label": "2-Order (Laplacian)",
        "fn": compute_laplacian,
        "viz_title": "Curvature / Blobs (Laplacian)",
    },
    "grad_orient": {
        "label": "Gradient Orient.",
        "fn": compute_grad_orient,
        "viz_title": "Edge Directions",
    },
    "gabor": {
        "label": "Gabor (Texture)",
        "fn": compute_gabor,
        "viz_title": "Oriented Texture (Gabor)",
    },
    "lbp": {
        "label": "LBP (Local Texture)",
        "fn": compute_lbp,
        "viz_title": "Local Binary Pattern",
    },
}
