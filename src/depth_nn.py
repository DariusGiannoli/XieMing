"""
src/depth_nn.py  —  Depth Anything V2 (Monocular Depth NN)
==========================================================
Loads Depth-Anything-V2-Small via HuggingFace ``transformers`` pipeline,
runs inference on CPU, and provides a scale+shift alignment against
ground-truth disparity so metrics are directly comparable with StereoSGBM.
"""

import time

import cv2
import numpy as np
import streamlit as st
import torch
from transformers import pipeline as hf_pipeline

_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ------------------------------------------------------------------
# Model loading  (cached across Streamlit reruns)
# ------------------------------------------------------------------
@st.cache_resource
def load_depth_anything():
    """Download (or use cached) Depth Anything V2 Small and return the
    HuggingFace depth-estimation pipeline."""
    return hf_pipeline(
        "depth-estimation",
        model=_MODEL_ID,
        device="cpu",
        dtype=torch.float32,
    )


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------
def predict_depth(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Run monocular depth estimation on a BGR image.

    Returns
    -------
    depth_raw : np.ndarray  (H, W) float32
        Raw relative inverse-depth output (not metric).
    elapsed_ms : float
        Wall-clock inference time in milliseconds.
    """
    pipe = load_depth_anything()
    from PIL import Image
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)

    t0 = time.perf_counter()
    result = pipe(pil_img)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    depth_pil = result["depth"]
    depth_raw = np.array(depth_pil, dtype=np.float32)

    # Resize to original resolution if the pipeline changed it
    h, w = img_bgr.shape[:2]
    if depth_raw.shape[:2] != (h, w):
        depth_raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_LINEAR)

    return depth_raw, elapsed_ms


# ------------------------------------------------------------------
# Scale + shift alignment against GT disparity
# ------------------------------------------------------------------
def align_to_gt(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Least-squares affine alignment of monocular prediction to GT disparity.

    Solves  ``scale * pred + shift ≈ gt``  over mutually valid pixels.

    Parameters
    ----------
    pred : (H, W) float, raw NN output
    gt   : (H, W) float, ground-truth disparity (Middlebury PFM)

    Returns
    -------
    aligned : (H, W) float, prediction in same disparity-pixel space as GT.
    """
    valid = np.isfinite(gt) & (gt > 0) & (pred > 0)
    if valid.sum() < 10:
        return pred  # not enough overlap
    A = np.stack([pred[valid], np.ones(valid.sum())], axis=1)
    params, *_ = np.linalg.lstsq(A, gt[valid], rcond=None)
    scale, shift = params
    return (scale * pred + shift).astype(np.float32)
