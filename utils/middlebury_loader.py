"""
Middlebury Dataset Loader
=========================
Scans, groups, loads and parses Middlebury stereo-pair data bundled at
``./data/middlebury/``.
"""

import io
import os
import re
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MIDDLEBURY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "middlebury",
)

BUNDLED_SCENES = {
    "artroom":  ["artroom1", "artroom2"],
    "curule":   ["curule1", "curule2", "curule3"],
    "skates":   ["skates1", "skates2"],
    "skiboots": ["skiboots1", "skiboots2", "skiboots3"],
}


# ------------------------------------------------------------------
# Scanning
# ------------------------------------------------------------------

def scan_dataset_root(root_path: str = DEFAULT_MIDDLEBURY_ROOT) -> list:
    """Return sorted list of scene names that contain im0.png, im1.png, calib.txt."""
    if not os.path.isdir(root_path):
        return []
    scenes = []
    for entry in sorted(os.listdir(root_path)):
        scene_dir = os.path.join(root_path, entry)
        if not os.path.isdir(scene_dir):
            continue
        required = ["im0.png", "im1.png", "calib.txt"]
        if all(os.path.isfile(os.path.join(scene_dir, f)) for f in required):
            scenes.append(entry)
    return scenes


def get_scene_groups(root_path: str = DEFAULT_MIDDLEBURY_ROOT) -> dict:
    """Group scenes by base name (strip trailing digits)."""
    scenes = scan_dataset_root(root_path)
    groups = {}
    for name in scenes:
        base = re.sub(r"\d+$", "", name)
        groups.setdefault(base, []).append(name)
    return {k: sorted(v) for k, v in sorted(groups.items())}


def get_available_views(scene_path: str) -> list:
    """Return available view variants.  Always single entry for this dataset."""
    return [{"suffix": "", "label": "Primary (im0/im1)"}]


# ------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------

def load_stereo_pair(scene_path: str, view_suffix: str = "") -> dict:
    """Load left + right images, calibration and optional GT disparity."""
    left = cv2.imread(os.path.join(scene_path, f"im0{view_suffix}.png"),
                      cv2.IMREAD_COLOR)
    right = cv2.imread(os.path.join(scene_path, f"im1{view_suffix}.png"),
                       cv2.IMREAD_COLOR)
    calib = parse_calib(os.path.join(scene_path, "calib.txt"))

    disp0_path = os.path.join(scene_path, "disp0.pfm")
    disparity_gt = load_pfm(disp0_path) if os.path.isfile(disp0_path) else None

    return {
        "left": left,
        "right": right,
        "calib": calib,
        "disparity_gt": disparity_gt,
    }


def load_single_view(scene_path: str, view_suffix: str = "") -> np.ndarray:
    """Load and return im0{suffix}.png from a scene folder."""
    return cv2.imread(os.path.join(scene_path, f"im0{view_suffix}.png"),
                      cv2.IMREAD_COLOR)


# ------------------------------------------------------------------
# Calibration parser
# ------------------------------------------------------------------

def parse_calib(calib_path: str) -> dict:
    """
    Parse Middlebury ``calib.txt``.
    Returns dict with at least: fx, baseline, doffs, width, height, ndisp.
    Also returns raw cam0/cam1 matrices and conf_raw text.
    """
    text = Path(calib_path).read_text()
    params = {}
    for line in text.strip().splitlines():
        line = line.strip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip()
        if "[" in val:
            nums = list(map(float,
                            re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", val)))
            params[key] = np.array(nums).reshape(3, 3) if len(nums) == 9 else nums
        else:
            try:
                params[key] = float(val)
            except ValueError:
                params[key] = val

    cam0 = params.get("cam0")
    fx = float(cam0[0, 0]) if isinstance(cam0, np.ndarray) and cam0.shape == (3, 3) else 0.0
    params["fx"] = fx
    params["conf_raw"] = text
    return params


# ------------------------------------------------------------------
# PFM loader
# ------------------------------------------------------------------

def load_pfm(filepath: str) -> np.ndarray:
    """Read a PFM (Portable FloatMap) and return a float32 ndarray."""
    with open(filepath, "rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in ("Pf", "PF"):
            raise ValueError(f"Not a valid PFM file (header: {header!r})")
        color = header == "PF"
        line = f.readline().decode("ascii").strip()
        while line.startswith("#"):
            line = f.readline().decode("ascii").strip()
        w, h = map(int, line.split())
        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        channels = 3 if color else 1
        data = np.frombuffer(f.read(), dtype=np.dtype(endian + "f4"))
        data = data.reshape((h, w, channels) if color else (h, w))
        return np.flipud(data.copy())


def read_pfm_bytes(file_bytes: bytes) -> np.ndarray:
    """Parse PFM from raw bytes (uploaded file)."""
    buf = io.BytesIO(file_bytes)
    header = buf.readline().decode("ascii").strip()
    if header not in ("Pf", "PF"):
        raise ValueError(f"Not a valid PFM file (header: {header!r})")
    color = header == "PF"
    line = buf.readline().decode("ascii").strip()
    while line.startswith("#"):
        line = buf.readline().decode("ascii").strip()
    w, h = map(int, line.split())
    scale = float(buf.readline().decode("ascii").strip())
    endian = "<" if scale < 0 else ">"
    channels = 3 if color else 1
    data = np.frombuffer(buf.read(), dtype=np.dtype(endian + "f4"))
    data = data.reshape((h, w, channels) if color else (h, w))
    return np.flipud(data.copy())
