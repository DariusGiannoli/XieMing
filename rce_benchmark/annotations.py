"""Annotation helpers for segmentation-first evaluation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from rce_benchmark.types import Annotation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_repo_path(path_str: str | None) -> Path | None:
    """Resolve a repo-relative or absolute path."""
    if path_str is None:
        return None
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def mask_from_polygon(image_shape: tuple[int, int], polygon: list[list[int]]) -> np.ndarray:
    """Fill a polygon into a binary mask."""
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def mask_from_bbox(image_shape: tuple[int, int], bbox: list[int]) -> np.ndarray:
    """Create a rectangular mask from a bbox."""
    x0, y0, x1, y1 = map(int, bbox)
    height, width = image_shape
    mask = np.zeros((height, width), dtype=np.uint8)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    mask[y0:y1, x0:x1] = 255
    return mask


def load_annotation_mask(annotation: Annotation, image_shape: tuple[int, int]) -> np.ndarray:
    """Load or derive an annotation mask."""
    resolved = resolve_repo_path(annotation.mask_path)
    if resolved is not None and resolved.is_file():
        mask = cv2.imread(str(resolved), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read annotation mask: {resolved}")
        if mask.shape != image_shape:
            raise ValueError(
                f"Mask/image shape mismatch for {resolved}: mask={mask.shape}, image={image_shape}"
            )
        return (mask > 0).astype(np.uint8) * 255
    if annotation.polygon:
        return mask_from_polygon(image_shape, annotation.polygon)
    if annotation.bbox:
        return mask_from_bbox(image_shape, annotation.bbox)
    raise ValueError("Annotation needs one of mask_path, polygon, or bbox.")


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    """Derive a bbox from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Cannot derive bbox from an empty mask.")
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_mask(mask: np.ndarray, path: str | Path) -> Path:
    """Persist a binary mask as PNG."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(out_path), mask_uint8):
        raise IOError(f"Failed to write mask: {out_path}")
    return out_path


def annotation_to_serializable(
    variant: str,
    view: str,
    mask_path: str | None,
    bbox: list[int] | None,
    polygon: list[list[int]] | None = None,
) -> dict[str, object]:
    """Create a serializable annotation payload."""
    payload: dict[str, object] = {
        "variant": variant,
        "view": view,
        "mask_path": mask_path,
        "bbox": bbox,
    }
    if polygon is not None:
        payload["polygon"] = polygon
    return payload
