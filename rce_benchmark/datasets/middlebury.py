"""Middlebury dataset utilities for the visual-first RCE benchmark."""

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from rce_benchmark.annotations import bbox_from_mask, load_annotation_mask
from rce_benchmark.types import Annotation, Episode


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MIDDLEBURY_ROOT = PROJECT_ROOT / "data" / "middlebury"

VIEW_TO_FILENAME = {
    "left": "im0.png",
    "right": "im1.png",
    "im0": "im0.png",
    "im1": "im1.png",
}


def discover_scene_groups(root: str | Path = DEFAULT_MIDDLEBURY_ROOT) -> dict[str, list[str]]:
    """Discover Middlebury scene groups by stripping numeric suffixes."""
    root_path = Path(root)
    groups: dict[str, list[str]] = {}
    for scene_dir in sorted(root_path.iterdir()):
        if not scene_dir.is_dir():
            continue
        if not (scene_dir / "im0.png").is_file():
            continue
        base = re.sub(r"\d+$", "", scene_dir.name)
        groups.setdefault(base, []).append(scene_dir.name)
    return {key: sorted(value) for key, value in sorted(groups.items())}


def _load_scene_view(
    scene_name: str,
    view: str,
    root: str | Path = DEFAULT_MIDDLEBURY_ROOT,
) -> np.ndarray:
    """Load one named scene/view pair."""
    filename = VIEW_TO_FILENAME.get(view)
    if filename is None:
        raise ValueError(f"Unsupported Middlebury view: {view}")
    image = cv2.imread(str(Path(root) / scene_name / filename), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing {filename} for scene {scene_name}")
    return image


def load_middlebury_stereo_pair(
    scene_name: str,
    root: str | Path = DEFAULT_MIDDLEBURY_ROOT,
) -> dict[str, np.ndarray]:
    """Load im0/im1 for a stereo scene."""
    return {
        "left": _load_scene_view(scene_name, "left", root=root),
        "right": _load_scene_view(scene_name, "right", root=root),
    }


def load_middlebury_single_view(scene_name: str, root: str | Path = DEFAULT_MIDDLEBURY_ROOT) -> np.ndarray:
    """Load im0 for a Middlebury scene variant."""
    return _load_scene_view(scene_name, "left", root=root)


def load_episode_manifest(path: str | Path) -> list[Episode]:
    """Load an episode manifest from JSON."""
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, list):
        raise ValueError(f"Manifest must be a list: {manifest_path}")
    return [Episode(**row) for row in raw]


def mask_to_bbox(mask: np.ndarray) -> list[int]:
    """Derive a bbox from a binary mask."""
    return bbox_from_mask(mask)


def _validate_annotation(
    episode: Episode,
    annotation: Annotation,
    split_name: str,
    root: Path,
) -> list[str]:
    """Validate that an annotation resolves to a valid image and mask."""
    errors: list[str] = []
    try:
        image = _load_scene_view(annotation.variant, annotation.view, root=root)
    except Exception as exc:
        return [f"{episode.episode_id}: failed to load {split_name} image: {exc}"]

    try:
        mask = load_annotation_mask(annotation, image.shape[:2])
    except Exception as exc:
        return [f"{episode.episode_id}: failed to load {split_name} mask: {exc}"]

    if not np.any(mask > 0):
        errors.append(f"{episode.episode_id}: empty {split_name} mask")
        return errors

    derived_bbox = mask_to_bbox(mask)
    x0, y0, x1, y1 = derived_bbox
    height, width = image.shape[:2]
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        errors.append(f"{episode.episode_id}: invalid {split_name} bbox {derived_bbox}")
    return errors


def validate_manifest(episodes: list[Episode], root: str | Path = DEFAULT_MIDDLEBURY_ROOT) -> list[str]:
    """Validate that scenes, masks, and derived boxes exist and are sane."""
    errors: list[str] = []
    root_path = Path(root)

    for episode in episodes:
        if not episode.object_id or not episode.object_name:
            errors.append(f"{episode.episode_id}: missing object identity")

        if episode.train_variant != episode.train.variant:
            errors.append(f"{episode.episode_id}: train_variant mismatch with train annotation")
        if episode.test_variant != episode.test.variant:
            errors.append(f"{episode.episode_id}: test_variant mismatch with test annotation")
        if episode.train_view != episode.train.view:
            errors.append(f"{episode.episode_id}: train_view mismatch with train annotation")
        if episode.test_view != episode.test.view:
            errors.append(f"{episode.episode_id}: test_view mismatch with test annotation")

        errors.extend(_validate_annotation(episode, episode.train, "train", root_path))
        errors.extend(_validate_annotation(episode, episode.test, "test", root_path))

        if episode.train.to_dict() == episode.test.to_dict():
            errors.append(f"{episode.episode_id}: train and test annotations are identical")

        train_bbox = episode.train_bbox
        test_bbox = episode.test_bbox
        if train_bbox and test_bbox and train_bbox == test_bbox:
            errors.append(f"{episode.episode_id}: test bbox reuses the train bbox")

    return errors


def load_episode_assets(
    episode: Episode,
    root: str | Path = DEFAULT_MIDDLEBURY_ROOT,
) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    """Load images, masks, and derived boxes for one episode."""
    root_path = Path(root)
    train_image = _load_scene_view(episode.train.variant, episode.train.view, root=root_path)
    test_image = _load_scene_view(episode.test.variant, episode.test.view, root=root_path)
    train_mask = load_annotation_mask(episode.train, train_image.shape[:2])
    test_mask = load_annotation_mask(episode.test, test_image.shape[:2])
    train_bbox = tuple(mask_to_bbox(train_mask))
    test_bbox = tuple(mask_to_bbox(test_mask))
    return {
        "train_image": train_image,
        "test_image": test_image,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "train_bbox": train_bbox,
        "test_bbox": test_bbox,
    }


def _sample_negative_boxes(
    image_shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    n_boxes: int,
    seed: int,
) -> list[tuple[int, int, int, int]]:
    """Sample non-overlapping negative boxes with the same window size."""
    height, width = image_shape
    x0, y0, x1, y1 = bbox
    win_w = x1 - x0
    win_h = y1 - y0
    rng = np.random.default_rng(seed)
    negatives: list[tuple[int, int, int, int]] = []
    attempts = 0
    max_attempts = max(200, n_boxes * 20)
    while len(negatives) < n_boxes and attempts < max_attempts:
        nx = int(rng.integers(0, max(width - win_w, 1)))
        ny = int(rng.integers(0, max(height - win_h, 1)))
        candidate = (nx, ny, nx + win_w, ny + win_h)
        if _boxes_overlap(candidate, bbox):
            attempts += 1
            continue
        negatives.append(candidate)
        attempts += 1
    return negatives


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """Whether two boxes overlap at all."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def crop_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a bounding box from an image."""
    x0, y0, x1, y1 = map(int, box)
    return image[y0:y1, x0:x1].copy()


def build_test_patch_dataset(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    class_label: str,
    n_negative_patches: int,
    seed: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Build a binary patch classification test set from one image and ROI."""
    positive = crop_box(image, bbox)
    negatives = [
        crop_box(image, box)
        for box in _sample_negative_boxes(image.shape[:2], bbox, n_negative_patches, seed)
    ]
    patches = [positive] + negatives
    labels = [class_label] + ["background"] * len(negatives)
    return patches, labels


def build_augmented_positive_patches(
    roi_patch: np.ndarray,
    target_count: int,
) -> list[np.ndarray]:
    """Deterministic positive patch pool for training."""
    transforms = [
        lambda img: img,
        lambda img: cv2.flip(img, 1),
        lambda img: cv2.GaussianBlur(img, (3, 3), 0),
        lambda img: cv2.convertScaleAbs(img, alpha=1.1, beta=8),
        lambda img: cv2.convertScaleAbs(img, alpha=0.9, beta=-8),
        lambda img: cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
        lambda img: cv2.rotate(img, cv2.ROTATE_180),
        lambda img: cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE),
    ]
    patches: list[np.ndarray] = []
    idx = 0
    while len(patches) < target_count:
        patches.append(transforms[idx % len(transforms)](roi_patch))
        idx += 1
    return patches[:target_count]


def build_train_patch_dataset(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    class_label: str,
    n_positive_patches: int,
    n_negative_patches: int,
    seed: int,
) -> tuple[list[np.ndarray], list[str]]:
    """Build deterministic training patch datasets for backbone models."""
    positive = build_augmented_positive_patches(crop_box(image, bbox), n_positive_patches)
    negatives = [
        crop_box(image, box)
        for box in _sample_negative_boxes(image.shape[:2], bbox, n_negative_patches, seed)
    ]
    patches = positive + negatives
    labels = [class_label] * len(positive) + ["background"] * len(negatives)
    return patches, labels
