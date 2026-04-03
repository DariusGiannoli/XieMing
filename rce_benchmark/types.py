"""Shared benchmark types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class Annotation:
    """Per-image annotation for one object instance."""

    variant: str
    view: str
    mask_path: str | None = None
    bbox: list[int] | None = None
    polygon: list[list[int]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Episode:
    """Benchmark episode description."""

    episode_id: str
    object_id: str
    object_name: str
    scene_group: str
    stage: str
    window_size: list[int]
    train_variant: str
    test_variant: str
    train_view: str
    test_view: str
    train_annotation: dict[str, Any]
    test_annotation: dict[str, Any]

    @property
    def train(self) -> Annotation:
        return Annotation(**self.train_annotation)

    @property
    def test(self) -> Annotation:
        return Annotation(**self.test_annotation)

    @property
    def train_bbox(self) -> list[int] | None:
        return self.train.bbox

    @property
    def test_bbox(self) -> list[int] | None:
        return self.test.bbox

    @property
    def win_w(self) -> int:
        return int(self.window_size[0])

    @property
    def win_h(self) -> int:
        return int(self.window_size[1])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResultRow:
    """One result row for CSV / JSON export."""

    stage: str
    task: str
    episode_id: str
    object_name: str
    scene_group: str
    train_variant: str
    test_variant: str
    model_id: str
    backbone: str
    head_type: str
    seed: int
    train_ms: float
    infer_ms_total: float
    infer_ms_per_window: float
    windows_per_sec: float
    n_train_samples: int
    feature_dim: int
    n_prototypes: int | None
    precision: float
    recall: float
    f1: float
    best_iou: float | None
    train_gt_bbox: list[int] | None = None
    test_gt_bbox: list[int] | None = None
    train_mask_path: str | None = None
    test_mask_path: str | None = None
    pred_mask_path: str | None = None
    heatmap_path: str | None = None
    overlay_path: str | None = None
    latent_plot_path: str | None = None
    prototype_gallery_path: str | None = None
    visual_summary_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
