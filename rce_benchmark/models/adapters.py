"""Model adapters for the benchmark."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from RCE.rce import RCE
from rce_benchmark.datasets.middlebury import build_train_patch_dataset
from rce_benchmark.models.backbones import BACKBONE_SPECS, FrozenBackboneExtractor


OBJECT_LABEL = "object"
BACKGROUND_LABEL = "background"


@dataclass
class ModelStats:
    model_id: str
    backbone: str
    head_type: str
    feature_dim: int
    n_train_samples: int
    n_prototypes: int | None
    train_ms: float


class BenchmarkModelAdapter:
    """Common binary benchmark model interface."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.stats = ModelStats(
            model_id=model_id,
            backbone="rgb",
            head_type="rce",
            feature_dim=3,
            n_train_samples=0,
            n_prototypes=None,
            train_ms=0.0,
        )
        self.train_mask_ = None
        self.train_bbox_ = None
        self.train_image_shape_ = None

    def fit(
        self,
        train_image: np.ndarray,
        train_mask: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        raise NotImplementedError

    def score_patch(self, patch_bgr: np.ndarray) -> float:
        raise NotImplementedError

    def classify_patch(self, patch_bgr: np.ndarray, threshold: float = 0.5) -> tuple[str, float]:
        conf = self.score_patch(patch_bgr)
        label = OBJECT_LABEL if conf >= threshold else BACKGROUND_LABEL
        return label, conf

    def collect_visual_state(self) -> dict[str, object]:
        """Return model-specific diagnostics for rendering."""
        return {}


class RawRCERGBModel(BenchmarkModelAdapter):
    """Proper RCE on sampled RGB pixels."""

    def __init__(self, model_id: str = "rce_rgb"):
        super().__init__(model_id)
        self.rce = RCE(R_max=80.0, default_label=BACKGROUND_LABEL)
        self.stats.backbone = "rgb"
        self.stats.head_type = "rce"
        self.stats.feature_dim = 3
        self.pixel_grid_side = 14
        self.positive_pixels_ = np.empty((0, 3), dtype=np.float64)
        self.background_pixels_ = np.empty((0, 3), dtype=np.float64)
        self.pixel_sample_count_ = 0

    def fit(
        self,
        train_image: np.ndarray,
        train_mask: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        start = time.perf_counter()
        self.train_mask_ = train_mask.copy()
        self.train_bbox_ = tuple(train_bbox)
        self.train_image_shape_ = train_image.shape[:2]

        roi_pixels = train_image[train_mask > 0].reshape(-1, 3).astype(np.float64)
        n_obj = int(training_cfg.get("rce_rgb_object_pixels", 1500))
        if len(roi_pixels) > n_obj:
            roi_pixels = roi_pixels[rng.choice(len(roi_pixels), n_obj, replace=False)]

        bg_pixels = train_image[train_mask == 0].reshape(-1, 3).astype(np.float64)
        n_bg = int(training_cfg.get("rce_rgb_background_pixels", 1500))
        if len(bg_pixels) > n_bg:
            bg_pixels = bg_pixels[rng.choice(len(bg_pixels), n_bg, replace=False)]

        x_train = np.vstack([roi_pixels, bg_pixels])
        y_train = np.concatenate(
            [
                np.full(len(roi_pixels), OBJECT_LABEL),
                np.full(len(bg_pixels), BACKGROUND_LABEL),
            ]
        )
        self.rce = RCE(
            R_max=float(training_cfg.get("rce_r_max", 80.0)),
            default_label=BACKGROUND_LABEL,
        )
        self.rce.fit(x_train, y_train)
        self.positive_pixels_ = roi_pixels
        self.background_pixels_ = bg_pixels
        self.pixel_sample_count_ = int(len(x_train))
        self.stats.n_train_samples = int(len(x_train))
        self.stats.n_prototypes = int(len(self.rce.centers_))
        self.stats.train_ms = (time.perf_counter() - start) * 1000

    def score_patch(self, patch_bgr: np.ndarray) -> float:
        h, w = patch_bgr.shape[:2]
        rows = np.linspace(0, h - 1, self.pixel_grid_side, dtype=int)
        cols = np.linspace(0, w - 1, self.pixel_grid_side, dtype=int)
        grid_r, grid_c = np.meshgrid(rows, cols, indexing="ij")
        pixels = patch_bgr[grid_r.ravel(), grid_c.ravel()].astype(np.float64)
        scores = self.rce.score_samples(pixels, OBJECT_LABEL, allow_nearest_margin=True)
        return float(np.mean(scores))

    def collect_visual_state(self) -> dict[str, object]:
        return {
            "positive_pixels": self.positive_pixels_,
            "background_pixels": self.background_pixels_,
            "rce": self.rce,
            "train_bbox": self.train_bbox_,
            "train_mask": self.train_mask_,
            "pixel_sample_count": self.pixel_sample_count_,
        }


class _BackboneStandardizer:
    """Train-fit z-score normalization."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "_BackboneStandardizer":
        self.mean_ = x_train.mean(axis=0)
        self.std_ = x_train.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x_data: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer is not fitted.")
        return (x_data - self.mean_) / self.std_


class _PatchBackboneModel(BenchmarkModelAdapter):
    """Shared patch-training flow for backbone models."""

    def __init__(self, model_id: str, backbone_id: str, head_type: str):
        super().__init__(model_id)
        self.backbone_id = backbone_id
        self.extractor = FrozenBackboneExtractor(backbone_id)
        self.standardizer = _BackboneStandardizer()
        self.stats.backbone = BACKBONE_SPECS[backbone_id].backbone_name
        self.stats.feature_dim = BACKBONE_SPECS[backbone_id].expected_dim
        self.stats.head_type = head_type
        self.train_features_ = np.empty((0, self.stats.feature_dim), dtype=np.float32)
        self.train_labels_ = np.empty((0,), dtype=object)
        self.train_patches_: list[np.ndarray] = []

    def _build_train_data(
        self,
        train_image: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        patches, labels = build_train_patch_dataset(
            image=train_image,
            bbox=train_bbox,
            class_label=OBJECT_LABEL,
            n_positive_patches=int(training_cfg.get("n_positive_patches", 8)),
            n_negative_patches=int(training_cfg.get("n_negative_patches", 24)),
            seed=seed,
        )
        features = np.array([self.extractor.extract(patch) for patch in patches], dtype=np.float32)
        features = self.standardizer.fit(features).transform(features)
        self.train_patches_ = [patch.copy() for patch in patches]
        self.train_features_ = features
        self.train_labels_ = np.array(labels, dtype=object)
        self.stats.n_train_samples = len(patches)
        return features, np.array(labels)

    def fit(
        self,
        train_image: np.ndarray,
        train_mask: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        self.train_mask_ = train_mask.copy()
        self.train_bbox_ = tuple(train_bbox)
        self.train_image_shape_ = train_image.shape[:2]
        self._fit_impl(train_image, train_bbox, training_cfg, seed)

    def _fit_impl(
        self,
        train_image: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        raise NotImplementedError

    def _transform_patch(self, patch_bgr: np.ndarray) -> np.ndarray:
        feats = self.extractor.extract(patch_bgr).reshape(1, -1).astype(np.float32)
        return self.standardizer.transform(feats)

    def collect_visual_state(self) -> dict[str, object]:
        return {
            "train_features": self.train_features_,
            "train_labels": self.train_labels_,
            "train_patches": self.train_patches_,
            "train_bbox": self.train_bbox_,
            "train_mask": self.train_mask_,
            "backbone_id": self.backbone_id,
            "head_type": self.stats.head_type,
        }


class LRBackboneModel(_PatchBackboneModel):
    """Frozen backbone + logistic regression head."""

    def __init__(self, model_id: str, backbone_id: str):
        super().__init__(model_id=model_id, backbone_id=backbone_id, head_type="lr")
        self.model = None

    def _fit_impl(
        self,
        train_image: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "scikit-learn is required for LR-backed benchmark models. "
                "Install the dependencies from requirements.txt before running these model ids."
            ) from exc

        start = time.perf_counter()
        x_train, y_train = self._build_train_data(train_image, train_bbox, training_cfg, seed)
        self.model = LogisticRegression(
            C=float(training_cfg.get("lr_c", 1.0)),
            max_iter=int(training_cfg.get("lr_max_iter", 1000)),
            random_state=seed,
        )
        self.model.fit(x_train, y_train)
        self.stats.n_prototypes = None
        self.stats.train_ms = (time.perf_counter() - start) * 1000

    def score_patch(self, patch_bgr: np.ndarray) -> float:
        feats = self._transform_patch(patch_bgr)
        probs = self.model.predict_proba(feats)[0]
        obj_idx = int(np.where(self.model.classes_ == OBJECT_LABEL)[0][0])
        return float(probs[obj_idx])

    def collect_visual_state(self) -> dict[str, object]:
        payload = super().collect_visual_state()
        payload["classifier"] = self.model
        return payload


class RCEBackboneModel(_PatchBackboneModel):
    """Frozen backbone + proper RCE head."""

    def __init__(self, model_id: str, backbone_id: str):
        super().__init__(model_id=model_id, backbone_id=backbone_id, head_type="rce")
        self.rce = RCE(R_max=80.0, default_label=BACKGROUND_LABEL)

    def _fit_impl(
        self,
        train_image: np.ndarray,
        train_bbox: tuple[int, int, int, int],
        training_cfg: dict,
        seed: int,
    ) -> None:
        start = time.perf_counter()
        x_train, y_train = self._build_train_data(train_image, train_bbox, training_cfg, seed)
        self.rce = RCE(
            R_max=float(training_cfg.get("rce_r_max", 80.0)),
            default_label=BACKGROUND_LABEL,
        )
        self.rce.fit(x_train, y_train)
        self.stats.n_prototypes = int(len(self.rce.centers_))
        self.stats.train_ms = (time.perf_counter() - start) * 1000

    def score_patch(self, patch_bgr: np.ndarray) -> float:
        feats = self._transform_patch(patch_bgr)
        return float(self.rce.score_samples(feats, OBJECT_LABEL, allow_nearest_margin=True)[0])

    def collect_visual_state(self) -> dict[str, object]:
        payload = super().collect_visual_state()
        payload["rce"] = self.rce
        return payload


MODEL_REGISTRY: dict[str, Callable[[], BenchmarkModelAdapter]] = {
    "rce_rgb": lambda: RawRCERGBModel("rce_rgb"),
    "lr_resnet18": lambda: LRBackboneModel("lr_resnet18", "resnet18"),
    "lr_mobilenetv3": lambda: LRBackboneModel("lr_mobilenetv3", "mobilenetv3"),
    "lr_mobilevit_xxs": lambda: LRBackboneModel("lr_mobilevit_xxs", "mobilevit_xxs"),
    "rce_resnet18": lambda: RCEBackboneModel("rce_resnet18", "resnet18"),
    "rce_mobilenetv3": lambda: RCEBackboneModel("rce_mobilenetv3", "mobilenetv3"),
    "rce_mobilevit_xxs": lambda: RCEBackboneModel("rce_mobilevit_xxs", "mobilevit_xxs"),
}


def create_model(model_id: str) -> BenchmarkModelAdapter:
    """Instantiate a benchmark model by id."""
    if model_id not in MODEL_REGISTRY:
        raise KeyError(f"Unknown benchmark model: {model_id}")
    return MODEL_REGISTRY[model_id]()
