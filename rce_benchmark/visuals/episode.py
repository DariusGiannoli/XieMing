"""Episode-level visual exports."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rce_benchmark.types import Episode


def _as_display_rgb(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _draw_bbox(image_bgr: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int]) -> np.ndarray:
    canvas = image_bgr.copy()
    x0, y0, x1, y1 = map(int, bbox)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 4)
    return canvas


def _overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    canvas = image_bgr.copy()
    overlay = np.zeros_like(canvas)
    overlay[mask > 0] = np.array(color, dtype=np.uint8)
    return cv2.addWeighted(overlay, alpha, canvas, 1.0, 0.0)


def _save_image(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError(f"Failed to save image: {path}")
    return str(path)


def _save_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    if not cv2.imwrite(str(path), mask_uint8):
        raise IOError(f"Failed to save mask: {path}")
    return str(path)


def _save_heatmap(path: Path, heatmap: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(heatmap, 0.0, 1.0)
    colored = cv2.applyColorMap((clipped * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    if not cv2.imwrite(str(path), colored):
        raise IOError(f"Failed to save heatmap: {path}")
    return str(path)


def _save_overview(
    path: Path,
    episode: Episode,
    train_image: np.ndarray,
    test_image: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    pred_mask: np.ndarray,
    heatmap: np.ndarray,
    train_bbox: tuple[int, int, int, int],
    test_bbox: tuple[int, int, int, int],
    detections: list[tuple],
) -> str:
    best_det = detections[0][:4] if detections else None
    tp = ((pred_mask > 0) & (test_mask > 0)).astype(np.uint8) * 255
    fp = ((pred_mask > 0) & (test_mask == 0)).astype(np.uint8) * 255
    fn = ((pred_mask == 0) & (test_mask > 0)).astype(np.uint8) * 255

    error_overlay = test_image.copy()
    error_overlay[tp > 0] = (0, 200, 0)
    error_overlay[fp > 0] = (0, 0, 255)
    error_overlay[fn > 0] = (255, 180, 0)
    error_overlay = cv2.addWeighted(error_overlay, 0.4, test_image, 0.8, 0.0)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    panels = [
        (_draw_bbox(_overlay_mask(train_image, train_mask, (0, 200, 0)), train_bbox, (0, 255, 255)), "Train view + mask"),
        (_draw_bbox(_overlay_mask(test_image, test_mask, (0, 200, 0)), test_bbox, (0, 255, 255)), "Test view + GT mask"),
        (cv2.applyColorMap((np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLORMAP_TURBO), "Confidence heatmap"),
        (_overlay_mask(test_image, pred_mask, (255, 0, 255)), "Predicted mask"),
        (error_overlay, "TP / FP / FN overlay"),
        (test_image.copy(), "Detections"),
    ]
    if best_det is not None:
        panels[-1] = (_draw_bbox(test_image.copy(), best_det, (255, 0, 255)), "Best detection")

    for ax, (panel, title) in zip(axes.flat, panels):
        ax.imshow(_as_display_rgb(panel))
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(f"{episode.episode_id} | {episode.object_name}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _pca_project(features: np.ndarray, n_components: int = 2) -> np.ndarray:
    if len(features) == 0:
        return np.empty((0, n_components), dtype=np.float32)
    centered = features - features.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components].T
    return centered @ basis


def _save_latent_plot(path: Path, model, visual_state: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))

    if model.stats.backbone == "rgb":
        ax = fig.add_subplot(111, projection="3d")
        positive = np.asarray(visual_state.get("positive_pixels", np.empty((0, 3))))
        background = np.asarray(visual_state.get("background_pixels", np.empty((0, 3))))
        rce = visual_state.get("rce")

        if len(background):
            sample = background[: min(len(background), 250)]
            ax.scatter(sample[:, 2], sample[:, 1], sample[:, 0], c=sample[:, ::-1] / 255.0, s=10, alpha=0.25, label="background")
        if len(positive):
            sample = positive[: min(len(positive), 250)]
            ax.scatter(sample[:, 2], sample[:, 1], sample[:, 0], c=sample[:, ::-1] / 255.0, s=14, alpha=0.6, label="object")

        if rce is not None and rce.centers_ is not None:
            support = getattr(rce, "support_counts_", np.ones(len(rce.centers_)))
            order = np.argsort(support)[::-1][: min(6, len(support))]
            for idx in order:
                center = rce.centers_[idx]
                label = rce.labels_[idx]
                color = "green" if label == "object" else "red"
                ax.scatter(center[2], center[1], center[0], c=color, s=60, marker="x")

                radius = float(rce.radii_[idx])
                if radius <= 0:
                    continue
                u = np.linspace(0, 2 * np.pi, 20)
                v = np.linspace(0, np.pi, 12)
                xs = center[2] + radius * np.outer(np.cos(u), np.sin(v))
                ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                zs = center[0] + radius * np.outer(np.ones_like(u), np.cos(v))
                ax.plot_wireframe(xs, ys, zs, color=color, alpha=0.12, linewidth=0.4)

        ax.set_xlabel("R")
        ax.set_ylabel("G")
        ax.set_zlabel("B")
        ax.set_title("Exact RGB latent geometry with selected hyperspheres")
        ax.legend(loc="upper left")
    else:
        ax = fig.add_subplot(111)
        features = np.asarray(visual_state.get("train_features", np.empty((0, model.stats.feature_dim))))
        labels = np.asarray(visual_state.get("train_labels", np.empty((0,), dtype=object)))
        projected = _pca_project(features, n_components=2)
        if len(projected):
            obj_mask = labels == "object"
            bg_mask = labels != "object"
            if np.any(bg_mask):
                ax.scatter(projected[bg_mask, 0], projected[bg_mask, 1], c="#b22222", s=25, alpha=0.55, label="background train patches")
            if np.any(obj_mask):
                ax.scatter(projected[obj_mask, 0], projected[obj_mask, 1], c="#0b7d45", s=30, alpha=0.75, label="object train patches")

        rce = visual_state.get("rce")
        if rce is not None and getattr(rce, "centers_", None) is not None and len(rce.centers_):
            centers = np.asarray(rce.centers_, dtype=np.float32)
            combo = np.vstack([features, centers]) if len(features) else centers
            combo_proj = _pca_project(combo, n_components=2)
            center_proj = combo_proj[-len(centers) :]
            ax.scatter(center_proj[:, 0], center_proj[:, 1], c="#1f3b82", s=70, marker="x", label="projected RCE prototypes")
            ax.text(
                0.01,
                0.01,
                "PCA projection only. Prototype markers are illustrative,\nnot exact hyperspheres in feature space.",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=9,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
            )

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title(f"{model.stats.backbone} feature latent view")
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_prototype_gallery(path: Path, model, visual_state: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(12, 6))
    fig.suptitle(f"{model.model_id} prototype / exemplar gallery")

    if model.stats.backbone == "rgb":
        rce = visual_state.get("rce")
        if rce is None or rce.centers_ is None or len(rce.centers_) == 0:
            plt.text(0.5, 0.5, "No prototypes", ha="center", va="center")
        else:
            support = getattr(rce, "support_counts_", np.ones(len(rce.centers_)))
            order = np.argsort(support)[::-1][: min(12, len(support))]
            for plot_idx, proto_idx in enumerate(order, start=1):
                ax = fig.add_subplot(3, 4, plot_idx)
                swatch = np.zeros((48, 48, 3), dtype=np.uint8)
                color = np.clip(rce.centers_[proto_idx], 0, 255).astype(np.uint8)
                swatch[:] = color
                ax.imshow(_as_display_rgb(swatch))
                ax.set_title(
                    f"{rce.labels_[proto_idx]}\nR={rce.radii_[proto_idx]:.1f} k={int(support[proto_idx])}",
                    fontsize=9,
                )
                ax.axis("off")
    else:
        patches = visual_state.get("train_patches", [])
        labels = np.asarray(visual_state.get("train_labels", []))
        rce = visual_state.get("rce")
        selections: list[int] = []
        titles: list[str] = []

        if rce is not None and len(patches) and len(getattr(rce, "centers_", [])):
            feats = np.asarray(visual_state.get("train_features", []))
            support = getattr(rce, "support_counts_", np.ones(len(rce.centers_)))
            for proto_idx in np.argsort(support)[::-1][: min(8, len(support))]:
                dists = np.linalg.norm(feats - rce.centers_[proto_idx], axis=1)
                nearest = int(np.argmin(dists))
                selections.append(nearest)
                titles.append(
                    f"proto {proto_idx}\n{rce.labels_[proto_idx]} k={int(support[proto_idx])}"
                )
        else:
            positive = list(np.where(labels == "object")[0][:4])
            negative = list(np.where(labels != "object")[0][:4])
            selections = positive + negative
            titles = [str(labels[idx]) for idx in selections]

        if not selections:
            plt.text(0.5, 0.5, "No train patches", ha="center", va="center")
        else:
            cols = min(4, len(selections))
            rows = int(np.ceil(len(selections) / cols))
            for plot_idx, (patch_idx, title) in enumerate(zip(selections, titles), start=1):
                ax = fig.add_subplot(rows, cols, plot_idx)
                ax.imshow(_as_display_rgb(patches[patch_idx]))
                ax.set_title(title, fontsize=9)
                ax.axis("off")

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _save_visual_summary(
    path: Path,
    episode: Episode,
    model,
    metrics: dict[str, float],
    artifact_paths: dict[str, str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = [
        f"# {episode.episode_id} visual summary",
        "",
        f"- Object: {episode.object_name} (`{episode.object_id}`)",
        f"- Model: {model.model_id}",
        f"- Backbone: {model.stats.backbone}",
        f"- Head: {model.stats.head_type}",
        f"- Train samples: {model.stats.n_train_samples}",
        f"- Prototypes: {model.stats.n_prototypes}",
        f"- Precision: {metrics['precision']:.3f}",
        f"- Recall: {metrics['recall']:.3f}",
        f"- F1: {metrics['f1']:.3f}",
        f"- Best IoU: {metrics.get('best_iou', 0.0):.3f}",
        "",
        "## Artifact Paths",
        "",
        f"- Overlay: {artifact_paths['overlay_path']}",
        f"- Heatmap: {artifact_paths['heatmap_path']}",
        f"- Predicted mask: {artifact_paths['pred_mask_path']}",
        f"- Latent plot: {artifact_paths['latent_plot_path']}",
        f"- Prototype gallery: {artifact_paths['prototype_gallery_path']}",
    ]
    path.write_text("\n".join(summary), encoding="utf-8")
    return str(path)


def export_episode_artifacts(
    output_root: str | Path,
    episode: Episode,
    model,
    seed: int,
    train_image: np.ndarray,
    test_image: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    train_bbox: tuple[int, int, int, int],
    test_bbox: tuple[int, int, int, int],
    heatmap: np.ndarray,
    pred_mask: np.ndarray,
    detections: list[tuple],
    metrics: dict[str, float],
) -> dict[str, str]:
    """Persist a complete artifact bundle for one episode/model/seed."""
    bundle_dir = _ensure_dir(Path(output_root) / "artifacts" / episode.episode_id / model.model_id / f"seed_{seed}")
    pred_mask_path = _save_mask(bundle_dir / "pred_mask.png", pred_mask)
    heatmap_path = _save_heatmap(bundle_dir / "heatmap.png", heatmap)
    overlay_path = _save_overview(
        bundle_dir / "overview.png",
        episode=episode,
        train_image=train_image,
        test_image=test_image,
        train_mask=train_mask,
        test_mask=test_mask,
        pred_mask=pred_mask,
        heatmap=heatmap,
        train_bbox=train_bbox,
        test_bbox=test_bbox,
        detections=detections,
    )
    visual_state = model.collect_visual_state()
    latent_plot_path = _save_latent_plot(bundle_dir / "latent.png", model, visual_state)
    prototype_gallery_path = _save_prototype_gallery(bundle_dir / "prototype_gallery.png", model, visual_state)
    artifact_paths = {
        "pred_mask_path": pred_mask_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "latent_plot_path": latent_plot_path,
        "prototype_gallery_path": prototype_gallery_path,
    }
    visual_summary_path = _save_visual_summary(
        bundle_dir / "visual_summary.md",
        episode=episode,
        model=model,
        metrics=metrics,
        artifact_paths=artifact_paths,
    )
    artifact_paths["visual_summary_path"] = visual_summary_path
    return artifact_paths
