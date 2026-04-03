"""Stage 2: Middlebury-first classification plus localization transfer."""

from __future__ import annotations

import time

from rce_benchmark.datasets import build_test_patch_dataset, load_episode_assets
from rce_benchmark.metrics import classification_metrics, eval_detection_metrics
from rce_benchmark.models import create_model
from rce_benchmark.tasks.common import sliding_window_detect
from rce_benchmark.types import ResultRow
from rce_benchmark.visuals import export_episode_artifacts


def run_stage2_middlebury(config: dict, episodes: list, model_ids: list[str]) -> list[ResultRow]:
    """Run Stage 2 tasks for each episode."""
    rows: list[ResultRow] = []
    stride = int(config["inference"].get("stride", 75))
    conf_thresh = float(config["inference"].get("conf_thresh", 0.5))
    nms_iou = float(config["inference"].get("nms_iou", 0.3))
    eval_iou = float(config["inference"].get("eval_iou", 0.5))
    class_threshold = float(config["inference"].get("classification_threshold", 0.5))
    test_negative_patches = int(config["training"].get("stage2_test_negative_patches", 12))
    output_dir = config["output_dir"]

    for episode in episodes:
        assets = load_episode_assets(episode)
        train_image = assets["train_image"]
        test_image = assets["test_image"]
        train_mask = assets["train_mask"]
        test_mask = assets["test_mask"]
        train_bbox = assets["train_bbox"]
        test_bbox = assets["test_bbox"]

        test_patches, test_labels = build_test_patch_dataset(
            image=test_image,
            bbox=test_bbox,
            class_label=episode.object_name,
            n_negative_patches=test_negative_patches,
            seed=0,
        )
        for seed in config["seeds"]:
            for model_id in model_ids:
                model = create_model(model_id)
                model.fit(train_image, train_mask, train_bbox, config["training"], int(seed))

                cls_start = time.perf_counter()
                predictions = []
                for patch in test_patches:
                    label, _ = model.classify_patch(patch_bgr=patch, threshold=class_threshold)
                    predictions.append(episode.object_name if label == "object" else "background")
                cls_ms = (time.perf_counter() - cls_start) * 1000
                cls_metrics = classification_metrics(
                    y_true=test_labels,
                    y_pred=predictions,
                    positive_label=episode.object_name,
                )

                detections, infer_ms, n_windows, heatmap, pred_mask = sliding_window_detect(
                    image=test_image,
                    model=model,
                    win_h=episode.win_h,
                    win_w=episode.win_w,
                    stride=stride,
                    conf_thresh=conf_thresh,
                    nms_iou=nms_iou,
                )
                loc_metrics = eval_detection_metrics(detections, test_bbox, eval_iou)
                metrics_dict = {
                    "precision": loc_metrics.precision,
                    "recall": loc_metrics.recall,
                    "f1": loc_metrics.f1,
                    "best_iou": loc_metrics.best_iou,
                }
                artifact_paths = export_episode_artifacts(
                    output_root=output_dir,
                    episode=episode,
                    model=model,
                    seed=int(seed),
                    train_image=train_image,
                    test_image=test_image,
                    train_mask=train_mask,
                    test_mask=test_mask,
                    train_bbox=train_bbox,
                    test_bbox=test_bbox,
                    heatmap=heatmap,
                    pred_mask=pred_mask,
                    detections=detections,
                    metrics=metrics_dict,
                )

                common = dict(
                    stage="stage2",
                    episode_id=episode.episode_id,
                    object_name=episode.object_name,
                    scene_group=episode.scene_group or "",
                    train_variant=str(episode.train_variant),
                    test_variant=str(episode.test_variant),
                    model_id=model.stats.model_id,
                    backbone=model.stats.backbone,
                    head_type=model.stats.head_type,
                    seed=int(seed),
                    train_ms=model.stats.train_ms,
                    n_train_samples=model.stats.n_train_samples,
                    feature_dim=model.stats.feature_dim,
                    n_prototypes=model.stats.n_prototypes,
                    train_gt_bbox=list(train_bbox),
                    test_gt_bbox=list(test_bbox),
                    train_mask_path=episode.train.mask_path,
                    test_mask_path=episode.test.mask_path,
                    pred_mask_path=artifact_paths["pred_mask_path"],
                    heatmap_path=artifact_paths["heatmap_path"],
                    overlay_path=artifact_paths["overlay_path"],
                    latent_plot_path=artifact_paths["latent_plot_path"],
                    prototype_gallery_path=artifact_paths["prototype_gallery_path"],
                    visual_summary_path=artifact_paths["visual_summary_path"],
                )

                rows.append(
                    ResultRow(
                        task="classification",
                        infer_ms_total=cls_ms,
                        infer_ms_per_window=(cls_ms / len(test_patches)) if test_patches else 0.0,
                        windows_per_sec=(len(test_patches) / (cls_ms / 1000.0)) if cls_ms else 0.0,
                        precision=cls_metrics["precision"],
                        recall=cls_metrics["recall"],
                        f1=cls_metrics["f1"],
                        best_iou=None,
                        **common,
                    )
                )
                rows.append(
                    ResultRow(
                        task="localization_transfer",
                        infer_ms_total=infer_ms,
                        infer_ms_per_window=(infer_ms / n_windows) if n_windows else 0.0,
                        windows_per_sec=(n_windows / (infer_ms / 1000.0)) if infer_ms else 0.0,
                        precision=loc_metrics.precision,
                        recall=loc_metrics.recall,
                        f1=loc_metrics.f1,
                        best_iou=loc_metrics.best_iou,
                        **common,
                    )
                )
    return rows
