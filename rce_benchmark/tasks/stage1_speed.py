"""Stage 1: reproduce and generalize the current speed comparison."""

from __future__ import annotations

from rce_benchmark.datasets import load_episode_assets
from rce_benchmark.metrics import eval_detection_metrics
from rce_benchmark.models import create_model
from rce_benchmark.tasks.common import sliding_window_detect
from rce_benchmark.types import ResultRow
from rce_benchmark.visuals import export_episode_artifacts


def run_stage1_speed(config: dict, episodes: list, model_ids: list[str]) -> list[ResultRow]:
    """Run Stage 1 on stereo train-left / test-right episodes."""
    rows: list[ResultRow] = []
    stride = int(config["inference"].get("stride", 75))
    conf_thresh = float(config["inference"].get("conf_thresh", 0.5))
    nms_iou = float(config["inference"].get("nms_iou", 0.3))
    eval_iou = float(config["inference"].get("eval_iou", 0.5))
    output_dir = config["output_dir"]

    for episode in episodes:
        assets = load_episode_assets(episode)
        train_image = assets["train_image"]
        test_image = assets["test_image"]
        train_mask = assets["train_mask"]
        test_mask = assets["test_mask"]
        train_bbox = assets["train_bbox"]
        test_bbox = assets["test_bbox"]

        for seed in config["seeds"]:
            for model_id in model_ids:
                model = create_model(model_id)
                model.fit(train_image, train_mask, train_bbox, config["training"], int(seed))
                detections, infer_ms, n_windows, heatmap, pred_mask = sliding_window_detect(
                    image=test_image,
                    model=model,
                    win_h=episode.win_h,
                    win_w=episode.win_w,
                    stride=stride,
                    conf_thresh=conf_thresh,
                    nms_iou=nms_iou,
                )
                det_metrics = eval_detection_metrics(detections, test_bbox, eval_iou)
                metrics_dict = {
                    "precision": det_metrics.precision,
                    "recall": det_metrics.recall,
                    "f1": det_metrics.f1,
                    "best_iou": det_metrics.best_iou,
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
                rows.append(
                    ResultRow(
                        stage="stage1",
                        task="localization",
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
                        infer_ms_total=infer_ms,
                        infer_ms_per_window=(infer_ms / n_windows) if n_windows else 0.0,
                        windows_per_sec=(n_windows / (infer_ms / 1000.0)) if infer_ms else 0.0,
                        n_train_samples=model.stats.n_train_samples,
                        feature_dim=model.stats.feature_dim,
                        n_prototypes=model.stats.n_prototypes,
                        precision=det_metrics.precision,
                        recall=det_metrics.recall,
                        f1=det_metrics.f1,
                        best_iou=det_metrics.best_iou,
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
                )
    return rows
