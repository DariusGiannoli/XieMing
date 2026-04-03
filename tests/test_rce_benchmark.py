"""Smoke tests for the RCE benchmark package."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from RCE.rce import RCE
from rce_benchmark.config import load_config
from rce_benchmark.datasets import load_episode_manifest, mask_to_bbox, validate_manifest
from rce_benchmark.models import MODEL_REGISTRY
from rce_benchmark.models.backbones import BACKBONE_SPECS
from rce_benchmark.notebook_tools import export_episode_payload
from rce_benchmark.tasks import run_stage1_speed, run_stage2_middlebury


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE1_CFG = PROJECT_ROOT / "rce_benchmark" / "configs" / "stage1_smoke.yaml"
STAGE2_CFG = PROJECT_ROOT / "rce_benchmark" / "configs" / "stage2_smoke.yaml"


class RCEBenchmarkTests(unittest.TestCase):
    """Benchmark smoke coverage."""

    def test_config_loads(self) -> None:
        config = load_config(STAGE1_CFG)
        self.assertEqual(config["stage"], "stage1")
        self.assertIn("model_ids", config)

    def test_model_registry_contains_expected_ids(self) -> None:
        expected = {
            "rce_rgb",
            "lr_resnet18",
            "lr_mobilenetv3",
            "lr_mobilevit_xxs",
            "rce_resnet18",
            "rce_mobilenetv3",
            "rce_mobilevit_xxs",
        }
        self.assertEqual(set(MODEL_REGISTRY), expected)

    def test_backbone_dims_match_contract(self) -> None:
        self.assertEqual(BACKBONE_SPECS["resnet18"].expected_dim, 512)
        self.assertEqual(BACKBONE_SPECS["mobilenetv3"].expected_dim, 576)
        self.assertEqual(BACKBONE_SPECS["mobilevit_xxs"].expected_dim, 320)

    def test_manifests_validate(self) -> None:
        for config_path in [STAGE1_CFG, STAGE2_CFG]:
            config = load_config(config_path)
            manifest = (config_path.parent.parent / config["episodes"]).resolve()
            episodes = load_episode_manifest(manifest)
            errors = validate_manifest(episodes)
            self.assertEqual(errors, [])
            for episode in episodes:
                self.assertNotEqual(episode.train_annotation, episode.test_annotation)
                self.assertNotEqual(episode.train.mask_path, episode.test.mask_path)

    def test_guard_against_reused_train_bbox(self) -> None:
        config = load_config(STAGE1_CFG)
        manifest = (STAGE1_CFG.parent.parent / config["episodes"]).resolve()
        episodes = load_episode_manifest(manifest)
        bad = episodes[0]
        bad.test_annotation["bbox"] = list(bad.train_annotation["bbox"])
        bad.test_annotation["mask_path"] = bad.train_annotation["mask_path"]
        errors = validate_manifest([bad])
        self.assertTrue(any("reuses the train bbox" in error or "identical" in error for error in errors))

    def test_mask_to_bbox_is_deterministic(self) -> None:
        import cv2

        mask_path = PROJECT_ROOT / "rce_benchmark" / "annotations" / "masks" / "stage1_curule1_train.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(mask_to_bbox(mask), [760, 420, 1060, 720])
        self.assertEqual(mask_to_bbox(mask), [760, 420, 1060, 720])

    def test_stage1_reproducible_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(STAGE1_CFG)
            config["episodes"] = str((STAGE1_CFG.parent.parent / config["episodes"]).resolve())
            config["output_dir"] = tmpdir
            episodes = load_episode_manifest(config["episodes"])[:1]
            rows_a = run_stage1_speed(config, episodes, ["rce_rgb"])
            rows_b = run_stage1_speed(config, episodes, ["rce_rgb"])
            self.assertEqual(len(rows_a), 1)
            a = rows_a[0].to_dict()
            b = rows_b[0].to_dict()
            for volatile in ["train_ms", "infer_ms_total", "infer_ms_per_window", "windows_per_sec"]:
                a.pop(volatile)
                b.pop(volatile)
            self.assertEqual(a, b)
            self.assertTrue(Path(rows_a[0].overlay_path).exists())
            self.assertTrue(Path(rows_a[0].latent_plot_path).exists())

    def test_stage2_smoke_and_report_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(STAGE2_CFG)
            config["episodes"] = str((STAGE2_CFG.parent.parent / config["episodes"]).resolve())
            config["output_dir"] = tmpdir
            episodes = load_episode_manifest(config["episodes"])[:1]
            rows = run_stage2_middlebury(config, episodes, ["rce_rgb"])
            tasks = {row.task for row in rows}
            self.assertEqual(tasks, {"classification", "localization_transfer"})
            for row in rows:
                self.assertTrue(row.object_name)
                self.assertTrue(Path(row.overlay_path).exists())
                self.assertTrue(Path(row.heatmap_path).exists())
                self.assertTrue(Path(row.pred_mask_path).exists())
                self.assertTrue(Path(row.visual_summary_path).exists())

            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "rce_benchmark.run",
                    "--config",
                    str(STAGE2_CFG),
                ],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(proc.returncode, 0)
            results_path = PROJECT_ROOT / "rce_benchmark" / "outputs" / "stage2_smoke" / "results.json"
            self.assertTrue(results_path.exists())
            records = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertTrue(any(record["task"] == "classification" for record in records))
            self.assertTrue(all(record["object_name"] for record in records))
            self.assertTrue(all(record["overlay_path"] for record in records))

    def test_notebook_export_episode_consumed_by_runner(self) -> None:
        config = load_config(STAGE1_CFG)
        manifest = (STAGE1_CFG.parent.parent / config["episodes"]).resolve()
        source_episode = load_episode_manifest(manifest)[0]
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_path = export_episode_payload(
                Path(tmpdir) / "episode.json",
                episode_id="exported_stage1_episode",
                object_id=source_episode.object_id,
                object_name=source_episode.object_name,
                scene_group=source_episode.scene_group,
                stage=source_episode.stage,
                window_size=source_episode.window_size,
                train_variant=source_episode.train_variant,
                test_variant=source_episode.test_variant,
                train_view=source_episode.train_view,
                test_view=source_episode.test_view,
                train_annotation=source_episode.train_annotation,
                test_annotation=source_episode.test_annotation,
            )
            config["episodes"] = str(episode_path)
            config["output_dir"] = tmpdir
            episodes = load_episode_manifest(episode_path)
            rows = run_stage1_speed(config, episodes, ["rce_rgb"])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].episode_id, "exported_stage1_episode")

    def test_naming_and_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = load_config(STAGE2_CFG)
            config["episodes"] = str((STAGE2_CFG.parent.parent / config["episodes"]).resolve())
            config["output_dir"] = tmpdir
            rows = run_stage2_middlebury(config, load_episode_manifest(config["episodes"])[:1], ["rce_rgb"])
            for row in rows:
                self.assertTrue(row.object_name)
                self.assertTrue(row.train_mask_path)
                self.assertTrue(row.test_mask_path)
                self.assertTrue(row.overlay_path)
                self.assertTrue(row.heatmap_path)
                self.assertTrue(row.latent_plot_path)
                self.assertTrue(row.prototype_gallery_path)

    def test_rce_support_counts_sum_to_train_samples(self) -> None:
        x_train = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.2, 1.1, 1.1],
                [200.0, 200.0, 200.0],
            ]
        )
        y_train = np.array(["background", "object", "object", "background"])
        model = RCE(R_max=20.0, default_label="background").fit(x_train, y_train)
        self.assertEqual(int(model.support_counts_.sum()), len(x_train))
        self.assertEqual(len(model.support_counts_), len(model.centers_))

    def test_naming_guard(self) -> None:
        readme = (PROJECT_ROOT / "rce_benchmark" / "README.md").read_text(encoding="utf-8")
        self.assertIn("Only `RCE/rce.py` is treated as the official RCE implementation.", readme)
        self.assertNotIn("repo handcrafted pipeline as RCE", readme.lower())


if __name__ == "__main__":
    unittest.main()
