# RecognitionBenchmark

This repository is now trimmed to the research-facing codepaths for comparing
proper RCE against the baseline architectures on Middlebury-style small-data
edge recognition tasks.

## Canonical Entry Points

- Proper RCE implementation:
  [rce.py](/Volumes/T7/RecognitionBenchmark/RCE/rce.py)
- Visual annotation and episode curation notebook:
  [rce_visual_workbench.ipynb](/Volumes/T7/RecognitionBenchmark/RCE/rce_visual_workbench.ipynb)
- Batch benchmark package:
  [run.py](/Volumes/T7/RecognitionBenchmark/rce_benchmark/run.py)
- Model weight download helper:
  [download_models.py](/Volumes/T7/RecognitionBenchmark/scripts/download_models.py)

## Repo Layout

- [RCE](/Volumes/T7/RecognitionBenchmark/RCE)
  Literature-style RCE implementation and the notebook-based visual workflow.
- [rce_benchmark](/Volumes/T7/RecognitionBenchmark/rce_benchmark)
  Segmentation-first benchmark harness, manifests, masks, reports, and visual exports.
- [training](/Volumes/T7/RecognitionBenchmark/training)
  Training scripts for the baseline models used in the comparison.
- [src](/Volumes/T7/RecognitionBenchmark/src)
  Shared baseline detectors and localization utilities.
- [scripts](/Volumes/T7/RecognitionBenchmark/scripts)
  Lightweight utility scripts such as model download/setup helpers.

## Research Notes

- Only `RCE/rce.py` is treated as proper RCE in the benchmark.
- The repo-level handcrafted feature pipeline under `src/detectors/rce/` is a
  legacy baseline and should not be reported as literature RCE.
- The practical annotation workflow is bbox-first. Each episode stores separate
  train/test annotations, and the benchmark can derive mask-like regions
  internally when a model path needs them.

## Running the Benchmark

Use the annotation notebook first to refine or lock masks, then run:

```bash
python -m rce_benchmark.run --config rce_benchmark/configs/stage1_full.yaml
python -m rce_benchmark.run --config rce_benchmark/configs/stage2_full.yaml
```

For smoke checks:

```bash
python -m unittest tests.test_rce_benchmark -v
python -m rce_benchmark.run --config rce_benchmark/configs/stage1_smoke.yaml
python -m rce_benchmark.run --config rce_benchmark/configs/stage2_smoke.yaml
```
