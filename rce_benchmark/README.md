# RCE Benchmark

This package provides a scriptable benchmark harness for proper literature-style
Restricted Coulomb Energy (RCE) benchmarking inside this repo.

## Scope

- Only `RCE/rce.py` is treated as the official RCE implementation.
- The repo's handcrafted grayscale feature pipeline is excluded from `RCE`
  benchmark tables and should not be reported as RCE.
- Benchmark timing is local-machine first. The package is structured so later
  device-runtime adapters can be added without redesigning the dataset and
  reporting layers.
- The practical research workflow is bbox-first. Episodes still support mask
  annotations, but train/test boxes can be used directly and any mask-like
  regions are derived internally when needed.

## Stages

- `stage1`: stereo-style train-left / test-right speed benchmark
- `stage2`: Middlebury-first classification benchmark plus localization transfer

## CLI

```bash
python -m rce_benchmark.run --config rce_benchmark/configs/stage1_smoke.yaml
```

## Visual Workflow

- Notebook-first visual exploration now lives in
  `RCE/rce_visual_workbench.ipynb`.
- The notebook and CLI share the same episode schema, mask files, and batch
  runner, so notebook-exported episodes can be consumed unchanged by the
  benchmark.
- The current notebook workflow is rectangle-based rather than freehand
  segmentation, which keeps annotation more consistent across views and scenes.
- Each result row now includes artifact paths for:
  - train/test overview overlays
  - confidence heatmaps
  - predicted masks
  - latent-space plots
  - prototype galleries

## Annotation Protocol

- Each episode carries explicit `object_id` and `object_name`.
- Train and test annotations are stored separately.
- The recommended protocol is a tight bbox around the visible target object in
  each image.
- Starter masks in `rce_benchmark/annotations/masks/` remain available for
  compatibility, but the bbox-first notebook flow is the preferred curation
  path.

## Notes

- Backbone benchmarks require local backbone weight files in `models/`.
- The current repo only ships trained heads, so raw-RCE smoke runs work
  immediately while backbone runs will fail loudly until those weights are
  downloaded.
