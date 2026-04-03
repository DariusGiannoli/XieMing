"""CLI entrypoint for the RCE benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from rce_benchmark.config import load_config
from rce_benchmark.datasets import load_episode_manifest, validate_manifest
from rce_benchmark.reports import write_reports
from rce_benchmark.tasks import run_stage1_speed, run_stage2_middlebury


def _resolve_manifest(config: dict) -> Path:
    manifest_path = Path(config["episodes"])
    if manifest_path.is_absolute():
        return manifest_path
    return Path(config["_config_path"]).resolve().parent.parent / manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RCE benchmark.")
    parser.add_argument("--config", required=True, help="Path to a benchmark YAML config.")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = _resolve_manifest(config)
    output_dir = Path(config["output_dir"])
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    config["output_dir"] = str(output_dir)
    episodes = load_episode_manifest(manifest_path)
    errors = validate_manifest(episodes)
    if errors:
        joined = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Manifest validation failed:\n{joined}")

    stage = str(config["stage"])
    model_ids = list(config["model_ids"])
    if stage == "stage1":
        rows = run_stage1_speed(config=config, episodes=episodes, model_ids=model_ids)
    elif stage == "stage2":
        rows = run_stage2_middlebury(config=config, episodes=episodes, model_ids=model_ids)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    write_reports(rows=rows, output_dir=output_dir)


if __name__ == "__main__":
    main()
