"""Configuration helpers for the RCE benchmark."""

from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_KEYS = {
    "stage",
    "dataset",
    "episodes",
    "model_ids",
    "training",
    "inference",
    "seeds",
    "output_dir",
}


def load_config(path: str | Path) -> dict:
    """Load and validate a YAML config."""
    cfg_path = Path(path)
    with cfg_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a mapping: {cfg_path}")
    missing = sorted(REQUIRED_KEYS - set(config))
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")
    config["_config_path"] = str(cfg_path.resolve())
    return config
