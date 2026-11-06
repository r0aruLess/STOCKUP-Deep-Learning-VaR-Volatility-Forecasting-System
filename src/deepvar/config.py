"""Configuration loading."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]  # repository root
RAW_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"


def load_config(path: str | Path | None = None) -> dict:
    """Load a YAML config (defaults to configs/default.yaml)."""
    path = Path(path) if path else ROOT / "configs" / "default.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
