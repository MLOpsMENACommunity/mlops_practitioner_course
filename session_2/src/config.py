"""Tiny config loader shared by the DVC pipeline stages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

#: Default location of the pipeline config, relative to the project root.
DEFAULT_CONFIG_PATH = "config/config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and parse the YAML pipeline config into a dict."""
    with open(path) as f:
        return yaml.safe_load(f)
