"""Configuration helpers for the CodeDNA CLI."""

from __future__ import annotations

from pathlib import Path


def get_default_config_path() -> Path:
    """Return the default location for the project config file."""

    return Path(".codedna") / "config.json"
