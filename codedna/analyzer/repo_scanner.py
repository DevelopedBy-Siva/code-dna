"""Repository scanning stubs for CodeDNA analyzer."""

from __future__ import annotations

from pathlib import Path


def scan_repo(root_path: str, config: dict) -> list[Path]:
    """Return source files discovered in a repository."""

    _ = (root_path, config)
    return []
