"""Repository scanning helpers for the CodeDNA analyzer."""

from __future__ import annotations

from pathlib import Path


def scan_repo(root_path: str, config: dict) -> list[Path]:
    """Walk a repository recursively and return supported source files."""

    root = Path(root_path).resolve()
    ignore_dirs = {
        "node_modules",
        ".git",
        "dist",
        "build",
        "__pycache__",
        ".venv",
    }
    configured_extensions = config.get("supported_extensions", [".py"])
    supported_extensions = {str(extension) for extension in configured_extensions}
    max_size_bytes = int(config.get("max_file_size_bytes", 100 * 1024))

    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_dir():
            continue
        if any(part in ignore_dirs for part in path.parts):
            continue
        if path.suffix not in supported_extensions:
            continue
        if path.stat().st_size > max_size_bytes:
            continue
        files.append(path)

    return sorted(files)
