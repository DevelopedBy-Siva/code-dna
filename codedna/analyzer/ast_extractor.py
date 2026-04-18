"""AST extraction stubs for CodeDNA analyzer."""

from __future__ import annotations

from pathlib import Path


def format_prompt(signature: str, docstring: str | None) -> str:
    """Format a prompt for a training example."""

    _ = docstring
    return signature


def extract_pairs_from_file(filepath: Path) -> list[dict]:
    """Extract prompt and completion pairs from a source file."""

    _ = filepath
    return []
