"""Checkpoint management stubs for CodeDNA."""

from __future__ import annotations


def list_checkpoints() -> list[dict]:
    """List available training checkpoints."""

    return []


def load_checkpoint(path: str) -> tuple[object, object]:
    """Load a training checkpoint."""

    _ = path
    return (object(), object())


def delete_checkpoint(path: str) -> None:
    """Delete a training checkpoint."""

    _ = path
