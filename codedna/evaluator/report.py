"""Evaluation reporting stubs for CodeDNA."""

from __future__ import annotations

from pathlib import Path


def print_report(base_metrics: dict, ft_metrics: dict) -> None:
    """Print an evaluation report."""

    _ = (base_metrics, ft_metrics)


def save_report(metrics: dict, path: Path) -> None:
    """Save evaluation metrics to disk."""

    _ = (metrics, path)
