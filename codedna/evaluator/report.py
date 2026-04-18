"""Evaluation reporting helpers for CodeDNA."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table


def print_report(base_metrics: dict, ft_metrics: dict) -> None:
    """Print an evaluation report."""

    table = Table(title="CodeDNA Evaluation Report")
    table.add_column("Metric")
    table.add_column("Base Model")
    table.add_column("Fine-Tuned")
    table.add_column("Delta")

    for label, key, lower_is_better in (
        ("Perplexity", "perplexity", True),
        ("HumanEval pass@1", "humaneval", False),
        ("Style Score", "style", False),
    ):
        base_value = float(base_metrics.get(key, 0.0))
        ft_value = float(ft_metrics.get(key, 0.0))
        delta = ft_value - base_value
        good = delta < 0 if lower_is_better else delta > 0
        arrow = "↑" if good else "↓"
        color = "green" if good else "red"
        table.add_row(label, f"{base_value:.2f}", f"{ft_value:.2f}", f"[{color}]{arrow} {delta:+.2f}[/{color}]")

    Console().print(table)


def save_report(metrics: dict, path: Path) -> None:
    """Save evaluation metrics to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
