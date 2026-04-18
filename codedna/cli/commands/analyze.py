"""CLI command stub for `codedna analyze`."""

from __future__ import annotations

from pathlib import Path

import typer


def analyze_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Analyze a repository for CodeDNA training data."""

    raise typer.Exit(code=0)
