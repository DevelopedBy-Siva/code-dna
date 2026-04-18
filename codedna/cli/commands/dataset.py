"""CLI command stubs for `codedna dataset` operations."""

from __future__ import annotations

from pathlib import Path

import typer


def preview_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Preview dataset examples."""

    raise typer.Exit(code=0)


def clean_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Clean dataset examples."""

    raise typer.Exit(code=0)


def export_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Export dataset files."""

    raise typer.Exit(code=0)
