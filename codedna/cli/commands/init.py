"""CLI command stub for `codedna init`."""

from __future__ import annotations

from pathlib import Path

import typer


def init_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Initialize a CodeDNA project."""

    raise typer.Exit(code=0)
