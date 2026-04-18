"""CLI command stubs for `codedna eval` operations."""

from __future__ import annotations

import typer


def run_command() -> None:
    """Run CodeDNA evaluation tasks."""

    raise typer.Exit(code=0)


def compare_command(prompt: str = typer.Option("", "--prompt")) -> None:
    """Compare base and fine-tuned models."""

    _ = prompt
    raise typer.Exit(code=0)
