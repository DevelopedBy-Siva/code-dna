"""CLI command implementation for `codedna train`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from codedna.trainer.checkpoints import list_checkpoints
from codedna.trainer.training_loop import run_training

console = Console()


def train_command(
    model: str = typer.Option("mistralai/Mistral-7B-v0.1", "--model"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    list_checkpoints_flag: bool = typer.Option(False, "--list-checkpoints"),
    output_dir: Path = typer.Option(Path(".codedna/checkpoints"), "--output-dir"),
) -> None:
    """Train a CodeDNA model or validate the setup."""

    if list_checkpoints_flag:
        checkpoints = list_checkpoints()
        table = Table(title="CodeDNA Checkpoints")
        table.add_column("Name")
        table.add_column("Path")
        table.add_column("Adapter")
        table.add_column("Tokenizer")
        for checkpoint in checkpoints:
            table.add_row(
                checkpoint["name"],
                checkpoint["path"],
                str(checkpoint["has_adapter"]),
                str(checkpoint["has_tokenizer"]),
            )
        console.print(table)
        return

    run_training(base_model=model, output_dir=str(output_dir), dry_run=dry_run)
