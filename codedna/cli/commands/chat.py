"""CLI command implementation for `codedna chat`."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from codedna.server.inference import generate, load_model

console = Console()


def chat_command(
    checkpoint_path: Path = typer.Option(Path("model/checkpoints/final"), "--checkpoint-path"),
    max_tokens: int = typer.Option(256, "--max-tokens"),
    temperature: float = typer.Option(0.2, "--temperature"),
) -> None:
    """Chat with the fine-tuned CodeDNA model from the terminal."""

    model, tokenizer = load_model(str(checkpoint_path))
    console.print("[bold green]CodeDNA chat[/bold green] - type `exit` to quit")

    while True:
        prompt = typer.prompt("You")
        if prompt.strip().lower() in {"exit", "quit"}:
            break
        completion = generate(
            model,
            tokenizer,
            f"user: {prompt}\nassistant:",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        console.print(completion)
