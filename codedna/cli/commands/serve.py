"""CLI command implementation for `codedna serve`."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

from codedna.server.app import create_app


def serve_command(
    port: int = typer.Option(8080, "--port"),
    host: str = typer.Option("127.0.0.1", "--host"),
    checkpoint_path: Path = typer.Option(Path("model/checkpoints/final"), "--checkpoint-path"),
) -> None:
    """Serve a CodeDNA model behind an OpenAI-compatible API."""

    app = create_app(str(checkpoint_path))
    uvicorn.run(app, host=host, port=port)
