"""Top-level CLI entry point for PyAssist."""

import typer
from . import generate, explain, review, chat

cli = typer.Typer(
    name="pyassist",
    help="Local Python code assistant for generation, explanation, review, and chat.",
    add_completion=False,
)

cli.command("generate")(generate.run)
cli.command("explain")(explain.run)
cli.command("review")(review.run)
cli.command("chat")(chat.run)

if __name__ == "__main__":
    cli()
