"""Generate Python code from a natural-language instruction."""

import typer
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from . import model as m
from .prompts import generate_prompt
from .config import cfg

console = Console()


def run(
    instruction: str = typer.Argument(..., help="What Python code to generate"),
    tokens:      int = typer.Option(cfg.max_new_tokens, "--tokens", "-t", help="Max tokens to generate"),
    temp:      float = typer.Option(cfg.temperature,    "--temp",   "-T", help="Sampling temperature (0=deterministic)"),
    raw:        bool = typer.Option(False,              "--raw",    "-r", help="Print raw output without formatting"),
):
    """Generate Python code from a natural language instruction."""
    prompt   = generate_prompt(instruction)
    response = m.generate(prompt, max_new_tokens=tokens, temperature=temp)

    if raw:
        print(response)
        return

    console.print()
    console.print(Panel(
        Syntax(response, "python", theme="monokai", line_numbers=True),
        title=f"[bold green]Generated Code[/bold green]",
        subtitle=f"[dim]{instruction[:60]}{'...' if len(instruction) > 60 else ''}[/dim]",
        border_style="green",
    ))
