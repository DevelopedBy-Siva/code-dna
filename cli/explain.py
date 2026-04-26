"""Explain what a Python file or snippet does."""

import sys
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from . import model as m
from .prompts import explain_prompt
from .config import cfg

console = Console()


def run(
    file:   str = typer.Argument(..., help="Python file to explain (use '-' for stdin)"),
    tokens: int = typer.Option(cfg.max_new_tokens, "--tokens", "-t", help="Max tokens to generate"),
    temp: float = typer.Option(cfg.temperature,    "--temp",   "-T", help="Sampling temperature"),
):
    """Explain what a Python file does."""

    if file == "-":
        code = sys.stdin.read()
        source = "stdin"
    else:
        path = Path(file)
        if not path.exists():
            console.print(f"[red]❌ File not found: {file}[/red]")
            raise typer.Exit(1)
        code   = path.read_text()
        source = path.name

    if not code.strip():
        console.print("[red]❌ File is empty[/red]")
        raise typer.Exit(1)

    if len(code) > 3000:
        console.print(f"[yellow]⚠ File is long — using first 3000 chars[/yellow]")
        code = code[:3000] + "\n# ... (truncated)"

    console.print(f"[dim]Explaining {source}...[/dim]")
    prompt   = explain_prompt(code)
    response = m.generate(prompt, max_new_tokens=tokens, temperature=temp)

    console.print()
    console.print(Panel(
        Markdown(response),
        title=f"[bold blue]Explanation — {source}[/bold blue]",
        border_style="blue",
    ))
