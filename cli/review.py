"""Review Python code and optionally generate a fixed version."""

import sys
import typer
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.syntax import Syntax
from . import model as m
from .prompts import review_prompt, fix_prompt
from .config import cfg

console = Console()


def run(
    file:   str  = typer.Argument(..., help="Python file to review (use '-' for stdin)"),
    tokens: int  = typer.Option(cfg.max_new_tokens, "--tokens", "-t", help="Max tokens"),
    temp: float  = typer.Option(cfg.temperature,    "--temp",   "-T", help="Temperature"),
    fix:   bool  = typer.Option(False, "--fix", "-f", help="Also generate a fixed version"),
):
    """Review Python code for bugs, issues, and improvements."""

    if file == "-":
        code   = sys.stdin.read()
        source = "stdin"
    else:
        path = Path(file)
        if not path.exists():
            console.print(f"[red]File not found: {file}[/red]")
            raise typer.Exit(1)
        code   = path.read_text()
        source = path.name

    if not code.strip():
        console.print("[red]File is empty[/red]")
        raise typer.Exit(1)

    if len(code) > 3000:
        console.print(f"[yellow]Warning: file is long; using first 3000 chars[/yellow]")
        code = code[:3000] + "\n# ... (truncated)"

    console.print(f"[dim]Reviewing {source}...[/dim]")
    review_response = m.generate(review_prompt(code), max_new_tokens=tokens, temperature=temp)

    console.print()
    console.print(Panel(
        Markdown(review_response),
        title=f"[bold yellow]Code Review - {source}[/bold yellow]",
        border_style="yellow",
    ))

    if fix:
        console.print(f"\n[dim]Generating fixed version...[/dim]")
        fix_response = m.generate(
            fix_prompt(code, "issues found during review"),
            max_new_tokens=tokens,
            temperature=temp,
        )
        console.print()
        console.print(Panel(
            Syntax(fix_response, "python", theme="monokai", line_numbers=True),
            title="[bold green]Fixed Version[/bold green]",
            border_style="green",
        ))
