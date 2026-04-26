"""Interactive chat mode for the local Python assistant."""

import typer
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown
from . import model as m
from .prompts import generate_prompt
from .config import cfg

console = Console()


def is_code_response(text: str) -> bool:
    """Heuristic: if response contains Python keywords, render as code."""
    code_markers = ["def ", "class ", "import ", "return ", "    "]
    return sum(1 for marker in code_markers if marker in text) >= 2


def run(
    temp: float = typer.Option(cfg.temperature, "--temp", "-T", help="Sampling temperature"),
    tokens: int = typer.Option(cfg.max_new_tokens, "--tokens", "-t", help="Max tokens per response"),
):
    """Start an interactive chat session with the Python assistant."""

    console.print()
    console.print(Panel(
        "[bold green]🐍 PyAssist — Python Code Chat[/bold green]\n"
        "[dim]Commands: /clear  /save  /quit[/dim]",
        border_style="green",
    ))
    console.print()

    last_response = ""
    turn          = 0

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]you[/bold cyan]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "/q"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            turn = 0
            console.clear()
            console.print("[dim]Conversation cleared[/dim]")
            continue

        if user_input.lower() == "/save":
            if not last_response:
                console.print("[yellow]Nothing to save yet[/yellow]")
                continue
            fname = f"pyassist_{datetime.now().strftime('%Y%m%d_%H%M%S')}.py"
            Path(fname).write_text(last_response)
            console.print(f"[green]Saved to {fname}[/green]")
            continue

        prompt   = generate_prompt(user_input)
        response = m.generate(prompt, max_new_tokens=tokens, temperature=temp)
        last_response = response
        turn += 1

        console.print()
        if is_code_response(response):
            console.print(Panel(
                Syntax(response, "python", theme="monokai", line_numbers=False),
                title="[bold green]assistant[/bold green]",
                border_style="dim",
            ))
        else:
            console.print(Panel(
                Markdown(response),
                title="[bold green]assistant[/bold green]",
                border_style="dim",
            ))
        console.print()
