"""Typer application entry point for the CodeDNA scaffold."""

from __future__ import annotations

import typer

from codedna.cli.commands import analyze, chat, dataset, eval, init, serve, train

app = typer.Typer(help="CodeDNA command-line interface.")
dataset_app = typer.Typer(help="Dataset operations.")
eval_app = typer.Typer(help="Evaluation operations.")

app.command("init")(init.init_command)
app.command("analyze")(analyze.analyze_command)
app.command("train")(train.train_command)
app.command("serve")(serve.serve_command)
app.command("chat")(chat.chat_command)

dataset_app.command("preview")(dataset.preview_command)
dataset_app.command("clean")(dataset.clean_command)
dataset_app.command("export")(dataset.export_command)
app.add_typer(dataset_app, name="dataset")

eval_app.command("run")(eval.run_command)
eval_app.command("compare")(eval.compare_command)
app.add_typer(eval_app, name="eval")


if __name__ == "__main__":
    app()
