"""CLI command implementations for `codedna eval` operations."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from peft import PeftModel
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from codedna.evaluator.humaneval_runner import run_humaneval
from codedna.evaluator.perplexity import compute_perplexity
from codedna.evaluator.report import print_report, save_report
from codedna.evaluator.style_scorer import score_samples

console = Console()


def run_command(
    quick: bool = typer.Option(False, "--quick"),
    adapter_path: Path = typer.Option(Path(".codedna/checkpoints/final"), "--adapter-path"),
) -> None:
    """Run CodeDNA evaluation tasks."""

    base_model = "mistralai/Mistral-7B-v0.1"
    dataset_path = Path(".codedna") / "dataset" / "val.jsonl"
    style_profile = json.loads((Path(".codedna") / "style_profile.json").read_text(encoding="utf-8"))

    if not dataset_path.exists():
        raise FileNotFoundError("Validation dataset missing. Run `codedna dataset export --repo .` first.")

    try:
        base_tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=quick)
        base_tokenizer.pad_token = base_tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=quick, device_map="auto")
    except Exception as exc:
        raise RuntimeError(f"Unable to load base model for evaluation: {exc}") from exc

    try:
        ft_tokenizer = base_tokenizer
        ft_base = AutoModelForCausalLM.from_pretrained(base_model, local_files_only=quick, device_map="auto")
        ft = PeftModel.from_pretrained(ft_base, adapter_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to load fine-tuned adapter for evaluation: {exc}") from exc

    base_metrics = {
        "perplexity": compute_perplexity(base, base_tokenizer, dataset_path),
        "style": score_samples(base, base_tokenizer, style_profile, n=10 if quick else 20),
        "humaneval": 0.0 if quick else run_humaneval(base, base_tokenizer, k=1),
    }
    ft_metrics = {
        "perplexity": compute_perplexity(ft, ft_tokenizer, dataset_path),
        "style": score_samples(ft, ft_tokenizer, style_profile, n=10 if quick else 20),
        "humaneval": 0.0 if quick else run_humaneval(ft, ft_tokenizer, k=1),
    }

    print_report(base_metrics, ft_metrics)
    save_report({"base": base_metrics, "finetuned": ft_metrics}, Path(".codedna") / "eval_report.json")


def compare_command(
    prompt: str = typer.Option("", "--prompt"),
    adapter_path: Path = typer.Option(Path(".codedna/checkpoints/final"), "--adapter-path"),
) -> None:
    """Compare base and fine-tuned models."""

    if not prompt:
        raise typer.BadParameter("Provide a prompt with --prompt.")

    base_model = "mistralai/Mistral-7B-v0.1"

    base_tokenizer = AutoTokenizer.from_pretrained(base_model)
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")

    ft_tokenizer = base_tokenizer
    ft_base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
    ft = PeftModel.from_pretrained(ft_base, adapter_path)

    base_output = _generate_text(base, base_tokenizer, prompt)
    ft_output = _generate_text(ft, ft_tokenizer, prompt)

    console.rule("Base Model")
    console.print(base_output)
    console.rule("Fine-Tuned Model")
    console.print(ft_output)


def _generate_text(model: object, tokenizer: object, prompt: str) -> str:
    """Generate text for an evaluation prompt."""

    import torch

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=192,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
