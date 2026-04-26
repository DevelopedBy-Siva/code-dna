"""Model loading and cached text generation helpers."""

import torch
from pathlib import Path
from rich.console import Console
from .config import cfg

console = Console()

_model     = None
_tokenizer = None


def get_model_and_tokenizer():
    """Load the model once and reuse it across commands."""
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    model_path = cfg.model_path

    if not Path(model_path).exists() and not model_path.startswith("Qwen/"):
        console.print(f"[red]❌ Model not found at: {model_path}[/red]")
        console.print("[yellow]Set the correct path in ~/.pyassist/config.toml or PYASSIST_MODEL_PATH env var[/yellow]")
        raise SystemExit(1)

    console.print(f"[dim]Loading model from {model_path}...[/dim]", end="")

    try:
        from unsloth import FastLanguageModel
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = model_path,
            max_seq_length = cfg.max_seq_length,
            load_in_4bit   = cfg.load_in_4bit,
            dtype          = torch.bfloat16,
        )
        FastLanguageModel.for_inference(model)
    except ImportError:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model     = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype   = torch.bfloat16,
            device_map    = "auto",
            trust_remote_code=True,
        )
        model.eval()

    _model, _tokenizer = model, tokenizer
    console.print(" [green]✓[/green]")
    return model, tokenizer


def generate(
    prompt:         str,
    max_new_tokens: int   = 512,
    temperature:    float = 0.2,
    top_p:          float = 0.95,
) -> str:
    """Generate text from a prompt."""
    model, tokenizer = get_model_and_tokenizer()

    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            temperature    = temperature,
            top_p          = top_p,
            do_sample      = temperature > 0,
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

    return tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
