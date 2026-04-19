"""Inference helpers for CodeDNA server and chat flows."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CHECKPOINT_CANDIDATES = [
    Path("model/checkpoints/final"),
    Path(".codedna/checkpoints/final"),
    Path(".codedna/checkpoints_run2/final"),
]


def resolve_checkpoint_path(checkpoint_path: str | None = None) -> Path:
    """Resolve the active checkpoint path for serving."""

    if checkpoint_path:
        path = Path(checkpoint_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Checkpoint path not found: {path}")

    for candidate in DEFAULT_CHECKPOINT_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No checkpoint directory found in known default locations.")


def load_model(checkpoint_path: str | None = None) -> tuple[object, object]:
    """Load the fine-tuned model and tokenizer from a checkpoint path."""

    checkpoint_dir = resolve_checkpoint_path(checkpoint_path)
    adapter_config_path = checkpoint_dir / "adapter_config.json"

    if adapter_config_path.exists():
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = adapter_config["base_model_name_or_path"]
        tokenizer_source = str(checkpoint_dir) if (checkpoint_dir / "tokenizer_config.json").exists() else base_model
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
        tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
        model = PeftModel.from_pretrained(base, str(checkpoint_dir))
        return model, tokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(str(checkpoint_dir), device_map="auto")
    return model, tokenizer


def generate(
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stop: list[str] | None = None,
) -> str:
    """Generate a completion for a prompt."""

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**generation_kwargs)

    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True).strip()
    if stop:
        for marker in stop:
            if marker and marker in generated:
                generated = generated.split(marker, 1)[0]
    return generated.strip()
