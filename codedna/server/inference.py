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
        tokenizer_source = (
            str(checkpoint_dir)
            if (checkpoint_dir / "tokenizer_config.json").exists()
            else base_model
        )
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
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stop: list[str] | None = None,
) -> str:
    # FIX: getattr(model, "device") fails silently for PeftModel / device_map="auto"
    # models — they have no single .device. Always derive from parameters instead.
    device = next(model.parameters()).device

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # FIX: encode each stop marker fully and collect all resulting token IDs rather
    # than grabbing only the first token of each string.  For multi-token special
    # tokens (e.g. "<|user|>" may encode to [1, 28766, 1404, 28766, 2]) this is the
    # difference between the stop working and being silently ignored.
    stop_token_ids: list[int] = [tokenizer.eos_token_id]
    for token in ["<|user|>", "<|system|>", "</s>"]:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        stop_token_ids.extend(encoded)
    stop_token_ids = list(set(stop_token_ids))  # deduplicate

    generation_kwargs: dict = {
        **inputs,
        "max_new_tokens": max_tokens,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": stop_token_ids,
        "do_sample": temperature > 0,
        # FIX: 1.15 penalises normal Python patterns (self, return, if, indentation).
        # 1.05 suppresses genuine repetition without mangling code structure.
        "repetition_penalty": 1.05,
        "top_p": 0.95,
    }
    if temperature > 0:
        generation_kwargs["temperature"] = temperature

    # FIX: torch.inference_mode() is strictly faster than torch.no_grad() — it
    # disables autograd tracking completely, not just gradient computation.
    with torch.inference_mode():
        outputs = model.generate(**generation_kwargs)

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    ).strip()

    # Trim at stop strings if provided
    if stop:
        for marker in stop:
            if marker and marker in generated:
                generated = generated.split(marker, 1)[0]

    return generated.strip()