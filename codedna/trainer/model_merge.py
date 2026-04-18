"""Model merge helpers for CodeDNA."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM


def merge_adapter(checkpoint_path: str, output_path: str, base_model: str) -> None:
    """Merge a LoRA adapter into the base model and save the merged weights."""

    checkpoint_dir = Path(checkpoint_path)
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
    model = PeftModel.from_pretrained(base, checkpoint_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(output_dir)
