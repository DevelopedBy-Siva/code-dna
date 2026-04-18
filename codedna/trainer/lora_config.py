"""LoRA configuration stubs for CodeDNA."""

from __future__ import annotations


def get_lora_config() -> object:
    """Return the LoRA configuration for training."""

    return {}


def get_qlora_quantization() -> object:
    """Return the QLoRA quantization configuration."""

    return {}


def get_training_args(output_dir: str) -> object:
    """Return training arguments for CodeDNA."""

    _ = output_dir
    return {}
