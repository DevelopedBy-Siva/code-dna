"""Inference stubs for CodeDNA server."""

from __future__ import annotations


def load_model(checkpoint_path: str) -> tuple[object, object]:
    """Load a model and tokenizer from a checkpoint path."""

    _ = checkpoint_path
    return (object(), object())


def generate(
    model: object,
    tokenizer: object,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.2,
    stop: list[str] | None = None,
) -> str:
    """Generate a completion for a prompt."""

    _ = (model, tokenizer, max_tokens, temperature, stop)
    return prompt
