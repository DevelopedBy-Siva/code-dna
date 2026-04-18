"""Style scoring stubs for CodeDNA."""

from __future__ import annotations


def compute_style_score(generated_code: str, style_profile: dict) -> dict:
    """Compute a style score for generated code."""

    _ = (generated_code, style_profile)
    return {"score": 0, "breakdown": {}}


def score_samples(model: object, tokenizer: object, style_profile: dict, n: int = 20) -> float:
    """Score a sample of generated completions."""

    _ = (model, tokenizer, style_profile, n)
    return 0.0
