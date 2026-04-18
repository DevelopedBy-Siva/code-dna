"""Perplexity evaluation helpers for CodeDNA."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def compute_perplexity(model: object, tokenizer: object, dataset_path: Path) -> float:
    """Compute perplexity for a validation dataset."""

    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return float("inf")

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    total_loss = 0.0
    total_tokens = 0

    for row in rows:
        tokens = tokenizer(row["text"], return_tensors="pt", truncation=True, max_length=2048)
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.no_grad():
            outputs = model(**tokens, labels=tokens["input_ids"])
        token_count = int(tokens["input_ids"].numel())
        total_loss += float(outputs.loss.item()) * token_count
        total_tokens += token_count

    return round(math.exp(total_loss / max(total_tokens, 1)), 2)
