"""Public dataset loading helpers for CodeDNA."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PUBLIC_DATASETS = [
    "sahil2801/CodeAlpaca-20k",
    "theblackcat102/evol-codealpaca-v1",
    "nampdn-ai/tiny-codes",
]


def load_public_datasets(language: str = "python", max_samples: int = 5000) -> list[dict]:
    """Load normalized public prompt and completion pairs from Hugging Face."""

    cache_path = Path(".codedna") / "dataset" / "public_pairs.json"
    if cache_path.exists():
        cached_pairs = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached_pairs[:max_samples]

    try:
        from datasets import load_dataset
    except ImportError:
        return []

    combined: list[dict] = []
    seen: set[tuple[str, str]] = set()
    per_dataset_limit = max(1, max_samples // len(PUBLIC_DATASETS))

    for dataset_name in PUBLIC_DATASETS:
        try:
            dataset = load_dataset(dataset_name, split="train")
        except Exception:
            continue

        sample_count = min(len(dataset), per_dataset_limit)
        sampled_rows = dataset.shuffle(seed=42).select(range(sample_count))
        for row in sampled_rows:
            normalized = _normalize_row(dict(row), language=language)
            if normalized is None:
                continue
            key = (normalized["prompt"], normalized["completion"])
            if key in seen:
                continue
            seen.add(key)
            combined.append(normalized)

    if combined:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    return combined


def _normalize_row(row: dict[str, Any], language: str) -> dict | None:
    """Normalize heterogeneous public dataset rows into a common schema."""

    row_language = str(row.get("language", language)).lower()
    if language.lower() not in row_language:
        return None

    prompt = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    completion = row.get("output") or row.get("completion") or row.get("response") or row.get("answer")
    if not prompt or not completion:
        return None

    return {
        "prompt": str(prompt).strip(),
        "completion": str(completion).strip(),
        "source": "public",
    }
