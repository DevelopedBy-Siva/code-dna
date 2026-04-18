"""Dataset mixing helpers for CodeDNA."""

from __future__ import annotations

import random
from copy import deepcopy


def mix_datasets(public: list[dict], private: list[dict], ratio: float = 0.30) -> tuple[list, list]:
    """Mix public and private datasets into train and validation splits."""

    if not public and not private:
        print("Dataset: 0 total, 0 train, 0 val")
        print("Private: 0 pairs (0.0%)")
        return [], []

    if not public:
        mixed = list(private)
    else:
        target_private = int(len(public) * ratio / max(1 - ratio, 1e-6))
        private_pool = list(private)
        if private and len(private_pool) < target_private:
            private_pool = _upsample_private_pairs(private_pool, target_private)
        actual_private = min(target_private, len(private_pool))
        mixed = list(public) + private_pool[:actual_private]

    random.Random(42).shuffle(mixed)
    split_index = int(len(mixed) * 0.9)
    train = mixed[:split_index]
    val = mixed[split_index:]
    private_count = sum(1 for pair in mixed if str(pair.get("source", "")).startswith("private"))
    private_pct = (private_count / len(mixed) * 100) if mixed else 0.0

    print(f"Dataset: {len(mixed)} total, {len(train)} train, {len(val)} val")
    print(f"Private: {private_count} pairs ({private_pct:.1f}%)")
    return train, val


def _upsample_private_pairs(private: list[dict], target_size: int) -> list[dict]:
    """Upsample private examples by repeating them with slight prompt variation."""

    augmented = list(private)
    cursor = 0
    while len(augmented) < target_size:
        base_pair = deepcopy(private[cursor % len(private)])
        base_pair["prompt"] = f"{base_pair['prompt']}\n# Preserve the developer's style."
        base_pair["source"] = "private_upsampled"
        augmented.append(base_pair)
        cursor += 1
    return augmented
