"""Dataset cleaning helpers for CodeDNA."""

from __future__ import annotations


def clean_pairs(pairs: list[dict], min_score: int = 60) -> list[dict]:
    """Filter low-scoring pairs and remove exact completion duplicates."""

    filtered_pairs: list[dict] = []
    seen_completions: set[str] = set()
    removed_count = 0

    for pair in pairs:
        if int(pair.get("score", 0)) < min_score:
            removed_count += 1
            continue

        completion = str(pair.get("completion", ""))
        if completion in seen_completions:
            removed_count += 1
            continue

        seen_completions.add(completion)
        filtered_pairs.append(pair)

    print(f"Removed {removed_count} pairs during cleaning")
    return filtered_pairs
