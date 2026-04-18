"""Quality scoring stubs for CodeDNA analyzer."""

from __future__ import annotations


def score_pair(pair: dict, style_profile: dict) -> int:
    """Score one prompt and completion pair."""

    _ = (pair, style_profile)
    return 0


def score_all(pairs: list[dict], style_profile: dict) -> list[dict]:
    """Score all extracted prompt and completion pairs."""

    _ = (pairs, style_profile)
    return []
