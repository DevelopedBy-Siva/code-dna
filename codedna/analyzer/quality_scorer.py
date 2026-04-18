"""Quality scoring helpers for CodeDNA analyzer."""

from __future__ import annotations

from difflib import SequenceMatcher

from codedna.analyzer.pattern_detector import detect_case_style, extract_identifiers, extract_imports


def score_pair(pair: dict, style_profile: dict) -> int:
    """Score one prompt and completion pair from 0 to 100."""

    score = 100
    completion = str(pair.get("completion", ""))
    line_count = len([line for line in completion.splitlines() if line.strip()])

    if line_count < 5:
        score -= 20
    if not pair.get("docstring"):
        score -= 15

    naming_style = detect_case_style(extract_identifiers(completion))
    if naming_style != style_profile.get("naming"):
        score -= 20

    overlap = set(extract_imports(completion)).intersection(style_profile.get("top_libraries", []))
    score += min(len(overlap) * 3, 15)

    return max(0, min(score, 100))


def score_all(pairs: list[dict], style_profile: dict) -> list[dict]:
    """Score all extracted prompt and completion pairs and filter near-duplicates."""

    scored_pairs: list[dict] = []
    existing_completions: list[str] = []

    for pair in pairs:
        completion = str(pair.get("completion", ""))
        similarity = max(
            (SequenceMatcher(a=completion, b=existing).ratio() for existing in existing_completions),
            default=0.0,
        )
        score = 0 if similarity > 0.85 else score_pair(pair, style_profile)

        enriched_pair = dict(pair)
        enriched_pair["score"] = score
        if score == 0:
            continue

        existing_completions.append(completion)
        scored_pairs.append(enriched_pair)

    return scored_pairs
