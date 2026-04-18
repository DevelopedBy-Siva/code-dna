"""Public dataset loading helpers for CodeDNA."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


PUBLIC_DATASETS = [
    "sahil2801/CodeAlpaca-20k",
    "theblackcat102/evol-codealpaca-v1",
    "nampdn-ai/tiny-codes",
]

PYTHON_HINTS = (
    "def ",
    "import ",
    "from ",
    "print(",
    "return ",
    "for ",
    "while ",
    "if ",
    "elif ",
    "try:",
    "except",
    "class ",
    "lambda ",
    "list(",
    "dict(",
)

NON_PYTHON_TERMS = (
    r"\bc\+\+\b",
    r"\bcpp\b",
    r"\bjavascript\b",
    r"\btypescript\b",
    r"\bjava\b",
    r"\bjulia\b",
    r"\bruby\b",
    r"\bgolang\b",
    r"\brust\b",
    r"\bphp\b",
    r"\bc#\b",
    r"\bin r\b",
    r"\br function\b",
    r"```r",
    r"\bswift\b",
    r"\bkotlin\b",
    r"\bsql\b",
    r"\bhtml\b",
    r"\bcss\b",
)

PYTHON_PROMPT_TERMS = (
    "python",
    "pandas",
    "numpy",
    "tkinter",
    "flask",
    "django",
    "matplotlib",
    "list comprehension",
    "dictionary",
    "tuple",
)

JUNK_PROMPT_PATTERNS = (
    r"^notice\b",
    r"\brole:",
    r"\bfor beginners\b",
    r"\bfor professionals\b",
)


def load_public_datasets(language: str = "python", max_samples: int = 5000) -> list[dict]:
    """Load normalized public prompt and completion pairs from Hugging Face."""

    cache_path = Path(".codedna") / "dataset" / "public_pairs.json"
    if cache_path.exists():
        cached_pairs = json.loads(cache_path.read_text(encoding="utf-8"))
        filtered_cache = [pair for pair in cached_pairs if _is_target_language_pair(pair, language)]
        if len(filtered_cache) != len(cached_pairs):
            cache_path.write_text(json.dumps(filtered_cache, indent=2), encoding="utf-8")
        return filtered_cache[:max_samples]

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

    prompt = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    completion = row.get("output") or row.get("completion") or row.get("response") or row.get("answer")
    if not prompt or not completion:
        return None

    normalized = {
        "prompt": str(prompt).strip(),
        "completion": str(completion).strip(),
        "source": "public",
    }
    row_language = str(row.get("language", "")).lower().strip()
    if row_language and language.lower() not in row_language:
        return None
    if not _is_target_language_pair(normalized, language):
        return None
    return normalized


def _is_target_language_pair(pair: dict[str, str], language: str) -> bool:
    """Return whether a normalized pair matches the target language tightly enough."""

    if language.lower() != "python":
        return True

    prompt = str(pair.get("prompt", "")).lower()
    completion = str(pair.get("completion", "")).lower()
    combined = f"{prompt}\n{completion}"

    if any(re.search(pattern, combined) for pattern in NON_PYTHON_TERMS):
        return False
    if any(re.search(pattern, prompt) for pattern in JUNK_PROMPT_PATTERNS):
        return False
    if not _looks_like_code(completion):
        return False
    if _looks_like_python(completion):
        return True
    return any(term in prompt for term in PYTHON_PROMPT_TERMS)


def _looks_like_code(text: str) -> bool:
    """Return whether text resembles source code rather than prose."""

    if "```" in text:
        return True
    indicators = [
        r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\breturn\b",
        r":\s*$",
        r"\bif\b.*:",
        r"\bfor\b.*:",
        r"\bwhile\b.*:",
        r"=\s*.+",
    ]
    return sum(bool(re.search(pattern, text, re.MULTILINE)) for pattern in indicators) >= 2


def _looks_like_python(text: str) -> bool:
    """Return whether code text is likely Python."""

    score = sum(hint in text for hint in PYTHON_HINTS)
    if re.search(r"^\s{4,}\S", text, re.MULTILINE):
        score += 1
    if re.search(r"^\s*#\s*", text, re.MULTILINE):
        score += 1
    return score >= 2
