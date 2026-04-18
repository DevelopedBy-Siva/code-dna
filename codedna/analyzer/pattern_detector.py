"""Pattern detection helpers for CodeDNA analyzer."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean

from codedna.analyzer.ast_extractor import extract_pairs_from_file


IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([A-Za-z0-9_\.]+)\s+import|import\s+([A-Za-z0-9_\.]+)|use\s+([A-Za-z0-9_:]+)|require\((['\"])([^'\"]+)\4\))",
    re.MULTILINE,
)


def detect_patterns(files: list[Path]) -> dict:
    """Detect repository-wide coding patterns and save the style profile."""

    repo_root = _infer_repo_root(files)
    function_pairs = [pair for path in files for pair in extract_pairs_from_file(path)]

    identifiers: list[str] = []
    imports: list[str] = []
    function_lengths: list[int] = []
    docstring_count = 0
    try_count = 0

    for path in files:
        source = path.read_text(encoding="utf-8", errors="ignore")
        identifiers.extend(extract_identifiers(source))
        imports.extend(extract_imports(source))

    for pair in function_pairs:
        completion = str(pair.get("completion", ""))
        function_lengths.append(len([line for line in completion.splitlines() if line.strip()]))
        docstring_count += int(bool(pair.get("docstring")))
        try_count += int("try:" in completion or "try {" in completion)

    total_functions = max(len(function_pairs), 1)
    profile = {
        "naming": detect_case_style(identifiers),
        "avg_function_length": round(mean(function_lengths), 2) if function_lengths else 0.0,
        "p90_function_length": _percentile(function_lengths, 90),
        "top_libraries": [name for name, _ in Counter(imports).most_common(20)],
        "docstring_rate": round(docstring_count / total_functions, 3),
        "uses_try_except": round(try_count / total_functions, 3),
    }

    output_path = repo_root / ".codedna" / "style_profile.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return profile


def extract_identifiers(text: str) -> list[str]:
    """Extract identifier tokens from a source string."""

    return [match.group(0) for match in IDENTIFIER_RE.finditer(text)]


def extract_imports(text: str) -> list[str]:
    """Extract imported library names from a source string."""

    imports: list[str] = []
    for left, right, third, _, fifth in IMPORT_RE.findall(text):
        name = left or right or third or fifth
        if name:
            imports.append(name)
    return imports


def detect_case_style(identifiers: list[str]) -> str:
    """Detect the dominant identifier naming convention."""

    if not identifiers:
        return "snake_case"

    snake_case = sum("_" in identifier and identifier.lower() == identifier for identifier in identifiers)
    camel_case = sum(
        "_" not in identifier
        and identifier[:1].islower()
        and any(character.isupper() for character in identifier[1:])
        for identifier in identifiers
    )
    pascal_case = sum(
        "_" not in identifier and identifier[:1].isupper() for identifier in identifiers
    )
    scores = {
        "snake_case": snake_case,
        "camelCase": camel_case,
        "PascalCase": pascal_case,
    }
    return max(scores, key=scores.get)


def _percentile(values: list[int], percentile: int) -> int:
    """Compute a rough percentile for a list of integer values."""

    if not values:
        return 0
    ordered = sorted(values)
    index = round((percentile / 100) * (len(ordered) - 1))
    return ordered[index]


def _infer_repo_root(files: list[Path]) -> Path:
    """Infer the repository root from a list of scanned files."""

    if not files:
        return Path(".").resolve()
    return Path(os.path.commonpath([str(path.parent) for path in files]))
