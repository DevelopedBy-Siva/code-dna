"""CLI command implementation for `codedna analyze`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from codedna.analyzer.ast_extractor import extract_pairs_from_file
from codedna.analyzer.pattern_detector import detect_patterns
from codedna.analyzer.quality_scorer import score_all
from codedna.analyzer.repo_scanner import scan_repo

console = Console()


def analyze_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Analyze a repository and save raw training pairs."""

    repo_root = repo.resolve()
    dataset_dir = repo_root / ".codedna" / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    files = scan_repo(str(repo_root), {"max_file_size_bytes": 100 * 1024})
    extracted_pairs = [pair for file_path in files for pair in extract_pairs_from_file(file_path)]
    style_profile = detect_patterns(files)
    scored_pairs = score_all(extracted_pairs, style_profile)

    output_path = dataset_dir / "raw_pairs.json"
    output_path.write_text(json.dumps(scored_pairs, indent=2), encoding="utf-8")

    average_score = (
        round(sum(pair["score"] for pair in scored_pairs) / len(scored_pairs), 2)
        if scored_pairs
        else 0.0
    )
    console.print(
        f"Scanned {len(files)} files, extracted {len(scored_pairs)} pairs, avg quality score {average_score}"
    )
    console.print(f"Saved raw pairs to {output_path}")
