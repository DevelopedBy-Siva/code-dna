"""CLI command implementations for `codedna dataset` operations."""

from __future__ import annotations

import json
import random
from pathlib import Path

import typer
from rich.console import Console

from codedna.analyzer.ast_extractor import extract_pairs_from_file
from codedna.analyzer.pattern_detector import detect_patterns
from codedna.analyzer.quality_scorer import score_all
from codedna.analyzer.repo_scanner import scan_repo
from codedna.dataset.cleaner import clean_pairs
from codedna.dataset.formatter import format_to_jsonl
from codedna.dataset.mixer import mix_datasets
from codedna.dataset.public_loader import load_public_datasets

console = Console()


def preview_command(repo: Path = typer.Option(Path("."), "--repo")) -> None:
    """Preview random private dataset examples."""

    repo_root = repo.resolve()
    raw_pairs = _load_or_build_private_pairs(repo_root)
    sample_size = min(5, len(raw_pairs))
    if sample_size == 0:
        console.print("No raw pairs available.")
        return

    for index, pair in enumerate(random.Random(42).sample(raw_pairs, sample_size), start=1):
        console.rule(f"Pair {index}")
        console.print(pair["prompt"])
        console.print(pair["completion"])


def clean_command(
    repo: Path = typer.Option(Path("."), "--repo"),
    min_score: int = typer.Option(60, "--min-score"),
) -> None:
    """Clean private dataset examples and save cleaned pairs."""

    repo_root = repo.resolve()
    raw_pairs = _load_or_build_private_pairs(repo_root)
    cleaned_pairs = clean_pairs(raw_pairs, min_score=min_score)

    cleaned_path = repo_root / ".codedna" / "dataset" / "cleaned_pairs.json"
    cleaned_path.write_text(json.dumps(cleaned_pairs, indent=2), encoding="utf-8")
    console.print(f"Saved {len(cleaned_pairs)} cleaned pairs to {cleaned_path}")


def export_command(
    repo: Path = typer.Option(Path("."), "--repo"),
    language: str = typer.Option("python", "--language"),
    max_public_samples: int = typer.Option(5000, "--max-public-samples"),
) -> None:
    """Export train and validation JSONL files for fine-tuning."""

    repo_root = repo.resolve()
    cleaned_pairs = _load_cleaned_or_build(repo_root)
    private_pairs = [dict(pair, source="private") for pair in cleaned_pairs]
    public_pairs = load_public_datasets(language=language, max_samples=max_public_samples)
    if not public_pairs:
        console.print("Public dataset unavailable, exporting with private data only.")

    train_pairs, val_pairs = mix_datasets(public_pairs, private_pairs, ratio=0.30)
    dataset_dir = repo_root / ".codedna" / "dataset"
    format_to_jsonl(train_pairs, dataset_dir / "train.jsonl")
    format_to_jsonl(val_pairs, dataset_dir / "val.jsonl")


def _load_or_build_private_pairs(repo_root: Path) -> list[dict]:
    """Load raw private pairs or build them from the private-repo directory."""

    raw_path = repo_root / ".codedna" / "dataset" / "raw_pairs.json"
    private_root = repo_root / "private-repo"
    if private_root.exists():
        files = scan_repo(str(private_root), {"max_file_size_bytes": 100 * 1024})
        extracted_pairs = [pair for file_path in files for pair in extract_pairs_from_file(file_path)]
        style_profile = detect_patterns(files)
        scored_pairs = score_all(extracted_pairs, style_profile)

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(scored_pairs, indent=2), encoding="utf-8")
        return scored_pairs

    if raw_path.exists():
        return json.loads(raw_path.read_text(encoding="utf-8"))

    return []


def _load_cleaned_or_build(repo_root: Path) -> list[dict]:
    """Load cleaned pairs or derive them from the raw private dataset."""

    cleaned_path = repo_root / ".codedna" / "dataset" / "cleaned_pairs.json"
    if not (repo_root / "private-repo").exists() and cleaned_path.exists():
        return json.loads(cleaned_path.read_text(encoding="utf-8"))

    raw_pairs = _load_or_build_private_pairs(repo_root)
    cleaned_pairs = clean_pairs(raw_pairs)
    cleaned_path.write_text(json.dumps(cleaned_pairs, indent=2), encoding="utf-8")
    return cleaned_pairs
