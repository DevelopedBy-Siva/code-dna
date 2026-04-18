"""Dataset formatting stubs for CodeDNA."""

from __future__ import annotations

from pathlib import Path


def format_to_jsonl(pairs: list[dict], output_path: Path) -> None:
    """Format dataset pairs into JSONL output."""

    _ = (pairs, output_path)
