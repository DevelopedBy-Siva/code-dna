"""Checkpoint management helpers for CodeDNA."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_CHECKPOINT_ROOT = Path(".codedna") / "checkpoints"


def list_checkpoints() -> list[dict]:
    """List saved checkpoints and lightweight metadata."""

    checkpoint_root = DEFAULT_CHECKPOINT_ROOT
    if not checkpoint_root.exists():
        return []

    checkpoints: list[dict] = []
    for path in sorted(checkpoint_root.iterdir()):
        if not path.is_dir():
            continue
        metadata_path = path / "metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        checkpoints.append(
            {
                "name": path.name,
                "path": str(path),
                "has_adapter": (path / "adapter_config.json").exists(),
                "has_tokenizer": (path / "tokenizer_config.json").exists(),
                "metadata": metadata,
            }
        )
    return checkpoints


def load_checkpoint(path: str) -> tuple[object, object]:
    """Load a saved model checkpoint and tokenizer."""

    checkpoint_path = Path(path)
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def delete_checkpoint(path: str) -> None:
    """Delete a saved checkpoint directory."""

    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        shutil.rmtree(checkpoint_path)
