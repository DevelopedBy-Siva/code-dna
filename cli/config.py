"""Runtime configuration for the CLI."""

import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    model_path: str = os.environ.get(
        "PYASSIST_MODEL_PATH",
        str(BASE_DIR / "model" / "qwen-python-finetuned" / "merged_model"),
    )

    max_seq_length: int   = 4096
    load_in_4bit:   bool  = False
    max_new_tokens: int   = int(os.environ.get("PYASSIST_MAX_TOKENS", "512"))
    temperature:    float = float(os.environ.get("PYASSIST_TEMPERATURE", "0.2"))


cfg = Config()
