"""Runtime configuration for the CLI."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    model_path: str = os.environ.get(
        "PYASSIST_MODEL_PATH",
        "./qwen-python-finetuned/merged_model",
    )

    max_seq_length: int   = 4096
    load_in_4bit:   bool  = False
    max_new_tokens: int   = int(os.environ.get("PYASSIST_MAX_TOKENS", "512"))
    temperature:    float = float(os.environ.get("PYASSIST_TEMPERATURE", "0.2"))


cfg = Config()
