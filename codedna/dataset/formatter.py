"""Dataset formatting helpers for CodeDNA."""

from __future__ import annotations

import json
from pathlib import Path


CHAT_TEMPLATE = (
    "<|system|>You are a coding assistant that writes code in the style of this developer.</s>\n"
    "<|user|>{prompt}</s>\n"
    "<|assistant|>{completion}</s>"
)


def format_to_jsonl(pairs: list[dict], output_path: Path) -> None:
    """Format dataset pairs into chat-style JSONL output."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for pair in pairs:
        text = CHAT_TEMPLATE.format(
            prompt=pair.get("prompt", ""),
            completion=pair.get("completion", ""),
        )
        lines.append(json.dumps({"text": text}, ensure_ascii=False))

    content = "\n".join(lines)
    if content:
        content = f"{content}\n"
    output_path.write_text(content, encoding="utf-8")
    print(f"Wrote {len(pairs)} examples to {output_path}")
