"""Style scoring helpers for CodeDNA."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from codedna.analyzer.pattern_detector import detect_case_style, extract_identifiers, extract_imports


def compute_style_score(generated_code: str, style_profile: dict) -> dict:
    """Compute a style score for generated code."""

    breakdown: dict[str, int] = {}

    naming = detect_case_style(extract_identifiers(generated_code))
    breakdown["naming"] = 30 if naming == style_profile.get("naming") else 0

    avg_length = max(float(style_profile.get("avg_function_length", 1) or 1), 1.0)
    line_count = max(1, len([line for line in generated_code.splitlines() if line.strip()]))
    deviation = abs(line_count - avg_length) / avg_length
    breakdown["length"] = int(25 * max(0.0, 1 - deviation))

    overlap = set(extract_imports(generated_code)).intersection(style_profile.get("top_libraries", []))
    breakdown["libraries"] = min(25, len(overlap) * 3)

    has_try = "try:" in generated_code
    expected_try = float(style_profile.get("uses_try_except", 0.0)) > 0.5
    breakdown["error_handling"] = 20 if has_try == expected_try else 0

    return {"score": sum(breakdown.values()), "breakdown": breakdown}


def score_samples(model: object, tokenizer: object, style_profile: dict, n: int = 20) -> float:
    """Score a sample of generated completions."""

    rows = _load_style_eval_rows()
    if not rows:
        return 0.0

    sample_rows = rows[: min(n, len(rows))]
    scores: list[int] = []
    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    for row in sample_rows:
        prompt = row["prompt"]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model.generate(**_deterministic_generation_kwargs(inputs, tokenizer))
        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        scores.append(compute_style_score(generated, style_profile)["score"])

    return round(sum(scores) / max(len(scores), 1), 2)


def _load_style_eval_rows() -> list[dict]:
    """Load style-evaluation prompts, preferring private cleaned pairs."""

    cleaned_path = Path(".codedna") / "dataset" / "cleaned_pairs.json"
    if cleaned_path.exists():
        cleaned_rows = json.loads(cleaned_path.read_text(encoding="utf-8"))
        return [{"prompt": row["prompt"], "completion": row.get("completion", "")} for row in cleaned_rows]

    dataset_path = Path(".codedna") / "dataset" / "val.jsonl"
    rows = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    prompts: list[dict] = []
    for row in rows:
        text = row["text"]
        if "<|assistant|>" not in text:
            continue
        prompt = text.split("<|assistant|>")[0] + "<|assistant|>"
        prompts.append({"prompt": prompt, "completion": ""})
    return prompts


def _deterministic_generation_kwargs(inputs: dict[str, torch.Tensor], tokenizer: object) -> dict:
    """Return deterministic generation kwargs without temperature warnings."""

    return {
        **inputs,
        "max_new_tokens": 160,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
