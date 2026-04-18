from datasets import load_dataset
import json
from pathlib import Path

PUBLIC_DATASETS = [
    "sahil2801/CodeAlpaca-20k",
    "theblackcat102/evol-codealpaca-v1",
    "nampdn-ai/tiny-codes",
]

def normalize_row(row, language="python"):
    row_language = str(row.get("language", language)).lower()
    if language.lower() not in row_language:
        return None
    prompt = row.get("instruction") or row.get("prompt") or row.get("question") or row.get("input")
    completion = row.get("output") or row.get("completion") or row.get("response") or row.get("answer")
    if not prompt or not completion:
        return None
    return {
        "prompt": str(prompt).strip(),
        "completion": str(completion).strip(),
        "source": "public",
    }

max_samples = 5000
per_dataset_limit = max(1, max_samples // len(PUBLIC_DATASETS))
combined = []
seen = set()

for dataset_name in PUBLIC_DATASETS:
    ds = load_dataset(dataset_name, split="train")
    sample_count = min(len(ds), per_dataset_limit)
    for row in ds.shuffle(seed=42).select(range(sample_count)):
        normalized = normalize_row(dict(row), language="python")
        if normalized is None:
            continue
        key = (normalized["prompt"], normalized["completion"])
        if key in seen:
            continue
        seen.add(key)
        combined.append(normalized)
    print(f"{dataset_name}: done")

out = Path(".codedna/dataset/public_pairs.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(combined, indent=2), encoding="utf-8")
print(f"Saved {len(combined)} public pairs to {out}")

