#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "CodeDNA A100 training bootstrap"
echo "Repo: $REPO_ROOT"

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-train.txt

echo
echo "Checking CUDA visibility..."
python3 - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"device_{i}:", torch.cuda.get_device_name(i))
PY

echo
echo "Validating training setup..."
python3 -m codedna.cli.main train --dry-run

echo
echo "Starting real training..."
python3 -m codedna.cli.main train --model mistralai/Mistral-7B-v0.1
