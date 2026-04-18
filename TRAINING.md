# Training on an A100

This repository is currently being edited from a macOS shell without visible CUDA, so real QLoRA training should be run from the A100 environment instead.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-train.txt
```

## Validate

```bash
python3 -m codedna.cli.main train --dry-run
```

You should no longer see the base-model warning once `mistralai/Mistral-7B-v0.1` is reachable or cached on that machine.

## Train

```bash
python3 -m codedna.cli.main train --model mistralai/Mistral-7B-v0.1
```

## One-shot helper

```bash
bash scripts/train_a100.sh
```
