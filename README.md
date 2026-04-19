# CodeDNA

CodeDNA is a CLI tool for building a personal coding assistant from your own repositories. It extracts function-level training pairs from your code, mixes them with filtered public coding data, fine-tunes a Mistral-based model with LoRA / QLoRA, evaluates the result, and serves it behind an OpenAI-compatible API.

## What It Does

- scans your repositories and extracts Python function-style prompt/completion pairs
- builds a private style profile from your code
- mixes private data with filtered public Python instruction/code pairs
- fine-tunes `mistralai/Mistral-7B-v0.1` with LoRA / QLoRA
- evaluates base vs fine-tuned performance
- serves the fine-tuned model through `/v1/chat/completions`

## Current Status

Implemented:

- project scaffold
- analyzer
- dataset pipeline
- trainer
- evaluator
- server

Current experiment summary:

- private cleaned pairs: `272`
- filtered public pairs: `1561`
- final mixed dataset: `2230`
- train split: `2007`
- validation split: `223`

Latest full evaluation on the stronger fine-tuned checkpoint:

- Perplexity: `3.69 -> 2.20`
- HumanEval pass@1: `1.20 -> 6.70`
- Style Score: `33.50 -> 41.05`

## Repository Layout

```text
codedna/
├── cli/
├── analyzer/
├── dataset/
├── trainer/
├── evaluator/
└── server/
```

Runtime artifacts are written under:

```text
.codedna/
```

Trained adapter artifacts currently live under:

```text
model/checkpoints/final
```

## Installation

Base dependencies:

```bash
pip install -r requirements.txt
```

Training-specific dependencies:

```bash
pip install -r requirements-train.txt
```

## Typical Workflow

### 1. Build the dataset

Private repositories should be placed under:

```text
private-repo/
```

Then run:

```bash
python -m codedna.cli.main dataset clean --repo .
python -m codedna.cli.main dataset export --repo .
```

Verify:

```bash
wc -l .codedna/dataset/train.jsonl .codedna/dataset/val.jsonl
```

### 2. Validate training setup

```bash
python -m codedna.cli.main train --dry-run
```

### 3. Train

```bash
python -m codedna.cli.main train --model mistralai/Mistral-7B-v0.1
```

For the A100 flow, see [TRAINING.md](TRAINING.md).

### 4. Evaluate

Quick evaluation:

```bash
python -m codedna.cli.main eval run --quick --adapter-path model/checkpoints/final
```

Full evaluation:

```bash
python -m codedna.cli.main eval run --adapter-path model/checkpoints/final
```

Prompt comparison:

```bash
python -m codedna.cli.main eval compare \
  --prompt "Write a Python retry helper with exponential backoff." \
  --adapter-path model/checkpoints/final
```

### 5. Serve locally

Start server:

```bash
python -m codedna.cli.main serve --checkpoint-path model/checkpoints/final --port 8080
```

Health check:

```bash
curl http://127.0.0.1:8080/health
```

List models:

```bash
curl http://127.0.0.1:8080/v1/models
```

Chat completions:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "codedna-local",
    "messages": [
      {"role": "user", "content": "Write a Python retry helper with exponential backoff."}
    ],
    "max_tokens": 200,
    "temperature": 0.2
  }'
```

Terminal chat:

```bash
python -m codedna.cli.main chat --checkpoint-path model/checkpoints/final
```

## Notes on Model Files

Large model files are expected to be stored with Git LFS.

If a checkpoint file such as `adapter_model.safetensors` looks like plain text containing:

```text
version https://git-lfs.github.com/spec/v1
```

then you only have the LFS pointer, not the real binary weights.

Fetch real LFS objects with:

```bash
git lfs install
git lfs pull
```

## Experiment Log

Detailed experiment notes, dataset formation decisions, training outcomes, and evaluation history are documented in:

- [README_exp.md](README_exp.md)

## Tech Stack

- `typer`
- `rich`
- `datasets`
- `transformers`
- `peft`
- `trl`
- `bitsandbytes`
- `accelerate`
- `fastapi`
- `uvicorn`
- `tree-sitter`
- `tree-sitter-languages`

## Current Caveats

- the base model must be downloadable or cached locally for training / evaluation
- full serving depends on the real adapter weights being present, not just Git LFS pointers
- the project is currently focused on Python-only private data
