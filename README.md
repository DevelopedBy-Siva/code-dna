# CodeDNA

CodeDNA is a local coding assistant for training and serving a model on your own repositories.

It extracts function-level examples from private Python projects, mixes them with filtered public data, fine-tunes a base model with QLoRA, evaluates the result, and serves it through a local API and browser interface.

---

## Results

Trained on **272 private function pairs** extracted from personal Python repositories, mixed with **1,561 filtered public Python pairs** (70/30 ratio), fine-tuned on `mistralai/Mistral-7B-v0.1` using QLoRA on an NVIDIA A100-SXM4-80GB.

| Metric | Base Mistral-7B | CodeDNA Fine-Tune | Delta |
|---|---|---|---|
| Perplexity ↓ | 3.69 | 2.20 | **−40.4%** |
| HumanEval pass@1 ↑ | 1.20% | 6.70% | **+5.5×** |
| Style Conformance ↑ | 33.50 / 100 | 41.05 / 100 | **+22.5%** |

- Lower perplexity means the fine-tuned model fits the validation set better
- HumanEval improvement suggests general coding ability held up alongside the style tuning
- The style score tracks naming, function length, library usage, and error-handling patterns

Successful A100 training run: **3 epochs · 2,007 training examples · ~33 minutes · train loss 0.5644 · val token accuracy 81.25%**

---

## Overview

The goal is simple: take a general-purpose open model and nudge it toward the way I actually write Python.

The pipeline focuses on:

- extracting reusable function examples from private repos
- building a mixed public/private training set
- fine-tuning a 7B model with QLoRA
- checking both general coding quality and style alignment
- serving the tuned model locally through an API and browser interface

---

## How It Works

```
Your GitHub Repos
      │
      ▼
 [Analyzer]  ──── AST-based extraction
      │            function-level prompt/completion pairs
      │            style profile detection
      │            quality scoring (0–100 per pair)
      ▼
 [Dataset]   ──── 70% public Python instruction data
      │            30% your private pairs (upsampled)
      │            chat-format JSONL
      ▼
 [Trainer]   ──── QLoRA on Mistral-7B-v0.1
      │            LoRA rank 16, alpha 32
      │            3 epochs, lr 2e-4, cosine schedule
      ▼
 [Evaluator] ──── Perplexity on held-out val set
      │            HumanEval pass@1 (164 problems)
      │            Custom style conformance score
      ▼
 [Server]    ──── FastAPI REST server
                   /v1/chat/completions endpoint
                   /v1/models, /health endpoints
                   browser playground at /
```

---

## Technical Decisions

### 1. 70/30 Public/Private Data Mix

Training only on a small private corpus risks overfitting to personal patterns and weakening general coding behavior. The final dataset mixes 70% public Python instruction/code data with 30% private pairs at the dataset level so both signals are present during fine-tuning.

In practice, this kept the model from becoming too narrow while still improving the style metric.

---

### 2. AST Extraction Over Line Splitting

Source files are parsed structurally rather than split by line count. In the current Python-first pipeline, extraction uses the standard library `ast` module for Python and keeps tree-sitter-backed helpers available for non-Python extensions. This produces cleaner, scope-correct pairs than naive chunking.

Prompt = function signature + optional docstring. Completion = body only. Early versions included the full signature inside the completion, causing the model to repeat it before generating — body-only completions fixed this.

Filters:
- skip functions with body < 3 lines
- skip functions with body > 80 lines
- skip near-duplicate completions (similarity > 0.85)

---

### 3. Quality Scoring Before Mixing

Every private pair is scored before entering the mix. Pairs below 60 are dropped. Scoring happens pre-mix so weak pairs never consume part of the private slice of the final dataset.

| Check | Points |
|---|---|
| Completion under 5 lines | −20 |
| No docstring | −15 |
| Naming mismatches repo convention | −20 |
| Library overlap with top-20 imports | up to +15 |
| Near-duplicate | score → 0 |

---

### 4. Style Conformance as a Custom Metric

Standard benchmarks (perplexity, HumanEval) don't measure whether the model writes code in your style. A custom scorer measures generated completions against the extracted style profile across four dimensions:

| Dimension | Weight | How Measured |
|---|---|---|
| Naming convention | 30 pts | snake_case / camelCase / PascalCase detection |
| Function length | 25 pts | deviation from repo average |
| Library usage | 25 pts | overlap with top-20 imported libraries |
| Error handling | 20 pts | try/except presence vs repo rate |

Perplexity and HumanEval are useful reference points. The style score is the metric that is most specific to this project.

---

## Repository Layout

```
codedna/
├── cli/
│   ├── main.py                  # Typer app, all command registration
│   └── commands/
│       ├── init.py              # codedna init
│       ├── analyze.py           # codedna analyze
│       ├── dataset.py           # codedna dataset preview/clean/export
│       ├── train.py             # codedna train
│       ├── eval.py              # codedna eval run/compare
│       ├── serve.py             # codedna serve
│       └── chat.py              # codedna chat
├── analyzer/
│   ├── repo_scanner.py          # file discovery + filtering
│   ├── ast_extractor.py         # AST-based function extraction
│   ├── pattern_detector.py      # style profile detection
│   └── quality_scorer.py        # pair scoring + dedup
├── dataset/
│   ├── public_loader.py         # HuggingFace public dataset download
│   ├── mixer.py                 # 70/30 mixing + upsampling
│   ├── formatter.py             # chat-format JSONL output
│   └── cleaner.py               # quality filtering
├── trainer/
│   ├── lora_config.py           # LoRA + QLoRA + TrainingArguments
│   ├── training_loop.py         # SFTTrainer wrapper
│   ├── checkpoints.py           # checkpoint management
│   └── model_merge.py           # adapter → standalone model
├── evaluator/
│   ├── perplexity.py            # validation set perplexity
│   ├── humaneval_runner.py      # HumanEval pass@k
│   ├── style_scorer.py          # custom style conformance
│   └── report.py                # rich terminal comparison table
└── server/
    ├── app.py                   # FastAPI REST server with /v1/chat/completions
    ├── inference.py             # model loading + generation
    └── playground.py            # browser interface
```

Runtime artifacts written to `.codedna/` (gitignored).
Trained adapter artifacts are expected under `model/checkpoints/final`.

---

## Installation

```bash
git clone https://github.com/your-username/codedna.git
cd codedna

# Base dependencies (analyzer, dataset, server)
pip install -r requirements.txt

# Training dependencies (GPU environment)
pip install -r requirements-train.txt
```

---

## Usage

### Analyze a repository

`codedna analyze` scans the path you pass directly. For example, to analyze the current repo:

```bash
codedna analyze --repo .
```

Example output:
```
✓ Scanned 47 files
✓ Extracted 312 function pairs
✓ Style profile saved → .codedna/style_profile.json
  naming: snake_case | avg length: 9.1 lines | docstring rate: 100%
✓ Quality scored: 272 pairs above threshold (score ≥ 60)
```

### Build the dataset

For dataset building, place your private Python repos under `private-repo/`, then run:

```bash
codedna dataset clean
codedna dataset export
```

Produces `.codedna/dataset/train.jsonl` and `val.jsonl`.

### Validate training config

```bash
codedna train --dry-run
```

### Train (GPU required)

```bash
codedna train --model mistralai/Mistral-7B-v0.1
```

For A100 / cloud GPU environments, see [TRAINING.md](TRAINING.md).

### Evaluate

```bash
# Quick (skips HumanEval)
codedna eval run --quick --adapter-path model/checkpoints/final

# Full evaluation
codedna eval run --adapter-path model/checkpoints/final
```

Output:
```
┌─────────────────────────────────────────────────────┐
│              CodeDNA Evaluation Report               │
├──────────────────────┬────────┬────────────┬────────┤
│ Metric               │ Base   │ Fine-Tuned │ Delta  │
├──────────────────────┼────────┼────────────┼────────┤
│ Perplexity ↓         │  3.69  │    2.20    │ ▼ 1.49 │
│ HumanEval pass@1 ↑   │  1.20% │    6.70%   │ ▲ 5.50 │
│ Style Score ↑        │ 33.50  │   41.05    │ ▲ 7.55 │
└──────────────────────┴────────┴────────────┴────────┘
```

### Side-by-side prompt comparison

```bash
codedna eval compare \
  --prompt "Write a Python retry helper with exponential backoff." \
  --adapter-path model/checkpoints/final
```

### Serve locally

```bash
codedna serve --checkpoint-path model/checkpoints/final --port 8080
```

The server exposes a `/v1/chat/completions` endpoint shaped after the OpenAI chat API. It can be pointed at from any client that accepts a configurable base URL, though compatibility with specific editors has not been verified.

It also serves a browser interface at `http://127.0.0.1:8080/`.

```json
{
  "model": "codedna-local",
  "apiBase": "http://localhost:8080/v1",
  "apiKey": "codedna"
}
```

### Terminal chat

```bash
codedna chat --checkpoint-path model/checkpoints/final
```

---

## Training Details

| Parameter | Value |
|---|---|
| Base model | `mistralai/Mistral-7B-v0.1` |
| Method | QLoRA (4-bit NF4 quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q/k/v/o proj, gate/up/down proj |
| Epochs | 3 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Batch size | 4 (effective 16 with grad accum) |
| Precision | bf16 (A100) |
| Hardware | NVIDIA A100-SXM4-80GB |
| Training time | ~33 minutes |
| Train loss | 0.5644 |
| Val token accuracy | 81.25% |

---

## Dataset Details

| Source | Count | Share |
|---|---|---|
| Private repos (unique cleaned pairs) | 272 pairs | — |
| Private share after upsampling | 669 examples | 30% of mix |
| Public: CodeAlpaca-20k | — | — |
| Public: evol-codealpaca-v1 | — | — |
| Public: tiny-codes | — | — |
| Public total (filtered) | 1,561 pairs | 70% of mix |
| **Final train set** | **2,007 examples** | — |
| **Final val set** | **223 examples** | — |

Private pairs are upsampled with lightweight prompt variation to hit the 30% target. Public data is filtered to Python-only, rejecting non-code instruction tasks, generic prompts, and mixed-language completions.

---

## Style Profile

The analyzer writes a repository style profile to `.codedna/style_profile.json`. The profile tracks:

- dominant naming convention
- average and p90 function length
- top imported libraries
- docstring rate
- try/except usage rate

---

## Tech Stack

| Layer | Tools |
|---|---|
| Application | `typer`, `rich` |
| Code parsing | `tree-sitter`, `tree-sitter-languages` |
| Datasets | `datasets` (HuggingFace) |
| Training | `transformers`, `peft`, `trl`, `bitsandbytes`, `accelerate` |
| Serving | `fastapi`, `uvicorn` |
| Evaluation | `evaluate`, `human-eval` |

---

## Limitations & Honest Caveats

- **Small private dataset.** 272 pairs is a real but modest private signal. Style conformance improvement (+22%) reflects the small size — a larger private corpus would push this further.
- **Python-only.** The current pipeline is restricted to `.py` files. Multi-language support requires additional tree-sitter grammars and language-specific filtering.
- **GPU required for training.** QLoRA reduces memory requirements, but CPU training is not supported and practical VRAM needs depend on the base model and runtime stack.
- **Serving requires real adapter weights.** If the repo was cloned without Git LFS, `adapter_model.safetensors` will be a pointer file. Run `git lfs pull` to fetch real weights.
- **HumanEval measures general coding, not style.** The +5.5x improvement on HumanEval is a positive signal, but this benchmark doesn't directly measure whether the model writes code in your style. The custom style score is the primary metric for that.

