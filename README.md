# CodeDNA

CodeDNA is a CLI tool scaffold for analyzing a developer codebase, preparing datasets, fine-tuning a coding model with LoRA or QLoRA, evaluating the result, and serving it behind an OpenAI-compatible API.

## Status

Phase 1 is complete.

- The project structure matches the planned module layout.
- The Typer CLI is registered and loadable.
- Analyzer, dataset, trainer, evaluator, and server files are present as typed stubs with module docstrings.

## Verify

```bash
pip install -e .
codedna --help
```
