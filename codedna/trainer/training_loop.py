"""Training loop helpers for CodeDNA."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from codedna.trainer.lora_config import get_lora_config, get_qlora_quantization, get_training_args

console = Console()


def run_training(base_model: str, output_dir: str, dry_run: bool = False) -> None:
    """Run the CodeDNA training loop or validate it with a dry run."""

    train_path = Path(".codedna") / "dataset" / "train.jsonl"
    val_path = Path(".codedna") / "dataset" / "val.jsonl"
    cache_dir = Path(".codedna") / "cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Missing train.jsonl or val.jsonl. Run `codedna dataset export --repo .` first.")

    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(val_path)},
        cache_dir=str(cache_dir),
    )

    qlora_config = None
    qlora_available = True
    qlora_error = ""
    try:
        qlora_config = get_qlora_quantization()
    except Exception as exc:
        qlora_available = False
        qlora_error = str(exc)

    tokenizer = None
    model = None
    model_load_error = ""
    model_config = None
    trainable_params = 0
    total_params = 0

    try:
        model_config = AutoConfig.from_pretrained(base_model, local_files_only=dry_run)
        tokenizer = AutoTokenizer.from_pretrained(base_model, local_files_only=dry_run)
        tokenizer.pad_token = tokenizer.eos_token

        model_kwargs = {"device_map": "auto"}
        if qlora_available and not dry_run:
            model_kwargs["quantization_config"] = qlora_config
        elif qlora_available and dry_run:
            model_kwargs["device_map"] = None
        if dry_run:
            model_kwargs["local_files_only"] = True

        model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)
        if qlora_available and not dry_run:
            model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, get_lora_config())
        trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total_params = sum(parameter.numel() for parameter in model.parameters())
    except Exception as exc:
        model_load_error = str(exc)

    tokenized_dataset = None
    if tokenizer is not None:
        tokenized_dataset = dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=True, max_length=2048),
            batched=True,
        )

    checkpoint_dir = Path(output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "base_model": base_model,
        "train_examples": len(dataset["train"]),
        "validation_examples": len(dataset["validation"]),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "qlora_available": qlora_available,
        "qlora_error": qlora_error,
        "model_load_error": model_load_error,
        "model_type": getattr(model_config, "model_type", ""),
    }
    (checkpoint_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    console.print(f"Base model: {base_model}")
    console.print(f"Dataset sizes: train={len(dataset['train'])}, val={len(dataset['validation'])}")
    if model_config is not None:
        console.print(f"Model type: {getattr(model_config, 'model_type', 'unknown')}")
    if model is not None:
        console.print(f"Trainable parameters: {trainable_params} / {total_params}")
    elif model_load_error:
        console.print(f"Model load warning: {model_load_error}")

    if dry_run:
        console.print("Dry run complete. Training configuration validated.")
        return

    if tokenizer is None or model is None or tokenized_dataset is None:
        raise RuntimeError(
            "Full training could not start because the base model/tokenizer could not be loaded."
        )

    try:
        from transformers import DataCollatorForLanguageModeling
        from trl import SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Full training requires `trl` and related training dependencies to be installed."
        ) from exc

    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        args=get_training_args(output_dir),
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    final_dir = checkpoint_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    console.print(f"Saved adapter checkpoint to {final_dir}")
