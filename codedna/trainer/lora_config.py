"""LoRA and QLoRA configuration helpers for CodeDNA."""

from __future__ import annotations

import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig, TrainingArguments


def get_lora_config() -> LoraConfig:
    """Return the LoRA configuration used for supervised fine-tuning."""

    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_qlora_quantization() -> BitsAndBytesConfig:
    """Return the 4-bit QLoRA quantization configuration."""

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def get_training_args(output_dir: str) -> TrainingArguments:
    """Return the Hugging Face training arguments for CodeDNA runs."""

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=10,
        report_to=[],
    )
