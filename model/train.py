"""Fine-tune Qwen2.5-Coder-14B on Python instruction data with Unsloth LoRA."""

import unsloth

import os
import inspect
import torch
import logging
from pathlib import Path
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from unsloth import FastLanguageModel

BASE_DIR = Path(__file__).resolve().parent
TRAINING_LOG = BASE_DIR / "training.log"

@dataclass
class Config:
    model_name: str     = "Qwen/Qwen2.5-Coder-14B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool  = False

    lora_r: int         = 64
    lora_alpha: int     = 128
    lora_dropout: float = 0.0
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    output_dir: str                  = str(BASE_DIR / "qwen-python-finetuned")
    num_train_epochs: int            = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int                = 100
    learning_rate: float             = 1e-4
    lr_scheduler_type: str           = "cosine"
    weight_decay: float              = 0.01
    bf16: bool                       = True
    fp16: bool                       = False
    logging_steps: int               = 25
    save_steps: int                  = 500
    eval_steps: int                  = 500
    save_total_limit: int            = 3
    load_best_model_at_end: bool     = True
    optim: str                       = "adamw_8bit"
    seed: int                        = 42

    val_split: float = 0.02
    max_samples: int = None


cfg = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(TRAINING_LOG),
    ]
)
logger = logging.getLogger(__name__)

FOREIGN_EOS = ["<EOS_TOKEN>", "</s>", "<eos>", "<|endoftext|>", "<|eot_id|>"]

def load_and_merge_datasets() -> Dataset:
    """Load, normalize, and combine the training datasets."""
    logger.info("Loading datasets...")
    all_datasets = []

    try:
        ds = load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")
        logger.info(f"  Vezora:          {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  Vezora failed: {e}")

    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        logger.info(f"  iamtarun:        {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  iamtarun failed: {e}")

    try:
        ds = load_dataset("flytech/python-codes-25k", split="train")
        logger.info(f"  flytech:         {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  flytech failed: {e}")

    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        ds = ds.filter(lambda x: x.get("lang", "").lower() == "python")
        ds = ds.map(lambda x: {
            "instruction": x.get("problem", ""),
            "input":       "",
            "output":      x.get("solution", ""),
        })
        logger.info(f"  Magicoder (py):  {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  Magicoder failed: {e}")

    # 5. CodeAlpaca — filter Python
    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        ds = ds.filter(lambda x: "python" in x.get("output", "").lower())
        logger.info(f"  CodeAlpaca (py): {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  CodeAlpaca failed: {e}")

    if not all_datasets:
        raise RuntimeError("No datasets loaded!")

    def normalize(example):
        return {
            "instruction": str(example.get("instruction", "") or ""),
            "input":       str(example.get("input",       "") or ""),
            "output":      str(example.get("output",      "") or ""),
        }

    normalized = [ds.map(normalize, remove_columns=ds.column_names) for ds in all_datasets]
    combined   = concatenate_datasets(normalized)
    logger.info(f"  Combined raw: {len(combined)} samples")
    return combined


def clean_dataset(dataset: Dataset) -> Dataset:
    """Drop noisy examples and simple near-duplicates."""
    logger.info("Cleaning dataset...")

    def strip_and_clean(example):
        out = example.get("output", "")
        ins = example.get("instruction", "")
        for tok in FOREIGN_EOS:
            out = out.replace(tok, "")
            ins = ins.replace(tok, "")
        return {
            "instruction": ins.strip(),
            "input":       example.get("input", ""),
            "output":      out.strip(),
        }

    dataset = dataset.map(strip_and_clean)

    python_kw = ["def ", "import ", "class ", "return ", "print(", "for ", "if "]

    def quality_ok(ex):
        out = ex.get("output", "")
        ins = ex.get("instruction", "")
        if len(out) < 50 or len(out) > 4000:        return False
        if len(ins) < 10:                             return False
        if not any(k in out for k in python_kw):     return False
        return True

    dataset = dataset.filter(quality_ok)
    logger.info(f"  After quality filter: {len(dataset)}")

    seen = set()
    def is_unique(ex):
        k = ex["instruction"][:100].strip().lower()
        if k in seen: return False
        seen.add(k); return True

    dataset = dataset.filter(is_unique)
    logger.info(f"  After dedup:          {len(dataset)}")
    return dataset

def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    """Pre-tokenize examples for the plain Hugging Face Trainer."""
    logger.info("Tokenizing...")
    EOS = tokenizer.eos_token

    def tokenize(example):
        inp = example.get("input", "").strip()
        input_section = f"\n\n### Input:\n{inp}" if inp else ""
        text = (
            f"### Instruction:\n{example['instruction'].strip()}"
            f"{input_section}\n\n"
            f"### Response:\n{example['output'].strip()}"
            f"{EOS}"
        )
        result = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized = dataset.map(
        tokenize,
        remove_columns=["instruction", "input", "output"],
        num_proc=4,
    )
    logger.info(f"  Tokenized: {len(tokenized)} samples")
    return tokenized


def prepare_dataset(dataset: Dataset, tokenizer):
    """Tokenize and split the dataset into train and eval sets."""
    if cfg.max_samples and len(dataset) > cfg.max_samples:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

    tokenized = tokenize_dataset(dataset, tokenizer)
    split     = tokenized.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    logger.info(f"  Train: {len(split['train'])}  |  Val: {len(split['test'])}")
    return split["train"], split["test"]

def load_model():
    """Load the base model and attach LoRA adapters."""
    logger.info(f"Loading: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.model_name,
        max_seq_length = cfg.max_seq_length,
        load_in_4bit   = cfg.load_in_4bit,
        dtype          = torch.bfloat16,
    )
    logger.info(f"EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

    model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg.lora_r,
        lora_alpha                 = cfg.lora_alpha,
        lora_dropout               = cfg.lora_dropout,
        target_modules             = list(cfg.target_modules),
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",
        random_state               = cfg.seed,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model, tokenizer

def train(model, tokenizer, train_dataset, eval_dataset):
    """Train the model with the standard Hugging Face Trainer."""
    logger.info("Configuring Trainer...")

    training_args = TrainingArguments(
        output_dir                  = cfg.output_dir,
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        warmup_steps                = cfg.warmup_steps,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        weight_decay                = cfg.weight_decay,
        optim                       = cfg.optim,
        bf16                        = cfg.bf16,
        fp16                        = cfg.fp16,
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        save_strategy               = "steps",
        save_total_limit            = cfg.save_total_limit,
        eval_strategy               = "steps",
        eval_steps                  = cfg.eval_steps,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,
        seed                        = cfg.seed,
        report_to                   = "none",
        dataloader_num_workers      = 4,
        remove_unused_columns       = False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model           = model,
        padding         = True,
        pad_to_multiple_of = 8,
        label_pad_token_id = -100,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": collator,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=3)],
    }

    trainer_init_params = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_init_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        logger.warning(
            "Trainer.__init__ accepts neither 'processing_class' nor 'tokenizer'; "
            "continuing without attaching the tokenizer."
        )

    trainer = Trainer(**trainer_kwargs)

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu_mem:.1f} GB")

    logger.info("Training started...")
    stats = trainer.train()
    logger.info(f"Done!")
    logger.info(f"  Loss:        {stats.metrics['train_loss']:.4f}")
    logger.info(f"  Runtime:     {stats.metrics['train_runtime']:.0f}s")
    logger.info(f"  Samples/sec: {stats.metrics['train_samples_per_second']:.1f}")
    return trainer

def save_model(model, tokenizer):
    """Save both the LoRA adapter and merged full model."""
    lora_path   = os.path.join(cfg.output_dir, "lora_adapter")
    merged_path = os.path.join(cfg.output_dir, "merged_model")

    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"LoRA adapter  → {lora_path}")

    logger.info("Merging LoRA weights into base model...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model  → {merged_path}")

def test_inference(model, tokenizer):
    """Run a small generation smoke test."""
    logger.info("Quick inference test...")
    FastLanguageModel.for_inference(model)

    prompts = [
        "Write a Python function to flatten a nested list",
        "Write a Python decorator that retries a function 3 times on exception",
        "Create a Python context manager for timing code blocks",
    ]

    for prompt in prompts:
        text   = f"### Instruction:\n{prompt}\n\n### Response:\n"
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens = 300,
                temperature    = 0.2,
                top_p          = 0.95,
                do_sample      = True,
                pad_token_id   = tokenizer.eos_token_id,
            )
        print(f"\n{'='*60}\n{prompt}\n{'='*60}")
        print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

def main():
    logger.info("=" * 60)
    logger.info("Qwen2.5-Coder-14B  |  Python Fine-tuning")
    logger.info("HuggingFace Trainer + Unsloth LoRA")
    logger.info("=" * 60)

    model, tokenizer  = load_model()

    raw               = load_and_merge_datasets()
    clean             = clean_dataset(raw)
    train_ds, eval_ds = prepare_dataset(clean, tokenizer)

    trainer           = train(model, tokenizer, train_ds, eval_ds)
    save_model(model, tokenizer)
    test_inference(model, tokenizer)

    logger.info(f"All done → {cfg.output_dir}")


if __name__ == "__main__":
    main()
