"""
Fine-tuning Qwen2.5-Coder-14B for Python Code Generation
=========================================================
Hardware:      A100 (40GB or 80GB)
TRL:           1.2.0
Transformers:  5.x
Unsloth:       2026.x
Python:        3.11

KEY FIX: Pre-tokenize the dataset before passing to SFTTrainer.
This bypasses TRL's internal chat-template processing which injects
<EOS_TOKEN> from Unsloth's patched tokenizer and causes a ValueError.
"""

import os
import torch
import logging
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, SFTConfig
from unsloth import FastLanguageModel

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

@dataclass
class Config:
    # Model
    model_name: str     = "Qwen/Qwen2.5-Coder-14B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool  = False

    # LoRA
    lora_r: int         = 64
    lora_alpha: int     = 128
    lora_dropout: float = 0.0           # 0.0 = Unsloth fast patching (faster)
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    output_dir: str                  = "./qwen-python-finetuned"
    num_train_epochs: int            = 3
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

    # Dataset
    val_split: float = 0.02
    max_samples: int = None


cfg = Config()

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FOREIGN EOS TOKENS TO STRIP
# datasets embed tokens from other model families
# ─────────────────────────────────────────────

FOREIGN_EOS = ["<EOS_TOKEN>", "</s>", "<eos>", "<|endoftext|>", "<|eot_id|>"]


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

def load_and_merge_datasets() -> Dataset:
    logger.info("Loading datasets...")
    all_datasets = []

    loaders = [
        ("Vezora/Tested-22k-Python-Alpaca",         "Vezora",      None),
        ("iamtarun/python_code_instructions_18k_alpaca", "iamtarun", None),
        ("flytech/python-codes-25k",                "flytech",     None),
        ("sahil2801/CodeAlpaca-20k",                "CodeAlpaca",  None),
    ]

    for hf_id, name, _ in loaders:
        try:
            ds = load_dataset(hf_id, split="train")
            if name == "CodeAlpaca":
                ds = ds.filter(lambda x: "python" in x.get("output", "").lower())
            logger.info(f"  {name:<16} {len(ds):>6} samples")
            all_datasets.append(ds)
        except Exception as e:
            logger.warning(f"  {name} failed: {e}")

    # Magicoder — different column names
    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        ds = ds.filter(lambda x: x.get("lang", "").lower() == "python")
        ds = ds.map(lambda x: {
            "instruction": x.get("problem", ""),
            "input":       "",
            "output":      x.get("solution", ""),
        })
        logger.info(f"  {'Magicoder':<16} {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  Magicoder failed: {e}")

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
    logger.info("Cleaning...")

    def strip_and_filter(example):
        out = example.get("output", "")
        ins = example.get("instruction", "")
        for tok in FOREIGN_EOS:
            out = out.replace(tok, "")
            ins = ins.replace(tok, "")
        return {"instruction": ins.strip(), "input": example.get("input", ""), "output": out.strip()}

    dataset = dataset.map(strip_and_filter)

    python_kw = ["def ", "import ", "class ", "return ", "print(", "for ", "if "]

    def quality_ok(ex):
        out, ins = ex.get("output",""), ex.get("instruction","")
        if len(out) < 50 or len(out) > 4000: return False
        if len(ins) < 10:                     return False
        if not any(k in out for k in python_kw): return False
        return True

    dataset = dataset.filter(quality_ok)
    logger.info(f"  After quality filter: {len(dataset)}")

    seen = set()
    def unique(ex):
        k = ex["instruction"][:100].strip().lower()
        if k in seen: return False
        seen.add(k); return True

    dataset = dataset.filter(unique)
    logger.info(f"  After dedup:          {len(dataset)}")
    return dataset


# ─────────────────────────────────────────────
# TOKENIZE UPFRONT
# This is the key fix: we tokenize before SFTTrainer
# so TRL never touches our text or applies its chat template.
# ─────────────────────────────────────────────

def tokenize_dataset(dataset: Dataset, tokenizer) -> Dataset:
    logger.info("Tokenizing dataset...")

    EOS = tokenizer.eos_token   # confirmed: <|im_end|> for Qwen2.5

    def tokenize(example):
        inp = example.get("input", "").strip()
        input_section = f"\n\n### Input:\n{inp}" if inp else ""
        text = (
            f"### Instruction:\n{example['instruction'].strip()}"
            f"{input_section}\n\n"
            f"### Response:\n{example['output'].strip()}"
            f"{EOS}"
        )
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=cfg.max_seq_length,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    dataset = dataset.map(
        tokenize,
        remove_columns=["instruction", "input", "output"],
        num_proc=4,
    )
    logger.info(f"  Tokenized: {len(dataset)} samples")
    return dataset


def prepare_dataset(dataset: Dataset, tokenizer):
    if cfg.max_samples and len(dataset) > cfg.max_samples:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

    tokenized = tokenize_dataset(dataset, tokenizer)
    split     = tokenized.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    logger.info(f"  Train: {len(split['train'])}  Val: {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def load_model():
    logger.info(f"Loading: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.model_name,
        max_seq_length = cfg.max_seq_length,
        load_in_4bit   = cfg.load_in_4bit,
        dtype          = torch.bfloat16,
    )
    logger.info(f"Tokenizer EOS token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id})")

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


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train(model, tokenizer, train_dataset, eval_dataset):
    logger.info("Configuring SFTTrainer...")

    sft_config = SFTConfig(
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
        # Since dataset is pre-tokenized, these tell TRL not to re-process text
        max_length                  = cfg.max_seq_length,
        dataset_kwargs              = {"skip_prepare_dataset": True},
    )

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        args             = sft_config,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu_mem:.1f} GB")

    logger.info("Training started...")
    stats = trainer.train()
    logger.info(f"Done! loss={stats.metrics['train_loss']:.4f}  "
                f"time={stats.metrics['train_runtime']:.0f}s  "
                f"samples/s={stats.metrics['train_samples_per_second']:.1f}")
    return trainer


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_model(model, tokenizer):
    lora_path   = os.path.join(cfg.output_dir, "lora_adapter")
    merged_path = os.path.join(cfg.output_dir, "merged_model")

    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"LoRA adapter  → {lora_path}")

    logger.info("Merging LoRA into base model...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model  → {merged_path}")


# ─────────────────────────────────────────────
# INFERENCE TEST
# ─────────────────────────────────────────────

def test_inference(model, tokenizer):
    logger.info("Quick inference test...")
    FastLanguageModel.for_inference(model)

    for prompt in [
        "Write a Python function to flatten a nested list",
        "Write a Python decorator that retries a function 3 times on exception",
        "Create a Python context manager for timing code blocks",
    ]:
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


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Qwen2.5-Coder-14B Python Fine-tuning")
    logger.info("TRL 1.2.0 | transformers 5.x | Unsloth 2026.x")
    logger.info("=" * 60)

    # Load model first so we have the tokenizer for dataset prep
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