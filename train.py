"""
Fine-tuning Qwen2.5-Coder-14B for Python Code Generation
=========================================================
Hardware:      A100 (40GB or 80GB)
Method:        LoRA via Unsloth
TRL:           1.2.0
Transformers:  5.x
Unsloth:       2026.x
Python:        3.11

Install:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    pip install trl==1.2.0 datasets accelerate bitsandbytes
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
    load_in_4bit: bool  = False         # A100 has enough VRAM

    # LoRA
    lora_r: int         = 64
    lora_alpha: int     = 128
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    output_dir: str                  = "./qwen-python-finetuned"
    num_train_epochs: int            = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4        # effective batch = 16
    warmup_steps: int                = 100
    learning_rate: float             = 1e-4     # conservative — baseline already at 80%
    lr_scheduler_type: str           = "cosine"
    weight_decay: float              = 0.01
    bf16: bool                       = True     # A100 native bf16
    fp16: bool                       = False
    logging_steps: int               = 25
    save_steps: int                  = 500
    eval_steps: int                  = 500
    save_total_limit: int            = 3
    load_best_model_at_end: bool     = True
    optim: str                       = "adamw_8bit"
    seed: int                        = 42

    # Dataset
    val_split: float  = 0.02
    max_samples: int  = None           # set to e.g. 5000 for a quick smoke test


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
# PROMPT TEMPLATE
# Qwen2.5 EOS token is <|im_end|>
# ─────────────────────────────────────────────

def format_prompt(example):
    input_section = (
        f"\n\n### Input:\n{example['input'].strip()}"
        if example.get("input", "").strip()
        else ""
    )
    return {
        "text": (
            f"### Instruction:\n{example.get('instruction', '').strip()}"
            f"{input_section}\n\n"
            f"### Response:\n{example.get('output', '').strip()}"
            f"<|im_end|>"          # Qwen2.5 chat EOS token
        )
    }


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

# Tokens from other model families that pollute our datasets
FOREIGN_EOS_TOKENS = [
    "<EOS_TOKEN>",   # Cohere / Command-R datasets
    "</s>",          # LLaMA / Mistral datasets
    "<eos>",         # Gemma datasets
    "<|endoftext|>", # GPT-2 style
    "<|eot_id|>",    # LLaMA-3 datasets
]


def load_and_merge_datasets() -> Dataset:
    logger.info("Loading datasets...")
    all_datasets = []

    # 1. Vezora — every sample is tested/verified Python
    try:
        ds = load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")
        logger.info(f"  Vezora:          {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  Vezora failed: {e}")

    # 2. Python code instructions (known to contain <EOS_TOKEN> — stripped below)
    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        logger.info(f"  iamtarun:        {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  iamtarun failed: {e}")

    # 3. flytech python codes
    try:
        ds = load_dataset("flytech/python-codes-25k", split="train")
        logger.info(f"  flytech:         {len(ds):>6} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"  flytech failed: {e}")

    # 4. Magicoder — filter Python only
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
        raise RuntimeError("No datasets loaded successfully!")

    # Normalize all datasets to instruction / input / output columns
    def normalize(example):
        return {
            "instruction": str(example.get("instruction", "") or ""),
            "input":       str(example.get("input",       "") or ""),
            "output":      str(example.get("output",      "") or ""),
        }

    normalized = [
        ds.map(normalize, remove_columns=ds.column_names)
        for ds in all_datasets
    ]
    combined = concatenate_datasets(normalized)
    logger.info(f"Combined raw total: {len(combined)} samples")
    return combined


def clean_dataset(dataset: Dataset) -> Dataset:
    logger.info("Cleaning dataset...")

    # ── Step 1: Strip foreign EOS tokens ──────────────────────────────────────
    # Some datasets (iamtarun, CodeAlpaca) embed tokens from other model families.
    # TRL 1.2 strictly validates EOS tokens and will crash if it finds a foreign one.
    def strip_foreign_eos(example):
        output = example.get("output", "")
        for tok in FOREIGN_EOS_TOKENS:
            output = output.replace(tok, "")
        instruction = example.get("instruction", "")
        for tok in FOREIGN_EOS_TOKENS:
            instruction = instruction.replace(tok, "")
        return {
            "instruction": instruction.strip(),
            "input":       example.get("input", ""),
            "output":      output.strip(),
        }

    dataset = dataset.map(strip_foreign_eos)
    logger.info(f"  Stripped foreign EOS tokens")

    # ── Step 2: Quality filter ────────────────────────────────────────────────
    python_keywords = ["def ", "import ", "class ", "return ", "print(", "for ", "if "]

    def quality_filter(example):
        out = example.get("output", "")
        ins = example.get("instruction", "")
        if len(out) < 50 or len(out) > 4000:             return False
        if len(ins) < 10:                                  return False
        if not any(k in out for k in python_keywords):    return False
        return True

    dataset = dataset.filter(quality_filter)
    logger.info(f"  After quality filter:  {len(dataset)}")

    # ── Step 3: Deduplicate by instruction prefix ─────────────────────────────
    seen = set()
    def is_unique(example):
        key = example["instruction"][:100].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    dataset = dataset.filter(is_unique)
    logger.info(f"  After deduplication:   {len(dataset)}")
    return dataset


def prepare_dataset(dataset: Dataset):
    if cfg.max_samples and len(dataset) > cfg.max_samples:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

    dataset = dataset.map(format_prompt)
    split   = dataset.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    logger.info(f"  Train: {len(split['train'])}  |  Val: {len(split['test'])}")
    return split["train"], split["test"]


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────

def load_model():
    logger.info(f"Loading model: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = cfg.model_name,
        max_seq_length = cfg.max_seq_length,
        load_in_4bit   = cfg.load_in_4bit,
        dtype          = torch.bfloat16,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r                          = cfg.lora_r,
        lora_alpha                 = cfg.lora_alpha,
        lora_dropout               = cfg.lora_dropout,
        target_modules             = list(cfg.target_modules),
        bias                       = "none",
        use_gradient_checkpointing = "unsloth",   # saves VRAM, no speed loss
        random_state               = cfg.seed,
    )

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model, tokenizer


# ─────────────────────────────────────────────
# TRAINING
# TRL 1.2: SFTConfig holds ALL args (training + SFT-specific)
# ─────────────────────────────────────────────

def train(model, tokenizer, train_dataset, eval_dataset):
    logger.info("Building SFTConfig...")

    sft_config = SFTConfig(
        # ── output ──────────────────────────────
        output_dir                  = cfg.output_dir,

        # ── schedule ────────────────────────────
        num_train_epochs            = cfg.num_train_epochs,
        per_device_train_batch_size = cfg.per_device_train_batch_size,
        gradient_accumulation_steps = cfg.gradient_accumulation_steps,
        warmup_steps                = cfg.warmup_steps,
        learning_rate               = cfg.learning_rate,
        lr_scheduler_type           = cfg.lr_scheduler_type,
        weight_decay                = cfg.weight_decay,
        optim                       = cfg.optim,

        # ── precision ───────────────────────────
        bf16                        = cfg.bf16,
        fp16                        = cfg.fp16,

        # ── logging / saving ────────────────────
        logging_steps               = cfg.logging_steps,
        save_steps                  = cfg.save_steps,
        save_strategy               = "steps",
        save_total_limit            = cfg.save_total_limit,

        # ── evaluation ──────────────────────────
        eval_strategy               = "steps",
        eval_steps                  = cfg.eval_steps,
        load_best_model_at_end      = cfg.load_best_model_at_end,
        metric_for_best_model       = "eval_loss",
        greater_is_better           = False,

        # ── misc ────────────────────────────────
        seed                        = cfg.seed,
        report_to                   = "none",       # swap to "wandb" if desired
        dataloader_num_workers      = 4,

        # ── SFT-specific (moved from SFTTrainer → SFTConfig in TRL 1.x) ──────
        dataset_text_field          = "text",
        max_length                  = cfg.max_seq_length,

        # ── CRITICAL for Qwen2.5: align EOS token with chat template ─────────
        # Without this TRL 1.2 raises ValueError about <EOS_TOKEN> not in vocab
        eos_token                   = "<|im_end|>",
    )

    trainer = SFTTrainer(
        model            = model,
        processing_class = tokenizer,       # renamed from `tokenizer` in TRL 1.x
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        args             = sft_config,
        callbacks        = [EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu_mem:.1f} GB")

    logger.info("Starting training...")
    stats = trainer.train()

    logger.info("Training complete!")
    logger.info(f"  Runtime:     {stats.metrics['train_runtime']:.0f}s")
    logger.info(f"  Samples/sec: {stats.metrics['train_samples_per_second']:.2f}")
    logger.info(f"  Final loss:  {stats.metrics['train_loss']:.4f}")
    return trainer


# ─────────────────────────────────────────────
# SAVE
# ─────────────────────────────────────────────

def save_model(model, tokenizer):
    lora_path   = os.path.join(cfg.output_dir, "lora_adapter")
    merged_path = os.path.join(cfg.output_dir, "merged_model")

    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"LoRA adapter saved  → {lora_path}")

    logger.info("Merging LoRA weights into base model (bf16)...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model saved  → {merged_path}")


# ─────────────────────────────────────────────
# QUICK INFERENCE TEST
# ─────────────────────────────────────────────

def test_inference(model, tokenizer):
    logger.info("Running quick inference test...")
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
        response = tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        print(f"\n{'='*60}\nPROMPT: {prompt}\n{'='*60}\n{response}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    logger.info("=" * 60)
    logger.info("Python Code Generation Fine-tuning")
    logger.info("TRL 1.2.0 | transformers 5.x | Unsloth 2026.x")
    logger.info("=" * 60)

    raw               = load_and_merge_datasets()
    clean             = clean_dataset(raw)
    train_ds, eval_ds = prepare_dataset(clean)
    model, tokenizer  = load_model()
    trainer           = train(model, tokenizer, train_ds, eval_ds)
    save_model(model, tokenizer)
    test_inference(model, tokenizer)

    logger.info(f"All done! Outputs → {cfg.output_dir}")


if __name__ == "__main__":
    main()