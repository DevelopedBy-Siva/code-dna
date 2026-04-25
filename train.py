
import os
import torch
import logging
from dataclasses import dataclass
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from unsloth import FastLanguageModel

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-Coder-14B-Instruct"
    max_seq_length: int = 4096
    load_in_4bit: bool = False

    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: tuple = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    )

    output_dir: str = "./qwen-python-finetuned"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    weight_decay: float = 0.01
    bf16: bool = True
    fp16: bool = False
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    optim: str = "adamw_8bit"
    seed: int = 42

    val_split: float = 0.02
    max_samples: int = None


cfg = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("training.log"),
    ]
)
logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """### Instruction:
{instruction}{input_section}

### Response:
{output}<|endoftext|>"""

INFERENCE_TEMPLATE = """### Instruction:
{instruction}{input_section}

### Response:
"""

def format_prompt(example, for_inference=False):
    """Build one prompt string."""
    input_section = f"\n\n### Input:\n{example['input']}" if example.get("input", "").strip() else ""
    template = INFERENCE_TEMPLATE if for_inference else PROMPT_TEMPLATE
    return template.format(
        instruction=example.get("instruction", "").strip(),
        input_section=input_section,
        output=example.get("output", "").strip(),
    )

def load_and_merge_datasets() -> Dataset:
    """Load and merge all datasets."""
    logger.info("Loading datasets...")
    all_datasets = []

    try:
        ds = load_dataset("Vezora/Tested-22k-Python-Alpaca", split="train")
        logger.info(f"Vezora: {len(ds)} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"Failed to load Vezora: {e}")

    try:
        ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
        logger.info(f"iamtarun: {len(ds)} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"Failed to load iamtarun: {e}")

    try:
        ds = load_dataset("flytech/python-codes-25k", split="train")
        logger.info(f"flytech: {len(ds)} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"Failed to load flytech: {e}")

    try:
        ds = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
        ds = ds.filter(lambda x: x.get("lang", "").lower() == "python")
        ds = ds.map(lambda x: {
            "instruction": x.get("problem", ""),
            "input": "",
            "output": x.get("solution", ""),
        })
        logger.info(f"Magicoder Python: {len(ds)} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"Failed to load Magicoder: {e}")

    try:
        ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
        ds = ds.filter(lambda x: "python" in x.get("output", "").lower())
        logger.info(f"CodeAlpaca Python: {len(ds)} samples")
        all_datasets.append(ds)
    except Exception as e:
        logger.warning(f"Failed to load CodeAlpaca: {e}")

    if not all_datasets:
        raise RuntimeError("No datasets loaded successfully!")

    def normalize(example):
        return {
            "instruction": str(example.get("instruction", "") or ""),
            "input":       str(example.get("input", "") or ""),
            "output":      str(example.get("output", "") or ""),
        }

    normalized = [ds.map(normalize, remove_columns=ds.column_names) for ds in all_datasets]
    combined = concatenate_datasets(normalized)
    logger.info(f"Combined total: {len(combined)} samples")
    return combined


def clean_dataset(dataset: Dataset) -> Dataset:
    """Filter and deduplicate the dataset."""
    logger.info("Cleaning dataset...")

    python_keywords = ["def ", "import ", "class ", "return ", "print(", "for ", "if "]

    def quality_filter(example):
        out = example.get("output", "")
        ins = example.get("instruction", "")
        if len(out) < 50 or len(out) > 4000:
            return False
        if len(ins) < 10:
            return False
        if not any(kw in out for kw in python_keywords):
            return False
        return True

    dataset = dataset.filter(quality_filter)
    logger.info(f"After quality filter: {len(dataset)} samples")

    seen = set()
    def is_unique(example):
        key = example["instruction"][:100].strip().lower()
        if key in seen:
            return False
        seen.add(key)
        return True

    dataset = dataset.filter(is_unique)
    logger.info(f"After deduplication: {len(dataset)} samples")

    return dataset


def prepare_dataset(dataset: Dataset):
    """Prepare train and validation splits."""
    # Keep the dataset manageable if we set a cap.
    if cfg.max_samples and len(dataset) > cfg.max_samples:
        dataset = dataset.shuffle(seed=cfg.seed).select(range(cfg.max_samples))

    # Turn rows into the training prompt format.
    dataset = dataset.map(lambda x: {"text": format_prompt(x)})

    # Split once so training and validation stay separate.
    split = dataset.train_test_split(test_size=cfg.val_split, seed=cfg.seed)
    logger.info(f"Train: {len(split['train'])} | Val: {len(split['test'])} samples")
    return split["train"], split["test"]

def load_model():
    """Load the model and tokenizer."""
    logger.info(f"Loading model: {cfg.model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.model_name,
        max_seq_length=cfg.max_seq_length,
        load_in_4bit=cfg.load_in_4bit,
        dtype=torch.bfloat16,
    )

    logger.info("Applying LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model, tokenizer

def train(model, tokenizer, train_dataset, eval_dataset):
    """Train the model."""
    logger.info("Starting training...")

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=cfg.warmup_steps,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        weight_decay=cfg.weight_decay,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim=cfg.optim,
        seed=cfg.seed,
        report_to="none",
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=cfg.max_seq_length,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu_mem:.1f}GB")

    trainer_stats = trainer.train()

    logger.info(f"Training complete!")
    logger.info(f"  Runtime:       {trainer_stats.metrics['train_runtime']:.0f}s")
    logger.info(f"  Samples/sec:   {trainer_stats.metrics['train_samples_per_second']:.2f}")
    logger.info(f"  Final loss:    {trainer_stats.metrics['train_loss']:.4f}")

    return trainer

def save_model(model, tokenizer, trainer):
    """Save adapters, merged model, and metrics."""
    # Save the adapter first.
    lora_path = os.path.join(cfg.output_dir, "lora_adapter")
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    logger.info(f"LoRA adapter saved to: {lora_path}")

    # Then save a merged copy too.
    merged_path = os.path.join(cfg.output_dir, "merged_model")
    logger.info("Merging LoRA weights into base model...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    logger.info(f"Merged model saved to: {merged_path}")

    metrics_path = os.path.join(cfg.output_dir, "training_metrics.txt")
    with open(metrics_path, "w") as f:
        for k, v in trainer.state.log_history[-1].items():
            f.write(f"{k}: {v}\n")
    logger.info(f"Metrics saved to: {metrics_path}")

def test_inference(model, tokenizer):
    """Run a quick inference test."""
    logger.info("Running inference test...")
    FastLanguageModel.for_inference(model)

    test_prompts = [
        "Write a Python function to flatten a nested list",
        "Write a Python decorator that retries a function up to 3 times on exception",
        "Create a Python context manager for timing code blocks",
    ]

    for prompt in test_prompts:
        example = {"instruction": prompt, "input": ""}
        input_text = format_prompt(example, for_inference=True)
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.2,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\n{'='*60}")
        print(f"PROMPT: {prompt}")
        print(f"{'='*60}")
        print(response)

def main():
    """Run the full training flow."""
    logger.info("=" * 60)
    logger.info("Python Code Generation Fine-tuning")
    logger.info("=" * 60)

    # Start with the data.
    raw_dataset = load_and_merge_datasets()
    clean = clean_dataset(raw_dataset)
    train_dataset, eval_dataset = prepare_dataset(clean)

    # Load the model and tokenizer.
    model, tokenizer = load_model()

    # Train on the prepared split.
    trainer = train(model, tokenizer, train_dataset, eval_dataset)

    # Save what we trained.
    save_model(model, tokenizer, trainer)

    # Run a quick sanity check.
    test_inference(model, tokenizer)

    logger.info("All done! Check ./qwen-python-finetuned for outputs.")


if __name__ == "__main__":
    main()
