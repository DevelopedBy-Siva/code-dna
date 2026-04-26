import os
import re
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_LOG = BASE_DIR / "benchmark.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BENCHMARK_LOG),
    ]
)
logger = logging.getLogger(__name__)

RESULTS_DIR = BASE_DIR / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

PROMPT_TEMPLATE = """### Instruction:
Complete the following Python function. Only return the completed function body, no explanation.

{prompt}

### Response:
"""

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-14B-Instruct"


def load_model(model_path: str, tokenizer_path: Optional[str] = None):
    logger.info(f"Loading model: {model_path}")
    tokenizer_source = tokenizer_path or model_path

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            trust_remote_code=True,
        )
    except Exception as e:
        if tokenizer_path:
            raise

        logger.warning(
            "Failed to load tokenizer from %s (%s). Falling back to base tokenizer: %s",
            model_path,
            e,
            DEFAULT_TOKENIZER,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            DEFAULT_TOKENIZER,
            trust_remote_code=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

def extract_code(response: str, prompt: str) -> str:
    """Clean the model output into code."""
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    elif "```" in response:
        response = response.split("```")[1].split("```")[0]

    fn_match = re.search(r"def\s+\w+\s*\(", response)
    if fn_match:
        response = response[fn_match.start():]

    lines = response.splitlines()
    clean_lines = []
    for line in lines:
        if clean_lines and re.match(r"^[a-zA-Z]", line) and not line.startswith(" "):
            break
        clean_lines.append(line)

    return "\n".join(clean_lines).strip()

def generate_solution(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.2,
    num_samples: int = 1,
) -> list[str]:
    """Generate one or more solutions."""
    input_text = PROMPT_TEMPLATE.format(prompt=prompt)
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    solutions = []
    with torch.no_grad():
        for _ in range(num_samples):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                do_sample=(temperature > 0),
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
            code = extract_code(response, prompt)
            full_solution = prompt + "\n" + code
            solutions.append(full_solution)

    return solutions

def run_tests_safely(solution: str, test_code: str, entry_point: str) -> dict:
    """Run the tests for one solution."""
    result = {"passed": False, "error": None}
    exec_globals = {}

    try:
        exec(solution, exec_globals)

        if entry_point not in exec_globals:
            result["error"] = f"Function '{entry_point}' not found in output"
            return result

        exec(test_code, exec_globals)
        exec(f"check({entry_point})", exec_globals)
        result["passed"] = True

    except AssertionError as e:
        result["error"] = f"AssertionError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"

    return result

def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k."""
    if n - c < k:
        return 1.0
    from math import comb
    return 1.0 - comb(n - c, k) / comb(n, k)

def run_benchmark(
    model_path: str,
    label: str,
    tokenizer_path: Optional[str] = None,
    num_samples: int = 1,
    temperature: float = 0.2,
    max_problems: Optional[int] = None,
    max_new_tokens: int = 512,
):
    """Run the benchmark and save results."""
    logger.info(f"{'='*60}")
    logger.info(f"Running HumanEval benchmark | label={label}")
    logger.info(f"{'='*60}")

    # Load the benchmark set first.
    logger.info("Loading HumanEval dataset...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    logger.info(f"Problems to evaluate: {len(dataset)}")

    # Then load the model we want to score.
    model, tokenizer = load_model(model_path, tokenizer_path=tokenizer_path)

    results = []
    start_time = time.time()

    for i, problem in enumerate(tqdm(dataset, desc="Evaluating")):
        task_id     = problem["task_id"]
        prompt      = problem["prompt"]
        test_code   = problem["test"]
        entry_point = problem["entry_point"]

        solutions = generate_solution(
            model, tokenizer, prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_samples=num_samples,
        )

        problem_results = []
        for sol in solutions:
            test_result = run_tests_safely(sol, test_code, entry_point)
            problem_results.append(test_result)

        num_correct = sum(1 for r in problem_results if r["passed"])
        p1 = pass_at_k(num_samples, num_correct, 1)

        results.append({
            "task_id":      task_id,
            "num_correct":  num_correct,
            "num_samples":  num_samples,
            "pass@1":       p1,
            "errors":       [r["error"] for r in problem_results if r["error"]],
        })

        if (i + 1) % 20 == 0:
            so_far = sum(r["pass@1"] for r in results) / len(results)
            logger.info(f"Progress {i+1}/{len(dataset)} | pass@1 so far: {so_far:.3f}")

    elapsed = time.time() - start_time
    total = len(results)
    pass1 = sum(r["pass@1"] for r in results) / total

    pass10 = None
    if num_samples >= 10:
        pass10 = sum(
            pass_at_k(r["num_samples"], r["num_correct"], 10)
            for r in results
        ) / total

    failed = [r for r in results if r["pass@1"] == 0]

    summary = {
        "label":          label,
        "model_path":     model_path,
        "timestamp":      datetime.now().isoformat(),
        "num_problems":   total,
        "num_samples":    num_samples,
        "temperature":    temperature,
        "pass@1":         round(pass1 * 100, 2),
        "pass@10":        round(pass10 * 100, 2) if pass10 else None,
        "num_passed":     sum(1 for r in results if r["pass@1"] > 0),
        "num_failed":     len(failed),
        "elapsed_sec":    round(elapsed, 1),
        "failed_tasks":   [r["task_id"] for r in failed],
        "per_problem":    results,
    }

    # Save one JSON report with the summary and per-problem details.
    out_path = RESULTS_DIR / f"{label}_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"RESULTS - {label.upper()}")
    logger.info(f"{'='*60}")
    logger.info(f"  pass@1:        {summary['pass@1']}%")
    if pass10:
        logger.info(f"  pass@10:       {summary['pass@10']}%")
    logger.info(f"  Passed:        {summary['num_passed']} / {total}")
    logger.info(f"  Time:          {elapsed/60:.1f} min")
    logger.info(f"  Results saved: {out_path}")

    return summary

def compare_results():
    """Compare saved benchmark results."""
    baseline_path  = RESULTS_DIR / "baseline_results.json"
    finetuned_path = RESULTS_DIR / "finetuned_results.json"

    if not baseline_path.exists() or not finetuned_path.exists():
        logger.error("Run both baseline and finetuned benchmarks first.")
        sys.exit(1)

    with open(baseline_path)  as f: baseline  = json.load(f)
    with open(finetuned_path) as f: finetuned = json.load(f)

    delta_pass1 = finetuned["pass@1"] - baseline["pass@1"]

    print("\n" + "="*60)
    print("HUMANEVAL BENCHMARK COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'Baseline':>12} {'Fine-tuned':>12} {'Delta':>10}")
    print("-"*60)
    print(f"{'pass@1 (%)':<20} {baseline['pass@1']:>12.2f} {finetuned['pass@1']:>12.2f} {delta_pass1:>+10.2f}")

    if baseline.get("pass@10") and finetuned.get("pass@10"):
        delta10 = finetuned["pass@10"] - baseline["pass@10"]
        print(f"{'pass@10 (%)':<20} {baseline['pass@10']:>12.2f} {finetuned['pass@10']:>12.2f} {delta10:>+10.2f}")

    print(f"{'Problems passed':<20} {baseline['num_passed']:>12} {finetuned['num_passed']:>12} {finetuned['num_passed']-baseline['num_passed']:>+10}")
    print(f"{'Total problems':<20} {baseline['num_problems']:>12} {finetuned['num_problems']:>12}")
    print("="*60)

    # Show what improved and what regressed.
    baseline_failed  = set(baseline["failed_tasks"])
    finetuned_failed = set(finetuned["failed_tasks"])
    fixed  = baseline_failed - finetuned_failed
    broken = finetuned_failed - baseline_failed

    print(f"\nProblems FIXED by fine-tuning:  {len(fixed)}")
    for t in sorted(fixed):
        print(f"   {t}")

    print(f"\nProblems BROKEN by fine-tuning: {len(broken)}")
    for t in sorted(broken):
        print(f"   {t}")

    print("\n" + "="*60)
    print("RESUME HEADLINE:")
    print(f'  "Fine-tuned Qwen2.5-Coder-14B on 75K Python samples,')
    print(f'   improving HumanEval pass@1 from {baseline["pass@1"]}% -> {finetuned["pass@1"]}%')
    print(f'   (+{delta_pass1:.1f}%)"')
    print("="*60 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="HumanEval Benchmarking for Fine-tuned LLMs")

    parser.add_argument("--model",       type=str,  default=None,       help="Model path or HuggingFace name")
    parser.add_argument("--tokenizer",   type=str,  default=None,       help="Optional tokenizer path or HuggingFace name")
    parser.add_argument("--label",       type=str,  default="baseline",  help="Label: 'baseline' or 'finetuned'")
    parser.add_argument("--samples",     type=int,  default=1,           help="Samples per problem (1=pass@1, 10=pass@1+pass@10)")
    parser.add_argument("--temperature", type=float,default=0.2,         help="Generation temperature")
    parser.add_argument("--max",         type=int,  default=None,        help="Max problems to eval (None=all 164)")
    parser.add_argument("--compare",     action="store_true",            help="Compare baseline vs finetuned results")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.compare:
        compare_results()
        return

    # For a normal run, we need a model path.
    if not args.model:
        print("Error: --model is required unless using --compare")
        print("\nExamples:")
        print("  python benchmark.py --model Qwen/Qwen2.5-Coder-14B-Instruct --label baseline")
        print(f"  python benchmark.py --model {BASE_DIR / 'qwen-python-finetuned' / 'merged_model'} --label finetuned")
        print("  python benchmark.py --compare")
        sys.exit(1)

    run_benchmark(
        model_path=args.model,
        label=args.label,
        tokenizer_path=args.tokenizer,
        num_samples=args.samples,
        temperature=args.temperature,
        max_problems=args.max,
    )


if __name__ == "__main__":
    main()
