"""HumanEval runner for CodeDNA."""

from __future__ import annotations

import tempfile

import torch


def run_humaneval(model: object, tokenizer: object, k: int = 1) -> float:
    """Run the HumanEval benchmark."""

    try:
        from human_eval.data import read_problems
        from human_eval.execution import check_correctness
    except ImportError:
        return 0.0

    problems = read_problems()
    if not problems:
        return 0.0

    device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    results: list[bool] = []
    for _, problem in list(problems.items()):
        passed = False
        for _ in range(k):
            inputs = tokenizer(problem["prompt"], return_tensors="pt", truncation=True, max_length=2048)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.2,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            completion = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
            candidate = problem["prompt"] + completion
            with tempfile.TemporaryDirectory() as tempdir:
                outcome = check_correctness(problem, candidate, timeout=3.0, tmp_dir=tempdir)
            if outcome["passed"]:
                passed = True
                break
        results.append(passed)

    return round((sum(results) / len(results)) * 100, 1)
