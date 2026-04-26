"""Prompt templates for each CLI mode."""


def generate_prompt(instruction: str) -> str:
    return (
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response:\n"
    )


def explain_prompt(code: str) -> str:
    return (
        f"### Instruction:\n"
        f"Explain what the following Python code does, step by step.\n\n"
        f"### Input:\n{code.strip()}\n\n"
        f"### Response:\n"
    )


def review_prompt(code: str) -> str:
    return (
        f"### Instruction:\n"
        f"Review the following Python code. Identify bugs, issues, and suggest improvements. "
        f"Be specific and actionable.\n\n"
        f"### Input:\n{code.strip()}\n\n"
        f"### Response:\n"
    )


def fix_prompt(code: str, error: str) -> str:
    return (
        f"### Instruction:\n"
        f"Fix the following Python code that produces this error: {error.strip()}\n\n"
        f"### Input:\n{code.strip()}\n\n"
        f"### Response:\n"
    )


def docstring_prompt(code: str) -> str:
    return (
        f"### Instruction:\n"
        f"Add a detailed docstring to the following Python function.\n\n"
        f"### Input:\n{code.strip()}\n\n"
        f"### Response:\n"
    )
