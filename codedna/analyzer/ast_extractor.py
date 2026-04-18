"""Tree-sitter extraction helpers for CodeDNA analyzer."""

from __future__ import annotations

import ast
import re
import warnings
from pathlib import Path

from tree_sitter import Node
from tree_sitter_languages import get_parser


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".ts": "typescript",
    ".js": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
}

FUNCTION_NODE_TYPES = {
    "function_definition",
    "function_declaration",
    "method_definition",
    "method_declaration",
}

BODY_NODE_TYPES = {
    "block",
    "statement_block",
    "declaration_list",
    "body",
}

DOCSTRING_PATTERNS = [
    re.compile(r'^\s*(?:"""|\'\'\')(?P<doc>.*?)(?:"""|\'\'\')', re.DOTALL),
    re.compile(r'^\s*"(?P<doc>[^"\n]+)"\s*$', re.MULTILINE),
    re.compile(r"^\s*//\s*(?P<doc>[^\n]+)", re.MULTILINE),
    re.compile(r"^\s*/\*\*?(?P<doc>.*?)\*/", re.DOTALL),
]

GENERIC_FUNCTION_PATTERN = re.compile(
    r"(?P<signature>(?:export\s+)?(?:async\s+)?function\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)|"
    r"func\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)|"
    r"(?:pub\s+)?fn\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\)|"
    r"(?:public\s+|private\s+|protected\s+)?[A-Za-z0-9_<>\[\]]+\s+[A-Za-z_][A-Za-z0-9_]*\s*\([^\)]*\))"
)


def format_prompt(signature: str, docstring: str | None) -> str:
    """Format a prompt for a training pair."""

    normalized_signature = signature.rstrip().rstrip(":")
    if docstring:
        summary = " ".join(docstring.strip().split())
        return f"# {summary}\n{normalized_signature}:"
    return f"# Complete the function\n{normalized_signature}:"


def extract_pairs_from_file(filepath: Path) -> list[dict]:
    """Extract prompt and completion pairs from a source file with tree-sitter."""

    language = LANGUAGE_BY_SUFFIX.get(filepath.suffix)
    if language is None:
        return []

    source = filepath.read_text(encoding="utf-8", errors="ignore")
    if filepath.suffix == ".py":
        return _extract_python_pairs(filepath, source)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            parser = get_parser(language)
        tree = parser.parse(source.encode("utf-8"))
        lines = source.splitlines()

        pairs: list[dict] = []
        for node in _walk(tree.root_node):
            if node.type not in FUNCTION_NODE_TYPES:
                continue
            pair = _build_pair(node, filepath, source, lines)
            if pair is not None:
                pairs.append(pair)
        return pairs
    except Exception:
        if filepath.suffix == ".py":
            return _extract_python_pairs(filepath, source)
        return _extract_generic_pairs(filepath, source)


def _walk(node: Node) -> list[Node]:
    """Return a flattened node list for traversal."""

    nodes = [node]
    for child in node.children:
        nodes.extend(_walk(child))
    return nodes


def _build_pair(node: Node, filepath: Path, source: str, lines: list[str]) -> dict | None:
    """Create a prompt and completion pair from a function node."""

    body_node = _find_body_node(node)
    if body_node is None:
        return None

    signature = source[node.start_byte : body_node.start_byte].strip().rstrip("{").strip()
    completion = source[body_node.start_byte : node.end_byte].strip()
    if not signature or not completion:
        return None

    body_line_count = len([line for line in completion.splitlines() if line.strip()])
    if body_line_count < 3 or body_line_count > 80:
        return None

    docstring = _extract_docstring(body_node, source, lines)
    return {
        "prompt": format_prompt(signature, docstring),
        "completion": completion,
        "file": str(filepath),
        "start_line": node.start_point[0] + 1,
        "end_line": node.end_point[0] + 1,
        "signature": signature,
        "docstring": docstring,
    }


def _find_body_node(node: Node) -> Node | None:
    """Find the node that represents the body of a function."""

    for child in reversed(node.children):
        if child.type in BODY_NODE_TYPES:
            return child
    for child in reversed(node.children):
        if child.is_named:
            return child
    return None


def _extract_docstring(body_node: Node, source: str, lines: list[str]) -> str | None:
    """Extract a likely docstring or leading comment from a function body."""

    body_text = source[body_node.start_byte : body_node.end_byte]
    for pattern in DOCSTRING_PATTERNS:
        match = pattern.search(body_text)
        if match:
            return " ".join(match.group("doc").replace("*", " ").split())

    first_line_index = body_node.start_point[0]
    if 0 <= first_line_index < len(lines):
        candidate = lines[first_line_index].strip()
        if candidate.startswith(("'''", '"""', "//", "/*")):
            return candidate.strip("/'* ")
    return None


def _extract_python_pairs(filepath: Path, source: str) -> list[dict]:
    """Fallback Python extractor using the standard library AST module."""

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    source_lines = source.splitlines()
    pairs: list[dict] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue

        start_line = getattr(node, "lineno", 0)
        end_line = getattr(node, "end_lineno", start_line)
        docstring = ast.get_docstring(node)
        body_start_line = _python_body_start_line(node)
        signature = _python_signature(source_lines, node, body_start_line)
        completion = _python_body_text(source_lines, body_start_line, end_line)
        body_line_count = len([line for line in completion.splitlines() if line.strip()])
        if body_line_count < 3 or body_line_count > 80:
            continue
        if completion.lstrip().startswith("def "):
            continue

        pairs.append(
            {
                "prompt": format_prompt(signature, docstring),
                "completion": completion,
                "file": str(filepath),
                "start_line": start_line,
                "end_line": end_line,
                "signature": signature,
                "docstring": docstring,
            }
        )
    return pairs


def _python_body_start_line(node: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    """Return the first source line that belongs to the executable function body."""

    if not node.body:
        return getattr(node, "end_lineno", getattr(node, "lineno", 1))

    first_statement = node.body[0]
    if (
        isinstance(first_statement, ast.Expr)
        and isinstance(first_statement.value, ast.Constant)
        and isinstance(first_statement.value.value, str)
    ):
        if len(node.body) == 1:
            return getattr(first_statement, "end_lineno", getattr(node, "lineno", 1))
        first_statement = node.body[1]
    return getattr(first_statement, "lineno", getattr(node, "lineno", 1))


def _python_signature(
    source_lines: list[str],
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    body_start_line: int,
) -> str:
    """Extract the function signature text without decorators or body."""

    _ = (source_lines, body_start_line)
    async_prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
    args_repr = ast.unparse(node.args)
    return_repr = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{async_prefix}def {node.name}({args_repr}){return_repr}:"


def _python_body_text(source_lines: list[str], body_start_line: int, end_line: int) -> str:
    """Extract only the function body text from the source lines."""

    if body_start_line <= 0 or end_line <= 0 or body_start_line > end_line:
        return ""
    return "\n".join(source_lines[body_start_line - 1 : end_line]).rstrip()


def _extract_generic_pairs(filepath: Path, source: str) -> list[dict]:
    """Fallback extractor for non-Python languages using heuristic matching."""

    lines = source.splitlines()
    pairs: list[dict] = []
    for index, line in enumerate(lines, start=1):
        if not GENERIC_FUNCTION_PATTERN.search(line):
            continue
        block = [line]
        for following in lines[index : index + 40]:
            block.append(following)
            if following.strip() in {"}", "};", "end"}:
                break

        completion = "\n".join(block).strip()
        body_line_count = len([entry for entry in completion.splitlines() if entry.strip()])
        if body_line_count < 3 or body_line_count > 80:
            continue

        signature = line.strip().rstrip("{").strip()
        pairs.append(
            {
                "prompt": format_prompt(signature, None),
                "completion": completion,
                "file": str(filepath),
                "start_line": index,
                "end_line": min(index + body_line_count - 1, len(lines)),
                "signature": signature,
                "docstring": None,
            }
        )
    return pairs
