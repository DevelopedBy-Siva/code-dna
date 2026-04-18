"""Tree-sitter extraction helpers for CodeDNA analyzer."""

from __future__ import annotations

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


def format_prompt(signature: str, docstring: str | None) -> str:
    """Format a prompt for a training pair."""

    if docstring:
        summary = " ".join(docstring.strip().split())
        return f"# {summary}\n{signature}:"
    return f"# Complete the function\n{signature}:"


def extract_pairs_from_file(filepath: Path) -> list[dict]:
    """Extract prompt and completion pairs from a source file with tree-sitter."""

    language = LANGUAGE_BY_SUFFIX.get(filepath.suffix)
    if language is None:
        return []

    source = filepath.read_text(encoding="utf-8", errors="ignore")
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
