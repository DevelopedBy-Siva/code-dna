"""FastAPI application for the CodeDNA local server."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from functools import partial
import re
from typing import Literal, TypedDict
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from codedna.server.inference import generate, load_model
from codedna.server.playground import render_playground


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """OpenAI-style chat message payload."""

    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """OpenAI-style chat completion request."""

    model: str = "codedna-local"
    messages: list[ChatMessage]
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    stop: list[str] | None = None


# ---------------------------------------------------------------------------
# Typed state so the model/tokenizer are never stored as bare `object`
# ---------------------------------------------------------------------------

class _ServerState(TypedDict, total=False):
    model_loaded: bool
    checkpoint_path: str | None
    model: object
    tokenizer: object
    load_error: str


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(checkpoint_path: str | None = None) -> FastAPI:
    """Create the FastAPI app and load the model once at startup."""

    state: _ServerState = {"model_loaded": False, "checkpoint_path": checkpoint_path}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            model, tokenizer = load_model(checkpoint_path)
            state["model"] = model
            state["tokenizer"] = tokenizer
            state["model_loaded"] = True
        except Exception as exc:
            state["load_error"] = str(exc)
            state["model_loaded"] = False
        app.state.codedna_state = state
        yield

    app = FastAPI(title="CodeDNA Local Server", lifespan=lifespan)

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    @app.get("/health")
    def health() -> dict:
        response: dict = {
            "status": "ok",
            "model_loaded": bool(state.get("model_loaded", False)),
        }
        if state.get("load_error"):
            response["load_error"] = str(state["load_error"])
        return response

    @app.get("/", response_class=HTMLResponse)
    def playground() -> str:
        return render_playground()

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"data": [{"id": "codedna-local", "object": "model"}]}

    # FIX: this was a plain `def`, which blocks FastAPI's event loop for the entire
    # duration of token generation (can be 5–30 s).  Every other request — including
    # /health — is frozen.  We make the route `async` and offload the CPU-bound
    # generate() call to the default ThreadPoolExecutor via run_in_executor so the
    # event loop stays free while the model runs.
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest) -> dict:
        if not state.get("model_loaded"):
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded: {state.get('load_error', 'unknown error')}",
            )

        prompt = _format_messages_to_prompt(request.messages)

        loop = asyncio.get_event_loop()
        # partial() pre-fills all keyword arguments so run_in_executor can call
        # generate() with no extra arguments (it only accepts a zero-arg callable).
        completion = await loop.run_in_executor(
            None,
            partial(
                generate,
                state["model"],
                state["tokenizer"],
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
            ),
        )

        completion = _clean_completion(completion)

        return {
            "id": f"chatcmpl-{uuid4().hex}",
            "object": "chat.completion",
            "model": "codedna-local",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": completion},
                    "finish_reason": "stop",
                }
            ],
        }

    return app


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Format using the same template used during fine-tuning."""
    parts = []
    for message in messages:
        if message.role == "system":
            parts.append(f"<|system|>{message.content}</s>")
        elif message.role == "user":
            parts.append(f"<|user|>{message.content}</s>")
        elif message.role == "assistant":
            parts.append(f"<|assistant|>{message.content}</s>")
    parts.append("<|assistant|>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Response cleaning (unchanged from original)
# ---------------------------------------------------------------------------

_DISCLAIMER_PATTERNS = (
    "for educational purposes only",
    'provided "as is" without warranty',
    "consult a professional developer",
    "consult a professional developer or security expert",
    "please use this code at your own risk",
    "in no event will the author be liable",
    "the entire risk as to the quality and performance",
    "please test and debug your code before using it",
    "this code assumes that the requests are coming from a single source",
    "does not handle any persistence or distributed locking",
    "please replace `max_requests` and `interval_seconds`",
)


def _clean_completion(content: str) -> str:
    text = content.strip()
    if not text:
        return text
    text = re.sub(r"^\s*assistant:\s*", "", text, flags=re.IGNORECASE)
    text = _truncate_on_role_marker(text)
    text = _normalize_inline_code_dump(text)
    text = _strip_flattened_code_lines(text)
    text = _dedupe_code_fences(text)
    text = _dedupe_paragraphs(text)
    text = _trim_repeated_tail(text)
    text = _trim_dangling_paragraph(text)
    text = _format_code_answer(text)
    return text.strip()


def _truncate_on_role_marker(text: str) -> str:
    markers = ("\nuser:", "\nsystem:", "\nassistant:")
    end = len(text)
    lowered = text.lower()
    for marker in markers:
        index = lowered.find(marker)
        if index != -1:
            end = min(end, index)
    return text[:end].strip()


def _dedupe_code_fences(text: str) -> str:
    pattern = re.compile(r"```([\w+-]*)\n([\s\S]*?)```")
    seen: set[tuple[str, str]] = set()
    pieces: list[str] = []
    last = 0
    for match in pattern.finditer(text):
        pieces.append(text[last : match.start()])
        language = (match.group(1) or "").strip().lower()
        code = match.group(2).strip()
        key = (language, code)
        if key not in seen:
            pieces.append(match.group(0))
            seen.add(key)
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def _dedupe_paragraphs(text: str) -> str:
    parts = re.split(r"(\n\s*\n)", text)
    seen: set[str] = set()
    cleaned: list[str] = []
    for part in parts:
        if not part or re.fullmatch(r"\n\s*\n", part):
            if cleaned and not re.fullmatch(r"\n\s*\n", cleaned[-1]):
                cleaned.append("\n\n")
            continue
        normalized = " ".join(part.split()).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if any(pattern in lowered for pattern in _DISCLAIMER_PATTERNS):
            continue
        if len(normalized) > 80 and normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(part.strip())
    result = "".join(cleaned).strip()
    return re.sub(r"\n{3,}", "\n\n", result)


def _normalize_inline_code_dump(text: str) -> str:
    if "```" in text:
        return text
    marker = "python\n"
    start = text.find(marker)
    if start == -1:
        return text
    end_markers = (
        "\n\nIn this code",
        "\n\nYou can use",
        "\n\nPlease note",
        "\n\nThis code",
        "\n\nAlso,",
        "\n\nFinally,",
        "\n\nRemember to",
    )
    end = len(text)
    for candidate in end_markers:
        index = text.find(candidate, start + len(marker))
        if index != -1:
            end = min(end, index)
    flat_duplicate = re.search(
        r"\n\s*import\s+\w+[^\n]*\bclass\b[^\n]*\bdef\b[^\n]*\breturn\b[^\n]*",
        text[start + len(marker) :],
    )
    if flat_duplicate:
        duplicate_start = start + len(marker) + flat_duplicate.start()
        end = min(end, duplicate_start)
    code = text[start + len(marker) : end].strip()
    if not code or "class " not in code and "def " not in code and "import " not in code:
        return text
    wrapped = f"```python\n{code}\n```"
    return text[:start] + wrapped + text[end:]


def _trim_repeated_tail(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(paragraphs) < 2:
        return text
    for index in range(1, len(paragraphs)):
        current = paragraphs[index]
        for previous in paragraphs[:index]:
            if current == previous:
                return "\n\n".join(paragraphs[:index]).strip()
            if len(current) > 40 and previous.startswith(current):
                return "\n\n".join(paragraphs[:index]).strip()
            if len(previous) > 40 and current.startswith(previous):
                return "\n\n".join(paragraphs[:index]).strip()
    return text


def _trim_dangling_paragraph(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return text
    last = paragraphs[-1]
    lowered = last.lower()
    looks_incomplete = last[-1:] not in {".", "!", "?", "`", ")", '"'}
    filler_prefixes = (
        "please note",
        "please replace",
        "this code assumes",
        "also, please note",
    )
    if looks_incomplete and any(lowered.startswith(prefix) for prefix in filler_prefixes):
        paragraphs = paragraphs[:-1]
    return "\n\n".join(paragraphs).strip()


def _strip_flattened_code_lines(text: str) -> str:
    lines = text.splitlines()
    cleaned: list[str] = []
    for line in lines:
        normalized = " ".join(line.split())
        looks_like_flat_code = (
            len(normalized) > 180
            and " class " in f" {normalized} "
            and " def " in f" {normalized} "
            and ("import " in normalized or "return " in normalized)
        )
        if looks_like_flat_code:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _format_code_answer(text: str) -> str:
    blocks = list(re.finditer(r"```([\w+-]*)\n([\s\S]*?)```", text))
    if not blocks:
        return text
    first = blocks[0]
    language = (first.group(1) or "").strip() or "code"
    code = first.group(2).strip()
    prose = (text[: first.start()] + "\n\n" + text[first.end() :]).strip()
    explanation = _short_explanation(prose)
    parts: list[str] = []
    if explanation:
        parts.append(explanation)
    parts.append(f"```{language}\n{code}\n```")
    return "\n\n".join(parts).strip()


def _short_explanation(text: str) -> str:
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if not paragraphs:
        return ""
    filtered: list[str] = []
    skip_prefixes = (
        "here is ",
        "below is ",
        "the code below",
        "please note",
        "also,",
        "finally,",
        "remember to",
        "this code is written in",
        "you can modify",
    )
    for paragraph in paragraphs:
        lowered = paragraph.lower()
        if any(lowered.startswith(prefix) for prefix in skip_prefixes):
            continue
        if any(pattern in lowered for pattern in _DISCLAIMER_PATTERNS):
            continue
        filtered.append(paragraph)
    chosen = filtered or paragraphs[:1]
    compact = chosen[0]
    compact = re.sub(r"\s+", " ", compact).strip()
    sentence_match = re.match(r"^(.+?[.!?])(?:\s|$)", compact)
    if sentence_match:
        compact = sentence_match.group(1).strip()
    if len(compact) > 180:
        compact = compact[:177].rsplit(" ", 1)[0] + "..."
    return compact