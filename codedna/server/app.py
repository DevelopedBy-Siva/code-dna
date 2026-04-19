"""FastAPI application for the CodeDNA local server."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from codedna.server.inference import generate, load_model
from codedna.server.playground import render_playground


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


def create_app(checkpoint_path: str | None = None) -> FastAPI:
    """Create the FastAPI app and load the model once at startup."""

    state: dict[str, object | bool | str] = {"model_loaded": False, "checkpoint_path": checkpoint_path}

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

    @app.get("/health")
    def health() -> dict:
        response = {"status": "ok", "model_loaded": bool(state.get("model_loaded", False))}
        if state.get("load_error"):
            response["load_error"] = str(state["load_error"])
        return response

    @app.get("/", response_class=HTMLResponse)
    def playground() -> str:
        return render_playground()

    @app.get("/v1/models")
    def list_models() -> dict:
        return {"data": [{"id": "codedna-local", "object": "model"}]}

    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest) -> dict:
        if not state.get("model_loaded"):
            raise HTTPException(status_code=503, detail=f"Model not loaded: {state.get('load_error', 'unknown error')}")
        prompt = _format_messages_to_prompt(request.messages)
        completion = generate(
            state["model"],
            state["tokenizer"],
            prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=request.stop,
        )
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


def _format_messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Flatten chat messages into a single prompt string."""

    parts = []
    for message in messages:
        parts.append(f"{message.role}: {message.content}")
    parts.append("assistant:")
    return "\n".join(parts)
