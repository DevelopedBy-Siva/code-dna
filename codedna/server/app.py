"""FastAPI application stubs for the CodeDNA server."""

from __future__ import annotations

from fastapi import FastAPI


def create_app() -> FastAPI:
    """Create the FastAPI application for CodeDNA."""

    return FastAPI(title="CodeDNA")
