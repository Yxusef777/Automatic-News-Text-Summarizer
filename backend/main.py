"""
FastAPI application for the Automatic News Text Summarizer.

Run with:
    uvicorn backend.main:app --reload

Endpoints
---------
GET  /           - Welcome message and API status.
GET  /health     - Health check (reports model status).
POST /summarize  - Generate an abstractive summary for a news article.
"""

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import (
    API_TITLE,
    API_VERSION,
    DEFAULT_LANGUAGE,
    MAX_OUTPUT_LENGTH,
)
from backend.model import summarizer

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=(
        "Bilingual (Arabic + English) abstractive news summarization service "
        "powered by Hugging Face Transformers."
    ),
)

# Allow all origins during development so the HTML frontend served on any
# local port can reach the API without CORS errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SummarizeRequest(BaseModel):
    """Request body for the /summarize endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        description="The news article text to summarise.",
    )
    language: str = Field(
        DEFAULT_LANGUAGE,
        description=(
            "Language of the input text. "
            "Use 'auto' for automatic detection, 'ar' for Arabic, 'en' for English."
        ),
    )
    max_length: int = Field(
        MAX_OUTPUT_LENGTH,
        ge=20,
        le=512,
        description="Maximum number of tokens in the generated summary.",
    )


class SummarizeResponse(BaseModel):
    """Response body returned by the /summarize endpoint."""

    summary: str = Field(..., description="The generated abstractive summary.")
    model_used: str = Field(..., description="Hugging Face model identifier used.")
    language: str = Field(..., description="Detected or specified language code.")
    original_length: int = Field(..., description="Character length of the original text.")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", summary="Root", tags=["General"])
def root() -> dict:
    """Return a welcome message and confirm the API is running."""
    return {
        "message": "Welcome to the Automatic News Text Summarizer API",
        "status": "running",
        "version": API_VERSION,
        "docs": "/docs",
    }


@app.get("/health", summary="Health Check", tags=["General"])
def health() -> dict:
    """Return the current health status including model load state."""
    model_info = summarizer.get_model_info()
    return {
        "status": "healthy",
        "model": model_info,
    }


@app.post(
    "/summarize",
    response_model=SummarizeResponse,
    summary="Summarize a news article",
    tags=["Summarization"],
)
def summarize_text(request: SummarizeRequest) -> SummarizeResponse:
    """Generate an abstractive summary for the provided news article.

    The endpoint accepts text in Arabic or English (or both) and returns a
    concise summary along with metadata about which model was used and what
    language was detected.
    """
    logger.info(
        "Summarize request: language=%s, max_length=%d, text_length=%d",
        request.language,
        request.max_length,
        len(request.text),
    )

    try:
        result = summarizer.summarize(
            text=request.text,
            language=request.language,
            max_length=request.max_length,
        )
    except RuntimeError as exc:
        logger.error("Summarization error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SummarizeResponse(
        summary=result["summary"],
        model_used=result["model_used"],
        language=result["language"],
        original_length=len(request.text),
    )
