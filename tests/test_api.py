"""
Pytest test suite for the Automatic News Text Summarizer API.

The tests use FastAPI's ``TestClient`` so no live server is required.
The model's ``summarize`` method is mocked so the tests run quickly without
downloading any model weights.

Run with:
    pytest tests/
"""

import sys
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# We inject a mock SummarizerModel singleton into sys.modules *before* any
# backend module is imported so that no real model weights are downloaded.
# ---------------------------------------------------------------------------

_MOCK_SUMMARIZE_RESULT = {
    "summary": "Test summary of the article.",
    "model_used": "csebuetnlp/mT5_multilingual_XLSum",
    "language": "en",
}

_MOCK_MODEL_INFO = {
    "model_name": "csebuetnlp/mT5_multilingual_XLSum",
    "status": "loaded",
}

# Build the mock model module before anything from `backend` is imported.
_mock_model_module = MagicMock()
_mock_summarizer = MagicMock()
_mock_summarizer.summarize.return_value = _MOCK_SUMMARIZE_RESULT
_mock_summarizer.get_model_info.return_value = _MOCK_MODEL_INFO
_mock_model_module.summarizer = _mock_summarizer
_mock_model_module.SummarizerModel = MagicMock(return_value=_mock_summarizer)

# Inject the mock so that `from backend.model import summarizer` in main.py
# picks up our mock instead of trying to load real Hugging Face weights.
sys.modules.setdefault("backend.model", _mock_model_module)

# Now it is safe to import the FastAPI app.
from backend.main import app  # noqa: E402  (import after mock injection)


@pytest.fixture(scope="module")
def client():
    """Return a TestClient backed by the FastAPI app with a mocked model."""
    yield TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_root_returns_200(client: TestClient) -> None:
    """GET / should return HTTP 200 and confirm the API is running."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"


def test_health_returns_200_and_healthy(client: TestClient) -> None:
    """GET /health should return HTTP 200 with a 'healthy' status field."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model" in data


def test_summarize_english_text_returns_200(client: TestClient) -> None:
    """POST /summarize with valid English text should return 200 and a summary."""
    payload = {
        "text": (
            "Scientists have discovered a new species of deep-sea fish in the "
            "Pacific Ocean. The fish, found at depths exceeding 3,000 metres, "
            "displays bioluminescent properties never observed before in any known "
            "vertebrate species. Researchers from three universities collaborated "
            "on the study, which was published in the journal Nature on Monday."
        ),
        "language": "en",
        "max_length": 100,
    }
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert isinstance(data["summary"], str)
    assert len(data["summary"]) > 0
    assert "model_used" in data
    assert "language" in data
    assert "original_length" in data


def test_summarize_empty_text_returns_422(client: TestClient) -> None:
    """POST /summarize with an empty text field should return HTTP 422."""
    payload = {"text": ""}
    response = client.post("/summarize", json=payload)
    assert response.status_code == 422
