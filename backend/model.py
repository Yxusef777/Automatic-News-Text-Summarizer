"""
Model service for the Automatic News Text Summarizer.

This module loads a Hugging Face Seq2Seq model once at startup and exposes a
`SummarizerModel` class with a `summarize()` method used by the FastAPI routes.
A module-level singleton `summarizer` is created so the model is shared across
all requests without reloading.
"""

import logging
import re

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.config import MAX_INPUT_LENGTH, MAX_OUTPUT_LENGTH, MODEL_NAME

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class SummarizerModel:
    """Wrapper around a Hugging Face Seq2Seq summarization model.

    Attributes
    ----------
    model_name:
        The Hugging Face model identifier used for this instance.
    tokenizer:
        Loaded ``AutoTokenizer`` instance.
    model:
        Loaded ``AutoModelForSeq2SeqLM`` instance.
    _is_loaded:
        ``True`` when the model has been successfully loaded.
    """

    # Regex that matches any Arabic Unicode character (U+0600–U+06FF)
    _ARABIC_PATTERN = re.compile(r"[\u0600-\u06FF]")

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """Load the tokenizer and model from Hugging Face Hub.

        Parameters
        ----------
        model_name:
            Hugging Face model identifier. Defaults to the value defined in
            ``backend.config.MODEL_NAME``.
        """
        self.model_name = model_name
        self._is_loaded = False
        self.tokenizer = None
        self.model = None

        logger.info("Loading model '%s' …", self.model_name)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._is_loaded = True
            logger.info("Model '%s' loaded successfully.", self.model_name)
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to load model '%s': %s", self.model_name, exc)
            raise RuntimeError(
                f"Could not load model '{self.model_name}'. "
                "Ensure the model name is correct and you have internet access."
            ) from exc

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarize(
        self,
        text: str,
        language: str = "auto",
        max_length: int = MAX_OUTPUT_LENGTH,
    ) -> dict:
        """Generate an abstractive summary for *text*.

        Parameters
        ----------
        text:
            The news article to summarise.
        language:
            Target language hint. Use ``"auto"`` (default) to let the method
            detect the language automatically, ``"ar"`` for Arabic, or
            ``"en"`` for English.
        max_length:
            Maximum number of tokens to generate for the summary.

        Returns
        -------
        dict
            A dictionary with keys ``summary``, ``model_used``, and
            ``language``.
        """
        if not self._is_loaded:  # pragma: no cover
            raise RuntimeError("Model is not loaded. Cannot summarize.")

        # ------------------------------------------------------------------
        # Language detection
        # ------------------------------------------------------------------
        detected_language = self._detect_language(text) if language == "auto" else language

        logger.debug(
            "Summarising %d-character text (language=%s).", len(text), detected_language
        )

        try:
            # Tokenise input, truncating to the maximum supported length
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=MAX_INPUT_LENGTH,
                truncation=True,
            )

            # Generate summary tokens
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=4,
                early_stopping=True,
            )

            # Decode the generated token IDs back to a string
            summary = self.tokenizer.decode(
                summary_ids[0], skip_special_tokens=True
            )
        except Exception as exc:  # pragma: no cover
            logger.error("Summarization failed: %s", exc)
            raise RuntimeError(f"Summarization failed: {exc}") from exc

        return {
            "summary": summary,
            "model_used": self.model_name,
            "language": detected_language,
        }

    def get_model_info(self) -> dict:
        """Return basic information about the loaded model.

        Returns
        -------
        dict
            A dictionary with ``model_name`` and ``status`` keys.
        """
        return {
            "model_name": self.model_name,
            "status": "loaded" if self._is_loaded else "not loaded",
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> str:
        """Detect whether *text* is primarily Arabic or English.

        Uses a simple heuristic: if any Arabic Unicode character (U+0600–U+06FF)
        is present in the text, the language is classified as Arabic; otherwise
        it defaults to English.

        Parameters
        ----------
        text:
            Input text whose language is to be detected.

        Returns
        -------
        str
            ``"ar"`` for Arabic, ``"en"`` otherwise.
        """
        if self._ARABIC_PATTERN.search(text):
            return "ar"
        return "en"


# ---------------------------------------------------------------------------
# Module-level singleton — shared across all FastAPI request handlers
# ---------------------------------------------------------------------------
summarizer = SummarizerModel()
