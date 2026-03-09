"""
Configuration constants for the Automatic News Text Summarizer.

Edit this file to switch model checkpoints or adjust generation parameters.
"""

# ---------------------------------------------------------------------------
# Model checkpoints
# ---------------------------------------------------------------------------

# Primary multilingual model fine-tuned on XL-Sum (44 languages incl. AR & EN)
MODEL_NAME: str = "csebuetnlp/mT5_multilingual_XLSum"

# Arabic-optimised BART-based model (139M parameters).
# To use this instead of the default, set MODEL_NAME = ARABIC_MODEL_NAME
# in your environment or replace the value above.
ARABIC_MODEL_NAME: str = "moussaKam/AraBART"

# ---------------------------------------------------------------------------
# Tokenisation / generation limits
# ---------------------------------------------------------------------------

# Maximum number of tokens accepted from the input article
MAX_INPUT_LENGTH: int = 1024

# Maximum number of tokens to generate for the summary
MAX_OUTPUT_LENGTH: int = 150

# ---------------------------------------------------------------------------
# API defaults
# ---------------------------------------------------------------------------

# Language to assume when the caller does not specify one.
# Accepted values: "auto", "ar", "en"
DEFAULT_LANGUAGE: str = "auto"

# Human-readable title shown in the OpenAPI docs
API_TITLE: str = "Automatic News Text Summarizer API"

# Semantic version exposed in the OpenAPI schema
API_VERSION: str = "1.0.0"
