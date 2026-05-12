"""
token_counter.py — Token Usage Tracking
=========================================
WHY THIS EXISTS:
  LLM APIs charge per token. Without tracking, you can accidentally
  send thousands of requests and incur unexpected costs.
  This module tracks input + output tokens per request and logs them.

PRODUCTION CONSIDERATIONS:
  - Aggregate token counts per user for billing/quota enforcement.
  - Set hard limits: refuse requests that would exceed a budget.
  - Log to a database for trend analysis and cost forecasting.
  - Use tiktoken for accurate pre-call token estimation.
"""

import tiktoken
from utils.logger import get_logger

logger = get_logger(__name__)

# Cache encoder instances — encoding is expensive to initialize
_ENCODER_CACHE: dict[str, tiktoken.Encoding] = {}


def get_encoder(model: str = "gemini-2.0-flash") -> tiktoken.Encoding:
    """
    Returns a cached tiktoken encoder for the given model.

    NOTE: Gemini models don't have a tiktoken encoder. We use cl100k_base
    (GPT-4's tokenizer) as an approximation — it gives counts within ~10-15%
    of Gemini's actual token counts, which is sufficient for cost estimation.

    Args:
        model: Model name (OpenAI or Gemini).

    Returns:
        tiktoken Encoding instance.
    """
    if model not in _ENCODER_CACHE:
        try:
            _ENCODER_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Gemini models and unknown models fall back to cl100k_base
            # (same tokenizer as GPT-4) — good enough for estimation
            _ENCODER_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _ENCODER_CACHE[model]


def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """
    Counts the number of tokens in a text string.

    WHY?
      Prevents sending prompts that exceed model context windows,
      which would cause API errors or truncated responses.

    Args:
        text: The text to count tokens for.
        model: The model to use for tokenization.

    Returns:
        Token count as integer.
    """
    encoder = get_encoder(model)
    return len(encoder.encode(text))


def estimate_prompt_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4o-mini",
) -> float:
    """
    Estimates the cost of a single LLM call in USD.

    WHY?
      Gives developers immediate feedback on the cost of each request
      during development, preventing billing surprises.

    Args:
        input_tokens: Number of input/prompt tokens.
        output_tokens: Number of generated output tokens.
        model: Model name for pricing lookup.

    Returns:
        Estimated cost in USD.
    """
    # Prices per 1M tokens (as of 2024 — update periodically)
    pricing = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4o": {"input": 5.00, "output": 15.00},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    rates = pricing.get(model, pricing["gpt-4o-mini"])
    cost = (input_tokens / 1_000_000 * rates["input"]) + (
        output_tokens / 1_000_000 * rates["output"]
    )
    logger.debug(
        f"Token usage — input: {input_tokens}, output: {output_tokens}, "
        f"estimated cost: ${cost:.6f}"
    )
    return cost
