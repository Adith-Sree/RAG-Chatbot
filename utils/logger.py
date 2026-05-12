"""
logger.py — Structured Logging Module
=======================================
WHY THIS EXISTS:
  print() is not enough in production. We need:
  - Log levels (DEBUG, INFO, WARNING, ERROR) to filter noise
  - Timestamps to debug timing issues
  - Structured output (JSON) so log aggregation tools (Datadog, ELK) can parse it
  - Module names so we know WHERE the log came from

PRODUCTION CONSIDERATIONS:
  - Use JSON logging in production so tools like Kibana/Splunk can parse fields.
  - Use log correlation IDs (request_id) to trace one request across all services.
  - Never log sensitive data (API keys, PII, passwords).
  - Implement log rotation to prevent disk exhaustion.

SCALING CONSIDERATIONS:
  - In a distributed system, ship logs to a centralized store (CloudWatch, Loki).
  - Add trace_id and span_id for distributed tracing.
"""

import logging
import sys
from utils.config import get_settings

settings = get_settings()


def get_logger(name: str) -> logging.Logger:
    """
    Returns a configured logger for the given module name.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Processing started", extra={"doc_id": "abc123"})

    Args:
        name: Typically pass __name__ to get the module's logger.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Console handler with human-readable format for development
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Don't propagate to root logger to avoid duplicate messages
    logger.propagate = False

    return logger
