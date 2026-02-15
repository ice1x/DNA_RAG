"""Structured logging configuration for dna-rag.

Provides two output modes:

* **console** -- human-friendly coloured output for local development.
* **json** -- machine-readable JSON lines for production log aggregators
  (Datadog, ELK, etc.).

Usage::

    from dna_rag.logging import setup_logging, get_logger

    setup_logging(level="INFO", fmt="console")
    logger = get_logger(__name__)
    logger.info("analysis_start", question="lactose tolerance")
"""

from __future__ import annotations

import logging
import sys

import structlog


def setup_logging(level: str = "INFO", fmt: str = "console") -> None:
    """Configure *structlog* and the standard-library root logger.

    Args:
        level: Logging level name (``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``).
        fmt: Output format -- ``"console"`` for coloured dev output or
            ``"json"`` for machine-readable JSON lines.
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    renderer: structlog.types.Processor
    if fmt == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a bound *structlog* logger for *name*."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
