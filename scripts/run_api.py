#!/usr/bin/env python3
"""Script to run the DNA RAG API server."""

from __future__ import annotations

import argparse
import logging
import sys

import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run DNA RAG API server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (production only, incompatible with --reload)",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Logging level",
    )

    args = parser.parse_args()

    if args.reload and args.workers > 1:
        logger.error("--reload and --workers are mutually exclusive")
        sys.exit(1)

    logger.info(f"Starting DNA RAG API server on {args.host}:{args.port}")

    uvicorn.run(
        "dna_rag.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
