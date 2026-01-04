"""Command-line interface for :class:`~ChatDNA`.

This module exposes a small CLI that allows users to ask questions about
their DNA data. It loads the DNA file once and can answer a single question
via command-line arguments or multiple questions in an interactive mode.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dna_rag.core.chat_dna import ChatDNA
from dna_rag.core.config import get_settings


def main() -> None:
    """Entry point for the ``dna-rag`` command-line tool."""

    parser = argparse.ArgumentParser(
        description="Ask questions about your DNA using an LLM",
    )
    parser.add_argument(
        "--dna-file",
        required=True,
        type=Path,
        help="Path to the DNA CSV file",
    )
    parser.add_argument(
        "--question",
        help="Question to ask about the provided DNA",
    )

    args = parser.parse_args()

    # Load configuration from environment
    settings = get_settings()
    try:
        settings.validate_api_keys()
    except ValueError as exc:
        parser.error(str(exc))

    # Get first available API key (backward compatibility)
    api_key = settings.deepseek_api_key or settings.openai_api_key
    chat = ChatDNA(api_key)

    if args.question:
        print(chat.ask(args.question, args.dna_file))
        return

    # Interactive mode: reuse the loaded DNA file for multiple questions
    while True:
        try:
            question = input("Question (blank to quit): ").strip()
        except EOFError:
            break
        if not question:
            break
        print(chat.ask(question, args.dna_file))


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
