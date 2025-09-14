"""Command-line interface for :class:`~ChatDNA`.

This module exposes a small CLI that allows users to ask questions about
their DNA data. It loads the DNA file once and can answer a single question
via command-line arguments or multiple questions in an interactive mode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from chat_dna import ChatDNA


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

    api_key = os.environ.get("API_KEY")
    if not api_key:
        parser.error("API_KEY environment variable is not set")

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

