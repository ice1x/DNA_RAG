"""Guarded entry-points for optional extras (api, ui).

These thin wrappers ensure that running ``dna-rag-api`` or ``dna-rag-ui``
without the corresponding extras installed produces a clear error message
instead of a raw ``ModuleNotFoundError`` traceback.
"""

from __future__ import annotations

import sys


def run_api() -> None:
    """Entry point for ``dna-rag-api`` console script."""
    try:
        from dna_rag.api.main import run
    except ImportError:
        print(
            "Error: FastAPI dependencies are not installed.\n"
            "Install them with: pip install dna-rag[api]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    run()


def run_ui() -> None:
    """Entry point for ``dna-rag-ui`` console script."""
    try:
        from dna_rag.ui.app import run
    except ImportError:
        print(
            "Error: Streamlit dependencies are not installed.\n"
            "Install them with: pip install dna-rag[ui]",
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    run()
