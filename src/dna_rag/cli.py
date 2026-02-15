"""Command-line interface for dna-rag.

Provides two sub-commands:

* ``dna-rag ask`` -- ask a single question.
* ``dna-rag interactive`` -- interactive session with multiple questions.
"""

from __future__ import annotations

from pathlib import Path

import click

from dna_rag.cache.memory import InMemoryCache
from dna_rag.config import Settings
from dna_rag.exceptions import (
    AnalysisError,
    ConfigurationError,
    DNARagError,
)
from dna_rag.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _make_llm_provider(settings: Settings):  # noqa: ANN202
    """Create an LLM provider from *settings*."""
    if settings.llm_provider == "deepseek":
        from dna_rag.llm.deepseek import DeepSeekProvider

        return DeepSeekProvider(settings)
    elif settings.llm_provider == "openai_compat":
        from dna_rag.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings)
    else:
        raise ConfigurationError(
            f"Unknown LLM provider: {settings.llm_provider}"
        )


def _build_engine(settings: Settings):  # noqa: ANN202
    """Wire up the :class:`~dna_rag.engine.DNAAnalysisEngine` from *settings*.

    If ``llm_interp_*`` settings are configured, a separate LLM provider
    is created for the interpretation step.
    """
    from dna_rag.engine import DNAAnalysisEngine

    snp_llm = _make_llm_provider(settings)

    interp_llm = None
    if settings.has_separate_interp_llm:
        interp_settings = settings.get_interp_settings_as_primary()
        interp_llm = _make_llm_provider(interp_settings)

    cache = (
        InMemoryCache(
            max_size=settings.cache_max_size,
            ttl_seconds=settings.cache_ttl_seconds,
        )
        if settings.cache_backend == "memory"
        else None
    )

    return DNAAnalysisEngine(
        snp_llm=snp_llm,
        interpretation_llm=interp_llm,
        cache=cache,
    )


@click.group()
@click.option(
    "--log-level",
    default=None,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
@click.option(
    "--log-format",
    default=None,
    type=click.Choice(["console", "json"]),
    help="Logging output format.",
)
@click.pass_context
def main(ctx: click.Context, log_level: str | None, log_format: str | None) -> None:
    """DNA RAG -- analyse your DNA data using LLMs."""
    ctx.ensure_object(dict)

    # Skip settings loading for --help on subcommands
    if ctx.invoked_subcommand is None:
        return

    try:
        settings = Settings()  # type: ignore[call-arg]
    except Exception as exc:
        click.echo(f"Configuration error: {exc}", err=True)
        ctx.exit(1)
        return

    if log_level:
        settings.log_level = log_level
    if log_format:
        settings.log_format = log_format  # type: ignore[assignment]
    setup_logging(settings.log_level, settings.log_format)

    ctx.obj["settings"] = settings


@main.command()
@click.option(
    "--dna-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the DNA data file.",
)
@click.option("--question", required=True, help="Question about your DNA.")
@click.option(
    "--output-format",
    default="text",
    type=click.Choice(["text", "json"]),
    help="Output format.",
)
@click.pass_context
def ask(
    ctx: click.Context,
    dna_file: Path,
    question: str,
    output_format: str,
) -> None:
    """Ask a single question about your DNA data."""
    settings: Settings = ctx.obj["settings"]
    engine = _build_engine(settings)

    try:
        result = engine.analyze(question, dna_file)
    except AnalysisError as exc:
        click.echo(str(exc), err=True)
        ctx.exit(0)
        return
    except DNARagError as exc:
        click.echo(f"Error: {exc}", err=True)
        ctx.exit(1)
        return

    if output_format == "json":
        click.echo(result.model_dump_json(indent=2))
    else:
        click.echo(result.interpretation)


@main.command()
@click.option(
    "--dna-file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the DNA data file.",
)
@click.pass_context
def interactive(ctx: click.Context, dna_file: Path) -> None:  # pragma: no cover
    """Interactive mode -- ask multiple questions about the same DNA file."""
    settings: Settings = ctx.obj["settings"]
    engine = _build_engine(settings)

    click.echo(
        "DNA RAG Interactive Mode. Type 'quit' or press Ctrl+D to exit.\n"
    )

    while True:
        try:
            question = click.prompt("Question", default="", show_default=False)
        except (EOFError, click.Abort):
            break

        if not question or question.lower() in ("quit", "exit", "q"):
            break

        try:
            result = engine.analyze(question, dna_file)
            click.echo(f"\n{result.interpretation}\n")
            click.echo(
                f"[Matched {result.snp_count_matched}/"
                f"{result.snp_count_requested} SNPs]\n"
            )
        except AnalysisError as exc:
            click.echo(f"Note: {exc}\n")
        except DNARagError as exc:
            click.echo(f"Error: {exc}\n")
