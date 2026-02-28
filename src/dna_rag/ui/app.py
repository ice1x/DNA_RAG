"""Streamlit UI for DNA RAG.

A thin adapter over :class:`~dna_rag.engine.DNAAnalysisEngine` --
file upload, question input, result display.

Run with::

    dna-rag-ui          # console entry-point
    make ui             # Makefile target
    streamlit run src/dna_rag/ui/app.py
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import streamlit as st
from pydantic import SecretStr

from dna_rag.cache.memory import InMemoryCache
from dna_rag.config import Settings
from dna_rag.exceptions import (
    AnalysisError,
    ConfigurationError,
    DNARagError,
)
from dna_rag.models import AnalysisResult

# ---------------------------------------------------------------------------
# Engine wiring (same pattern as cli.py)
# ---------------------------------------------------------------------------


def _make_llm_provider(settings: Settings):  # noqa: ANN202
    if settings.llm_provider == "deepseek":
        from dna_rag.llm.deepseek import DeepSeekProvider

        return DeepSeekProvider(settings)
    elif settings.llm_provider == "openai_compat":
        from dna_rag.llm.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider(settings)
    else:
        raise ConfigurationError(f"Unknown LLM provider: {settings.llm_provider}")


def _build_vector_store(settings: Settings):  # noqa: ANN202
    """Attempt to create a :class:`SNPVectorStore`.  Returns ``None`` on failure."""
    try:
        from dna_rag.vector_store import SNPVectorStore
    except ImportError:
        return None

    persist_dir = Path(settings.rag_persist_directory) if settings.rag_persist_directory else None
    return SNPVectorStore(
        persist_directory=persist_dir,
        embedding_model=settings.rag_embedding_model,
        collection_name=settings.rag_collection_name,
    )


def _build_engine(settings: Settings):  # noqa: ANN202
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

    vector_store = None
    if settings.rag_enabled:
        vector_store = _build_vector_store(settings)

    snp_database = None
    if settings.validation_enabled:
        from dna_rag.snp_database import SNPDatabase

        snp_database = SNPDatabase(
            cache=cache,
            request_timeout=settings.validation_timeout,
            rate_limit_delay=settings.validation_rate_limit_delay,
        )

    return DNAAnalysisEngine(
        snp_llm=snp_llm,
        interpretation_llm=interp_llm,
        cache=cache,
        vector_store=vector_store,
        snp_database=snp_database,
        rag_search_results=settings.rag_search_results,
        rag_min_similarity=settings.rag_min_similarity,
        medical_disclaimer=settings.medical_disclaimer,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _render_result(result: AnalysisResult) -> None:
    """Render a single analysis result."""
    st.subheader(result.question)
    if result.cached:
        st.caption("(cached result)")

    st.markdown(result.interpretation)

    col1, col2 = st.columns(2)
    col1.metric("SNPs requested", result.snp_count_requested)
    col2.metric("SNPs matched", result.snp_count_matched)

    if result.matched_snps:
        with st.expander(f"Matched SNPs ({len(result.matched_snps)})"):
            rows = [snp.model_dump() for snp in result.matched_snps]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()


def _format_history(history: list[AnalysisResult]) -> str:
    """Format chat history as plain text for download."""
    lines: list[str] = []
    lines.append("DNA RAG — Chat History")
    lines.append(f"Exported: {datetime.now():%Y-%m-%d %H:%M:%S}")
    lines.append("=" * 60)

    for i, result in enumerate(reversed(history), 1):
        ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        lines.append("")
        lines.append(f"[{ts}] Question #{i}")
        lines.append(f"Q: {result.question}")
        lines.append("")
        lines.append("Answer:")
        lines.append(result.interpretation)
        lines.append("")
        lines.append(
            f"SNPs: {result.snp_count_matched} matched"
            f" / {result.snp_count_requested} requested"
        )
        if result.matched_snps:
            lines.append("Matched SNPs:")
            for snp in result.matched_snps:
                lines.append(
                    f"  - {snp.rsid} | chr{snp.chromosome}:{snp.position}"
                    f" | {snp.genotype} | {snp.gene} | {snp.trait}"
                )
        if result.cached:
            lines.append("(cached result)")
        lines.append("-" * 60)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _has_env_api_key() -> bool:
    """Check whether ``DNA_RAG_LLM_API_KEY`` is set in the environment."""
    return bool(os.environ.get("DNA_RAG_LLM_API_KEY"))


def _init_engine_from_env() -> bool:
    """Try building the engine from environment / ``.env`` settings.

    Returns ``True`` on success, ``False`` if API key is missing.
    """
    try:
        settings = Settings()  # type: ignore[call-arg]
        st.session_state.engine = _build_engine(settings)
        return True
    except (ConfigurationError, Exception):
        return False


def _init_engine_from_input(
    api_key: str,
    provider: Literal["deepseek", "openai_compat"],
    model: str,
    base_url: str,
) -> bool:
    """Build the engine from user-supplied values.

    Returns ``True`` on success.
    """
    try:
        settings = Settings(  # type: ignore[call-arg]
            llm_api_key=SecretStr(api_key),
            llm_provider=provider,
            llm_model=model,
            llm_base_url=base_url,
        )
        st.session_state.engine = _build_engine(settings)
        return True
    except (ConfigurationError, Exception) as exc:
        st.error(f"Configuration error: {exc}")
        return False


def main() -> None:
    st.set_page_config(page_title="DNA RAG", page_icon="\U0001f9ec")
    st.title("\U0001f9ec DNA RAG")
    st.caption(
        "\u26a0\ufe0f **Not medical advice.** For educational and research purposes only. "
        "Consult a healthcare provider for medical interpretation of genetic data."
    )

    # --- Init session state ------------------------------------------------
    defaults: dict[str, object] = {
        "engine": None,
        "dna_path": None,
        "dna_df": None,
        "file_id": None,
        "history": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Init engine: env first, then sidebar input -----------------------
    if st.session_state.engine is None and _has_env_api_key():
        _init_engine_from_env()

    # --- Sidebar ----------------------------------------------------------
    with st.sidebar:
        # --- API key input (only when env key is absent) ------------------
        if not _has_env_api_key():
            st.header("LLM Settings")
            provider = st.selectbox(
                "Provider",
                ["deepseek", "openai_compat"],
                help="deepseek or any OpenAI-compatible API",
            )
            api_key = st.text_input(
                "API Key",
                type="password",
                help="Your LLM provider API key",
            )
            model = st.text_input(
                "Model",
                value="deepseek-r1:free" if provider == "deepseek" else "gpt-4o-mini",
            )
            default_url = (
                "https://api.deepseek.com/v1"
                if provider == "deepseek"
                else "https://api.openai.com/v1"
            )
            base_url = st.text_input("Base URL", value=default_url)

            if api_key and st.session_state.engine is None:
                _init_engine_from_input(
                    api_key,
                    provider,  # type: ignore[arg-type]
                    model,
                    base_url,
                )

            if not api_key:
                st.warning("Enter your API key to start.")

            st.caption(
                "\u26a0\ufe0f **Privacy:** Your DNA data is sent to the selected LLM "
                "provider and is subject to their privacy policy and data retention "
                "rules. This tool does not store your data."
            )

            st.divider()

        st.header("DNA file")
        uploaded = st.file_uploader(
            "Upload your DNA data",
            type=["txt", "csv", "vcf"],
            help="23andMe, AncestryDNA, MyHeritage, or VCF format",
        )

        if uploaded is not None and uploaded.file_id != st.session_state.file_id:
            suffix = Path(uploaded.name).suffix or ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = Path(tmp.name)
            try:
                from dna_rag.parsers.detector import detect_and_parse

                df = detect_and_parse(tmp_path)
                st.session_state.dna_path = tmp_path
                st.session_state.dna_df = df
                st.session_state.file_id = uploaded.file_id
                st.session_state.history = []
            except Exception as exc:
                st.error(f"Could not parse DNA file: {exc}")
                tmp_path.unlink(missing_ok=True)

        if st.session_state.dna_df is not None:
            st.success(f"{len(st.session_state.dna_df):,} variants loaded")

            # --- Polygenic Risk Scores ---
            st.header("Polygenic Risk Scores")
            from dna_rag.polygenic import PolygenicScoreCalculator

            calc = PolygenicScoreCalculator()
            available = calc.list_available_scores()
            score_name = st.selectbox("Score", available)
            if st.button("Calculate"):
                try:
                    prs = calc.calculate(score_name, st.session_state.dna_df)
                    c1, c2 = st.columns(2)
                    c1.metric("Percentile", f"{prs.percentile:.1f}%")
                    c2.metric("SNPs", f"{prs.snps_matched}/{prs.snps_total}")
                    st.info(prs.interpretation)
                except Exception as exc:
                    st.error(f"PRS error: {exc}")


    # --- Guard: engine must be ready --------------------------------------
    if st.session_state.engine is None:
        st.info("Configure your LLM API key in the sidebar to get started.")
        st.stop()

    # --- Main: question + results -----------------------------------------
    question = st.text_input(
        "Ask a question about your DNA",
        placeholder="e.g. lactose tolerance, caffeine metabolism",
    )

    if question and st.session_state.dna_path is not None:
        with st.spinner("Analyzing..."):
            try:
                result = st.session_state.engine.analyze(
                    question, st.session_state.dna_path
                )
                st.session_state.history.insert(0, result)
            except AnalysisError as exc:
                st.warning(str(exc))
            except DNARagError as exc:
                st.error(f"Error: {exc}")

    if not st.session_state.dna_path:
        st.info("Upload a DNA file in the sidebar to get started.")

    # --- Download chat history --------------------------------------------
    if st.session_state.history:
        st.download_button(
            label="\u2b07 Download chat history",
            data=_format_history(st.session_state.history),
            file_name=f"dna_rag_chat_{datetime.now():%Y%m%d_%H%M%S}.txt",
            mime="text/plain",
        )

    # --- Render history ---------------------------------------------------
    for result in st.session_state.history:
        _render_result(result)


main()


# ---------------------------------------------------------------------------
# Entry point for `dna-rag-ui` console script
# ---------------------------------------------------------------------------


def run() -> None:
    """Launch the Streamlit app via ``dna-rag-ui``."""
    import sys

    from streamlit.web.cli import main as st_main

    sys.argv = ["streamlit", "run", __file__, "--server.headless", "true"]
    st_main()
