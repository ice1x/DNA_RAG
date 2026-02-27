"""Streamlit UI for DNA RAG.

A thin adapter over :class:`~dna_rag.engine.DNAAnalysisEngine` --
file upload, question input, result display.

Run with::

    dna-rag-ui          # console entry-point
    make ui             # Makefile target
    streamlit run src/dna_rag/ui/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

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

    return DNAAnalysisEngine(
        snp_llm=snp_llm,
        interpretation_llm=interp_llm,
        cache=cache,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="DNA RAG", page_icon="\U0001f9ec", layout="wide")
    st.title("\U0001f9ec DNA RAG")

    # --- Init engine (once) ------------------------------------------------
    if "engine" not in st.session_state:
        try:
            settings = Settings()  # type: ignore[call-arg]
            st.session_state.engine = _build_engine(settings)
        except (ConfigurationError, Exception) as exc:
            st.error(f"Configuration error: {exc}")
            st.info("Check your `.env` file or `DNA_RAG_*` environment variables.")
            st.stop()

    # --- Init session state ------------------------------------------------
    defaults: dict[str, object] = {
        "dna_path": None,
        "dna_df": None,
        "file_id": None,
        "history": [],
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # --- Sidebar: file upload + PRS ---------------------------------------
    with st.sidebar:
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

    # --- Main: question + results -----------------------------------------
    question = st.text_input(
        "Ask a question about your DNA",
        placeholder="e.g. lactose tolerance, caffeine metabolism",
    )

    analyze_disabled = not question or st.session_state.dna_path is None
    if st.button("Analyze", disabled=analyze_disabled, type="primary"):
        if st.session_state.dna_path is None:
            st.warning("Upload a DNA file first.")
        else:
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
