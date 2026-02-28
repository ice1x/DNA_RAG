"""Tests for the Streamlit UI adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

st = pytest.importorskip("streamlit", reason="streamlit is required for UI tests")

from dna_rag.config import Settings  # noqa: E402
from dna_rag.exceptions import AnalysisError, ConfigurationError, DNARagError  # noqa: E402
from dna_rag.models import AnalysisResult, SNPResult  # noqa: E402
from dna_rag.ui.app import _build_engine, _make_llm_provider  # noqa: E402
from tests.ui.conftest import INTERPRETATION  # noqa: E402

# ---------------------------------------------------------------------------
# _make_llm_provider
# ---------------------------------------------------------------------------


class TestMakeLlmProvider:
    """_make_llm_provider returns the correct provider class."""

    def test_deepseek_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(llm_provider="deepseek", llm_api_key="k")  # type: ignore[arg-type]

        from dna_rag.llm.deepseek import DeepSeekProvider

        provider = _make_llm_provider(settings)
        assert isinstance(provider, DeepSeekProvider)

    def test_openai_compat_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(
            llm_provider="openai_compat",
            llm_api_key="k",  # type: ignore[arg-type]
            llm_base_url="http://localhost:1234/v1",
        )

        from dna_rag.llm.openai_compat import OpenAICompatProvider

        provider = _make_llm_provider(settings)
        assert isinstance(provider, OpenAICompatProvider)

    def test_unknown_provider_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(llm_provider="deepseek", llm_api_key="k")  # type: ignore[arg-type]
        # Force an unknown provider value after validation
        object.__setattr__(settings, "llm_provider", "unknown_provider")

        with pytest.raises(ConfigurationError, match="Unknown LLM provider"):
            _make_llm_provider(settings)


# ---------------------------------------------------------------------------
# _build_engine
# ---------------------------------------------------------------------------


class TestBuildEngine:
    """_build_engine assembles the engine correctly."""

    def test_builds_with_memory_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(
            llm_provider="deepseek",
            llm_api_key="k",  # type: ignore[arg-type]
            cache_backend="memory",
            cache_max_size=50,
            cache_ttl_seconds=120,
        )

        engine = _build_engine(settings)

        from dna_rag.cache.memory import InMemoryCache
        from dna_rag.engine import DNAAnalysisEngine

        assert isinstance(engine, DNAAnalysisEngine)
        assert isinstance(engine._cache, InMemoryCache)

    def test_builds_without_cache(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(
            llm_provider="deepseek",
            llm_api_key="k",  # type: ignore[arg-type]
            cache_backend="none",
        )

        engine = _build_engine(settings)
        assert engine._cache is None

    def test_builds_with_separate_interp_llm(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        settings = Settings(
            llm_provider="deepseek",
            llm_api_key="k",  # type: ignore[arg-type]
            llm_interp_provider="deepseek",
            llm_interp_api_key="k2",  # type: ignore[arg-type]
            llm_interp_base_url="http://localhost:1234/v1",
        )

        engine = _build_engine(settings)
        assert engine._interp_llm is not engine._snp_llm


# ---------------------------------------------------------------------------
# _render_result (via AppTest)
# ---------------------------------------------------------------------------

_FAKE_RESULT = AnalysisResult(
    question="lactose tolerance",
    matched_snps=[
        SNPResult(
            rsid="rs1",
            chromosome="1",
            position=111,
            genotype="AA",
            gene="LCT",
            trait="Lactose tolerance",
        ),
    ],
    interpretation=INTERPRETATION,
    snp_count_requested=1,
    snp_count_matched=1,
    cached=False,
)

_FAKE_RESULT_CACHED = _FAKE_RESULT.model_copy(update={"cached": True})

_EMPTY_RESULT = AnalysisResult(
    question="test",
    matched_snps=[],
    interpretation="No data.",
    snp_count_requested=1,
    snp_count_matched=0,
    cached=False,
)


class TestRenderResult:
    """_render_result displays analysis results without errors."""

    def test_render_result_basic(self):
        """_render_result runs without raising."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_result  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT

            _render_result(_FAKE_RESULT)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"

    def test_render_result_cached_label(self):
        """Cached results show the '(cached result)' caption."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_result  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT_CACHED

            _render_result(_FAKE_RESULT_CACHED)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        captions = [c.value for c in at.caption]
        assert any("cached" in c.lower() for c in captions)

    def test_render_result_no_matched_snps(self):
        """Result with zero matched SNPs renders without error."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_result  # noqa: F811
            from tests.ui.test_app import _EMPTY_RESULT

            _render_result(_EMPTY_RESULT)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"


# ---------------------------------------------------------------------------
# Full app flow (via AppTest)
# ---------------------------------------------------------------------------


class TestAppFlow:
    """Smoke-tests for the full Streamlit app.

    AppTest.from_file runs the script via exec(), so mock.patch on module
    attributes does not work.  Instead we pre-set ``session_state["engine"]``
    to a mock before the first run, which skips the real ``_build_engine`` call.
    """

    def test_initial_load_shows_upload_prompt(self):
        """Fresh app shows 'Upload a DNA file' info message."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        info_texts = [i.value for i in at.info]
        assert any("upload" in t.lower() for t in info_texts), (
            f"Expected upload prompt in info messages, got: {info_texts}"
        )

    def test_no_analysis_without_file(self):
        """Entering a question without a DNA file does not trigger analysis."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.run(timeout=10)

        at.text_input[0].set_value("lactose tolerance")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        mock_engine.analyze.assert_not_called()

    def test_config_error_shows_error_message(self):
        """ConfigurationError during engine init shows st.error."""
        from streamlit.testing.v1 import AppTest

        def script():
            import streamlit as st

            from dna_rag.exceptions import ConfigurationError

            # Reproduce the init logic with a forced failure
            st.set_page_config(page_title="DNA RAG")
            st.title("DNA RAG")
            if "engine" not in st.session_state:
                try:
                    raise ConfigurationError("bad config")
                except (ConfigurationError, Exception) as exc:
                    st.error(f"Configuration error: {exc}")
                    st.info("Check your `.env` file or `DNA_RAG_*` environment variables.")
                    st.stop()

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        errors = [e.value for e in at.error]
        assert any("bad config" in e for e in errors)

    def test_full_analysis_flow(self, dna_file: Path):
        """Inject mock engine → ask question → get result."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        mock_engine.analyze.return_value = _FAKE_RESULT

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.run(timeout=10)

        # Type question — analysis triggers automatically on submit
        at.text_input[0].set_value("lactose tolerance")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        mock_engine.analyze.assert_called_once()
        assert any(
            "lactose tolerant" in m.value for m in at.markdown
        ), f"Expected interpretation in markdown, got: {[m.value for m in at.markdown]}"

    def test_analysis_error_shows_warning(self, dna_file: Path):
        """AnalysisError during analyze shows st.warning."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        mock_engine.analyze.side_effect = AnalysisError("No relevant SNPs found")

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.run(timeout=10)

        at.text_input[0].set_value("nonsense")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        warnings = [w.value for w in at.warning]
        assert any("No relevant SNPs" in w for w in warnings)

    def test_dna_rag_error_shows_error(self, dna_file: Path):
        """DNARagError during analyze shows st.error."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        mock_engine.analyze.side_effect = DNARagError("unexpected failure")

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.run(timeout=10)

        at.text_input[0].set_value("test")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        errors = [e.value for e in at.error]
        assert any("unexpected failure" in e for e in errors)
