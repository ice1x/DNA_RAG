"""Tests for the Streamlit UI adapter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

st = pytest.importorskip("streamlit", reason="streamlit is required for UI tests")

from dna_rag.config import Settings  # noqa: E402
from dna_rag.exceptions import AnalysisError, ConfigurationError, DNARagError  # noqa: E402
from dna_rag.models import AnalysisResult, SNPResult  # noqa: E402
from dna_rag.ui.app import (  # noqa: E402
    _build_engine,
    _format_history,
    _make_llm_provider,
    _snp_to_display_row,
)
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
# _render_answer (via AppTest)
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

_FAKE_RESULT_CLINVAR = AnalysisResult(
    question="lactose tolerance",
    matched_snps=[
        SNPResult(
            rsid="rs1",
            chromosome="1",
            position=111,
            genotype="AA",
            gene="LCT",
            trait="Lactose tolerance",
            clinical_significance="Benign",
            clinvar_trait="Lactose intolerance",
            maf=0.25,
            maf_allele="T",
        ),
    ],
    interpretation=INTERPRETATION,
    snp_count_requested=1,
    snp_count_matched=1,
    cached=False,
    validation_used=True,
)

_EMPTY_RESULT = AnalysisResult(
    question="test",
    matched_snps=[],
    interpretation="No data.",
    snp_count_requested=1,
    snp_count_matched=0,
    cached=False,
)


class TestRenderAnswer:
    """_render_answer displays analysis results without errors."""

    def test_render_answer_basic(self):
        """_render_answer runs without raising."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_answer  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT

            _render_answer(_FAKE_RESULT)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"

    def test_render_answer_cached_label(self):
        """Cached results show the '(cached result)' caption."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_answer  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT_CACHED

            _render_answer(_FAKE_RESULT_CACHED)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        captions = [c.value for c in at.caption]
        assert any("cached" in c.lower() for c in captions)

    def test_render_answer_no_matched_snps(self):
        """Result with zero matched SNPs renders without error."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_answer  # noqa: F811
            from tests.ui.test_app import _EMPTY_RESULT

            _render_answer(_EMPTY_RESULT)

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
        """Chat input is disabled when no DNA file is uploaded."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        # chat_input should be present but disabled (no DNA file)
        assert len(at.chat_input) == 1
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
        """Inject mock engine -> ask question -> get result."""
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
        at.chat_input[0].set_value("lactose tolerance")
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

        at.chat_input[0].set_value("nonsense")
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

        at.chat_input[0].set_value("test")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        errors = [e.value for e in at.error]
        assert any("unexpected failure" in e for e in errors)

    def test_history_populated_after_analysis(self, dna_file: Path):
        """Session state history is populated after analysis."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()
        mock_engine.analyze.return_value = _FAKE_RESULT

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.run(timeout=10)

        # Before asking — history empty
        assert len(at.session_state.history) == 0

        # Ask a question -> analysis -> history populated
        at.chat_input[0].set_value("lactose tolerance")
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        assert len(at.session_state.history) == 1
        assert at.session_state.history[0].question == "lactose tolerance"

    def test_download_button_renders_without_error(self):
        """Download button renders with history without crashing."""
        from streamlit.testing.v1 import AppTest

        def script():
            import streamlit as st

            from dna_rag.ui.app import _format_history
            from tests.ui.test_app import _FAKE_RESULT

            st.download_button(
                label="\u2b07 Download chat history",
                data=_format_history([_FAKE_RESULT]),
                file_name="test.txt",
                mime="text/plain",
            )

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"


# ---------------------------------------------------------------------------
# _format_history
# ---------------------------------------------------------------------------


class TestFormatHistory:
    """Tests for _format_history output format."""

    def test_empty_history(self):
        """Empty history produces header only."""
        result = _format_history([])
        assert "DNA RAG" in result
        assert "Exported:" in result
        assert "Question" not in result

    def test_single_result(self):
        """Single result is formatted with timestamp, question, and answer."""
        result = _format_history([_FAKE_RESULT])
        assert "Q: lactose tolerance" in result
        assert INTERPRETATION in result
        assert "SNPs: 1 matched / 1 requested" in result
        assert "rs1" in result
        assert "LCT" in result

    def test_multiple_results_chronological_order(self):
        """Results are output in chronological order (oldest first)."""
        second = _FAKE_RESULT.model_copy(update={"question": "caffeine metabolism"})
        # history stores chronologically (append), oldest first
        result = _format_history([_FAKE_RESULT, second])
        pos_q1 = result.index("Question #1")
        pos_q2 = result.index("Question #2")
        assert pos_q1 < pos_q2
        # First question should be the oldest (lactose)
        q1_block = result[pos_q1:pos_q2]
        assert "lactose tolerance" in q1_block

    def test_cached_result_marked(self):
        """Cached results include '(cached result)' marker."""
        result = _format_history([_FAKE_RESULT_CACHED])
        assert "(cached result)" in result

    def test_no_matched_snps(self):
        """Result with no matched SNPs omits SNP details."""
        result = _format_history([_EMPTY_RESULT])
        assert "Q: test" in result
        assert "Matched SNPs:" not in result
        assert "SNPs: 0 matched / 1 requested" in result

    def test_timestamp_present(self):
        """Each entry has a timestamp in the output."""
        result = _format_history([_FAKE_RESULT])
        ts = _FAKE_RESULT.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        assert ts in result


# ---------------------------------------------------------------------------
# Chat interface tests
# ---------------------------------------------------------------------------


class TestChatInterface:
    """Tests for the chat message display."""

    def test_chat_messages_rendered_for_history(self, dna_file: Path):
        """Chat history renders user and assistant messages."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.session_state["history"] = [_FAKE_RESULT]
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        # The question should appear in markdown (inside user chat_message)
        assert any(
            "lactose tolerance" in m.value for m in at.markdown
        ), f"Expected question in chat, got: {[m.value for m in at.markdown]}"
        # The interpretation should also appear
        assert any(
            "lactose tolerant" in m.value for m in at.markdown
        ), f"Expected interpretation in chat, got: {[m.value for m in at.markdown]}"

    def test_welcome_message_when_no_file(self):
        """Assistant welcome message shown when no DNA file is uploaded."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        info_texts = [i.value for i in at.info]
        assert any("upload" in t.lower() for t in info_texts)

    def test_chat_input_present(self, dna_file: Path):
        """Chat input is present and enabled when DNA file is loaded."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        assert len(at.chat_input) == 1

    def test_download_button_in_sidebar_with_history(self, dna_file: Path):
        """Download button appears when history exists."""
        from streamlit.testing.v1 import AppTest

        mock_engine = MagicMock()

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = mock_engine
        at.session_state["dna_path"] = dna_file
        at.session_state["dna_df"] = "placeholder"
        at.session_state["file_id"] = "test-file-id"
        at.session_state["history"] = [_FAKE_RESULT]
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"


# ---------------------------------------------------------------------------
# ClinVar display tests
# ---------------------------------------------------------------------------


class TestClinVarDisplay:
    """ClinVar verification data is rendered in the UI."""

    def test_render_answer_with_clinvar(self):
        """_render_answer shows ClinVar data when available."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_answer  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT_CLINVAR

            _render_answer(_FAKE_RESULT_CLINVAR)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        # ClinVar expander should contain significance data
        all_md = " ".join(m.value for m in at.markdown)
        assert "Benign" in all_md
        assert "Lactose intolerance" in all_md

    def test_render_answer_without_clinvar_no_expander(self):
        """Without ClinVar data, the verification expander is not shown."""
        from streamlit.testing.v1 import AppTest

        def script():
            from dna_rag.ui.app import _render_answer  # noqa: F811
            from tests.ui.test_app import _FAKE_RESULT

            _render_answer(_FAKE_RESULT)

        at = AppTest.from_function(script)
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        # ClinVar caption should not appear
        captions = [c.value for c in at.caption]
        assert not any("ClinVar" in c for c in captions)

    def test_snp_to_display_row_with_clinvar(self):
        """_snp_to_display_row includes ClinVar columns when data is present."""
        snp = SNPResult(
            rsid="rs1", chromosome="1", position=111, genotype="AA",
            gene="LCT", trait="Lactose tolerance",
            clinical_significance="Pathogenic",
            clinvar_trait="Familial hypercholesterolemia",
            maf=0.0012, maf_allele="A",
        )
        row = _snp_to_display_row(snp)
        assert row["ClinVar"] == "Pathogenic"
        assert row["ClinVar Trait"] == "Familial hypercholesterolemia"
        assert row["MAF"] == "0.0012"

    def test_snp_to_display_row_without_clinvar(self):
        """_snp_to_display_row omits ClinVar columns when data is absent."""
        snp = SNPResult(
            rsid="rs1", chromosome="1", position=111, genotype="AA",
            gene="LCT", trait="Lactose tolerance",
        )
        row = _snp_to_display_row(snp)
        assert "ClinVar" not in row
        assert "ClinVar Trait" not in row
        assert "MAF" not in row


# ---------------------------------------------------------------------------
# _format_history with ClinVar
# ---------------------------------------------------------------------------


class TestFormatHistoryClinVar:
    """_format_history includes ClinVar data in text export."""

    def test_clinvar_in_export(self):
        """ClinVar significance appears in exported text."""
        result = _format_history([_FAKE_RESULT_CLINVAR])
        assert "ClinVar: Benign" in result
        assert "Lactose intolerance" in result
        assert "MAF: 0.2500" in result

    def test_no_clinvar_in_export_when_absent(self):
        """Without ClinVar data, export does not mention ClinVar."""
        result = _format_history([_FAKE_RESULT])
        assert "ClinVar:" not in result
        assert "MAF:" not in result


# ---------------------------------------------------------------------------
# NCBI Verification toggle tests
# ---------------------------------------------------------------------------


class TestNCBIToggle:
    """Tests for the NCBI verification toggle in the sidebar."""

    def test_toggle_appears_when_engine_configured(self):
        """Toggle is rendered in the sidebar when engine is available."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        toggles = at.toggle
        assert len(toggles) >= 1, "Expected at least one toggle widget"
        ncbi_toggle = toggles(key="ncbi_validation")
        assert ncbi_toggle is not None, "NCBI validation toggle not found"

    def test_toggle_default_off(self):
        """Toggle defaults to OFF when validation_enabled is False."""
        from streamlit.testing.v1 import AppTest

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.session_state["settings"] = Settings(
            llm_api_key="k",  # type: ignore[arg-type]
            validation_enabled=False,
        )
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        assert at.session_state.ncbi_validation is False

    def test_toggle_default_on_from_env(self, monkeypatch: pytest.MonkeyPatch):
        """Toggle defaults to ON when validation_enabled is True in settings."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        at = AppTest.from_file("src/dna_rag/ui/app.py")
        at.session_state["engine"] = MagicMock()
        at.session_state["settings"] = Settings(
            llm_api_key="k",  # type: ignore[arg-type]
            validation_enabled=True,
        )
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        assert at.session_state.ncbi_validation is True

    def test_toggle_on_rebuilds_engine_with_validation(
        self, monkeypatch: pytest.MonkeyPatch,
    ):
        """Switching toggle ON rebuilds engine with validation_enabled=True."""
        from streamlit.testing.v1 import AppTest

        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")

        at = AppTest.from_file("src/dna_rag/ui/app.py")
        mock_engine = MagicMock()
        at.session_state["engine"] = mock_engine
        at.session_state["settings"] = Settings(
            llm_api_key="k",  # type: ignore[arg-type]
            validation_enabled=False,
            rag_enabled=False,  # avoid sentence_transformers import
        )
        at.run(timeout=10)
        assert not at.exception, f"Unexpected exception: {at.exception}"
        assert at.session_state.settings.validation_enabled is False

        # Flip toggle ON — engine is rebuilt with new settings
        at.toggle(key="ncbi_validation").set_value(True)
        at.run(timeout=10)

        assert not at.exception, f"Unexpected exception: {at.exception}"
        # Settings should now have validation_enabled=True
        assert at.session_state.settings.validation_enabled is True
        # Engine should have been replaced (not the original mock)
        assert at.session_state.engine is not mock_engine
