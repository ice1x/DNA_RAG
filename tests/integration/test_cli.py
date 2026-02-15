"""Integration tests for the CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from dna_rag.cli import main
from tests.conftest import FakeLLMProvider

_SNP_JSON = json.dumps(
    {
        "rs1": {
            "gene": "LCT",
            "chromosome": "1",
            "position": 111,
            "trait": "Lactose tolerance",
        }
    }
)
_INTERPRETATION = "You are lactose tolerant."


def _make_dna_file(tmp_path: Path) -> Path:
    f = tmp_path / "dna.txt"
    f.write_text("# 23andMe data\nrs1\t1\t111\tAA\n")
    return f


class TestAskCommand:
    """Tests for `dna-rag ask`."""

    def test_text_output(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        dna = _make_dna_file(tmp_path)
        fake = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])

        with patch("dna_rag.cli._make_llm_provider", return_value=fake):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["ask", "--dna-file", str(dna), "--question", "lactose"],
            )

        assert result.exit_code == 0, f"output: {result.output}"
        assert "lactose tolerant" in result.output

    def test_json_output(self, tmp_path: Path, monkeypatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        monkeypatch.setenv("DNA_RAG_LOG_LEVEL", "CRITICAL")  # suppress log noise
        dna = _make_dna_file(tmp_path)
        fake = FakeLLMProvider([_SNP_JSON, _INTERPRETATION])

        with patch("dna_rag.cli._make_llm_provider", return_value=fake):
            runner = CliRunner()
            result = runner.invoke(
                main,
                [
                    "--log-level",
                    "CRITICAL",
                    "ask",
                    "--dna-file",
                    str(dna),
                    "--question",
                    "lactose",
                    "--output-format",
                    "json",
                ],
            )

        assert result.exit_code == 0, f"output: {result.output}"
        # Extract JSON from output (may contain log lines on stderr leaking)
        output = result.output.strip()
        # Find the JSON object in the output
        json_start = output.find("{")
        assert json_start >= 0, f"No JSON found in output: {output}"
        data = json.loads(output[json_start:])
        assert data["question"] == "lactose"
        assert len(data["matched_snps"]) == 1

    def test_missing_api_key(self, tmp_path: Path, monkeypatch):
        monkeypatch.delenv("DNA_RAG_LLM_API_KEY", raising=False)
        dna = _make_dna_file(tmp_path)
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ask", "--dna-file", str(dna), "--question", "test"],
        )
        assert result.exit_code != 0

    def test_nonexistent_file(self, monkeypatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["ask", "--dna-file", "/nonexistent/file.txt", "--question", "test"],
        )
        assert result.exit_code != 0


class TestHelpOutput:
    """Ensure --help works for all commands."""

    def test_main_help(self):
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "DNA RAG" in result.output

    def test_ask_help(self, monkeypatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(main, ["ask", "--help"])
        assert result.exit_code == 0
        assert "--dna-file" in result.output

    def test_interactive_help(self, monkeypatch):
        monkeypatch.setenv("DNA_RAG_LLM_API_KEY", "test-key")
        runner = CliRunner()
        result = runner.invoke(main, ["interactive", "--help"])
        assert result.exit_code == 0
        assert "--dna-file" in result.output
