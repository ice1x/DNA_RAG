import gzip
from pathlib import Path

import pytest

from chat_dna import ChatDNA
from fake_llm import FakeLLM


def test_chat_dna_load_and_filter(sample_dna_file):
    chat = ChatDNA(api_key="test")
    file_hash = chat._hash_file(sample_dna_file)
    df = chat._load_dna(sample_dna_file, file_hash)
    snp_dict = {"rs1": {"gene": "GENE", "chromosome": "1", "position": 111}}
    filtered = chat._filter_snps(df, snp_dict)

    assert filtered.shape[0] == 1
    row = filtered.iloc[0]
    assert row["RSID"] == "rs1"
    assert row["GENOTYPE"] == "AA"
    assert row["gene"] == "GENE"


def test_load_dna_missing_file(tmp_path: Path) -> None:
    chat = ChatDNA(api_key="test")
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        chat._load_dna(missing, "hash")


def test_load_dna_bad_extension(tmp_path: Path) -> None:
    chat = ChatDNA(api_key="test")
    bad = tmp_path / "dna.bad"
    bad.write_text("rs1,1,111,AA\n")
    file_hash = chat._hash_file(bad)
    with pytest.raises(ValueError):
        chat._load_dna(bad, file_hash)


def test_load_dna_gz(sample_dna_file: Path) -> None:
    gz_file = sample_dna_file.with_suffix(sample_dna_file.suffix + ".gz")
    with gzip.open(gz_file, "wb") as f:
        f.write(sample_dna_file.read_bytes())

    chat = ChatDNA(api_key="test")
    file_hash = chat._hash_file(gz_file)
    df = chat._load_dna(gz_file, file_hash)
    assert df.shape[0] == 2


def test_chat_dna_caching(sample_dna_file):
    responses = [
        '{"rs1": {"gene": "GENE", "chromosome": "1", "position": 111}}',
        "interpretation",
    ]
    fake = FakeLLM(responses)
    chat = ChatDNA(api_key="test")
    chat.llm = fake

    q = "lactose tolerance"
    assert chat.ask(q, sample_dna_file) == "interpretation"
    assert chat.ask(q, sample_dna_file) == "interpretation"
    assert fake.calls == 2


def test_invalid_json_logs_error(sample_dna_file, caplog):
    fake = FakeLLM(["not json"])
    chat = ChatDNA(api_key="test")
    chat.llm = fake

    with caplog.at_level("ERROR"):
        answer = chat.ask("lactose", sample_dna_file)

    assert answer == "No relevant SNPs found."
    assert "Failed to decode SNP JSON" in caplog.text
