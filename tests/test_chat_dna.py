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
