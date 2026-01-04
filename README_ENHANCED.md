# DNA_RAG - Enhanced PoC

[![CI](https://github.com/ice1x/DNA_RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/ice1x/DNA_RAG/actions/workflows/ci.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An advanced toolkit for analyzing DNA data using Large Language Models (LLMs) with RAG (Retrieval-Augmented Generation) capabilities.

## Features

### Core Capabilities
- **RAG-Based SNP Retrieval**: Vector database for semantic search over SNP descriptions
- **SNP Validation**: Real-time validation through NCBI dbSNP API
- **Structured Output**: Confidence scores, source citations, and caveats
- **Conversation History**: Multi-turn dialogues with context
- **Polygenic Risk Scores**: Calculate complex trait risk scores
- **Multiple File Formats**: Support for CSV, VCF, gzipped files

### Architecture

```
DNA_RAG/
├── chat_dna.py              # Original implementation
├── chat_dna_enhanced.py     # Enhanced with RAG & validation
├── snp_database.py          # dbSNP integration
├── vector_store.py          # ChromaDB-based RAG
├── polygenic_scores.py      # Polygenic risk score calculation
├── vcf_parser.py            # VCF file support
└── tests/                   # Comprehensive test suite
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Setup pre-commit hooks (optional)
pre-commit install
```

## Usage

### Basic Usage (Original)

```bash
export API_KEY="your-deepseek-api-key"
python cli.py --dna-file path/to/dna.csv --question "lactose tolerance"
```

### Enhanced Usage (with RAG)

```python
from pathlib import Path
from chat_dna_enhanced import ChatDNAEnhanced

# Initialize with RAG and validation
chat = ChatDNAEnhanced(
    api_key="your-key",
    use_vector_store=True,
    use_validation=True,
    vector_store_path=Path("./vector_db")
)

# Ask question with structured output
result = chat.ask("lactose tolerance", Path("dna.csv"), return_structured=True)

print(f"Interpretation: {result.interpretation}")
print(f"Confidence: {result.confidence}")
print(f"SNPs found: {len(result.snps_found)}")
print(f"Sources: {result.sources}")
print(f"Caveats: {result.caveats}")
```

### Polygenic Risk Scores

```python
from polygenic_scores import PolygenicScoreCalculator
import pandas as pd

# Load DNA data
df = pd.read_csv("dna.csv", names=["RSID", "CHROMOSOME", "POSITION", "GENOTYPE"])

# Calculate risk score
calculator = PolygenicScoreCalculator()
result = calculator.calculate("alzheimers_risk", df)

print(f"Score: {result.normalized_score}")
print(f"Percentile: {result.percentile}")
print(f"Interpretation: {result.interpretation}")
```

### VCF File Support

```python
from vcf_parser import VCFParser, convert_vcf_to_csv
from pathlib import Path

# Parse VCF file
parser = VCFParser(Path("genome.vcf.gz"))
df = parser.parse()

# Or convert to CSV
convert_vcf_to_csv(Path("genome.vcf"), Path("output.csv"))
```

## Development

### Running Tests

```bash
# Run all tests with coverage
pytest

# Run specific test file
pytest tests/test_vector_store.py

# Run with verbose output
pytest -v
```

### Code Quality

```bash
# Format code
black .

# Lint code
ruff check .

# Type checking
mypy *.py
```

### CI/CD

The project uses GitHub Actions for:
- Automated testing on Python 3.10, 3.11, 3.12
- Code formatting checks (black)
- Linting (ruff)
- Type checking (mypy)
- Coverage reporting

## Architecture Details

### RAG Pipeline

1. **Query** → User asks a question
2. **Retrieval** → Vector store finds relevant SNPs
3. **Validation** → dbSNP confirms SNP metadata
4. **Filtering** → User's DNA matched against SNPs
5. **Augmentation** → LLM interprets with context
6. **Response** → Structured output with confidence

### Data Flow

```
User Question
     ↓
Vector Store Search (semantic)
     ↓
SNP Validation (dbSNP API)
     ↓
DNA File Filtering
     ↓
LLM Interpretation
     ↓
Structured Result + Confidence
```

## Modules

### `snp_database.py`
- Validates RS IDs through NCBI dbSNP
- Caches results (TTL cache)
- Extracts chromosome, position, gene, alleles

### `vector_store.py`
- ChromaDB integration
- Sentence transformers for embeddings
- Semantic search over SNP traits
- Persistent storage support

### `chat_dna_enhanced.py`
- Extends original ChatDNA
- RAG-based SNP retrieval
- Conversation history
- Structured outputs with confidence scores
- Source citation

### `polygenic_scores.py`
- Polygenic risk score calculation
- Multiple trait support
- Percentile estimation
- Interpretation generation

### `vcf_parser.py`
- VCF file parsing
- Gzip support
- RS ID extraction
- Genotype conversion

## Configuration

### Vector Store

```python
store = SNPVectorStore(
    persist_directory=Path("./data"),
    embedding_model="all-MiniLM-L6-v2",
    collection_name="snp_traits"
)
```

### SNP Database

```python
db = SNPDatabase(
    cache_ttl=3600,        # 1 hour cache
    max_cache_size=1000,   # 1000 SNPs
    request_timeout=10.0   # 10 seconds
)
```

## Limitations & Disclaimers

**IMPORTANT**: This is a research/educational tool. Not for medical use.

- SNP associations are simplified
- LLM responses may contain hallucinations
- Genetic risk is probabilistic, not deterministic
- Many factors beyond genetics affect traits
- **Always consult healthcare professionals for medical decisions**

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run linters and tests
5. Submit pull request

## License

See LICENSE file for details.
