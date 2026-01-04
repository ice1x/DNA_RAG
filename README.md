# DNA_RAG

A toolkit for analyzing DNA data with language models, featuring RAG (Retrieval-Augmented Generation), SNP validation, and polygenic risk scores.

## Project Structure

```
DNA_RAG/
├── src/
│   └── dna_rag/
│       ├── api/              # FastAPI endpoints
│       │   ├── __init__.py
│       │   └── app.py
│       ├── core/             # Core business logic
│       │   ├── __init__.py
│       │   ├── chat_dna.py
│       │   ├── chat_dna_enhanced.py
│       │   ├── config.py
│       │   └── llm_providers.py
│       ├── models/           # Data models
│       │   ├── __init__.py
│       │   └── api_models.py
│       └── utils/            # Utilities
│           ├── __init__.py
│           ├── langchain_deepseek.py
│           ├── polygenic_scores.py
│           ├── snp_database.py
│           ├── vcf_parser.py
│           └── vector_store.py
├── scripts/                  # CLI scripts
│   ├── cli.py
│   ├── cli_enhanced.py
│   └── run_api.py
├── tests/                    # Test suite
├── docs/                     # Documentation
├── manual_PoC/              # Proof of concept files
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Features

- **DNA Analysis**: Ask questions about your DNA data using LLMs
- **RAG Integration**: Vector store for semantic SNP retrieval
- **SNP Validation**: Validate SNPs against dbSNP database
- **Polygenic Risk Scores**: Calculate risk scores for various conditions
- **Multi-Provider Support**: OpenAI, DeepSeek, and more
- **REST API**: FastAPI endpoints for web integration
- **Type Safety**: Full type hints and Pydantic validation

## Installation

```bash
# Clone repository
git clone https://github.com/ice1x/DNA_RAG.git
cd DNA_RAG

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# Install package in editable mode
pip install -e .
```

## Configuration

Create a `.env` file:

```bash
# LLM Provider API Keys
DNA_RAG_OPENAI_API_KEY=your_openai_key
DNA_RAG_DEEPSEEK_API_KEY=your_deepseek_key

# Provider Priority (comma-separated)
DNA_RAG_LLM_PROVIDERS=openai,deepseek

# Models
DNA_RAG_OPENAI_MODEL=gpt-4o-mini
DNA_RAG_DEEPSEEK_MODEL=deepseek-chat

# Optional Settings
DNA_RAG_USE_VECTOR_STORE=true
DNA_RAG_USE_VALIDATION=true
DNA_RAG_CACHE_TTL=3600
```

See [CONFIGURATION.md](CONFIGURATION.md) for all available settings.

## Usage

### Command Line Interface

#### Basic Usage

```bash
# Ask a question about your DNA
python scripts/cli.py --dna-file path/to/dna.csv --question "lactose tolerance"

# Interactive mode
python scripts/cli.py --dna-file path/to/dna.csv
```

#### Enhanced Mode (with RAG)

```bash
# Enhanced analysis with RAG and validation
python scripts/cli_enhanced.py --dna-file path/to/dna.csv --question "eye color"

# JSON output
python scripts/cli_enhanced.py --dna-file path/to/dna.csv --question "eye color" --json

# Calculate polygenic risk score
python scripts/cli_enhanced.py --dna-file path/to/dna.csv --prs alzheimers_risk
```

### Python API

```python
from pathlib import Path
from dna_rag import ChatDNA

# Basic usage
chat = ChatDNA(api_key="your-key")
answer = chat.ask("lactose tolerance", Path("dna.csv"))
print(answer)

# Enhanced usage with RAG
from dna_rag.core.chat_dna_enhanced import ChatDNAEnhanced

chat = ChatDNAEnhanced(api_key="your-key")
result = chat.ask("eye color", Path("dna.csv"))
print(f"Interpretation: {result.interpretation}")
print(f"Confidence: {result.confidence}")
print(f"SNPs found: {len(result.snps_found)}")
```

### REST API

#### Start the API server

```bash
# Development mode (with auto-reload)
python scripts/run_api.py --reload

# Production mode (with multiple workers)
python scripts/run_api.py --workers 4
```

#### Use the API

```bash
# Health check
curl http://localhost:8000/health

# Ask a question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Am I lactose intolerant?",
    "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n"
  }'

# Enhanced question with RAG
curl -X POST "http://localhost:8000/ask-enhanced" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Am I lactose intolerant?",
    "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n",
    "use_rag": true
  }'

# Calculate polygenic score
curl -X POST "http://localhost:8000/polygenic-score" \
  -H "Content-Type: application/json" \
  -d '{
    "score_name": "alzheimers_risk",
    "dna_data": "rsid,chromosome,position,genotype\nrs429358,19,45411941,CT\n"
  }'
```

#### Interactive API Documentation

Visit http://localhost:8000/docs for Swagger UI or http://localhost:8000/redoc for ReDoc.

See [docs/API.md](docs/API.md) for complete API documentation.

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run API tests only
pytest tests/test_api.py -v
```

### Code Quality

```bash
# Format code
black src tests scripts

# Lint code
ruff check src tests scripts

# Type checking
mypy src
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

## Documentation

- [Configuration Guide](CONFIGURATION.md) - Environment variables and settings
- [LLM Providers](LLM_PROVIDERS.md) - Supported LLM providers and setup
- [API Documentation](docs/API.md) - REST API reference
- [Enhanced Features](README_ENHANCED.md) - RAG, validation, and advanced features
- [Changelog](CHANGES.md) - Version history and changes

## DNA Data Format

### CSV Format (Recommended)

```csv
rsid,chromosome,position,genotype
rs4988235,2,136608646,AG
rs1815739,11,66560624,CT
```

### VCF Format

VCF files are automatically converted to CSV format.

```bash
# Convert VCF to CSV
from dna_rag.utils.vcf_parser import convert_vcf_to_csv
convert_vcf_to_csv("input.vcf", "output.csv")
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Disclaimer

**This tool is for research and educational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment.**

- Always consult qualified healthcare professionals for medical decisions
- Genetic data interpretation is complex and context-dependent
- Results should not be used to make health decisions without professional guidance

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{dna_rag,
  title = {DNA RAG: DNA Analysis with Language Models},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/ice1x/DNA_RAG}
}
```
