# DNA_RAG Configuration Guide

DNA_RAG uses environment variables for configuration management. This guide explains all available configuration options.

## Quick Start

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and set your API key:
```bash
DEEPSEEK_API_KEY=your_actual_api_key_here
```

3. Run the application - it will automatically load settings from `.env`

## Configuration Methods

DNA_RAG supports three ways to configure settings (in order of precedence):

1. **Direct environment variables** (highest priority)
2. **.env file** in the project root
3. **Default values** (lowest priority)

## Required Settings

### DEEPSEEK_API_KEY
- **Required**: Yes
- **Description**: Your DeepSeek API key
- **How to get**: Sign up at https://platform.deepseek.com
- **Example**: `DEEPSEEK_API_KEY=sk-abc123...`

**Note**: For backward compatibility, `API_KEY` is also supported but `DEEPSEEK_API_KEY` is preferred.

## Optional Settings

All optional settings have sensible defaults and use the `DNA_RAG_` prefix.

### SNP Database Settings

#### DNA_RAG_CACHE_TTL
- **Default**: `3600` (1 hour)
- **Unit**: Seconds
- **Description**: How long to cache SNP validation results from dbSNP
- **Example**: `DNA_RAG_CACHE_TTL=7200` (2 hours)

#### DNA_RAG_MAX_CACHE_SIZE
- **Default**: `1000`
- **Min**: `1`
- **Description**: Maximum number of SNPs to cache in memory
- **Example**: `DNA_RAG_MAX_CACHE_SIZE=500`

#### DNA_RAG_REQUEST_TIMEOUT
- **Default**: `10.0`
- **Unit**: Seconds
- **Min**: `1.0`
- **Description**: Timeout for HTTP requests to external APIs
- **Example**: `DNA_RAG_REQUEST_TIMEOUT=15.0`

### Vector Store Settings

#### DNA_RAG_VECTOR_STORE_PATH
- **Default**: `./data/vector_store`
- **Description**: Path to ChromaDB vector store directory
- **Example**: `DNA_RAG_VECTOR_STORE_PATH=/var/lib/dna_rag/vectors`

#### DNA_RAG_EMBEDDING_MODEL
- **Default**: `all-MiniLM-L6-v2`
- **Description**: Sentence transformer model for generating embeddings
- **Options**:
  - `all-MiniLM-L6-v2` (fast, 384 dimensions)
  - `all-mpnet-base-v2` (better quality, 768 dimensions)
  - `paraphrase-MiniLM-L6-v2` (optimized for paraphrasing)
- **Example**: `DNA_RAG_EMBEDDING_MODEL=all-mpnet-base-v2`

#### DNA_RAG_USE_VECTOR_STORE
- **Default**: `true`
- **Options**: `true`, `false`
- **Description**: Enable/disable RAG (Retrieval-Augmented Generation)
- **Example**: `DNA_RAG_USE_VECTOR_STORE=false`

### Validation Settings

#### DNA_RAG_USE_VALIDATION
- **Default**: `true`
- **Options**: `true`, `false`
- **Description**: Enable/disable SNP validation through NCBI dbSNP API
- **Example**: `DNA_RAG_USE_VALIDATION=false`

**Note**: Disabling validation speeds up queries but reduces accuracy.

### LLM Settings

#### DNA_RAG_LLM_MODEL
- **Default**: `deepseek-r1:free`
- **Description**: DeepSeek model to use
- **Example**: `DNA_RAG_LLM_MODEL=deepseek-chat`

#### DNA_RAG_LLM_TEMPERATURE
- **Default**: `0.0`
- **Range**: `0.0` to `2.0`
- **Description**: LLM temperature (higher = more creative, lower = more deterministic)
- **Example**: `DNA_RAG_LLM_TEMPERATURE=0.5`

#### DNA_RAG_LLM_MAX_RETRIES
- **Default**: `2`
- **Min**: `0`
- **Description**: Maximum number of retries for failed LLM API calls
- **Example**: `DNA_RAG_LLM_MAX_RETRIES=3`

## Using Configuration in Code

### Basic Usage

```python
from config import get_settings

# Get global settings instance
settings = get_settings()

# Validate API key is set
settings.validate_api_key()

# Access settings
print(f"Using model: {settings.llm_model}")
print(f"Cache TTL: {settings.cache_ttl} seconds")
```

### With ChatDNAEnhanced

```python
from chat_dna_enhanced import ChatDNAEnhanced
from config import get_settings
from pathlib import Path

settings = get_settings()
settings.validate_api_key()

chat = ChatDNAEnhanced(
    api_key=settings.deepseek_api_key,
    use_vector_store=settings.use_vector_store,
    use_validation=settings.use_validation,
    vector_store_path=settings.get_vector_store_path(),
)

result = chat.ask("lactose tolerance", Path("dna.csv"))
```

### Reload Settings

Useful for testing or when environment changes:

```python
from config import reload_settings

# Reload from environment
settings = reload_settings()
```

## Environment File (.env)

Create a `.env` file in the project root:

```bash
# Required
DEEPSEEK_API_KEY=sk-your-key-here

# Optional overrides
DNA_RAG_CACHE_TTL=7200
DNA_RAG_USE_VECTOR_STORE=true
DNA_RAG_LLM_TEMPERATURE=0.0
```

**Important**:
- Never commit `.env` to version control (already in `.gitignore`)
- Use `.env.example` as a template
- Sensitive values (API keys) should only be in `.env`

## Command-Line Usage

### Basic CLI (cli.py)
```bash
export DEEPSEEK_API_KEY=your_key
python cli.py --dna-file dna.csv --question "lactose tolerance"
```

### Enhanced CLI (cli_enhanced.py)
```bash
# Uses settings from .env automatically
python cli_enhanced.py --dna-file dna.csv --question "eye color"

# Override RAG
python cli_enhanced.py --dna-file dna.csv --question "test" --no-rag

# Disable validation (faster but less accurate)
python cli_enhanced.py --dna-file dna.csv --question "test" --no-validation
```

## Configuration Profiles

For different environments, you can use different .env files:

```bash
# Development
cp .env.example .env.dev
# Set dev values in .env.dev

# Production
cp .env.example .env.prod
# Set prod values in .env.prod

# Load specific file
export $(cat .env.dev | xargs)
python cli_enhanced.py ...
```

## Troubleshooting

### "DEEPSEEK_API_KEY environment variable is not set"
- Check that `.env` file exists in project root
- Verify `DEEPSEEK_API_KEY` is set in `.env`
- Try setting directly: `export DEEPSEEK_API_KEY=your_key`

### Vector store path errors
- Ensure the directory exists or is creatable
- Check write permissions
- Use absolute path if relative path fails

### LLM timeouts
- Increase `DNA_RAG_REQUEST_TIMEOUT`
- Check internet connection
- Verify API key is valid

### Cache not working
- Check that `DNA_RAG_CACHE_TTL` > 0
- Verify `DNA_RAG_MAX_CACHE_SIZE` > 0
- Cache is in-memory, so restarts clear it

## Security Best Practices

1. **Never commit API keys**: Always use `.env` (already in `.gitignore`)
2. **Rotate keys regularly**: Update `DEEPSEEK_API_KEY` periodically
3. **Use environment-specific configs**: Different keys for dev/prod
4. **Restrict file permissions**: `chmod 600 .env`
5. **Use secrets management**: For production, consider vault services

## Performance Tuning

### For Speed
```bash
DNA_RAG_USE_VALIDATION=false       # Skip dbSNP validation
DNA_RAG_REQUEST_TIMEOUT=5.0        # Faster timeouts
DNA_RAG_CACHE_TTL=86400            # Cache for 24 hours
```

### For Accuracy
```bash
DNA_RAG_USE_VALIDATION=true        # Enable validation
DNA_RAG_USE_VECTOR_STORE=true      # Enable RAG
DNA_RAG_LLM_TEMPERATURE=0.0        # Deterministic responses
```

### For Development
```bash
DNA_RAG_USE_VALIDATION=false       # Skip external API calls
DNA_RAG_CACHE_TTL=30               # Short cache for testing
```

## Reference

See `.env.example` for a complete, annotated configuration template.

For programmatic access, see the `DNARAGSettings` class in `config.py`.
