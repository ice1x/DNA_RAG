# DNA RAG API Documentation

## Overview

DNA RAG provides a RESTful API built with FastAPI for analyzing DNA data using large language models with Retrieval-Augmented Generation (RAG).

## Running the API Server

### Development Mode

```bash
uvicorn dna_rag.api.app:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode

```bash
uvicorn dna_rag.api.app:app --host 0.0.0.0 --port 8000 --workers 4
```

## Environment Variables

Set up your API keys in `.env` file:

```bash
DNA_RAG_OPENAI_API_KEY=your_openai_key
DNA_RAG_DEEPSEEK_API_KEY=your_deepseek_key
DNA_RAG_LLM_PROVIDERS=openai,deepseek
```

## API Endpoints

### Health Check

**GET** `/health`

Check API health and available providers.

**Response:**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "providers": {
    "openai": {
      "available": true,
      "model": "gpt-4o-mini"
    },
    "deepseek": {
      "available": true,
      "model": "deepseek-chat"
    }
  }
}
```

### Ask DNA Question (Basic)

**POST** `/ask`

Ask a question about DNA data using basic LLM analysis.

**Request Body:**
```json
{
  "question": "Am I lactose intolerant?",
  "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n..."
}
```

**Response:**
```json
{
  "question": "Am I lactose intolerant?",
  "answer": "Based on your DNA, you are likely lactose tolerant...",
  "provider": "openai"
}
```

### Ask DNA Question (Enhanced)

**POST** `/ask-enhanced`

Ask a question with enhanced features including RAG, SNP validation, and structured output.

**Request Body:**
```json
{
  "question": "Am I lactose intolerant?",
  "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n...",
  "use_rag": true
}
```

**Response:**
```json
{
  "question": "Am I lactose intolerant?",
  "interpretation": "You are likely lactose tolerant based on your genetics.",
  "confidence": 0.85,
  "snps_found": [
    {
      "rsid": "rs4988235",
      "genotype": "AG",
      "gene": "LCT",
      "trait": "lactose tolerance",
      "validated": true,
      "similarity": 0.95
    }
  ],
  "sources": ["dbSNP", "ClinVar"],
  "caveats": ["This is not medical advice"],
  "provider": "openai"
}
```

### Calculate Polygenic Score

**POST** `/polygenic-score`

Calculate a polygenic risk score from DNA data.

**Request Body:**
```json
{
  "score_name": "alzheimers_risk",
  "dna_data": "rsid,chromosome,position,genotype\nrs429358,19,45411941,CT\n..."
}
```

**Response:**
```json
{
  "score_name": "alzheimers_risk",
  "score": 1.25,
  "interpretation": "Elevated risk compared to population average",
  "percentile": 75.5
}
```

## Error Responses

All endpoints return error responses in the following format:

```json
{
  "error": "ValidationError",
  "detail": "DNA data must contain columns: rsid, chromosome, position, genotype",
  "status_code": 400
}
```

### HTTP Status Codes

- `200 OK` - Request succeeded
- `400 Bad Request` - Invalid request data
- `500 Internal Server Error` - Server error
- `503 Service Unavailable` - Service not initialized

## Interactive API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json

## Example Usage

### Python

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "Am I lactose intolerant?",
        "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n"
    }
)
print(response.json())
```

### cURL

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Am I lactose intolerant?",
    "dna_data": "rsid,chromosome,position,genotype\nrs4988235,2,136608646,AG\n"
  }'
```

## Rate Limiting

Currently, there are no rate limits. In production, consider implementing rate limiting using middleware.

## Authentication

Currently, the API does not require authentication. In production, consider implementing API key authentication or OAuth2.
