# DNA RAG API Reference

> REST + WebSocket API built with FastAPI.

Start the server:

```bash
# Development
dna-rag-api

# Or directly
uvicorn dna_rag.api.main:app --reload --host 0.0.0.0 --port 8000

# Docker
make docker-build && make docker-up
```

Interactive docs: `http://localhost:8000/docs` (Swagger UI).

---

## Endpoints

### Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/analyze` | Synchronous analysis (upload + question) |
| `POST` | `/api/v1/analyze/async` | Submit background job, returns `job_id` |
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status / retrieve result |
| `WS` | `/ws/v1/analyze` | WebSocket with step-by-step streaming |

### File Management

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/files` | Upload DNA file, returns `file_id` |
| `GET` | `/api/v1/files/{file_id}` | File metadata |
| `DELETE` | `/api/v1/files/{file_id}` | Remove uploaded file |

### System

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness probe |
| `GET` | `/ready` | Readiness probe (checks engine) |
| `GET` | `/api/v1/formats` | Supported DNA file formats |

---

## `POST /api/v1/analyze`

Synchronous DNA analysis. Provide **either** `file` (upload) **or** `file_id`.

**Request** (`multipart/form-data`):

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | yes | Question about the DNA data |
| `file` | binary | one of | DNA file upload |
| `file_id` | string | one of | Reference to a previously uploaded file |

**Response** (`200 OK`):

```json
{
  "id": "ana_01J8K...",
  "question": "What is my caffeine metabolism?",
  "matched_snps": [
    {
      "rsid": "rs762551",
      "chromosome": "15",
      "position": 75041917,
      "genotype": "AC"
    }
  ],
  "interpretation": "Based on your CYP1A2 genotype ...",
  "snp_count_requested": 5,
  "snp_count_matched": 3,
  "cached": false,
  "processing_time_ms": 4200,
  "llm_models": {
    "snp_identification": "deepseek-r1:free",
    "interpretation": "gpt-4o-mini"
  }
}
```

**Errors**:

| Status | When |
|--------|------|
| `400` | Missing file/file_id, parsing error, invalid format |
| `404` | `file_id` not found |
| `422` | No matching variants in DNA file |
| `502` | LLM provider error |

---

## `POST /api/v1/analyze/async`

Submit analysis as a background job.

**Request**: same as `/api/v1/analyze`.

**Response** (`202 Accepted`):

```json
{
  "job_id": "job_01J8K...",
  "status": "pending",
  "poll_url": "/api/v1/jobs/job_01J8K...",
  "websocket_url": "/ws/v1/jobs/job_01J8K..."
}
```

---

## `GET /api/v1/jobs/{job_id}`

Poll status of an async job.

**Response** (`200 OK`):

```json
{
  "job_id": "job_01J8K...",
  "status": "completed",
  "created_at": "2025-01-15T10:30:00Z",
  "completed_at": "2025-01-15T10:30:04Z",
  "result": { "...same as /api/v1/analyze response..." },
  "error": null
}
```

Status values: `pending` → `running` → `completed` | `failed`.

---

## `POST /api/v1/files`

Upload a DNA data file.

**Request**: `multipart/form-data` with a single `file` field.

**Response** (`201 Created`):

```json
{
  "file_id": "file_abc123",
  "filename": "genome_data.txt",
  "format": "23andme",
  "row_count": 610000,
  "size_bytes": 23456789,
  "uploaded_at": "2025-01-15T10:30:00Z"
}
```

---

## `GET /api/v1/files/{file_id}`

Returns metadata for a previously uploaded file (same shape as upload response).

---

## `DELETE /api/v1/files/{file_id}`

Removes an uploaded file. Returns `204 No Content` on success, `404` if not found.

---

## WebSocket `/ws/v1/analyze`

Real-time progress for the two-step pipeline.

**Client sends**:

```json
{ "question": "caffeine metabolism", "file_id": "file_abc123" }
```

**Server streams** (`ProgressEvent`):

```
→ { "step": "parsing",             "status": "started" }
→ { "step": "parsing",             "status": "done",    "data": {"rows": 610000} }
→ { "step": "snp_identification",  "status": "started", "data": {"model": "deepseek-r1:free"} }
→ { "step": "snp_identification",  "status": "done",    "data": {"snps_found": 5} }
→ { "step": "filtering",           "status": "started" }
→ { "step": "filtering",           "status": "done",    "data": {"matched": 3} }
→ { "step": "interpretation",      "status": "started", "data": {"model": "gpt-4o-mini"} }
→ { "step": "interpretation",      "status": "done" }
→ { "step": "complete",            "status": "done",    "data": { ...result... } }
```

On error:

```json
{ "step": "error", "status": "error", "data": {"message": "..."} }
```

---

## Error Format

All errors follow RFC 7807-inspired format:

```json
{
  "error": {
    "code": "NO_MATCHING_VARIANTS",
    "message": "None of the identified SNPs were found in your DNA file.",
    "details": { "snps_requested": 8, "snps_matched": 0 }
  }
}
```

---

## Configuration

See the [README](../README.md#configuration) for all `DNA_RAG_*` environment variables.

API-specific settings (extend core config):

| Variable | Default | Description |
|----------|---------|-------------|
| `DNA_RAG_API_HOST` | `0.0.0.0` | Server bind address |
| `DNA_RAG_API_PORT` | `8000` | Server port |
| `DNA_RAG_API_WORKERS` | `1` | Uvicorn worker count |
| `DNA_RAG_CORS_ORIGINS` | `["*"]` | Allowed CORS origins |
| `DNA_RAG_AUTH_ENABLED` | `false` | Enable API key auth |
| `DNA_RAG_API_KEYS` | — | Comma-separated valid API keys |
| `DNA_RAG_UPLOAD_DIR` | `/tmp/dna_rag_uploads` | File upload directory |
| `DNA_RAG_FILE_MAX_SIZE_MB` | `50` | Max upload size (1–200 MB) |
| `DNA_RAG_FILE_RETENTION_HOURS` | `24` | Auto-cleanup after N hours |
| `DNA_RAG_RATE_LIMIT_PER_MINUTE` | `60` | Requests per minute per key |
| `DNA_RAG_JOB_TTL_SECONDS` | `3600` | Job result lifetime |

---

## Examples

### cURL

```bash
# Health check
curl http://localhost:8000/health

# Upload file
curl -X POST http://localhost:8000/api/v1/files \
  -F "file=@genome_data.csv"

# Analyze with file upload
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "question=lactose tolerance" \
  -F "file=@genome_data.csv"

# Analyze with file_id
curl -X POST http://localhost:8000/api/v1/analyze \
  -F "question=caffeine metabolism" \
  -F "file_id=file_abc123"

# Supported formats
curl http://localhost:8000/api/v1/formats
```

### Python

```python
import httpx

base = "http://localhost:8000"

# Upload
with open("genome.csv", "rb") as f:
    r = httpx.post(f"{base}/api/v1/files", files={"file": f})
file_id = r.json()["file_id"]

# Analyze
r = httpx.post(
    f"{base}/api/v1/analyze",
    data={"question": "lactose tolerance", "file_id": file_id},
)
print(r.json()["interpretation"])
```
