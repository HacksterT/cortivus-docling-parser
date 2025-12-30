# PRD: Cortivus Docling Parser

## Document Control

| Field | Value |
|-------|-------|
| Version | 3.0 |
| Status | Phase 3 Complete - Intelligent Parser |
| Created | 2024-12-30 |
| Author | Product Team |
| Repository | cortivus-docling-parser |

---

## Overview

### Problem Statement

Applications in the Cortivus ecosystem need to extract text from various document formats (PDF, DOCX, HTML, audio) and split the content into semantically meaningful chunks for downstream processing such as embeddings, RAG pipelines, and text analysis. Currently, each application would need to implement its own parsing logic, leading to:

- Duplicated code across projects
- Inconsistent parsing quality
- Maintenance burden for multiple implementations
- Complex dependencies in each application

### Solution

A standalone, stateless document parsing microservice that:

- Accepts documents via REST API
- Parses content to markdown format using Docling
- Chunks content intelligently with configurable parameters
- Returns structured JSON responses
- Runs as a Docker container for easy deployment

### Key Differentiators

| Feature | Description |
|---------|-------------|
| **Stateless** | No database, no persistent storage - pure function of input to output |
| **Multi-format** | PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, and audio files |
| **Intelligent Chunking** | Uses Docling HybridChunker respecting document structure |
| **Token-aware** | Chunks respect embedding model token limits |
| **Plug-and-play** | Single Docker container, no external dependencies |
| **Table Extraction** | Structured JSON output for tables (headers + rows) |
| **Image Detection** | Identifies figures, charts, diagrams for AI description |
| **Granite Vision** | AI-powered image descriptions (requires GPU) |
| **Intelligent Routing** | Auto-selects optimal processing pipeline per element type |

---

## User Stories

### Story 1: Parse Uploaded Document

**As a** backend developer
**I want to** upload a document file to a parsing service
**So that** I can receive structured markdown content with chunks for further processing

**Acceptance Criteria:**
- Can upload PDF, DOCX, PPTX, XLSX, HTML, TXT, MD files
- Receives markdown content extracted from document
- Receives array of chunks with token counts
- Receives document metadata (title, word count)
- Processing time is included in response

### Story 2: Parse Raw Text

**As a** backend developer
**I want to** send raw text content to a parsing service
**So that** I can receive chunked content for embedding generation

**Acceptance Criteria:**
- Can POST JSON with text content
- Can specify chunking parameters
- Receives array of chunks with token counts
- Title is auto-extracted or can be provided

### Story 3: Parse Audio File

**As a** backend developer
**I want to** upload audio files for transcription
**So that** I can process spoken content alongside documents

**Acceptance Criteria:**
- Can upload MP3, WAV, M4A, FLAC files
- Audio is transcribed using Whisper ASR
- Transcription is chunked like other documents
- Response format is identical to document parsing

### Story 4: Configure Chunking Parameters

**As a** backend developer
**I want to** customize chunk size and overlap
**So that** I can optimize for my specific embedding model and use case

**Acceptance Criteria:**
- Can specify chunk_size (100-5000 characters)
- Can specify chunk_overlap (0-500 characters)
- Can specify max_tokens (100-2000)
- Defaults are sensible for common use cases

### Story 5: Check Service Health

**As a** DevOps engineer
**I want to** check if the parser service is healthy
**So that** I can configure health checks in my orchestration

**Acceptance Criteria:**
- GET /health returns status
- Response includes service name and version
- Can be used for Docker/Kubernetes health checks

---

## Functional Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| FR-1 | Accept file uploads via multipart form POST | P0 |
| FR-2 | Accept raw text content via JSON POST | P0 |
| FR-3 | Parse PDF documents to markdown using Docling | P0 |
| FR-4 | Parse DOCX/DOC documents to markdown | P0 |
| FR-5 | Parse PPTX/PPT presentations to markdown | P1 |
| FR-6 | Parse XLSX/XLS spreadsheets to markdown | P1 |
| FR-7 | Parse HTML/HTM pages to markdown | P1 |
| FR-8 | Parse plain text and markdown files | P0 |
| FR-9 | Transcribe audio files using Whisper ASR | P1 |
| FR-10 | Chunk parsed content with configurable size/overlap | P0 |
| FR-11 | Return token counts per chunk | P0 |
| FR-12 | Extract document metadata (title, word count) | P0 |
| FR-13 | Provide health check endpoint | P0 |
| FR-14 | Return processing time in response | P1 |
| FR-15 | List supported file types via API | P2 |

---

## Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-1 | Response time for typical documents (<50 pages) | < 5 seconds |
| NFR-2 | Response time for audio files (<10 minutes) | < 60 seconds |
| NFR-3 | Container memory footprint | < 2GB idle |
| NFR-4 | Concurrent request handling | 10+ simultaneous |
| NFR-5 | Uptime when deployed | 99.9% |
| NFR-6 | Startup time from container start | < 30 seconds |

---

## API Specification

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/parse/file` | POST | Parse uploaded document file |
| `/parse/text` | POST | Parse raw text content |
| `/supported-types` | GET | List supported file formats |

### Request: POST /parse/file

**Content-Type:** `multipart/form-data`

**Form Fields:**
- `file` (required): The document file to parse

**Query Parameters:**
- `chunk_size` (optional, default=1000): Target chunk size in characters (100-5000)
- `chunk_overlap` (optional, default=200): Overlap between chunks (0-500)
- `max_tokens` (optional, default=512): Maximum tokens per chunk (100-2000)

### Request: POST /parse/text

**Content-Type:** `application/json`

```json
{
  "content": "string (required)",
  "title": "string (optional)",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_tokens": 512
}
```

### Response: ParseResponse

```json
{
  "success": true,
  "filename": "document.pdf",
  "file_type": "document",
  "title": "Document Title",
  "markdown_content": "# Title\n\nContent...",
  "chunks": [
    {
      "index": 0,
      "content": "Chunk text content...",
      "token_count": 128,
      "start_char": 0,
      "end_char": 450,
      "metadata": {}
    }
  ],
  "metadata": {
    "word_count": 1500,
    "char_count": 8500
  },
  "processing_time_ms": 1234.56,
  "error": null
}
```

### Response: SupportedTypesResponse

```json
{
  "document": [".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls", ".html", ".htm"],
  "text": [".txt", ".md", ".markdown"],
  "audio": [".mp3", ".wav", ".m4a", ".flac"]
}
```

---

## Technical Architecture

### Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | FastAPI |
| Runtime | Python 3.11 |
| Parser | Docling with HybridChunker |
| Audio | OpenAI Whisper (local) |
| Tokenizer | sentence-transformers/all-MiniLM-L6-v2 |
| Container | Docker |
| Package Manager | UV |

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                  Cortivus Docling Parser                    │
│                  (Stateless Microservice)                   │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐         ┌─────▼─────┐       ┌─────▼─────┐
    │ /parse/ │         │ /parse/   │       │/supported-│
    │  file   │         │   text    │       │   types   │
    └────┬────┘         └─────┬─────┘       └───────────┘
         │                    │
    ┌────▼────────────────────▼────┐
    │       Parser Service         │
    │  ┌────────┐  ┌────────────┐  │
    │  │Docling │  │ Whisper    │  │
    │  │(docs)  │  │ (audio)    │  │
    │  └────────┘  └────────────┘  │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │      Chunker Service         │
    │  ┌────────┐  ┌────────────┐  │
    │  │Hybrid  │  │  Simple    │  │
    │  │Chunker │  │  Chunker   │  │
    │  └────────┘  └────────────┘  │
    └──────────────┬───────────────┘
                   │
    ┌──────────────▼───────────────┐
    │     JSON Response            │
    │     (ParseResponse)          │
    └──────────────────────────────┘
```

### Project Structure

```
cortivus-docling-parser/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # REST endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parser.py        # Document parsing
│   │   └── chunker.py       # Text chunking
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models
├── tests/
│   └── sample_documents/
├── tasks/
│   └── prd-docling-parser.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Non-Goals (MVP)

The following are explicitly out of scope for this service:

| Non-Goal | Rationale |
|----------|-----------|
| Database storage | Caller's responsibility |
| Embedding generation | Caller's responsibility |
| Authentication/Authorization | Internal service, protected by network |
| Rate limiting | Handled by API gateway or caller |
| File size validation | Memory constraints natural limit |
| Advanced OCR for scanned PDFs | Docling handles basic OCR |
| Caching | Stateless design, caller can cache |
| Batch processing | Single-request model for simplicity |

---

## Integration Guide

### Docker Compose Integration

Add to your application's `docker-compose.yml`:

```yaml
services:
  docling-parser:
    build: ../cortivus-docling-parser
    container_name: cortivus_docling_parser
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
    networks:
      - your-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8001/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  backend:
    environment:
      DOCLING_PARSER_URL: http://docling-parser:8001
    depends_on:
      docling-parser:
        condition: service_healthy
```

### Backend Configuration

Add to your backend's config:

```python
# config.py
DOCLING_PARSER_URL: str = "http://docling-parser:8001"
```

### Usage Example

```python
import httpx

async def parse_document(file_content: bytes, filename: str):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DOCLING_PARSER_URL}/parse/file",
            files={"file": (filename, file_content)},
            params={"chunk_size": 1000, "max_tokens": 512}
        )
        return response.json()
```

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Parser container starts successfully | Pass/Fail | docker-compose up |
| Health endpoint responds | HTTP 200 | curl /health |
| PDF parsing works | Returns chunks | Test with sample PDF |
| DOCX parsing works | Returns chunks | Test with sample DOCX |
| Text parsing works | Returns chunks | Test with JSON payload |
| Chunk token counts accurate | Within 10% of max_tokens | Verify with tokenizer |
| Processing time reported | Non-zero ms | Check response field |
| Error handling works | Returns error message | Test with invalid file |

---

## Testing Plan

### Unit Tests

- Parser service functions
- Chunker logic
- Pydantic model validation

### Integration Tests

```bash
# Health check
curl http://localhost:8001/health

# Parse PDF
curl -X POST "http://localhost:8001/parse/file" \
  -F "file=@tests/sample_documents/sample.pdf"

# Parse text
curl -X POST "http://localhost:8001/parse/text" \
  -H "Content-Type: application/json" \
  -d '{"content": "Test content...", "chunk_size": 500}'

# Get supported types
curl http://localhost:8001/supported-types
```

### Load Tests

- 10 concurrent document uploads
- Memory usage under load
- Response time consistency

---

## Deployment

### Standalone

```bash
cd cortivus-docling-parser
docker-compose up --build
```

### As Part of Cortivus Stack

```bash
cd cortivus-career-intelligence
# Parser is included in docker-compose.yml
docker-compose up --build
```

---

## Future Enhancements (Post-MVP)

| Enhancement | Description | Priority |
|-------------|-------------|----------|
| Batch endpoint | Parse multiple files in one request | P2 |
| Streaming response | Stream chunks as they're generated | P2 |
| Custom tokenizers | Support different embedding models | P3 |
| OCR enhancement | Better scanned document handling | P3 |
| URL parsing | Parse documents from URLs | P3 |
| Caching layer | Optional Redis caching | P3 |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12-30 | Initial PRD - Implementation complete |
