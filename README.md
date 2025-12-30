# Cortivus Docling Parser

A stateless, intelligent document parsing microservice that converts various document formats (PDF, DOCX, TXT, HTML, audio) into structured markdown with intelligent chunking, table extraction, and AI-powered image descriptions. Designed as a plug-and-play service for the Cortivus ecosystem and other applications.

---

## For AI Vibe Coding Connections

This section provides everything you need to connect a UI or another service to this document parser.

### Quick Start - Which Docker Compose?

| Your Hardware | Command |
|---------------|---------|
| **NVIDIA GPU** | `docker-compose -f docker-compose.gpu.yml up -d` |
| **No NVIDIA GPU** | `docker-compose up -d` |

**If you have an NVIDIA GPU, always use the GPU version.** It's faster for everything and enables AI image descriptions. The CPU version exists only for machines without NVIDIA GPUs.

**Auto-detection:** The GPU version auto-detects CUDA. No configuration needed - just run the command.

### API Endpoint

```
Base URL: http://localhost:8001
```

### Primary Endpoint for Parsing Documents

```
POST http://localhost:8001/parse/file
Content-Type: multipart/form-data
```

**Form Data:**
- `file`: The document file (required)

**Query Parameters (all optional):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `mode` | `auto` | Processing mode: `auto`, `ocr_heavy`, `table_focus`, `vision_enabled` |
| `chunk_size` | `1000` | Target chunk size in characters |
| `chunk_overlap` | `200` | Overlap between chunks |
| `max_tokens` | `512` | Max tokens per chunk |
| `extract_tables` | `true` | Extract tables as structured JSON |
| `detect_images` | `true` | Detect images in document |

### Example Requests

**JavaScript/Fetch:**
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8001/parse/file?mode=auto', {
  method: 'POST',
  body: formData
});
const result = await response.json();
```

**Python:**
```python
import requests

with open('document.pdf', 'rb') as f:
    response = requests.post(
        'http://localhost:8001/parse/file',
        files={'file': f},
        params={'mode': 'auto'}
    )
result = response.json()
```

**cURL:**
```bash
curl -X POST "http://localhost:8001/parse/file?mode=auto" \
  -F "file=@document.pdf"
```

### Response Structure

```json
{
  "success": true,
  "filename": "document.pdf",
  "file_type": "pdf",
  "title": "Document Title",
  "markdown_content": "# Full markdown content...",
  "chunks": [
    {"index": 0, "content": "...", "token_count": 150}
  ],
  "tables": [
    {"index": 0, "headers": ["Col1", "Col2"], "rows": [["a", "b"]]}
  ],
  "images": [
    {"index": 0, "description": "AI description of image", "image_type": "picture"}
  ],
  "docling_document": { ... },
  "metadata": {"word_count": 500, "table_count": 1, "image_count": 1},
  "processing_time_ms": 1250.5,
  "error": null
}
```

### The DoclingDocument (Native Docling Structure)

The `docling_document` field contains the **full native Docling document structure**. This is the recommended way to extract structured data from documents - use this instead of regex parsing the markdown.

**Key properties in DoclingDocument:**
- `texts` - All text items (paragraphs, headings, equations)
- `tables` - Table items with full structure and annotations
- `pictures` - Pictures with captions and AI descriptions
- `key_value_items` - Extracted key-value pairs (great for forms/JDs)
- `body` - Tree structure with reading order hierarchy
- `groups` - Grouped items (lists, chapters, sections)
- `furniture` - Headers, footers, and non-body content

**Example: Extracting structured data from a Job Description:**
```python
result = response.json()
doc = result["docling_document"]

# Get all text content with semantic structure
for text_item in doc.get("texts", []):
    print(f"Type: {text_item.get('label')}, Text: {text_item.get('text')}")

# Get key-value pairs (job title, salary, requirements, etc.)
for kv in doc.get("key_value_items", []):
    print(f"Key: {kv.get('key')}, Value: {kv.get('value')}")

# Get tables with full structure
for table in doc.get("tables", []):
    print(f"Table: {table}")
```

**Why use DoclingDocument instead of markdown?**
- Native semantic structure (headings, lists, sections)
- Key-value extraction (no regex needed)
- Reading order preserved
- Table structure with cell coordinates
- Figure-caption matching
- Language detection, title, authors

For simple use cases, `markdown_content` and `chunks` work fine. For structured extraction (like JD parsing), use `docling_document`.

### Health Check

```
GET http://localhost:8001/health
```

Returns service status and available features.

### Get Processing Modes (for UI dropdowns)

```
GET http://localhost:8001/processing-modes
```

Returns:
```json
{
  "modes": [
    {"value": "auto", "name": "AUTO", "description": "Default - works for most documents, balanced settings"},
    {"value": "ocr_heavy", "name": "OCR_HEAVY", "description": "Optimized for scanned documents and image-based PDFs"},
    {"value": "table_focus", "name": "TABLE_FOCUS", "description": "Prioritize table extraction accuracy"},
    {"value": "vision_enabled", "name": "VISION_ENABLED", "description": "Include AI descriptions of images/charts (GPU recommended)"}
  ]
}
```

Use this to dynamically populate mode selection dropdowns in your UI.

### Processing Modes Explained

| Mode | Use When |
|------|----------|
| `auto` | Default - works for most documents |
| `vision_enabled` | You want AI descriptions of images/charts (GPU recommended) |
| `ocr_heavy` | Scanned documents or image-based PDFs |
| `table_focus` | Documents with lots of tables/data |

### No Manual Configuration Required

- **OCR:** Auto-enabled, auto-selects best engine
- **Tables:** Auto-extracted from all documents
- **Images:** Auto-detected; descriptions generated when `mode=vision_enabled`
- **GPU:** Auto-detected when using `docker-compose.gpu.yml`

---

## Key Features

- **Stateless Architecture**: No database, no embeddings - just parse and return
- **Multi-Format Support**: PDF, DOCX, PPTX, XLSX, HTML, TXT, MD, and audio files
- **Intelligent Chunking**: Uses Docling HybridChunker for semantic boundaries
- **Token-Aware**: Chunks respect embedding model token limits
- **Table Extraction**: Structured JSON output with headers and rows
- **Image Detection**: Identifies figures, charts, and diagrams
- **Granite Vision**: AI-powered image descriptions (GPU optional)
- **Intelligent Routing**: Auto-selects optimal processing per element type
- **REST API**: Simple HTTP endpoints for easy integration
- **Docker Ready**: Single container deployment

## Supported File Types

| Category | Formats |
|----------|---------|
| Document | `.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`, `.html`, `.htm` |
| Text | `.txt`, `.md`, `.markdown` |
| Audio | `.mp3`, `.wav`, `.m4a`, `.flac` |

## Quick Start

### Running with Docker Compose (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd cortivus-docling-parser

# Start the service
docker-compose up --build

# Service will be available at http://localhost:8001
```

### Running Locally (Development)

```bash
# Install dependencies with UV
pip install uv
uv sync

# Start the server
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

---

## Processing Modes

The parser supports intelligent routing with different processing modes:

| Mode | Description | Use Case |
|------|-------------|----------|
| `auto` | Balanced settings for general documents | Default - works well for most documents |
| `ocr_heavy` | Optimized for scanned documents | Scanned PDFs, image-based documents |
| `table_focus` | Prioritize table extraction accuracy | Data-heavy documents, spreadsheets |
| `vision_enabled` | Include AI image descriptions | Documents with charts, figures, medical images |

```bash
# Example: Parse with table focus mode
curl -X POST "http://localhost:8001/parse/file?mode=table_focus" \
  -F "file=@document.pdf"

# Example: Parse with vision enabled (requires Granite Vision)
curl -X POST "http://localhost:8001/parse/file?mode=vision_enabled" \
  -F "file=@medical_report.pdf"
```

---

## Granite Vision Setup (Optional)

Granite Vision provides AI-powered descriptions for images, charts, and figures detected in documents. This is especially useful for:

- Medical images (X-rays, CT scans, MRIs)
- Charts and graphs (extracting data trends)
- Diagrams (describing processes and architectures)
- Figures (understanding visual content)

### Prerequisites

1. **GPU Access** (Recommended): Granite Vision runs much faster with GPU
2. **Docker Desktop with GPU Support**: Enable GPU passthrough in Docker Desktop settings

### Step 1: Enable GPU in Docker Desktop (Windows)

1. Open Docker Desktop
2. Go to **Settings** → **Resources** → **WSL Integration**
3. Ensure WSL 2 is enabled
4. Go to **Settings** → **Docker Engine**
5. The GPU should be automatically available if you have NVIDIA drivers installed

### Step 2: Update docker-compose.yml for GPU

Modify `docker-compose.yml` to include GPU access:

```yaml
version: '3.8'

services:
  parser:
    build: .
    container_name: cortivus_docling_parser
    ports:
      - "8001:8001"
    environment:
      - LOG_LEVEL=INFO
      - GRANITE_MODEL=ibm-granite/granite-vision-3.2-2b
      - GRANITE_DEVICE=auto  # Will use GPU if available
    volumes:
      - ./app:/app/app  # Hot reload for development
      - granite-models:/root/.cache/huggingface  # Cache downloaded models
    # Enable GPU access
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    command: uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload

volumes:
  granite-models:  # Persistent storage for downloaded models
```

### Step 3: Pre-download Granite Vision Model

The model will download automatically on first use, but you can pre-download it:

```bash
# Enter the container
docker exec -it cortivus_docling_parser bash

# Pre-download the model (inside container)
python -c "
from transformers import AutoProcessor, AutoModelForVision2Seq
model_name = 'ibm-granite/granite-vision-3.2-2b'
print('Downloading Granite Vision model...')
AutoProcessor.from_pretrained(model_name)
AutoModelForVision2Seq.from_pretrained(model_name)
print('Model downloaded successfully!')
"
```

### Step 4: Verify Granite Vision is Available

```bash
curl http://localhost:8001/health
```

Response should show:
```json
{
  "status": "healthy",
  "service": "cortivus-docling-parser",
  "version": "0.3.0",
  "features": {
    "ocr_engine": "auto",
    "table_extraction": true,
    "image_detection": true,
    "granite_vision": true,
    "intelligent_routing": true
  },
  "processing_modes": ["auto", "ocr_heavy", "table_focus", "vision_enabled"]
}
```

### Step 5: Use Vision-Enabled Parsing

```bash
# Parse with image descriptions
curl -X POST "http://localhost:8001/parse/file?mode=vision_enabled" \
  -F "file=@document_with_images.pdf"
```

### Running Without GPU (CPU Mode)

Granite Vision can run on CPU, but will be slower:

```yaml
environment:
  - GRANITE_DEVICE=cpu  # Force CPU mode
```

### Model Size and Requirements

| Model | Size | VRAM Required | CPU RAM |
|-------|------|---------------|---------|
| granite-vision-3.2-2b | ~4GB | 8GB+ GPU | 16GB+ |

---

## API Endpoints

### Health Check

Check service status and available features.

```
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "cortivus-docling-parser",
  "version": "0.3.0",
  "features": {
    "ocr_engine": "auto",
    "table_extraction": true,
    "image_detection": true,
    "granite_vision": true,
    "intelligent_routing": true
  },
  "processing_modes": ["auto", "ocr_heavy", "table_focus", "vision_enabled"]
}
```

---

### Parse File

Parse an uploaded document file and return markdown content with chunks, tables, and images.

```
POST /parse/file
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `chunk_size` | int | 1000 | Target chunk size in characters (100-5000) |
| `chunk_overlap` | int | 200 | Overlap between chunks (0-500) |
| `max_tokens` | int | 512 | Maximum tokens per chunk (100-2000) |
| `mode` | string | auto | Processing mode: auto, ocr_heavy, table_focus, vision_enabled |
| `psm_mode` | string | automatic | Tesseract PSM: automatic, single_column, uniform_block, sparse_text |
| `extract_tables` | bool | true | Extract tables as structured JSON |
| `detect_images` | bool | true | Detect images for description |

**Response:**
```json
{
  "success": true,
  "filename": "medical_report.pdf",
  "file_type": "pdf",
  "title": "Patient Lab Results",
  "markdown_content": "# Lab Results\n\n| Test | Value |...",
  "chunks": [
    {
      "index": 0,
      "content": "# Lab Results...",
      "token_count": 150,
      "start_char": 0,
      "end_char": 500,
      "chunk_type": "text",
      "metadata": {}
    }
  ],
  "tables": [
    {
      "index": 0,
      "markdown": "| Test | Value | Range |\n|---|---|---|...",
      "headers": ["Test", "Value", "Range"],
      "rows": [["Hemoglobin", "14.2", "12-16"]],
      "page": 1,
      "confidence": 0.95
    }
  ],
  "images": [
    {
      "index": 0,
      "page": 2,
      "description": "Chest X-ray showing clear lung fields...",
      "image_type": "medical"
    }
  ],
  "metadata": {
    "word_count": 450,
    "table_count": 2,
    "image_count": 1,
    "ocr_applied": true,
    "ocr_engine": "tesseract"
  },
  "processing_time_ms": 1250.5,
  "error": null
}
```

**Examples:**

```bash
# Basic parsing
curl -X POST "http://localhost:8001/parse/file" \
  -F "file=@document.pdf"

# With table extraction and custom chunking
curl -X POST "http://localhost:8001/parse/file?extract_tables=true&chunk_size=500" \
  -F "file=@spreadsheet.pdf"

# Medical document with vision
curl -X POST "http://localhost:8001/parse/file?mode=vision_enabled&psm_mode=single_column" \
  -F "file=@medical_report.pdf"

# Scanned document
curl -X POST "http://localhost:8001/parse/file?mode=ocr_heavy" \
  -F "file=@scanned_document.pdf"
```

**Python Example:**

```python
import requests

# Parse with table extraction
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8001/parse/file",
        files={"file": f},
        params={
            "mode": "auto",
            "extract_tables": True,
            "detect_images": True
        }
    )
    result = response.json()

    if result["success"]:
        print(f"Title: {result['title']}")
        print(f"Chunks: {len(result['chunks'])}")
        print(f"Tables: {len(result['tables'])}")
        print(f"Images: {len(result['images'])}")

        # Access structured table data
        for table in result["tables"]:
            print(f"\nTable {table['index']}:")
            print(f"  Headers: {table['headers']}")
            for row in table["rows"]:
                print(f"  Row: {row}")
```

---

### Parse Text

Parse raw text content and return it chunked with any markdown tables extracted.

```
POST /parse/text
```

**Request Body:**
```json
{
  "content": "Your text content with | tables | here |...",
  "title": "Optional title",
  "chunk_size": 1000,
  "chunk_overlap": 200,
  "max_tokens": 512
}
```

---

### Get Supported Types

List all supported file formats.

```
GET /supported-types
```

---

## Response Schema

### ParseResponse

| Field | Type | Description |
|-------|------|-------------|
| `success` | boolean | Whether parsing succeeded |
| `filename` | string | Original filename (null for text input) |
| `file_type` | string | Detected file type |
| `title` | string | Extracted or provided document title |
| `markdown_content` | string | Full parsed content as markdown |
| `chunks` | array | Chunked content with token counts |
| `tables` | array | Extracted tables with structured data |
| `images` | array | Detected images with descriptions |
| `docling_document` | object | Full native DoclingDocument with texts, tables, pictures, body hierarchy, key_value_items, groups, and furniture. Use for structured extraction. |
| `metadata` | object | Document metadata |
| `processing_time_ms` | float | Processing time in milliseconds |
| `error` | string | Error message if parsing failed |

### TableResponse

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Table index in document |
| `markdown` | string | Table in markdown format |
| `headers` | array | Column headers |
| `rows` | array | Table rows as list of lists |
| `page` | int | Page number (if available) |
| `confidence` | float | Extraction confidence (0-1) |

### ImageResponse

| Field | Type | Description |
|-------|------|-------------|
| `index` | int | Image index in document |
| `page` | int | Page number (if available) |
| `description` | string | AI-generated description (if vision enabled) |
| `image_type` | string | Type: figure, chart, medical, diagram, etc. |

---

## Tesseract PSM Modes

For OCR optimization, you can specify the Tesseract Page Segmentation Mode:

| Mode | Description | Best For |
|------|-------------|----------|
| `automatic` | Fully automatic page segmentation | General documents |
| `single_column` | Single column of variable text | Medical reports, letters |
| `uniform_block` | Uniform block of text | Dense paragraphs |
| `sparse_text` | Sparse text, find as much as possible | Forms with scattered fields |
| `sparse_with_osd` | Sparse text with orientation detection | Tables with gaps |

```bash
# Use single column mode for medical reports
curl -X POST "http://localhost:8001/parse/file?psm_mode=single_column" \
  -F "file=@medical_report.pdf"
```

---

## Architecture

```
                    ┌─────────────────────────────────────────────────────────┐
                    │            Cortivus Docling Parser v0.3.0               │
                    │            (Intelligent Document Parser)                 │
                    └─────────────────────────────────────────────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
        ┌─────▼─────┐                  ┌──────▼──────┐               ┌────────▼────────┐
        │  /parse/  │                  │  /parse/    │               │  /health        │
        │   file    │                  │   text      │               │  /supported-    │
        └─────┬─────┘                  └──────┬──────┘               └─────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │  Mode Router      │
                    │  (Intelligent)    │
                    └─────────┬─────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
   ┌─────▼─────┐        ┌─────▼─────┐        ┌─────▼─────┐
   │  Docling  │        │ Tesseract │        │  Granite  │
   │  Parser   │        │   OCR     │        │  Vision   │
   └─────┬─────┘        └─────┬─────┘        └─────┬─────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
        ┌─────▼─────┐   ┌─────▼─────┐   ┌─────▼─────┐
        │  Tables   │   │  Images   │   │   Text    │
        │ Extractor │   │ Detector  │   │  Chunker  │
        └─────┬─────┘   └─────┬─────┘   └─────┬─────┘
              │               │               │
              └───────────────┼───────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   JSON Response   │
                    │   (ParseResponse) │
                    └───────────────────┘
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | INFO | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `GRANITE_MODEL` | ibm-granite/granite-vision-3.2-2b | Granite Vision model |
| `GRANITE_DEVICE` | auto | Device: auto, cuda, cpu |
| `DEFAULT_CHUNK_SIZE` | 1000 | Default chunk size |
| `DEFAULT_CHUNK_OVERLAP` | 200 | Default overlap |
| `DEFAULT_MAX_TOKENS` | 512 | Default max tokens |

---

## Development

### Project Structure

```
cortivus-docling-parser/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application entry
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # REST endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── parser.py        # Document parsing logic
│   │   ├── chunker.py       # Text chunking logic
│   │   └── vision.py        # Granite Vision integration
│   └── models/
│       ├── __init__.py
│       └── schemas.py       # Pydantic models
├── tests/
│   └── sample_documents/    # Test files
├── tasks/
│   └── prd-docling-parser.md
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## License

Proprietary - Cortivus Career Intelligence
