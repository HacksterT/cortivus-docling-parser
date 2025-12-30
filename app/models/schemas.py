"""
Pydantic models for request/response validation.

Phase 2 Enhanced:
- Structured table data (JSON output)
- Image detection placeholders (for Phase 3 Granite Vision)
- Processing mode configuration
"""

from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime
from enum import Enum


class ProcessingMode(str, Enum):
    """
    Document processing modes.

    - auto: Default - works for most documents, balanced settings
    - ocr_heavy: Optimized for scanned documents and image-based PDFs
    - table_focus: Prioritize table extraction accuracy
    - vision_enabled: Include AI descriptions of images/charts (GPU recommended)
    """
    AUTO = "auto"
    OCR_HEAVY = "ocr_heavy"
    TABLE_FOCUS = "table_focus"
    VISION_ENABLED = "vision_enabled"


# Human-readable descriptions for each processing mode
PROCESSING_MODE_DESCRIPTIONS = {
    ProcessingMode.AUTO: "Default - works for most documents, balanced settings",
    ProcessingMode.OCR_HEAVY: "Optimized for scanned documents and image-based PDFs",
    ProcessingMode.TABLE_FOCUS: "Prioritize table extraction accuracy",
    ProcessingMode.VISION_ENABLED: "Include AI descriptions of images/charts (GPU recommended)",
}


class TesseractPSMMode(str, Enum):
    """Tesseract Page Segmentation Modes for OCR."""
    AUTOMATIC = "automatic"         # PSM 3: Fully automatic
    SINGLE_COLUMN = "single_column" # PSM 4: Single column, variable text
    UNIFORM_BLOCK = "uniform_block" # PSM 6: Uniform text block
    SPARSE_TEXT = "sparse_text"     # PSM 11: Sparse text, find all
    SPARSE_WITH_OSD = "sparse_with_osd"  # PSM 12: Sparse with orientation


class ChunkResponse(BaseModel):
    """Response model for a single document chunk."""
    index: int = Field(..., description="Chunk index (0-based)")
    content: str = Field(..., description="Chunk text content")
    token_count: int = Field(..., description="Estimated token count")
    start_char: int = Field(..., description="Start character position in original document")
    end_char: int = Field(..., description="End character position in original document")
    chunk_type: str = Field(default="text", description="Chunk type: text, table, or mixed")
    metadata: dict = Field(default_factory=dict, description="Additional chunk metadata")


class TableResponse(BaseModel):
    """Structured table data extracted from documents."""
    index: int = Field(..., description="Table index in document")
    markdown: str = Field(..., description="Table in markdown format")
    headers: List[str] = Field(default_factory=list, description="Column headers")
    rows: List[List[str]] = Field(default_factory=list, description="Table rows as list of lists")
    page: Optional[int] = Field(None, description="Page number where table appears")
    confidence: float = Field(default=1.0, description="Extraction confidence (0-1)")


class ImageResponse(BaseModel):
    """Image/figure data detected in documents (Phase 3: Granite Vision)."""
    index: int = Field(..., description="Image index in document")
    page: Optional[int] = Field(None, description="Page number where image appears")
    description: Optional[str] = Field(None, description="AI-generated description (Phase 3)")
    image_type: str = Field(default="detected", description="Type: figure, chart, photo, diagram")
    bounding_box: Optional[dict] = Field(None, description="Image location coordinates")


class ParseResponse(BaseModel):
    """Response model for document parsing."""
    success: bool = Field(..., description="Whether parsing succeeded")
    filename: Optional[str] = Field(None, description="Original filename if uploaded")
    file_type: str = Field(..., description="Detected file type (e.g., 'pdf', 'docx')")
    title: str = Field(..., description="Extracted or inferred document title")
    markdown_content: str = Field(..., description="Full parsed content as markdown")
    chunks: List[ChunkResponse] = Field(default_factory=list, description="Chunked content")

    # Phase 2: Structured extractions
    tables: List[TableResponse] = Field(default_factory=list, description="Extracted tables with structure")
    images: List[ImageResponse] = Field(default_factory=list, description="Detected images (Phase 3 adds descriptions)")

    # Full DoclingDocument - native Docling structure with all semantic information
    docling_document: Optional[dict] = Field(
        None,
        description="Full DoclingDocument as JSON - contains texts, tables, pictures, body hierarchy, key_value_items, groups, and furniture. Use this for structured extraction rather than regex parsing."
    )

    metadata: dict = Field(default_factory=dict, description="Document metadata")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    error: Optional[str] = Field(None, description="Error message if parsing failed")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "filename": "medical_report.pdf",
                "file_type": "pdf",
                "title": "Patient Lab Results",
                "markdown_content": "# Lab Results\n\n| Test | Value | Range |\n|---|---|---|\n...",
                "chunks": [
                    {
                        "index": 0,
                        "content": "# Lab Results\n\nPatient: John Doe...",
                        "token_count": 150,
                        "start_char": 0,
                        "end_char": 500,
                        "chunk_type": "text",
                        "metadata": {"chunk_method": "hybrid"}
                    }
                ],
                "tables": [
                    {
                        "index": 0,
                        "markdown": "| Test | Value | Range |\n|---|---|---|\n| Hemoglobin | 14.2 | 12-16 |",
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
                        "description": None,
                        "image_type": "detected"
                    }
                ],
                "metadata": {
                    "word_count": 450,
                    "table_count": 2,
                    "image_count": 1,
                    "ocr_applied": True,
                    "ocr_engine": "tesseract"
                },
                "processing_time_ms": 1250.5,
                "error": None
            }
        }


class ParseTextRequest(BaseModel):
    """Request model for parsing raw text content."""
    content: str = Field(..., min_length=1, description="Text content to parse")
    chunk_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Target chunk size in characters"
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    )
    max_tokens: int = Field(
        default=512,
        ge=100,
        le=2000,
        description="Maximum tokens per chunk for embedding models"
    )
    title: Optional[str] = Field(
        None,
        description="Optional title for the document"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Job Description: Senior Software Engineer...",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "max_tokens": 512,
                "title": "Software Engineer JD"
            }
        }


class ParseFileRequest(BaseModel):
    """Request configuration for file parsing (query params)."""
    chunk_size: int = Field(default=1000, ge=100, le=5000, description="Target chunk size")
    chunk_overlap: int = Field(default=200, ge=0, le=500, description="Chunk overlap")
    max_tokens: int = Field(default=512, ge=100, le=2000, description="Max tokens per chunk")
    mode: ProcessingMode = Field(default=ProcessingMode.AUTO, description="Processing mode")
    psm_mode: TesseractPSMMode = Field(
        default=TesseractPSMMode.AUTOMATIC,
        description="Tesseract PSM mode for OCR"
    )
    extract_tables: bool = Field(default=True, description="Extract tables as structured JSON")
    detect_images: bool = Field(default=True, description="Detect images for Phase 3")


class SupportedTypesResponse(BaseModel):
    """Response model for supported file types."""
    document: List[str] = Field(..., description="Supported document formats")
    text: List[str] = Field(..., description="Supported text formats")
    audio: List[str] = Field(..., description="Supported audio formats")

    class Config:
        json_schema_extra = {
            "example": {
                "document": [".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".html", ".htm"],
                "text": [".txt", ".md", ".markdown"],
                "audio": [".mp3", ".wav", ".m4a", ".flac"]
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(default="0.2.0", description="Service version")
    features: dict = Field(
        default_factory=lambda: {
            "ocr_engine": "tesseract",
            "table_extraction": True,
            "image_detection": True,
            "granite_vision": False  # Phase 3
        },
        description="Available features"
    )
