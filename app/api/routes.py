"""
API routes for document parsing endpoints.

Phase 2 Enhanced:
- Explicit Tesseract OCR configuration
- Structured table extraction (JSON output)
- Image detection for Phase 3
- Configurable PSM modes
"""

import os
import logging
import tempfile
import time
from typing import Optional, Dict

from fastapi import APIRouter, UploadFile, File, Query, HTTPException

from app.services.parser import (
    parse_document,
    parse_text,
    extract_title,
    get_file_type,
    is_supported_format,
    get_supported_formats,
    ParseConfig,
    TesseractPSM
)
from app.services.chunker import ChunkingConfig, create_chunker, DocumentChunk
from app.models.schemas import (
    ParseResponse,
    ParseTextRequest,
    ChunkResponse,
    TableResponse,
    ImageResponse,
    SupportedTypesResponse,
    HealthResponse,
    ProcessingMode,
    TesseractPSMMode,
    PROCESSING_MODE_DESCRIPTIONS
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Map API PSM modes to internal TesseractPSM enum
PSM_MODE_MAP = {
    TesseractPSMMode.AUTOMATIC: TesseractPSM.AUTOMATIC,
    TesseractPSMMode.SINGLE_COLUMN: TesseractPSM.SINGLE_COLUMN,
    TesseractPSMMode.UNIFORM_BLOCK: TesseractPSM.UNIFORM_BLOCK,
    TesseractPSMMode.SPARSE_TEXT: TesseractPSM.SPARSE_TEXT,
    TesseractPSMMode.SPARSE_WITH_OSD: TesseractPSM.SPARSE_WITH_OSD,
}


def _convert_chunks_to_response(chunks: list[DocumentChunk]) -> list[ChunkResponse]:
    """Convert internal DocumentChunk objects to API response format."""
    return [
        ChunkResponse(
            index=chunk.index,
            content=chunk.content,
            token_count=chunk.token_count or 0,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            chunk_type=chunk.metadata.get("chunk_type", "text"),
            metadata=chunk.metadata
        )
        for chunk in chunks
    ]


def _convert_tables_to_response(tables) -> list[TableResponse]:
    """Convert internal TableData objects to API response format."""
    return [
        TableResponse(
            index=table.index,
            markdown=table.markdown,
            headers=table.headers,
            rows=table.rows,
            page=table.page,
            confidence=table.confidence
        )
        for table in tables
    ]


def _convert_images_to_response(images) -> list[ImageResponse]:
    """Convert internal ImageData objects to API response format."""
    return [
        ImageResponse(
            index=image.index,
            page=image.page,
            description=image.description,
            image_type=image.image_type,
            bounding_box=image.bounding_box
        )
        for image in images
    ]


def _serialize_docling_document(docling_doc) -> Optional[dict]:
    """
    Serialize DoclingDocument to JSON-compatible dict.

    DoclingDocument is a Pydantic model from docling_core.types.doc,
    so it has native serialization support.

    Args:
        docling_doc: The DoclingDocument object from Docling conversion

    Returns:
        JSON-serializable dict or None if document is not available
    """
    if docling_doc is None:
        return None

    try:
        # DoclingDocument is a Pydantic model - use model_dump() for Pydantic v2
        # or .dict() for Pydantic v1
        if hasattr(docling_doc, 'model_dump'):
            return docling_doc.model_dump(mode='json')
        elif hasattr(docling_doc, 'dict'):
            return docling_doc.dict()
        else:
            # Fallback: try JSON export if available
            if hasattr(docling_doc, 'export_to_dict'):
                return docling_doc.export_to_dict()
            logger.warning("DoclingDocument has no serialization method available")
            return None
    except Exception as e:
        logger.error(f"Failed to serialize DoclingDocument: {e}")
        return None


@router.post("/parse/file", response_model=ParseResponse)
async def parse_file_endpoint(
    file: UploadFile = File(..., description="Document file to parse"),
    chunk_size: int = Query(
        default=1000,
        ge=100,
        le=5000,
        description="Target chunk size in characters"
    ),
    chunk_overlap: int = Query(
        default=200,
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    ),
    max_tokens: int = Query(
        default=512,
        ge=100,
        le=2000,
        description="Maximum tokens per chunk"
    ),
    mode: ProcessingMode = Query(
        default=ProcessingMode.AUTO,
        description="Processing mode: auto, ocr_heavy, table_focus, vision_enabled"
    ),
    psm_mode: TesseractPSMMode = Query(
        default=TesseractPSMMode.AUTOMATIC,
        description="Tesseract PSM mode for OCR"
    ),
    extract_tables: bool = Query(
        default=True,
        description="Extract tables as structured JSON"
    ),
    detect_images: bool = Query(
        default=True,
        description="Detect images (Phase 3 will add descriptions)"
    )
):
    """
    Parse an uploaded document file.

    Supports PDF, DOCX, DOC, PPTX, XLSX, HTML, TXT, MD, and audio files (MP3, WAV, M4A, FLAC).

    Returns the parsed content as markdown with intelligent chunking,
    plus structured table extraction and image detection.

    **Processing Modes:**
    - `auto`: Automatic detection and optimal processing
    - `ocr_heavy`: Optimized for scanned documents
    - `table_focus`: Prioritize table extraction accuracy
    - `vision_enabled`: Phase 3 - include AI image descriptions

    **Tesseract PSM Modes:**
    - `automatic`: General documents, mixed layouts (default)
    - `single_column`: Medical reports, single-column documents
    - `uniform_block`: Dense text paragraphs
    - `sparse_text`: Forms with scattered fields
    - `sparse_with_osd`: Tables with gaps
    """
    start_time = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    if not is_supported_format(file.filename):
        supported = get_supported_formats()
        all_formats = supported["document"] + supported["text"] + supported["audio"]
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported formats: {', '.join(all_formats)}"
        )

    temp_path = None
    try:
        # Save uploaded file to temporary location
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = temp_file.name
            content = await file.read()
            temp_file.write(content)

        logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")

        # Create parse configuration
        # Enable image descriptions when mode is vision_enabled
        describe_images = (mode == ProcessingMode.VISION_ENABLED)

        parse_config = ParseConfig(
            ocr_enabled=True,
            ocr_engine="tesseract",
            tesseract_psm=PSM_MODE_MAP.get(psm_mode, TesseractPSM.AUTOMATIC),
            extract_tables=extract_tables,
            detect_images=detect_images,
            describe_images=describe_images,
            mode=mode.value
        )

        # Parse the document with Phase 2 enhancements
        markdown_content, docling_doc, metadata, tables, images = parse_document(
            temp_path,
            file.filename,
            config=parse_config
        )

        # Check for parsing errors
        if markdown_content.startswith("[Error:"):
            return ParseResponse(
                success=False,
                filename=file.filename,
                file_type=get_file_type(file.filename),
                title=file.filename,
                markdown_content="",
                chunks=[],
                tables=[],
                images=[],
                docling_document=None,
                metadata=metadata,
                processing_time_ms=(time.time() - start_time) * 1000,
                error=markdown_content
            )

        # Extract title
        title = extract_title(markdown_content, file.filename)

        # Create chunker and chunk the document
        config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_tokens=max_tokens,
            use_semantic_splitting=docling_doc is not None
        )
        chunker = create_chunker(config)

        chunks = await chunker.chunk_document(
            content=markdown_content,
            title=title,
            source=file.filename,
            metadata=metadata,
            docling_doc=docling_doc
        )

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(
            f"Parsed {file.filename}: {len(chunks)} chunks, "
            f"{len(tables)} tables, {len(images)} images in {processing_time_ms:.1f}ms"
        )

        return ParseResponse(
            success=True,
            filename=file.filename,
            file_type=get_file_type(file.filename),
            title=title,
            markdown_content=markdown_content,
            chunks=_convert_chunks_to_response(chunks),
            tables=_convert_tables_to_response(tables),
            images=_convert_images_to_response(images),
            docling_document=_serialize_docling_document(docling_doc),
            metadata=metadata,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Error parsing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing file: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_path}: {e}")


@router.post("/parse/text", response_model=ParseResponse)
async def parse_text_endpoint(request: ParseTextRequest):
    """
    Parse raw text content.

    Chunks the provided text content using configurable parameters.
    Also extracts any markdown tables found in the text.
    """
    start_time = time.time()

    try:
        # Create parse configuration
        parse_config = ParseConfig(
            extract_tables=True,
            detect_images=False  # No images in raw text
        )

        # Parse the text with Phase 2 enhancements
        content, _, metadata, tables, images = parse_text(
            request.content,
            request.title,
            config=parse_config
        )

        # Extract or use provided title
        title = request.title or extract_title(content, "text_input")

        # Create chunker and chunk the content
        config = ChunkingConfig(
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            max_tokens=request.max_tokens,
            use_semantic_splitting=False  # Use simple chunker for raw text
        )
        chunker = create_chunker(config)

        chunks = await chunker.chunk_document(
            content=content,
            title=title,
            source="text_input",
            metadata=metadata
        )

        processing_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Parsed text input: {len(chunks)} chunks, {len(tables)} tables in {processing_time_ms:.1f}ms")

        return ParseResponse(
            success=True,
            filename=None,
            file_type="text",
            title=title,
            markdown_content=content,
            chunks=_convert_chunks_to_response(chunks),
            tables=_convert_tables_to_response(tables),
            images=[],  # No images in text input
            docling_document=None,  # Text parsing doesn't use Docling
            metadata=metadata,
            processing_time_ms=processing_time_ms
        )

    except Exception as e:
        logger.error(f"Error parsing text: {e}")
        raise HTTPException(status_code=500, detail=f"Error parsing text: {str(e)}")


@router.get("/supported-types", response_model=SupportedTypesResponse)
async def get_supported_types():
    """
    Get list of supported file types.

    Returns categorized lists of supported document, text, and audio formats.
    """
    formats = get_supported_formats()
    return SupportedTypesResponse(
        document=formats["document"],
        text=formats["text"],
        audio=formats["audio"]
    )


@router.get("/processing-modes")
async def get_processing_modes():
    """
    Get available processing modes with descriptions.

    Returns a list of processing modes and their descriptions for UI display.
    """
    return {
        "modes": [
            {
                "value": mode.value,
                "name": mode.name,
                "description": PROCESSING_MODE_DESCRIPTIONS[mode]
            }
            for mode in ProcessingMode
        ]
    }
