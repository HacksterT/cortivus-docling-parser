"""
Document parsing service using Docling.

Phase 2 Enhanced Version:
- Explicit Tesseract OCR configuration (no RapidOCR ambiguity)
- Configurable PSM modes for different document types
- Table detection and structured extraction
- Image/figure detection for future Granite Vision integration

Handles parsing of various document formats (PDF, DOCX, TXT, HTML, audio)
into markdown format with metadata extraction.
"""

import os
import re
import logging
import tempfile
from pathlib import Path
from typing import Optional, Any, Tuple, List, Dict
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Supported file extensions
DOCLING_FORMATS = ['.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls', '.html', '.htm']
TEXT_FORMATS = ['.txt', '.md', '.markdown']
AUDIO_FORMATS = ['.mp3', '.wav', '.m4a', '.flac']


class TesseractPSM(Enum):
    """
    Tesseract Page Segmentation Modes.

    Choose based on document type:
    - AUTOMATIC (3): General documents, mixed layouts
    - SINGLE_COLUMN (4): Medical reports, single-column docs
    - UNIFORM_BLOCK (6): Dense text blocks, paragraphs
    - SPARSE_TEXT (11): Forms with scattered fields
    - SPARSE_WITH_OSD (12): Tables with gaps, structured data
    """
    AUTOMATIC = 3           # Fully automatic page segmentation
    SINGLE_COLUMN = 4       # Single column of variable text sizes
    UNIFORM_BLOCK = 6       # Uniform block of text
    SINGLE_LINE = 7         # Single text line
    SINGLE_WORD = 8         # Single word
    SPARSE_TEXT = 11        # Sparse text, find as much as possible
    SPARSE_WITH_OSD = 12    # Sparse text with OSD (orientation/script detection)


@dataclass
class TableData:
    """Structured table data extracted from documents."""
    index: int
    markdown: str
    headers: List[str] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)
    page: Optional[int] = None
    confidence: float = 1.0


@dataclass
class ImageData:
    """Image/figure data detected in documents (for Phase 3 Granite Vision)."""
    index: int
    page: Optional[int] = None
    description: Optional[str] = None  # Filled by Granite Vision when enabled
    bounding_box: Optional[Dict] = None
    image_type: str = "unknown"  # figure, chart, photo, diagram, etc.
    confidence: float = 0.0  # Confidence of image description


@dataclass
class ParseConfig:
    """Configuration for document parsing."""
    # OCR settings
    ocr_enabled: bool = True
    ocr_engine: str = "tesseract"  # Explicitly use Tesseract
    tesseract_psm: TesseractPSM = TesseractPSM.AUTOMATIC

    # Table extraction
    extract_tables: bool = True
    tables_as_json: bool = True  # Also return structured JSON for tables

    # Image detection and description (Phase 3)
    detect_images: bool = True
    describe_images: bool = False  # Phase 3: Enable Granite Vision descriptions

    # Processing mode - intelligent routing
    mode: str = "auto"  # auto, ocr_heavy, table_focus, vision_enabled

    @classmethod
    def from_mode(cls, mode: str) -> "ParseConfig":
        """
        Create configuration optimized for a specific processing mode.

        Intelligent routing based on document characteristics:
        - auto: Balanced settings for general documents
        - ocr_heavy: Optimized for scanned documents
        - table_focus: Prioritize table extraction accuracy
        - vision_enabled: Include AI image descriptions (Phase 3)
        """
        if mode == "ocr_heavy":
            return cls(
                ocr_enabled=True,
                tesseract_psm=TesseractPSM.AUTOMATIC,
                extract_tables=True,
                detect_images=True,
                describe_images=False,
                mode=mode
            )
        elif mode == "table_focus":
            return cls(
                ocr_enabled=True,
                tesseract_psm=TesseractPSM.SPARSE_WITH_OSD,
                extract_tables=True,
                detect_images=False,
                describe_images=False,
                mode=mode
            )
        elif mode == "vision_enabled":
            return cls(
                ocr_enabled=True,
                tesseract_psm=TesseractPSM.AUTOMATIC,
                extract_tables=True,
                detect_images=True,
                describe_images=True,  # Enable Granite Vision
                mode=mode
            )
        else:  # auto
            return cls(
                ocr_enabled=True,
                tesseract_psm=TesseractPSM.AUTOMATIC,
                extract_tables=True,
                detect_images=True,
                describe_images=False,
                mode="auto"
            )


def get_file_type(filename: str) -> str:
    """
    Get the file type from filename extension.

    Args:
        filename: Original filename

    Returns:
        File type string (e.g., 'pdf', 'docx', 'txt')
    """
    ext = os.path.splitext(filename)[1].lower()
    if ext.startswith('.'):
        return ext[1:]  # Remove leading dot
    return ext or "unknown"


def _create_pdf_converter(config: ParseConfig):
    """
    Create a Docling DocumentConverter with OCR and vision configuration.

    Uses Docling's native pipeline options and VLM for image descriptions.

    Args:
        config: Parse configuration with OCR and vision settings

    Returns:
        Configured DocumentConverter instance
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    # Configure PDF pipeline
    pipeline_options = PdfPipelineOptions()

    if config.ocr_enabled:
        pipeline_options.do_ocr = True
        logger.info("OCR enabled")
    else:
        pipeline_options.do_ocr = False

    # Enable table structure detection
    if config.extract_tables:
        pipeline_options.do_table_structure = True
        logger.info("Table structure detection enabled")

    # Enable picture/image description with VLM (Granite Vision)
    if config.describe_images:
        try:
            from docling.datamodel.pipeline_options import granite_picture_description

            pipeline_options.do_picture_description = True
            pipeline_options.picture_description_options = granite_picture_description
            pipeline_options.images_scale = 2.0
            pipeline_options.generate_picture_images = True
            logger.info("Picture description enabled with Granite Vision")

        except ImportError as e:
            logger.warning(f"VLM picture description not available: {e}")
            logger.info("Install docling[vlm] for image description support")

    # Create converter with PDF-specific options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    return converter


def _extract_tables_from_markdown(markdown_content: str) -> List[TableData]:
    """
    Extract tables from markdown content and parse into structured format.

    Args:
        markdown_content: Markdown text containing tables

    Returns:
        List of TableData objects with structured table information
    """
    tables = []

    # Regex to find markdown tables
    # Matches: | header1 | header2 |\n|---|---|\n| data1 | data2 |
    table_pattern = r'(\|[^\n]+\|\n\|[-:\s|]+\|\n(?:\|[^\n]+\|\n?)+)'

    matches = re.finditer(table_pattern, markdown_content)

    for idx, match in enumerate(matches):
        table_md = match.group(1).strip()
        lines = table_md.split('\n')

        if len(lines) < 3:  # Need header, separator, and at least one data row
            continue

        # Parse header row
        header_line = lines[0]
        headers = [cell.strip() for cell in header_line.split('|') if cell.strip()]

        # Skip separator line (lines[1])

        # Parse data rows
        rows = []
        for row_line in lines[2:]:
            if row_line.strip():
                cells = [cell.strip() for cell in row_line.split('|') if cell.strip()]
                if cells:
                    rows.append(cells)

        tables.append(TableData(
            index=idx,
            markdown=table_md,
            headers=headers,
            rows=rows,
            confidence=0.95  # High confidence for explicit markdown tables
        ))

    logger.info(f"Extracted {len(tables)} tables from markdown")
    return tables


def _extract_images_from_docling(docling_doc: Any) -> List[ImageData]:
    """
    Extract images and descriptions from Docling's native document structure.

    Uses doc.pictures[].annotations to get VLM-generated descriptions,
    which is the proper Docling-native approach.

    Args:
        docling_doc: The DoclingDocument object from conversion

    Returns:
        List of ImageData objects with descriptions from Granite Vision
    """
    images = []

    if docling_doc is None:
        return images

    try:
        from docling_core.types.doc.document import PictureDescriptionData

        pictures = getattr(docling_doc, 'pictures', [])

        for idx, pic in enumerate(pictures):
            caption = None
            description = None
            provenance = None

            # Get caption if available
            if hasattr(pic, 'caption_text'):
                caption = pic.caption_text(doc=docling_doc)

            # Get VLM annotations (descriptions from Granite Vision)
            annotations = getattr(pic, 'annotations', [])
            for annotation in annotations:
                if isinstance(annotation, PictureDescriptionData):
                    description = annotation.text
                    provenance = annotation.provenance
                    break

            # Use description if available, otherwise caption
            final_description = description or caption

            images.append(ImageData(
                index=idx,
                page=None,  # Could extract from pic.prov if needed
                description=final_description,
                bounding_box=None,
                image_type="picture" if description else "detected",
                confidence=0.85 if description else 0.0
            ))

        if images:
            described_count = sum(1 for img in images if img.description)
            logger.info(f"Extracted {len(images)} pictures ({described_count} with descriptions)")

    except ImportError as e:
        logger.warning(f"Could not import PictureDescriptionData: {e}")
    except Exception as e:
        logger.warning(f"Error extracting images from Docling document: {e}")

    return images


def _detect_images_in_markdown(markdown_content: str) -> List[ImageData]:
    """
    Fallback: Detect image placeholders in markdown content.

    Used when DoclingDocument is not available or for non-PDF formats.
    Docling marks images with <!-- image --> comments.

    Args:
        markdown_content: Markdown text

    Returns:
        List of ImageData objects (no descriptions)
    """
    images = []
    simple_pattern = r'<!-- image -->'

    for idx, match in enumerate(re.finditer(simple_pattern, markdown_content)):
        images.append(ImageData(
            index=idx,
            description=None,
            image_type="detected"
        ))

    if images:
        logger.info(f"Detected {len(images)} image markers in markdown")

    return images


def parse_document(
    file_path: str,
    filename: Optional[str] = None,
    config: Optional[ParseConfig] = None
) -> Tuple[str, Optional[Any], dict, List[TableData], List[ImageData]]:
    """
    Parse a document file and return markdown content with structured data.

    Args:
        file_path: Path to the document file
        filename: Optional original filename (for type detection)
        config: Optional parsing configuration

    Returns:
        Tuple of (markdown_content, docling_document, metadata, tables, images)
    """
    if config is None:
        config = ParseConfig()

    if filename is None:
        filename = os.path.basename(file_path)

    file_ext = os.path.splitext(filename)[1].lower()
    tables: List[TableData] = []
    images: List[ImageData] = []

    # Audio formats - transcribe with Whisper ASR
    if file_ext in AUDIO_FORMATS:
        content = _transcribe_audio(file_path)
        metadata = _extract_metadata(content, file_path, filename)
        metadata['ocr_applied'] = False
        return (content, None, metadata, tables, images)

    # Docling-supported formats (convert to markdown)
    if file_ext in DOCLING_FORMATS:
        try:
            logger.info(f"Converting {file_ext} file using Docling with Tesseract OCR: {filename}")

            # Create converter with explicit Tesseract configuration
            converter = _create_pdf_converter(config)
            result = converter.convert(file_path)

            # Export to markdown for consistent processing
            markdown_content = result.document.export_to_markdown()
            logger.info(f"Successfully converted {filename} to markdown")

            metadata = _extract_metadata(markdown_content, file_path, filename)
            metadata['ocr_applied'] = config.ocr_enabled
            metadata['ocr_engine'] = config.ocr_engine
            metadata['tesseract_psm'] = config.tesseract_psm.name

            # Extract structured tables
            if config.extract_tables:
                tables = _extract_tables_from_markdown(markdown_content)
                metadata['table_count'] = len(tables)

            # Extract images from Docling's native structure
            if config.detect_images:
                # Use Docling's doc.pictures for proper image extraction
                images = _extract_images_from_docling(result.document)

                # Fallback to markdown parsing if no pictures found
                if not images:
                    images = _detect_images_in_markdown(markdown_content)

                metadata['image_count'] = len(images)
                metadata['has_images'] = len(images) > 0
                metadata['vlm_enabled'] = config.describe_images

                # Count images with descriptions
                described = sum(1 for img in images if img.description)
                metadata['images_with_descriptions'] = described

            # Return both markdown and DoclingDocument for HybridChunker
            return (markdown_content, result.document, metadata, tables, images)

        except Exception as e:
            logger.error(f"Failed to convert {file_path} with Docling: {e}")
            # For binary formats, don't attempt raw text fallback
            error_msg = f"[Error: Failed to parse {filename} with Docling: {e}]"
            return (error_msg, None, {"error": str(e)}, [], [])

    # Text-based formats (read directly)
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Try with different encoding
        with open(file_path, 'r', encoding='latin-1') as f:
            content = f.read()

    metadata = _extract_metadata(content, file_path, filename)
    metadata['ocr_applied'] = False

    # Still extract tables from markdown/text files
    if config.extract_tables:
        tables = _extract_tables_from_markdown(content)
        metadata['table_count'] = len(tables)

    return (content, None, metadata, tables, images)


def parse_text(
    content: str,
    title: Optional[str] = None,
    config: Optional[ParseConfig] = None
) -> Tuple[str, None, dict, List[TableData], List[ImageData]]:
    """
    Parse raw text content.

    Args:
        content: Raw text content
        title: Optional document title
        config: Optional parsing configuration

    Returns:
        Tuple of (content, None, metadata, tables, images)
    """
    if config is None:
        config = ParseConfig()

    metadata = {
        "file_type": "text",
        "word_count": len(content.split()),
        "line_count": len(content.split('\n')),
        "char_count": len(content),
        "parsed_at": datetime.now().isoformat(),
        "ocr_applied": False
    }

    if title:
        metadata["title"] = title

    # Extract tables from text if it contains markdown tables
    tables: List[TableData] = []
    if config.extract_tables:
        tables = _extract_tables_from_markdown(content)
        metadata['table_count'] = len(tables)

    return (content, None, metadata, tables, [])


def _transcribe_audio(file_path: str) -> str:
    """
    Transcribe audio file using Whisper ASR via Docling.

    Args:
        file_path: Path to audio file

    Returns:
        Transcribed text as markdown
    """
    try:
        from pathlib import Path
        from docling.document_converter import DocumentConverter, AudioFormatOption
        from docling.datamodel.pipeline_options import AsrPipelineOptions
        from docling.datamodel import asr_model_specs
        from docling.datamodel.base_models import InputFormat
        from docling.pipeline.asr_pipeline import AsrPipeline

        # Use Path object - Docling expects this
        audio_path = Path(file_path).resolve()
        logger.info(f"Transcribing audio file using Whisper Turbo: {audio_path.name}")

        # Verify file exists
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Configure ASR pipeline with Whisper Turbo model
        pipeline_options = AsrPipelineOptions()
        pipeline_options.asr_options = asr_model_specs.WHISPER_TURBO

        converter = DocumentConverter(
            format_options={
                InputFormat.AUDIO: AudioFormatOption(
                    pipeline_cls=AsrPipeline,
                    pipeline_options=pipeline_options,
                )
            }
        )

        # Transcribe the audio file - pass Path object
        result = converter.convert(audio_path)

        # Export to markdown with timestamps
        markdown_content = result.document.export_to_markdown()
        logger.info(f"Successfully transcribed {audio_path.name}")
        return markdown_content

    except Exception as e:
        logger.error(f"Failed to transcribe {file_path} with Whisper ASR: {e}")
        return f"[Error: Could not transcribe audio file {os.path.basename(file_path)}: {e}]"


def extract_title(content: str, filename: str) -> str:
    """
    Extract title from document content or filename.

    Args:
        content: Document content
        filename: Original filename

    Returns:
        Extracted or inferred title
    """
    # Try to find markdown title
    lines = content.split('\n')
    for line in lines[:10]:  # Check first 10 lines
        line = line.strip()
        if line.startswith('# '):
            return line[2:].strip()

    # Fallback to filename (without extension)
    return os.path.splitext(os.path.basename(filename))[0]


def _extract_metadata(content: str, file_path: str, filename: str) -> dict:
    """
    Extract metadata from document content.

    Args:
        content: Document content
        file_path: Path to file
        filename: Original filename

    Returns:
        Metadata dictionary
    """
    file_ext = os.path.splitext(filename)[1].lower()

    metadata = {
        "file_path": file_path,
        "filename": filename,
        "file_type": file_ext[1:] if file_ext.startswith('.') else file_ext,
        "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else len(content),
        "parsed_at": datetime.now().isoformat()
    }

    # Extract basic content metadata
    lines = content.split('\n')
    metadata['line_count'] = len(lines)
    metadata['word_count'] = len(content.split())
    metadata['char_count'] = len(content)

    # Try to extract YAML frontmatter
    if content.startswith('---'):
        try:
            import yaml
            end_marker = content.find('\n---\n', 4)
            if end_marker != -1:
                frontmatter = content[4:end_marker]
                yaml_metadata = yaml.safe_load(frontmatter)
                if isinstance(yaml_metadata, dict):
                    metadata.update(yaml_metadata)
        except ImportError:
            logger.debug("PyYAML not installed, skipping frontmatter extraction")
        except Exception as e:
            logger.debug(f"Failed to parse frontmatter: {e}")

    return metadata


def is_supported_format(filename: str) -> bool:
    """
    Check if a file format is supported.

    Args:
        filename: Filename to check

    Returns:
        True if format is supported
    """
    ext = os.path.splitext(filename)[1].lower()
    return ext in DOCLING_FORMATS + TEXT_FORMATS + AUDIO_FORMATS


def get_supported_formats() -> dict:
    """
    Get dictionary of all supported formats.

    Returns:
        Dictionary with document, text, and audio format lists
    """
    return {
        "document": DOCLING_FORMATS,
        "text": TEXT_FORMATS,
        "audio": AUDIO_FORMATS
    }
