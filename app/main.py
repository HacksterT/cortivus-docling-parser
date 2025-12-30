"""
Cortivus Docling Parser - Stateless document parsing microservice.

This service converts documents (PDF, DOCX, TXT, HTML, audio) into structured
markdown with intelligent chunking. Designed as a plug-and-play service for
the Cortivus ecosystem and other projects.
"""

import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.api.routes import router

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Cortivus Docling Parser",
    description="Stateless document parsing microservice using Docling. "
                "Converts documents to markdown with intelligent chunking.",
    version="0.3.0",  # Phase 3: Granite Vision, intelligent routing
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS - allow all origins for microservice use
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration."""
    # Check if Granite Vision is available
    try:
        from app.services.vision import is_granite_available
        granite_available = is_granite_available()
    except ImportError:
        granite_available = False

    return {
        "status": "healthy",
        "service": "cortivus-docling-parser",
        "version": "0.3.0",
        "features": {
            "ocr_engine": "auto",  # Auto-selected best available
            "table_extraction": True,
            "image_detection": True,
            "granite_vision": granite_available,
            "intelligent_routing": True
        },
        "processing_modes": ["auto", "ocr_heavy", "table_focus", "vision_enabled"]
    }


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Cortivus Docling Parser starting up...")
    logger.info(f"Log level: {log_level}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Cortivus Docling Parser shutting down...")
