# Cortivus Docling Parser - Stateless document parsing microservice
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Docling + audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager for faster dependency installation
RUN pip install --no-cache-dir uv

# Copy dependency files
COPY pyproject.toml ./

# Install Python dependencies
RUN uv pip install --system -e .

# Pre-download OCR models while still running as root
# This ensures models are available and writable
RUN python -c "from rapidocr import RapidOCR; ocr = RapidOCR(); print('RapidOCR models downloaded')" || true

# Set permissions on model directories for non-root access
RUN chmod -R 777 /usr/local/lib/python3.11/site-packages/rapidocr/models/ 2>/dev/null || true

# Copy application code
COPY app/ ./app/

# Create non-root user for security
RUN useradd -m -u 1000 parser && chown -R parser:parser /app
USER parser

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
