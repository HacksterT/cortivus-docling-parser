#!/usr/bin/env python3
"""
Simple CLI script to test the Cortivus Docling Parser.

Usage:
    python scripts/test_parser.py

The script will prompt for a document path, send it to the parser service,
and save the JSON output to an output file.

Requirements:
    - Parser service must be running (docker-compose up)
    - requests library (pip install requests)
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: requests library required. Install with: pip install requests")
    sys.exit(1)


# Configuration
PARSER_URL = os.getenv("PARSER_URL", "http://localhost:8001")
OUTPUT_DIR = Path(__file__).parent.parent / "output"


def check_service_health():
    """Check if the parser service is running."""
    try:
        response = requests.get(f"{PARSER_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"\n[OK] Parser service is healthy")
            print(f"     Version: {health.get('version', 'unknown')}")
            print(f"     Features:")
            features = health.get('features', {})
            for feature, enabled in features.items():
                status = "enabled" if enabled else "disabled"
                print(f"       - {feature}: {status}")
            print(f"     Processing modes: {', '.join(health.get('processing_modes', []))}")
            return True
        else:
            print(f"\n[ERROR] Parser service returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Cannot connect to parser service at {PARSER_URL}")
        print("        Make sure the service is running: docker-compose up")
        return False
    except Exception as e:
        print(f"\n[ERROR] Health check failed: {e}")
        return False


def parse_document(file_path: str, mode: str = "auto") -> dict:
    """
    Send a document to the parser service.

    Args:
        file_path: Path to the document file
        mode: Processing mode (auto, ocr_heavy, table_focus, vision_enabled)

    Returns:
        Parsed response as dictionary
    """
    file_path = Path(file_path).resolve()

    if not file_path.exists():
        return {"success": False, "error": f"File not found: {file_path}"}

    print(f"\n[INFO] Parsing: {file_path.name}")
    print(f"       Mode: {mode}")
    print(f"       Size: {file_path.stat().st_size:,} bytes")

    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path.name, f)}
            params = {
                "mode": mode,
                "extract_tables": True,
                "detect_images": True
            }

            print(f"\n[INFO] Sending to parser...")
            response = requests.post(
                f"{PARSER_URL}/parse/file",
                files=files,
                params=params,
                timeout=300  # 5 minutes for large documents
            )

            if response.status_code == 200:
                result = response.json()
                print(f"[OK] Parsing complete!")
                print(f"     Processing time: {result.get('processing_time_ms', 0):.1f}ms")
                print(f"     Chunks: {len(result.get('chunks', []))}")
                print(f"     Tables: {len(result.get('tables', []))}")
                print(f"     Images: {len(result.get('images', []))}")
                return result
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }

    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out (document may be too large)"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_output(result: dict, source_file: str) -> str:
    """Save parsing result to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    source_name = Path(source_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = OUTPUT_DIR / f"{source_name}_{timestamp}.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(output_file)


def print_summary(result: dict):
    """Print a summary of the parsing result."""
    print("\n" + "=" * 60)
    print("PARSING SUMMARY")
    print("=" * 60)

    if not result.get("success", False):
        print(f"\n[FAILED] {result.get('error', 'Unknown error')}")
        return

    print(f"\nTitle: {result.get('title', 'Unknown')}")
    print(f"File Type: {result.get('file_type', 'Unknown')}")

    # Metadata
    metadata = result.get("metadata", {})
    print(f"\nMetadata:")
    print(f"  Word count: {metadata.get('word_count', 'N/A')}")
    print(f"  OCR applied: {metadata.get('ocr_applied', False)}")
    if metadata.get('ocr_engine'):
        print(f"  OCR engine: {metadata.get('ocr_engine')}")

    # Tables
    tables = result.get("tables", [])
    if tables:
        print(f"\nTables ({len(tables)}):")
        for table in tables[:3]:  # Show first 3
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            print(f"  Table {table.get('index', '?')}: {len(headers)} columns, {len(rows)} rows")
            if headers:
                print(f"    Headers: {', '.join(headers[:5])}")
                if len(headers) > 5:
                    print(f"             ... and {len(headers) - 5} more")
        if len(tables) > 3:
            print(f"  ... and {len(tables) - 3} more tables")

    # Images
    images = result.get("images", [])
    if images:
        print(f"\nImages ({len(images)}):")
        for img in images[:3]:
            desc = img.get("description") or "[No description yet]"
            if len(desc) > 50:
                desc = desc[:50] + "..."
            print(f"  Image {img.get('index', '?')}: {img.get('image_type', 'unknown')} - {desc}")
        if len(images) > 3:
            print(f"  ... and {len(images) - 3} more images")

    # Chunks preview
    chunks = result.get("chunks", [])
    if chunks:
        print(f"\nChunks ({len(chunks)}):")
        total_tokens = sum(c.get("token_count", 0) for c in chunks)
        print(f"  Total tokens: ~{total_tokens:,}")
        print(f"  Average chunk size: ~{total_tokens // len(chunks) if chunks else 0} tokens")

    # Content preview
    content = result.get("markdown_content", "")
    if content:
        preview = content[:500].replace("\n", " ")
        if len(content) > 500:
            preview += "..."
        print(f"\nContent preview:")
        print(f"  {preview}")


def select_mode() -> str:
    """Let user select processing mode."""
    modes = {
        "1": ("auto", "Balanced settings for general documents"),
        "2": ("ocr_heavy", "Optimized for scanned documents"),
        "3": ("table_focus", "Prioritize table extraction accuracy"),
        "4": ("vision_enabled", "Include AI image descriptions (requires Granite Vision)")
    }

    print("\nSelect processing mode:")
    for key, (mode, desc) in modes.items():
        print(f"  {key}. {mode}: {desc}")

    while True:
        choice = input("\nEnter choice [1-4, default=1]: ").strip() or "1"
        if choice in modes:
            return modes[choice][0]
        print("Invalid choice. Please enter 1-4.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("CORTIVUS DOCLING PARSER - Test Script")
    print("=" * 60)

    # Check service health
    if not check_service_health():
        sys.exit(1)

    # Get document path
    print("\n" + "-" * 60)

    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Using provided file: {file_path}")
    else:
        file_path = input("\nEnter document path: ").strip()

        # Handle quoted paths
        if file_path.startswith('"') and file_path.endswith('"'):
            file_path = file_path[1:-1]
        if file_path.startswith("'") and file_path.endswith("'"):
            file_path = file_path[1:-1]

    if not file_path:
        print("No file path provided. Exiting.")
        sys.exit(1)

    # Resolve path
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        # Try relative to project root
        project_root = Path(__file__).parent.parent
        alt_path = project_root / file_path.name
        if alt_path.exists():
            file_path = alt_path
        else:
            print(f"[ERROR] File not found: {file_path}")
            sys.exit(1)

    # Select mode
    mode = select_mode()

    # Parse document
    result = parse_document(str(file_path), mode)

    # Save output
    output_file = save_output(result, str(file_path))
    print(f"\n[SAVED] Output written to: {output_file}")

    # Print summary
    print_summary(result)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
