@echo off
REM Test script for Cortivus Docling Parser (Windows)
REM Usage: test_parser.bat [optional_document_path]

cd /d "%~dp0"
python test_parser.py %*
pause
