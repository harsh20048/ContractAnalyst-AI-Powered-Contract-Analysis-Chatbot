"""
HuggingFace PDF Extraction Framework

A comprehensive framework for deploying Hugging Face models and extracting
information from PDF documents.
"""

__version__ = "1.0.0"
__author__ = "HF Framework Team"

from .config import settings, MODEL_CONFIGS, EXTRACTION_TASKS
from .models import ModelManager
from .pdf_processor import PDFProcessor
from .extraction_engine import ExtractionEngine

__all__ = [
    "settings",
    "MODEL_CONFIGS", 
    "EXTRACTION_TASKS",
    "ModelManager",
    "PDFProcessor",
    "ExtractionEngine"
]