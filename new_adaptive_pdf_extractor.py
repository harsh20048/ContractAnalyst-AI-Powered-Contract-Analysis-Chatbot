"""
Updated Adaptive PDF Table Extraction Application
Refined version with consolidated imports, improved error handling, and cleaner structure.
"""

import sys
import copy
import os
import tempfile
import sqlite3
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Adaptive PDF Data Extraction System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adaptive table extraction imports with comprehensive fallbacks
# Adaptive table extraction imports with comprehensive fallbacks
try:
    from extraction.field_extraction import extract_field_with_learning
    extract_learning_available = True
except ImportError:
    extract_learning_available = False

try:
    from secure_pdf_extractor.extraction.Table_extraction import (
        extract_tables_with_learning,
        save_table_corrections,
        get_learning_statistics,
        AdaptiveTableExtractor,
        extract_tables
    )
    ADAPTIVE_AVAILABLE = True
    print("✅ Table extraction imports successful")
except ImportError as e:
    ADAPTIVE_AVAILABLE = False
    print(f"❌ Table extraction import error: {e}")
    # (keep your fallback AdaptiveTableExtractor here)
    # Complete fallback implementation
    class AdaptiveTableExtractor:
        def __init__(self, *args, **kwargs):
            self.debug = kwargs.get('debug', False)
        
        def extract_tables_with_learning(self, *args, **kwargs):
            return []
        
        def bulk_save_corrections(self, *args, **kwargs):
            return 0
        
        def get_learning_statistics(self, *args, **kwargs):
            return {'templates_learned': 0, 'header_corrections': 0}
    
    def extract_tables_with_learning(*args, **kwargs):
        return []

# Field extraction imports
try:
    from secure_pdf_extractor.extraction.field_extraction import (
        PDFFieldExtractor,
        extract_fields_from_pdf,
        process_pdf_documents
    )
    FIELD_EXTRACTION_AVAILABLE = True
    print("✅ Field extraction imports successful")
except ImportError as e:
    FIELD_EXTRACTION_AVAILABLE = False
    print(f"❌ Field extraction import error: {e}")
    # (keep your fallback PDFFieldExtractor here)
    # Fallback PDFFieldExtractor
    class PDFFieldExtractor:
        def __init__(self, *args, **kwargs):
            pass
        
        def extract_fields(self, text, **kwargs):
            return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

# Text extraction imports
try:
    from secure_pdf_extractor.extraction.text_extraction import extract_text_from_pdf, detect_language
    print("✅ Text extraction imports successful")
except ImportError as e:
    print(f"❌ Text extraction import error: {e}")
    # (keep your pdfplumber fallback here)
except ImportError:
    pdfplumber = None
    def extract_text_from_pdf(file_path):
        return "Sample text", False
    
    def detect_language(text):
        return "en"
    
    class PDFFieldExtractor:
        def __init__(self, *args, **kwargs):
            pass
        
        def extract_fields(self, text, **kwargs):
            return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

# Learning system imports with comprehensive fallbacks
# ✅ NEW - Correct and comprehensive
# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from learning.database import LearningDatabase  # ✅ Correct filename
    from learning.pattern_learner import PatternLearner
    print("✅ Learning system imports successful")
    LEARNING_AVAILABLE = True
except ImportError as e:
    print(f"❌ Learning system import failed: {e}")
    LEARNING_AVAILABLE = False
    # Fallback classes if needed
    class LearningDatabase:
        def __init__(self, *args, **kwargs):
            pass
        
        def get_table_correction_stats(self):
            return {'total_table_corrections': 0}
    
    class PatternLearner:
        def __init__(self, *args, **kwargs):
            pass
    
    print("Learning system modules imported successfully")
except ImportError:
    print(" Learning system modules missing, using comprehensive fallbacks")
    
    class LearningDatabase:
        def __init__(self, db_path=None):
            self.db_path = db_path
            print(f" Fallback LearningDatabase initialized with {db_path}")
        
        def get_table_correction_stats(self):
            return {
                'total_table_corrections': 0,
                'average_confidence_improvement': 0.0,
                'documents_with_table_corrections': 0,
                'learned_pattern_count': 0
            }
        
        def save_document_metadata(self, doc_hash, filename, file_size):
            return True
        
        def get_corrections_for_document(self, doc_hash):
            return {}
        
        def save_correction(self, *args, **kwargs):
            return True
        
        def get_learned_table_patterns(self):
            return []
    
    class PatternLearner:
        def __init__(self, db_path=None):
            self.db_path = db_path
            print(f" Fallback PatternLearner initialized with {db_path}")
        
        def learn_from_feedback(self, *args, **kwargs):
            return True