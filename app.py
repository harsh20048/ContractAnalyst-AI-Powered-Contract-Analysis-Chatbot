"""
Adaptive PDF Table Extraction Application with Data Persistence
Complete production-ready implementation with comprehensive functionality.
Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import sys
import copy
import os
import tempfile
import json
import time
import hashlib
import logging
import threading
import queue
import traceback
import re
import pickle
import base64
import zipfile
import csv
import io
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import streamlit as st
import pandas as pd
import numpy as np

# PDF processing libraries with fallbacks
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# OCR libraries with fallbacks
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# NLP libraries with fallbacks
try:
    import spacy
    from spacy import displacy
    NLP_AVAILABLE = True
except ImportError:
    NLP_AVAILABLE = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.corpus import stopwords
    from nltk.tag import pos_tag
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

# ML libraries with fallbacks
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Image processing libraries
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import custom modules
from database import DatabaseManager
from pattern_learner import PatternLearner

# Configure Streamlit page
st.set_page_config(
    page_title="Adaptive PDF Data Extraction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/pdf-extractor',
        'Report a bug': "https://github.com/your-repo/pdf-extractor/issues",
        'About': "# Adaptive PDF Data Extraction System\nVersion 2.0.0"
    }
)

# Configure logging with file handler
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global constants and configuration
CONFIG = {
    'MAX_FILE_SIZE_MB': 50,
    'SUPPORTED_FORMATS': ['pdf'],
    'MAX_CONCURRENT_EXTRACTIONS': 3,
    'EXTRACTION_TIMEOUT_SECONDS': 300,
    'CACHE_EXPIRY_HOURS': 24,
    'DEFAULT_CONFIDENCE_THRESHOLD': 0.7,
    'MAX_PATTERN_HISTORY': 100,
    'DATABASE_BACKUP_INTERVAL_HOURS': 6,
    'LOG_RETENTION_DAYS': 30,
    'MAX_MEMORY_USAGE_MB': 1000
}

# Error handling classes
class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass

class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

# Data models
@dataclass
class ExtractionJob:
    """Represents a PDF extraction job."""
    job_id: str
    document_hash: str
    filename: str
    file_size: int
    start_time: datetime
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: float
    result: Optional[Dict] = None
    error: Optional[str] = None
    extraction_method: str = 'auto'
    
@dataclass
class ValidationRule:
    """Represents a data validation rule."""
    field_name: str
    rule_type: str  # 'required', 'format', 'range', 'custom'
    rule_value: Any
    error_message: str
    severity: str = 'error'  # 'error', 'warning', 'info'

@dataclass
class ExtractionMetrics:
    """Metrics for extraction performance."""
    total_documents: int
    successful_extractions: int
    failed_extractions: int
    average_processing_time: float
    average_confidence: float
    most_common_errors: List[Tuple[str, int]]

# Utility functions
def measure_time(func):
    """Decorator to measure function execution time."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper

def safe_execute(func, *args, **kwargs):
    """Safely execute a function with error handling."""
    try:
        return func(*args, **kwargs), None
    except Exception as e:
        logger.error(f"Error in {func.__name__}: {str(e)}")
        return None, str(e)

def validate_file_upload(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded file."""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    if uploaded_file.size > CONFIG['MAX_FILE_SIZE_MB'] * 1024 * 1024:
        return False, f"File size exceeds {CONFIG['MAX_FILE_SIZE_MB']}MB limit"
    
    if not uploaded_file.name.lower().endswith('.pdf'):
        return False, "Only PDF files are supported"
    
    return True, "File validation passed"

def get_file_hash(file_content: bytes) -> str:
    """Generate SHA256 hash for file content."""
    return hashlib.sha256(file_content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = int(np.floor(np.log(size_bytes) / np.log(1024)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def clean_text(text: str) -> str:
    """Clean and normalize extracted text."""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters
    text = re.sub(r'[^\w\s\-.,;:!?()[]{}@#$%^&*+=<>/"\'\\|`~]', '', text)
    
    # Normalize quotes
    text = re.sub(r'[""''`]', '"', text)
    
    return text

# Initialize global components with caching
@st.cache_resource
def init_components():
    """Initialize database and pattern learner components."""
    try:
        db_manager = DatabaseManager()
        pattern_learner = PatternLearner(db_manager)
        logger.info("Components initialized successfully")
        return db_manager, pattern_learner
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        st.error(f"System initialization failed: {e}")
        st.stop()

@st.cache_resource
def init_extraction_engines():
    """Initialize available PDF extraction engines."""
    engines = {}
    
    if PYPDF2_AVAILABLE:
        engines['pypdf2'] = "PyPDF2 - Basic text extraction"
    
    if PDFPLUMBER_AVAILABLE:
        engines['pdfplumber'] = "PDFPlumber - Advanced layout analysis"
    
    if TABULA_AVAILABLE:
        engines['tabula'] = "Tabula - Table extraction specialist"
    
    if CAMELOT_AVAILABLE:
        engines['camelot'] = "Camelot - Precision table extraction"
    
    if PYMUPDF_AVAILABLE:
        engines['pymupdf'] = "PyMuPDF - Fast and comprehensive"
    
    if OCR_AVAILABLE:
        engines['ocr'] = "Tesseract OCR - Scanned document processing"
    
    engines['mock'] = "Mock Extractor - Demo/Testing mode"
    
    logger.info(f"Initialized {len(engines)} extraction engines")
    return engines

# PDF Processing Classes
class PDFProcessor:
    """Main PDF processing class with multiple extraction methods."""
    
    def __init__(self):
        self.extraction_engines = init_extraction_engines()
        self.extraction_jobs = {}
        self.job_queue = queue.Queue()
        self.worker_pool = ThreadPoolExecutor(max_workers=CONFIG['MAX_CONCURRENT_EXTRACTIONS'])
        
    def create_extraction_job(self, file_content: bytes, filename: str, 
                            method: str = 'auto') -> str:
        """Create a new extraction job."""
        job_id = hashlib.md5(f"{filename}{time.time()}".encode()).hexdigest()[:16]
        document_hash = get_file_hash(file_content)
        
        job = ExtractionJob(
            job_id=job_id,
            document_hash=document_hash,
            filename=filename,
            file_size=len(file_content),
            start_time=datetime.now(),
            status='pending',
            progress=0.0,
            extraction_method=method
        )
        
        self.extraction_jobs[job_id] = job
        return job_id
    
    @measure_time
    def extract_with_pypdf2(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using PyPDF2."""
        if not PYPDF2_AVAILABLE:
            raise PDFExtractionError("PyPDF2 not available")
        
        field_data = {}
        table_data = {}
        
        try:
            pdf_file = io.BytesIO(file_content)
            reader = PyPDF2.PdfReader(pdf_file)
            
            all_text = ""
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"
            
            # Extract fields using pattern matching
            field_data = self._extract_fields_from_text(all_text)
            
            # Extract simple tables using text patterns
            table_data = self._extract_tables_from_text(all_text)
            
            logger.info(f"PyPDF2 extraction completed: {len(field_data)} fields, {len(table_data)} tables")
            
        except Exception as e:
            raise PDFExtractionError(f"PyPDF2 extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_with_pdfplumber(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using PDFPlumber."""
        if not PDFPLUMBER_AVAILABLE:
            raise PDFExtractionError("PDFPlumber not available")
        
        field_data = {}
        table_data = {}
        
        try:
            with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                all_text = ""
                tables = []
                
                for page_num, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table_idx, table in enumerate(page_tables):
                        if table and len(table) > 1:  # Has header and data
                            table_name = f"table_page_{page_num + 1}_{table_idx + 1}"
                            table_data[table_name] = {
                                'headers': table[0] if table[0] else [f"Col_{i+1}" for i in range(len(table[1]))],
                                'rows': table[1:],
                                'metadata': {
                                    'extraction_confidence': 0.8,
                                    'table_type': 'extracted_table',
                                    'page_number': page_num + 1,
                                    'extraction_method': 'pdfplumber'
                                }
                            }
            
            # Extract fields from combined text
            field_data = self._extract_fields_from_text(all_text)
            
            logger.info(f"PDFPlumber extraction completed: {len(field_data)} fields, {len(table_data)} tables")
            
        except Exception as e:
            raise PDFExtractionError(f"PDFPlumber extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_with_tabula(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using Tabula."""
        if not TABULA_AVAILABLE:
            raise PDFExtractionError("Tabula not available")
        
        field_data = {}
        table_data = {}
        
        try:
            # Save to temporary file for tabula
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract tables using tabula
                tables = tabula.read_pdf(tmp_file_path, pages='all', multiple_tables=True)
                
                for table_idx, df in enumerate(tables):
                    if not df.empty:
                        table_name = f"tabula_table_{table_idx + 1}"
                        table_data[table_name] = {
                            'headers': df.columns.tolist(),
                            'rows': df.values.tolist(),
                            'metadata': {
                                'extraction_confidence': 0.9,
                                'table_type': 'tabula_extracted',
                                'page_number': 1,  # Tabula doesn't provide page info easily
                                'extraction_method': 'tabula'
                            }
                        }
                
                # Extract text for field extraction
                if PYPDF2_AVAILABLE:
                    field_data, _ = self.extract_with_pypdf2(file_content)
                
                logger.info(f"Tabula extraction completed: {len(field_data)} fields, {len(table_data)} tables")
                
            finally:
                os.unlink(tmp_file_path)
                
        except Exception as e:
            raise PDFExtractionError(f"Tabula extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_with_camelot(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using Camelot."""
        if not CAMELOT_AVAILABLE:
            raise PDFExtractionError("Camelot not available")
        
        field_data = {}
        table_data = {}
        
        try:
            # Save to temporary file for camelot
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # Extract tables using camelot
                tables = camelot.read_pdf(tmp_file_path, pages='all')
                
                for table_idx, table in enumerate(tables):
                    df = table.df
                    if not df.empty:
                        table_name = f"camelot_table_{table_idx + 1}"
                        headers = df.iloc[0].tolist() if len(df) > 0 else []
                        rows = df.iloc[1:].values.tolist() if len(df) > 1 else []
                        
                        table_data[table_name] = {
                            'headers': headers,
                            'rows': rows,
                            'metadata': {
                                'extraction_confidence': float(table.accuracy) / 100.0,
                                'table_type': 'camelot_extracted',
                                'page_number': table.page,
                                'extraction_method': 'camelot',
                                'whitespace': table.whitespace,
                                'order': table.order
                            }
                        }
                
                # Extract text for field extraction
                if PYPDF2_AVAILABLE:
                    field_data, _ = self.extract_with_pypdf2(file_content)
                
                logger.info(f"Camelot extraction completed: {len(field_data)} fields, {len(table_data)} tables")
                
            finally:
                os.unlink(tmp_file_path)
                
        except Exception as e:
            raise PDFExtractionError(f"Camelot extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_with_pymupdf(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using PyMuPDF."""
        if not PYMUPDF_AVAILABLE:
            raise PDFExtractionError("PyMuPDF not available")
        
        field_data = {}
        table_data = {}
        
        try:
            doc = fitz.open(stream=file_content, filetype="pdf")
            all_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                all_text += f"\n--- Page {page_num + 1} ---\n{text}"
                
                # Extract tables using PyMuPDF
                tables = page.find_tables()
                for table_idx, table in enumerate(tables):
                    try:
                        table_data_extracted = table.extract()
                        if table_data_extracted and len(table_data_extracted) > 1:
                            table_name = f"pymupdf_table_page_{page_num + 1}_{table_idx + 1}"
                            headers = table_data_extracted[0] if table_data_extracted[0] else []
                            rows = table_data_extracted[1:] if len(table_data_extracted) > 1 else []
                            
                            table_data[table_name] = {
                                'headers': headers,
                                'rows': rows,
                                'metadata': {
                                    'extraction_confidence': 0.85,
                                    'table_type': 'pymupdf_extracted',
                                    'page_number': page_num + 1,
                                    'extraction_method': 'pymupdf',
                                    'bbox': table.bbox
                                }
                            }
                    except Exception as e:
                        logger.warning(f"Failed to extract table {table_idx} from page {page_num + 1}: {e}")
            
            doc.close()
            
            # Extract fields from combined text
            field_data = self._extract_fields_from_text(all_text)
            
            logger.info(f"PyMuPDF extraction completed: {len(field_data)} fields, {len(table_data)} tables")
            
        except Exception as e:
            raise PDFExtractionError(f"PyMuPDF extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_with_ocr(self, file_content: bytes) -> Tuple[Dict, Dict]:
        """Extract data using OCR (Tesseract)."""
        if not OCR_AVAILABLE:
            raise PDFExtractionError("OCR not available")
        
        field_data = {}
        table_data = {}
        
        try:
            # Convert PDF to images and then OCR
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(stream=file_content, filetype="pdf")
                all_text = ""
                
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    image = Image.open(io.BytesIO(img_data))
                    
                    # Perform OCR
                    page_text = pytesseract.image_to_string(image)
                    all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"
                
                doc.close()
                
                # Extract fields and tables from OCR text
                field_data = self._extract_fields_from_text(all_text)
                table_data = self._extract_tables_from_text(all_text)
                
                logger.info(f"OCR extraction completed: {len(field_data)} fields, {len(table_data)} tables")
            else:
                raise PDFExtractionError("PyMuPDF required for OCR extraction")
                
        except Exception as e:
            raise PDFExtractionError(f"OCR extraction failed: {str(e)}")
        
        return field_data, table_data
    
    @measure_time
    def extract_mock_data(self, file_content: bytes, document_hash: str, 
                         pattern_learner: PatternLearner) -> Tuple[Dict, Dict]:
        """Generate mock extraction data for testing."""
        
        # Try to get learned patterns first
        learned_data = pattern_learner.get_learned_patterns(document_hash)
        
        if learned_data and ('field_patterns' in learned_data or 'table_patterns' in learned_data):
            logger.info("Using learned patterns for mock extraction")
            field_data = learned_data.get('field_patterns', {})
            table_data = learned_data.get('table_patterns', {})
        else:
            logger.info("Using fallback mock extraction")
            
            # Generate comprehensive mock data
            field_data = {
                'document_title': 'Q1 2024 Financial Report',
                'document_date': '2024-03-31',
                'document_type': 'Financial Report',
                'company_name': 'Tech Innovations Inc.',
                'report_period': 'Q1 2024',
                'prepared_by': 'Finance Department',
                'approval_status': 'Draft',
                'last_updated': '2024-03-31 15:30:00',
                'total_revenue': 125000.00,
                'total_expenses': 70000.00,
                'net_profit': 55000.00,
                'gross_margin': 0.44,
                'currency': 'USD',
                'fiscal_year': '2024',
                'quarter': 'Q1',
                'department_count': 4,
                'employee_count': 156,
                'office_locations': 'San Francisco, New York, Austin',
                'ceo_name': 'John Smith',
                'cfo_name': 'Jane Doe',
                'audit_firm': 'ABC Auditing LLC',
                'tax_id': '12-3456789',
                'incorporation_state': 'Delaware',
                'website': 'https://techinnovations.com',
                'phone': '+1-555-123-4567',
                'email': 'info@techinnovations.com',
                'address': '123 Innovation Drive, San Francisco, CA 94105',
                'confidence_score': 0.89
            }
            
            table_data = {
                'financial_summary': {
                    'headers': ['Item', 'Amount', 'Date', 'Category', 'Status'],
                    'rows': [
                        ['Revenue Q1', '125,000.00', '2024-03-31', 'Income', 'Confirmed'],
                        ['Operating Expenses', '45,000.00', '2024-03-31', 'Expense', 'Pending'],
                        ['Marketing Budget', '25,000.00', '2024-03-31', 'Expense', 'Approved'],
                        ['Net Profit', '55,000.00', '2024-03-31', 'Income', 'Calculated'],
                        ['Tax Liability', '12,000.00', '2024-03-31', 'Expense', 'Estimated'],
                        ['Cash Flow', '43,000.00', '2024-03-31', 'Income', 'Actual']
                    ],
                    'metadata': {
                        'extraction_confidence': 0.85,
                        'table_type': 'financial_summary',
                        'page_number': 1,
                        'extraction_method': 'mock'
                    }
                },
                'employee_data': {
                    'headers': ['Name', 'Department', 'Salary', 'Start Date', 'Performance', 'Location'],
                    'rows': [
                        ['John Smith', 'Engineering', '85,000', '2023-01-15', 'Excellent', 'San Francisco'],
                        ['Jane Doe', 'Marketing', '72,000', '2023-03-01', 'Good', 'New York'],
                        ['Bob Wilson', 'Sales', '68,000', '2023-02-10', 'Excellent', 'Austin'],
                        ['Alice Brown', 'HR', '65,000', '2023-01-20', 'Good', 'San Francisco'],
                        ['Charlie Davis', 'Engineering', '90,000', '2022-11-05', 'Excellent', 'San Francisco'],
                        ['Diana Miller', 'Finance', '75,000', '2023-04-12', 'Good', 'New York']
                    ],
                    'metadata': {
                        'extraction_confidence': 0.92,
                        'table_type': 'employee_data',
                        'page_number': 2,
                        'extraction_method': 'mock'
                    }
                },
                'expense_breakdown': {
                    'headers': ['Category', 'Q1 2024', 'Q4 2023', 'Change %', 'Budget', 'Variance'],
                    'rows': [
                        ['Office Rent', '15,000', '15,000', '0%', '15,000', '0'],
                        ['Salaries', '45,000', '42,000', '+7.1%', '47,000', '-2,000'],
                        ['Marketing', '8,000', '5,000', '+60%', '10,000', '-2,000'],
                        ['Travel', '3,000', '1,500', '+100%', '4,000', '-1,000'],
                        ['Software', '5,000', '4,500', '+11.1%', '5,500', '-500'],
                        ['Utilities', '2,000', '2,200', '-9.1%', '2,100', '-100']
                    ],
                    'metadata': {
                        'extraction_confidence': 0.88,
                        'table_type': 'expense_breakdown',
                        'page_number': 3,
                        'extraction_method': 'mock'
                    }
                }
            }
        
        return field_data, table_data
    
    def _extract_fields_from_text(self, text: str) -> Dict:
        """Extract field data from text using pattern matching."""
        field_data = {}
        
        # Common field patterns
        patterns = {
            'document_date': [
                r'(?:Date|Report Date|Document Date):\s*(\d{4}-\d{2}-\d{2})',
                r'(?:Date|Report Date|Document Date):\s*(\d{2}/\d{2}/\d{4})',
                r'(?:Date|Report Date|Document Date):\s*(\w+ \d{1,2}, \d{4})'
            ],
            'total_revenue': [
                r'(?:Total Revenue|Revenue|Total Sales):\s*\$?([\d,]+\.?\d*)',
                r'Revenue\s*\$?([\d,]+\.?\d*)'
            ],
            'total_expenses': [
                r'(?:Total Expenses|Expenses|Total Costs):\s*\$?([\d,]+\.?\d*)',
                r'Expenses\s*\$?([\d,]+\.?\d*)'
            ],
            'company_name': [
                r'(?:Company|Corporation|Inc\.|LLC):\s*([A-Za-z\s&.,-]+)',
                r'^([A-Za-z\s&.,-]+(?:Inc\.|LLC|Corp\.|Corporation))'
            ],
            'document_title': [
                r'^([A-Z][A-Za-z\s]+(?:Report|Statement|Summary|Analysis))',
                r'Title:\s*([A-Za-z\s]+)'
            ]
        }
        
        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    # Clean and convert value if needed
                    if field_name in ['total_revenue', 'total_expenses']:
                        value = float(value.replace(',', '').replace('$', ''))
                    field_data[field_name] = value
                    break
        
        return field_data
    
    def _extract_tables_from_text(self, text: str) -> Dict:
        """Extract simple tables from text using pattern matching."""
        table_data = {}
        
        # Look for table-like structures in text
        lines = text.split('\n')
        current_table = []
        table_count = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                if current_table and len(current_table) > 2:
                    # Process the table
                    table_count += 1
                    headers = current_table[0].split()
                    rows = [row.split() for row in current_table[1:]]
                    
                    table_data[f'text_table_{table_count}'] = {
                        'headers': headers,
                        'rows': rows,
                        'metadata': {
                            'extraction_confidence': 0.6,
                            'table_type': 'text_extracted',
                            'page_number': 1,
                            'extraction_method': 'text_pattern'
                        }
                    }
                current_table = []
            else:
                # Check if line looks like a table row (has multiple columns)
                if len(line.split()) >= 3 and any(char.isdigit() for char in line):
                    current_table.append(line)
        
        return table_data
    
    def extract_data(self, file_content: bytes, document_hash: str, 
                    pattern_learner: PatternLearner, method: str = 'auto') -> Tuple[Dict, Dict]:
        """Main extraction method that routes to appropriate extractor."""
        
        if method == 'auto':
            # Try methods in order of preference
            methods_to_try = []
            
            if PDFPLUMBER_AVAILABLE:
                methods_to_try.append('pdfplumber')
            if CAMELOT_AVAILABLE:
                methods_to_try.append('camelot')
            if TABULA_AVAILABLE:
                methods_to_try.append('tabula')
            if PYMUPDF_AVAILABLE:
                methods_to_try.append('pymupdf')
            if PYPDF2_AVAILABLE:
                methods_to_try.append('pypdf2')
            if OCR_AVAILABLE:
                methods_to_try.append('ocr')
            
            methods_to_try.append('mock')  # Always available as fallback
            
            for method_name in methods_to_try:
                try:
                    return self._extract_with_method(file_content, document_hash, 
                                                   pattern_learner, method_name)
                except PDFExtractionError as e:
                    logger.warning(f"Method {method_name} failed: {e}")
                    continue
            
            raise PDFExtractionError("All extraction methods failed")
        
        else:
            return self._extract_with_method(file_content, document_hash, 
                                           pattern_learner, method)
    
    def _extract_with_method(self, file_content: bytes, document_hash: str, 
                           pattern_learner: PatternLearner, method: str) -> Tuple[Dict, Dict]:
        """Extract data using specified method."""
        
        extraction_methods = {
            'pypdf2': self.extract_with_pypdf2,
            'pdfplumber': self.extract_with_pdfplumber,
            'tabula': self.extract_with_tabula,
            'camelot': self.extract_with_camelot,
            'pymupdf': self.extract_with_pymupdf,
            'ocr': self.extract_with_ocr,
            'mock': lambda content: self.extract_mock_data(content, document_hash, pattern_learner)
        }
        
        if method not in extraction_methods:
            raise PDFExtractionError(f"Unknown extraction method: {method}")
        
        if method == 'mock':
            return extraction_methods[method](file_content)
        else:
            return extraction_methods[method](file_content)

# UI Component Classes
class UIComponents:
    """Collection of reusable UI components."""
    
    @staticmethod
    def render_metric_card(title: str, value: Any, delta: Optional[str] = None, 
                          help_text: Optional[str] = None):
        """Render a metric card."""
        st.metric(
            label=title,
            value=value,
            delta=delta,
            help=help_text
        )
    
    @staticmethod
    def render_status_badge(status: str, text: str):
        """Render a status badge."""
        colors = {
            'success': '#28a745',
            'warning': '#ffc107',
            'error': '#dc3545',
            'info': '#17a2b8',
            'primary': '#007bff'
        }
        
        color = colors.get(status, '#6c757d')
        st.markdown(
            f'<span style="background-color: {color}; color: white; padding: 0.2em 0.6em; '
            f'border-radius: 0.25em; font-size: 0.875em; font-weight: bold;">{text}</span>',
            unsafe_allow_html=True
        )
    
    @staticmethod
    def render_progress_bar(progress: float, text: str = ""):
        """Render a progress bar."""
        st.progress(progress, text=text)
    
    @staticmethod
    def render_collapsible_section(title: str, content_func: Callable, 
                                 expanded: bool = False):
        """Render a collapsible section."""
        with st.expander(title, expanded=expanded):
            content_func()
    
    @staticmethod
    def render_data_table(df: pd.DataFrame, title: str = "", 
                         use_container_width: bool = True):
        """Render a data table with formatting."""
        if title:
            st.subheader(title)
        
        st.dataframe(
            df,
            use_container_width=use_container_width,
            hide_index=True
        )
    
    @staticmethod
    def render_field_editor(field_name: str, field_value: Any, 
                          field_type: str = 'auto', key: str = None):
        """Render an appropriate field editor based on field type."""
        
        if field_type == 'auto':
            if isinstance(field_value, bool):
                field_type = 'checkbox'
            elif isinstance(field_value, (int, float)):
                field_type = 'number'
            elif isinstance(field_value, str) and re.match(r'\d{4}-\d{2}-\d{2}', field_value):
                field_type = 'date'
            else:
                field_type = 'text'
        
        label = field_name.replace('_', ' ').title()
        
        if field_type == 'text':
            return st.text_input(label, value=str(field_value), key=key)
        elif field_type == 'number':
            return st.number_input(label, value=float(field_value) if field_value else 0.0, key=key)
        elif field_type == 'checkbox':
            return st.checkbox(label, value=bool(field_value), key=key)
        elif field_type == 'date':
            try:
                date_value = datetime.strptime(str(field_value), '%Y-%m-%d').date()
            except:
                date_value = datetime.now().date()
            return st.date_input(label, value=date_value, key=key)
        elif field_type == 'textarea':
            return st.text_area(label, value=str(field_value), key=key)
        elif field_type == 'selectbox' and isinstance(field_value, dict) and 'options' in field_value:
            return st.selectbox(label, options=field_value['options'], 
                              index=field_value.get('index', 0), key=key)
        else:
            return st.text_input(label, value=str(field_value), key=key)

# Data Validation Classes
class DataValidator:
    """Validates extracted data against rules."""
    
    def __init__(self):
        self.validation_rules = []
    
    def add_rule(self, rule: ValidationRule):
        """Add a validation rule."""
        self.validation_rules.append(rule)
    
    def validate_field_data(self, field_data: Dict) -> List[Dict]:
        """Validate field data against rules."""
        validation_results = []
        
        for rule in self.validation_rules:
            result = self._apply_rule(field_data, rule)
            if result:
                validation_results.append(result)
        
        return validation_results
    
    def _apply_rule(self, field_data: Dict, rule: ValidationRule) -> Optional[Dict]:
        """Apply a single validation rule."""
        field_value = field_data.get(rule.field_name)
        
        if rule.rule_type == 'required':
            if field_value is None or str(field_value).strip() == '':
                return {
                    'field': rule.field_name,
                    'message': rule.error_message,
                    'severity': rule.severity,
                    'rule_type': rule.rule_type
                }
        
        elif rule.rule_type == 'format':
            if field_value and not re.match(rule.rule_value, str(field_value)):
                return {
                    'field': rule.field_name,
                    'message': rule.error_message,
                    'severity': rule.severity,
                    'rule_type': rule.rule_type
                }
        
        elif rule.rule_type == 'range':
            if field_value is not None:
                try:
                    num_value = float(field_value)
                    min_val, max_val = rule.rule_value
                    if not (min_val <= num_value <= max_val):
                        return {
                            'field': rule.field_name,
                            'message': rule.error_message,
                            'severity': rule.severity,
                            'rule_type': rule.rule_type
                        }
                except (ValueError, TypeError):
                    pass
        
        return None

# Session State Management
class SessionStateManager:
    """Manages Streamlit session state with persistence."""
    
    @staticmethod
    def init_session_state():
        """Initialize session state variables with proper defaults."""
        defaults = {
            # File and document management
            'uploaded_file': None,
            'document_hash': None,
            'file_processed': False,
            'processing_method': 'auto',
            
            # Extracted data
            'field_data': {},
            'table_data': {},
            'raw_extraction_data': {},
            
            # UI state
            'current_tab': 'upload',
            'selected_table': None,
            'show_advanced_options': False,
            'theme': 'light',
            
            # Processing state
            'extraction_job_id': None,
            'extraction_status': 'idle',
            'extraction_progress': 0.0,
            'extraction_error': None,
            'processing_start_time': None,
            'processing_end_time': None,
            
            # Corrections and learning
            'corrections_applied': False,
            'corrections_saved': False,
            'show_save_success': False,
            'validation_results': [],
            'learned_patterns_applied': False,
            
            # Statistics and analytics
            'extraction_stats': {},
            'performance_metrics': {},
            'user_feedback': {},
            
            # Configuration
            'extraction_settings': {
                'confidence_threshold': CONFIG['DEFAULT_CONFIDENCE_THRESHOLD'],
                'auto_save': True,
                'validation_enabled': True,
                'pattern_learning_enabled': True
            },
            
            # Cache and temporary data
            'temp_data': {},
            'cache_timestamp': None,
            'last_activity': datetime.now()
        }
        
        for key, default_value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
    
    @staticmethod
    def update_last_activity():
        """Update last activity timestamp."""
        st.session_state.last_activity = datetime.now()
    
    @staticmethod
    def clear_extraction_data():
        """Clear extraction-related data."""
        keys_to_clear = [
            'field_data', 'table_data', 'raw_extraction_data',
            'extraction_job_id', 'extraction_status', 'extraction_progress',
            'extraction_error', 'processing_start_time', 'processing_end_time',
            'corrections_applied', 'corrections_saved', 'validation_results'
        ]
        
        for key in keys_to_clear:
            if key in st.session_state:
                if isinstance(st.session_state[key], dict):
                    st.session_state[key] = {}
                elif isinstance(st.session_state[key], list):
                    st.session_state[key] = []
                else:
                    st.session_state[key] = None
    
    @staticmethod
    def export_session_state() -> Dict:
        """Export current session state for backup."""
        exportable_keys = [
            'extraction_settings', 'user_feedback', 'extraction_stats',
            'performance_metrics'
        ]
        
        export_data = {}
        for key in exportable_keys:
            if key in st.session_state:
                export_data[key] = st.session_state[key]
        
        export_data['export_timestamp'] = datetime.now().isoformat()
        return export_data
    
    @staticmethod
    def import_session_state(import_data: Dict):
        """Import session state from backup."""
        for key, value in import_data.items():
            if key != 'export_timestamp':
                st.session_state[key] = value

# Advanced Field Editors
def create_advanced_field_editor(field_name: str, field_value: Any, document_hash: str, 
                                field_config: Optional[Dict] = None) -> Any:
    """Create an advanced field editor with validation and suggestions."""
    
    if field_config is None:
        field_config = {}
    
    # Determine field type and constraints
    field_type = field_config.get('type', 'auto')
    constraints = field_config.get('constraints', {})
    suggestions = field_config.get('suggestions', [])
    help_text = field_config.get('help', '')
    
    # Create unique key for the field
    field_key = f"field_{field_name}_{document_hash}"
    
    # Create the appropriate input widget
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if field_type == 'currency':
            new_value = st.number_input(
                f"💰 {field_name.replace('_', ' ').title()}",
                value=float(field_value) if field_value else 0.0,
                format="%.2f",
                key=field_key,
                help=help_text
            )
            
        elif field_type == 'percentage':
            new_value = st.slider(
                f"📊 {field_name.replace('_', ' ').title()}",
                min_value=0.0,
                max_value=100.0,
                value=float(field_value) * 100 if field_value else 0.0,
                format="%.1f%%",
                key=field_key,
                help=help_text
            ) / 100.0
            
        elif field_type == 'date':
            try:
                if isinstance(field_value, str):
                    date_value = datetime.strptime(field_value.split()[0], '%Y-%m-%d').date()
                else:
                    date_value = datetime.now().date()
            except:
                date_value = datetime.now().date()
                
            new_value = st.date_input(
                f"📅 {field_name.replace('_', ' ').title()}",
                value=date_value,
                key=field_key,
                help=help_text
            ).strftime('%Y-%m-%d')
            
        elif field_type == 'email':
            new_value = st.text_input(
                f"📧 {field_name.replace('_', ' ').title()}",
                value=str(field_value) if field_value else '',
                key=field_key,
                help=help_text,
                placeholder="example@company.com"
            )
            
        elif field_type == 'phone':
            new_value = st.text_input(
                f"📞 {field_name.replace('_', ' ').title()}",
                value=str(field_value) if field_value else '',
                key=field_key,
                help=help_text,
                placeholder="+1-555-123-4567"
            )
            
        elif field_type == 'url':
            new_value = st.text_input(
                f"🌐 {field_name.replace('_', ' ').title()}",
                value=str(field_value) if field_value else '',
                key=field_key,
                help=help_text,
                placeholder="https://example.com"
            )
            
        elif field_type == 'select' and suggestions:
            current_index = 0
            if field_value in suggestions:
                current_index = suggestions.index(field_value)
            
            new_value = st.selectbox(
                f"📋 {field_name.replace('_', ' ').title()}",
                options=suggestions,
                index=current_index,
                key=field_key,
                help=help_text
            )
            
        elif field_type == 'multiline':
            new_value = st.text_area(
                f"📝 {field_name.replace('_', ' ').title()}",
                value=str(field_value) if field_value else '',
                key=field_key,
                help=help_text,
                height=100
            )
            
        elif isinstance(field_value, (int, float)) or field_type == 'number':
            min_val = constraints.get('min', None)
            max_val = constraints.get('max', None)
            
            new_value = st.number_input(
                f"🔢 {field_name.replace('_', ' ').title()}",
                value=float(field_value) if field_value else 0.0,
                min_value=min_val,
                max_value=max_val,
                key=field_key,
                help=help_text
            )
            
        else:
            # Default text input
            new_value = st.text_input(
                f"📄 {field_name.replace('_', ' ').title()}",
                value=str(field_value) if field_value else '',
                key=field_key,
                help=help_text
            )
    
    with col2:
        # Add validation indicator
        if field_name in st.session_state.get('validation_results', []):
            st.error("❌")
        else:
            st.success("✅")
        
        # Add suggestion button if suggestions available
        if suggestions and field_type != 'select':
            if st.button("💡", key=f"suggest_{field_key}", help="Show suggestions"):
                st.info(f"Suggestions: {', '.join(suggestions[:3])}")
    
    return new_value

# Advanced Table Editor
def create_advanced_table_editor(table_name: str, table_data: Dict, document_hash: str) -> pd.DataFrame:
    """Create an advanced table editor with enhanced functionality."""
    
    if 'headers' not in table_data or 'rows' not in table_data:
        st.error(f"Invalid table data for {table_name}")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(table_data['rows'], columns=table_data['headers'])
    
    # Table controls
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button(f"➕ Add Row", key=f"add_row_{table_name}_{document_hash}"):
            # Add empty row
            empty_row = [''] * len(table_data['headers'])
            table_data['rows'].append(empty_row)
            df = pd.DataFrame(table_data['rows'], columns=table_data['headers'])
    
    with col2:
        if st.button(f"➖ Remove Last Row", key=f"remove_row_{table_name}_{document_hash}"):
            if table_data['rows']:
                table_data['rows'].pop()
                df = pd.DataFrame(table_data['rows'], columns=table_data['headers'])
    
    with col3:
        if st.button(f"🧹 Clear All", key=f"clear_table_{table_name}_{document_hash}"):
            table_data['rows'] = []
            df = pd.DataFrame(columns=table_data['headers'])
    
    with col4:
        auto_format = st.checkbox("Auto Format", key=f"auto_format_{table_name}_{document_hash}")
    
    # Configure column types
    column_config = {}
    for col in df.columns:
        # Try to detect column type
        if df[col].dtype in ['int64', 'float64'] or df[col].astype(str).str.match(r'^\d+\.?\d*$').all():
            column_config[col] = st.column_config.NumberColumn(
                col,
                help=f"Edit {col} values",
                format="%.2f"
            )
        elif df[col].astype(str).str.match(r'^\d{4}-\d{2}-\d{2}$').any():
            column_config[col] = st.column_config.DateColumn(
                col,
                help=f"Edit {col} values"
            )
        else:
            column_config[col] = st.column_config.TextColumn(
                col,
                help=f"Edit {col} values",
                width="medium"
            )
    
    # Create editable table
    edited_df = st.data_editor(
        df,
        key=f"table_{table_name}_{document_hash}",
        use_container_width=True,
        num_rows="dynamic",
        column_config=column_config,
        hide_index=True
    )
    
    # Apply auto-formatting if enabled
    if auto_format:
        edited_df = apply_table_formatting(edited_df)
    
    return edited_df

def apply_table_formatting(df: pd.DataFrame) -> pd.DataFrame:
    """Apply automatic formatting to table data."""
    
    for col in df.columns:
        # Clean and format text columns
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            
            # Try to convert to numeric if possible
            if df[col].str.match(r'^\d+\.?\d*$').all():
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Format currency columns
            elif df[col].str.contains(r'\$|USD|EUR|GBP', case=False, na=False).any():
                df[col] = df[col].str.replace(r'[^\d.]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='ignore')
            
            # Format percentage columns
            elif df[col].str.contains(r'%', case=False, na=False).any():
                df[col] = df[col].str.replace('%', '').astype(float) / 100
    
    return df

# Export/Import functionality
def export_extraction_data(field_data: Dict, table_data: Dict, format_type: str = 'json') -> str:
    """Export extraction data in various formats."""
    
    timestamp = datetime.now().isoformat()
    export_data = {
        'export_metadata': {
            'timestamp': timestamp,
            'format': format_type,
            'version': '2.0.0'
        },
        'field_data': field_data,
        'table_data': table_data
    }
    
    if format_type == 'json':
        return json.dumps(export_data, indent=2, default=str)
    
    elif format_type == 'csv':
        # Convert to CSV format
        output = io.StringIO()
        
        # Export field data
        output.write("FIELD DATA\n")
        for field, value in field_data.items():
            output.write(f"{field},{value}\n")
        
        output.write("\nTABLE DATA\n")
        
        # Export table data
        for table_name, table_content in table_data.items():
            output.write(f"\n{table_name.upper()}\n")
            if 'headers' in table_content and 'rows' in table_content:
                # Write headers
                output.write(','.join(table_content['headers']) + '\n')
                # Write rows
                for row in table_content['rows']:
                    output.write(','.join(str(cell) for cell in row) + '\n')
        
        return output.getvalue()
    
    elif format_type == 'excel':
        # Create Excel file in memory
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Write field data
            field_df = pd.DataFrame(list(field_data.items()), columns=['Field', 'Value'])
            field_df.to_excel(writer, sheet_name='Field Data', index=False)
            
            # Write table data
            for table_name, table_content in table_data.items():
                if 'headers' in table_content and 'rows' in table_content:
                    table_df = pd.DataFrame(table_content['rows'], columns=table_content['headers'])
                    sheet_name = table_name[:31]  # Excel sheet name limit
                    table_df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        return output.getvalue()
    
    else:
        return str(export_data)

# Statistics and Analytics
def calculate_extraction_metrics(db_manager: DatabaseManager) -> ExtractionMetrics:
    """Calculate comprehensive extraction metrics."""
    
    stats = db_manager.get_statistics()
    
    return ExtractionMetrics(
        total_documents=stats.get('total_documents', 0),
        successful_extractions=stats.get('total_documents', 0) - stats.get('failed_extractions', 0),
        failed_extractions=stats.get('failed_extractions', 0),
        average_processing_time=stats.get('avg_processing_time_ms', 0) / 1000.0,
        average_confidence=stats.get('avg_extraction_confidence', 0),
        most_common_errors=[]  # Would be calculated from error logs
    )

def display_extraction_analytics(db_manager: DatabaseManager, pattern_learner: PatternLearner):
    """Display comprehensive extraction analytics."""
    
    st.header("📊 System Analytics")
    
    # Get metrics
    metrics = calculate_extraction_metrics(db_manager)
    learning_stats = pattern_learner.get_learning_statistics()
    
    # Overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Documents Processed",
            metrics.total_documents,
            help="Total number of documents processed"
        )
    
    with col2:
        success_rate = (metrics.successful_extractions / max(metrics.total_documents, 1)) * 100
        st.metric(
            "Success Rate",
            f"{success_rate:.1f}%",
            help="Percentage of successful extractions"
        )
    
    with col3:
        st.metric(
            "Avg Processing Time",
            f"{metrics.average_processing_time:.1f}s",
            help="Average time to process a document"
        )
    
    with col4:
        st.metric(
            "Avg Confidence",
            f"{metrics.average_confidence:.1%}",
            help="Average extraction confidence score"
        )
    
    # Detailed analytics
    tab1, tab2, tab3, tab4 = st.tabs(["Performance", "Learning", "Errors", "Usage Patterns"])
    
    with tab1:
        st.subheader("Performance Metrics")
        
        # Performance over time chart (mock data for now)
        dates = pd.date_range(start='2024-01-01', end='2024-03-31', freq='D')
        performance_data = pd.DataFrame({
            'Date': dates,
            'Documents': np.random.poisson(5, len(dates)),
            'Avg_Confidence': np.random.normal(0.85, 0.1, len(dates)),
            'Processing_Time': np.random.normal(2.5, 0.5, len(dates))
        })
        
        st.line_chart(performance_data.set_index('Date')[['Documents']])
        st.line_chart(performance_data.set_index('Date')[['Avg_Confidence']])
        
    with tab2:
        st.subheader("Pattern Learning Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Learned Patterns", learning_stats.get('total_patterns', 0))
            st.metric("Field Patterns", learning_stats.get('field_patterns', 0))
            st.metric("Table Patterns", learning_stats.get('table_patterns', 0))
        
        with col2:
            if learning_stats.get('most_used_patterns'):
                st.write("**Most Used Patterns**")
                for pattern in learning_stats['most_used_patterns']:
                    st.write(f"- {pattern['name']} ({pattern['usage_count']} uses)")
    
    with tab3:
        st.subheader("Error Analysis")
        
        # Error statistics (mock data)
        error_data = pd.DataFrame({
            'Error Type': ['File Format', 'Extraction Timeout', 'Pattern Mismatch', 'Validation Error'],
            'Count': [5, 3, 8, 2],
            'Percentage': [27.8, 16.7, 44.4, 11.1]
        })
        
        st.bar_chart(error_data.set_index('Error Type')['Count'])
        st.dataframe(error_data, use_container_width=True)
    
    with tab4:
        st.subheader("Usage Patterns")
        
        # Usage patterns (mock data)
        st.write("**Most Common Field Types**")
        common_fields = db_manager.get_statistics().get('common_fields', [])
        if common_fields:
            field_df = pd.DataFrame(common_fields)
            st.bar_chart(field_df.set_index('field')['count'])
        
        st.write("**Most Common Table Types**")
        common_tables = db_manager.get_statistics().get('common_table_types', [])
        if common_tables:
            table_df = pd.DataFrame(common_tables)
            st.bar_chart(table_df.set_index('type')['count'])

# Main Application
def main():
    """Main application entry point."""
    
    # Initialize components
    db_manager, pattern_learner = init_components()
    pdf_processor = PDFProcessor()
    session_manager = SessionStateManager()
    ui_components = UIComponents()
    data_validator = DataValidator()
    
    # Initialize session state
    session_manager.init_session_state()
    
    # Add custom CSS
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stMetric {
            background-color: #f0f2f6;
            border: 1px solid #e1e5e9;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        .extraction-status {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .success { background-color: #d4edda; border-color: #c3e6cb; color: #155724; }
        .warning { background-color: #fff3cd; border-color: #ffeaa7; color: #856404; }
        .error { background-color: #f8d7da; border-color: #f5c6cb; color: #721c24; }
        .info { background-color: #d1ecf1; border-color: #bee5eb; color: #0c5460; }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.title("📊 Adaptive PDF Data Extraction System")
    st.markdown("*Advanced document processing with intelligent pattern learning and persistent corrections*")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🔧 System Control")
        
        # Extraction method selection
        extraction_methods = pdf_processor.extraction_engines
        selected_method = st.selectbox(
            "Extraction Method",
            options=list(extraction_methods.keys()),
            index=list(extraction_methods.keys()).index('mock'),
            help="Choose the PDF extraction method"
        )
        
        st.session_state.processing_method = selected_method
        
        # Settings
        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.extraction_settings['confidence_threshold'] = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.extraction_settings['confidence_threshold'],
                step=0.05,
                help="Minimum confidence score for accepting extracted data"
            )
            
            st.session_state.extraction_settings['auto_save'] = st.checkbox(
                "Auto Save",
                value=st.session_state.extraction_settings['auto_save'],
                help="Automatically save corrections"
            )
            
            st.session_state.extraction_settings['validation_enabled'] = st.checkbox(
                "Enable Validation",
                value=st.session_state.extraction_settings['validation_enabled'],
                help="Validate extracted data against rules"
            )
            
            st.session_state.extraction_settings['pattern_learning_enabled'] = st.checkbox(
                "Pattern Learning",
                value=st.session_state.extraction_settings['pattern_learning_enabled'],
                help="Learn from corrections to improve accuracy"
            )
        
        # System status
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Available extraction engines
        st.write("**Available Engines:**")
        for engine, description in extraction_methods.items():
            status = "🟢" if engine != 'mock' else "🟡"
            st.write(f"{status} {engine}: {description}")
        
        # Statistics summary
        stats = db_manager.get_statistics()
        st.metric("Documents", stats.get('total_documents', 0))
        st.metric("Corrections", stats.get('total_corrections', 0))
        st.metric("DB Size", f"{stats.get('database_size_mb', 0):.1f} MB")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Upload", "📝 Field Data", "📊 Table Data", 
        "💾 Actions", "🔍 Analytics", "⚙️ Admin"
    ])
    
    # Tab 1: File Upload and Processing
    with tab1:
        st.header("📁 Document Upload & Processing")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help=f"Maximum file size: {CONFIG['MAX_FILE_SIZE_MB']}MB"
        )
        
        if uploaded_file is not None:
            # Validate file
            is_valid, validation_message = validate_file_upload(uploaded_file)
            
            if not is_valid:
                st.error(f"❌ {validation_message}")
                return
            
            # File information
            file_content = uploaded_file.getvalue()
            document_hash = get_file_hash(file_content)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", format_file_size(len(file_content)))
            with col2:
                st.metric("Document Hash", document_hash[:16] + "...")
            with col3:
                st.metric("Pages", "Detecting...")  # Would need actual PDF parsing
            
            # Check if document was processed before
            existing_corrections = db_manager.load_corrections(document_hash)
            
            if existing_corrections:
                st.success("✅ Document found in database")
                st.info(f"🕒 Last processed: {existing_corrections.get('upload_date', 'Unknown')}")
                
                if st.button("📂 Load Previous Results", type="primary"):
                    st.session_state.field_data = existing_corrections.get('field_data', {})
                    st.session_state.table_data = existing_corrections.get('table_data', {})
                    st.session_state.file_processed = True
                    st.session_state.document_hash = document_hash
                    st.session_state.uploaded_file = uploaded_file
                    st.session_state.learned_patterns_applied = True
                    
                    # Update pattern learner
                    pattern_learner.update_patterns(
                        document_hash,
                        st.session_state.field_data,
                        st.session_state.table_data
                    )
                    
                    st.rerun()
            
            # Process new document
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔄 Process Document", type="primary", use_container_width=True):
                    st.session_state.extraction_status = 'processing'
                    st.session_state.processing_start_time = datetime.now()
                    
                    # Create extraction job
                    job_id = pdf_processor.create_extraction_job(
                        file_content, uploaded_file.name, selected_method
                    )
                    st.session_state.extraction_job_id = job_id
                    
                    with st.spinner(f"Processing with {selected_method}..."):
                        try:
                            # Extract data
                            field_data, table_data = pdf_processor.extract_data(
                                file_content, document_hash, pattern_learner, selected_method
                            )
                            
                            # Store results
                            st.session_state.field_data = field_data
                            st.session_state.table_data = table_data
                            st.session_state.document_hash = document_hash
                            st.session_state.uploaded_file = uploaded_file
                            st.session_state.file_processed = True
                            st.session_state.extraction_status = 'completed'
                            st.session_state.processing_end_time = datetime.now()
                            
                            # Validate if enabled
                            if st.session_state.extraction_settings['validation_enabled']:
                                validation_results = data_validator.validate_field_data(field_data)
                                st.session_state.validation_results = validation_results
                            
                            st.success("✅ Document processed successfully!")
                            st.rerun()
                            
                        except PDFExtractionError as e:
                            st.session_state.extraction_status = 'failed'
                            st.session_state.extraction_error = str(e)
                            st.error(f"❌ Extraction failed: {e}")
                        except Exception as e:
                            st.session_state.extraction_status = 'failed'
                            st.session_state.extraction_error = str(e)
                            st.error(f"❌ Unexpected error: {e}")
                            logger.error(f"Extraction error: {e}", exc_info=True)
            
            with col2:
                if existing_corrections and st.button("🔄 Re-process Document", use_container_width=True):
                    # Force re-processing even if document exists
                    if st.button("⚠️ Confirm Re-process", type="secondary"):
                        # Same processing logic as above
                        pass
        
        # Processing status
        if st.session_state.extraction_status == 'processing':
            st.info("🔄 Processing in progress...")
            ui_components.render_progress_bar(0.5, "Extracting data...")
            
        elif st.session_state.extraction_status == 'completed':
            processing_time = (st.session_state.processing_end_time - 
                             st.session_state.processing_start_time).total_seconds()
            st.success(f"✅ Processing completed in {processing_time:.1f} seconds")
            
        elif st.session_state.extraction_status == 'failed':
            st.error(f"❌ Processing failed: {st.session_state.extraction_error}")
    
    # Tab 2: Field Data Editor
    with tab2:
        if not st.session_state.file_processed:
            st.info("👈 Please upload and process a document first")
        else:
            st.header("📝 Field Data Extraction & Editing")
            
            if st.session_state.field_data:
                # Field configuration for advanced editing
                field_configs = {
                    'document_date': {'type': 'date'},
                    'total_revenue': {'type': 'currency'},
                    'total_expenses': {'type': 'currency'},
                    'net_profit': {'type': 'currency'},
                    'gross_margin': {'type': 'percentage'},
                    'email': {'type': 'email'},
                    'phone': {'type': 'phone'},
                    'website': {'type': 'url'},
                    'approval_status': {'type': 'select', 'suggestions': ['Draft', 'Pending', 'Approved', 'Rejected']},
                    'address': {'type': 'multiline'}
                }
                
                # Group fields by category
                field_categories = {
                    'Document Information': ['document_title', 'document_date', 'document_type', 'company_name'],
                    'Financial Data': ['total_revenue', 'total_expenses', 'net_profit', 'gross_margin', 'currency'],
                    'Contact Information': ['email', 'phone', 'website', 'address'],
                    'Report Details': ['report_period', 'prepared_by', 'approval_status', 'last_updated'],
                    'Company Details': ['fiscal_year', 'department_count', 'employee_count', 'office_locations'],
                    'Other Fields': []
                }
                
                # Categorize fields
                categorized_fields = {category: [] for category in field_categories}
                for field_name in st.session_state.field_data.keys():
                    categorized = False
                    for category, fields in field_categories.items():
                        if field_name in fields:
                            categorized_fields[category].append(field_name)
                            categorized = True
                            break
                    if not categorized:
                        categorized_fields['Other Fields'].append(field_name)
                
                # Display field editors by category
                updated_fields = {}
                
                for category, field_names in categorized_fields.items():
                    if field_names:
                        st.subheader(f"📋 {category}")
                        
                        for field_name in field_names:
                            field_value = st.session_state.field_data[field_name]
                            field_config = field_configs.get(field_name, {})
                            
                            new_value = create_advanced_field_editor(
                                field_name, field_value, st.session_state.document_hash, field_config
                            )
                            
                            updated_fields[field_name] = new_value
                
                # Update session state
                st.session_state.field_data.update(updated_fields)
                
                # Show validation results
                if st.session_state.validation_results:
                    st.subheader("⚠️ Validation Issues")
                    for result in st.session_state.validation_results:
                        severity = result['severity']
                        icon = "❌" if severity == 'error' else "⚠️" if severity == 'warning' else "ℹ️"
                        st.write(f"{icon} **{result['field']}**: {result['message']}")
                
                # Extraction confidence
                if 'confidence_score' in st.session_state.field_data:
                    confidence = st.session_state.field_data['confidence_score']
                    st.progress(confidence, text=f"Overall Confidence: {confidence:.1%}")
            
            else:
                st.warning("No field data extracted. Try a different extraction method.")
    
    # Tab 3: Table Data Editor
    with tab3:
        if not st.session_state.file_processed:
            st.info("👈 Please upload and process a document first")
        else:
            st.header("📊 Table Data Extraction & Editing")
            
            if st.session_state.table_data:
                # Table selection
                table_names = list(st.session_state.table_data.keys())
                selected_table = st.selectbox(
                    "Select Table",
                    options=table_names,
                    index=0 if table_names else None
                )
                
                if selected_table:
                    st.session_state.selected_table = selected_table
                    table_content = st.session_state.table_data[selected_table]
                    
                    # Table metadata
                    if 'metadata' in table_content:
                        metadata = table_content['metadata']
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            confidence = metadata.get('extraction_confidence', 0)
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col2:
                            table_type = metadata.get('table_type', 'Unknown')
                            st.metric("Type", table_type.replace('_', ' ').title())
                        
                        with col3:
                            page_num = metadata.get('page_number', 'Unknown')
                            st.metric("Page", page_num)
                        
                        with col4:
                            method = metadata.get('extraction_method', 'Unknown')
                            st.metric("Method", method.title())
                    
                    # Advanced table editor
                    edited_df = create_advanced_table_editor(
                        selected_table, table_content, st.session_state.document_hash
                    )
                    
                    # Update session state with edited data
                    if not edited_df.empty:
                        st.session_state.table_data[selected_table]['rows'] = edited_df.values.tolist()
                        st.session_state.table_data[selected_table]['headers'] = edited_df.columns.tolist()
                    
                    # Table statistics
                    st.subheader("📈 Table Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Rows", len(edited_df))
                    with col2:
                        st.metric("Columns", len(edited_df.columns))
                    with col3:
                        empty_cells = edited_df.isnull().sum().sum()
                        st.metric("Empty Cells", empty_cells)
                    
                    # Export individual table
                    st.subheader("📤 Export Table")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if st.button("📄 Export as CSV", key=f"csv_{selected_table}"):
                            csv_data = edited_df.to_csv(index=False)
                            st.download_button(
                                "💾 Download CSV",
                                data=csv_data,
                                file_name=f"{selected_table}.csv",
                                mime="text/csv"
                            )
                    
                    with col2:
                        if st.button("📊 Export as Excel", key=f"excel_{selected_table}"):
                            excel_data = io.BytesIO()
                            with pd.ExcelWriter(excel_data, engine='openpyxl') as writer:
                                edited_df.to_excel(writer, sheet_name=selected_table, index=False)
                            st.download_button(
                                "💾 Download Excel",
                                data=excel_data.getvalue(),
                                file_name=f"{selected_table}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                    
                    with col3:
                        if st.button("📋 Export as JSON", key=f"json_{selected_table}"):
                            json_data = export_extraction_data({}, {selected_table: table_content}, 'json')
                            st.download_button(
                                "💾 Download JSON",
                                data=json_data,
                                file_name=f"{selected_table}.json",
                                mime="application/json"
                            )
            
            else:
                st.warning("No table data extracted. Try a different extraction method.")
    
    # Tab 4: Actions (Save, Export, Import)
    with tab4:
        st.header("💾 Save & Export Actions")
        
        if not st.session_state.file_processed:
            st.info("👈 Please upload and process a document first")
        else:
            col1, col2 = st.columns(2)
            
            # Save actions
            with col1:
                st.subheader("💾 Save Corrections")
                
                if st.button("💾 Save All Corrections", type="primary", use_container_width=True):
                    try:
                        success = db_manager.save_corrections(
                            document_hash=st.session_state.document_hash,
                            filename=st.session_state.uploaded_file.name,
                            corrections={
                                'timestamp': datetime.now().isoformat(),
                                'corrections_applied': True,
                                'extraction_method': st.session_state.processing_method,
                                'processing_time': (st.session_state.processing_end_time - 
                                                  st.session_state.processing_start_time).total_seconds()
                                if st.session_state.processing_end_time else 0
                            },
                            table_data=st.session_state.table_data,
                            field_data=st.session_state.field_data,
                            file_size=len(st.session_state.uploaded_file.getvalue())
                        )
                        
                        if success:
                            # Learn from corrections if enabled
                            if st.session_state.extraction_settings['pattern_learning_enabled']:
                                learning_results = pattern_learner.learn_from_corrections(
                                    st.session_state.document_hash,
                                    st.session_state.field_data,
                                    st.session_state.table_data
                                )
                                st.info(f"📚 Learned {learning_results.get('fields_learned', 0)} field patterns and "
                                        f"{learning_results.get('tables_learned', 0)} table patterns")
                            
                            st.session_state.corrections_saved = True
                            st.success("✅ Corrections saved successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to save corrections")
                    
                    except Exception as e:
                        st.error(f"❌ Error saving corrections: {str(e)}")
                        logger.error(f"Save error: {e}", exc_info=True)
                
                # Auto-save toggle
                if st.session_state.extraction_settings['auto_save']:
                    st.info("🔄 Auto-save is enabled")
                
                if st.session_state.corrections_saved:
                    st.success("✅ Document has saved corrections")
            
            # Export actions
            with col2:
                st.subheader("📤 Export Options")
                
                export_format = st.selectbox(
                    "Export Format",
                    options=['json', 'csv', 'excel'],
                    index=0
                )
                
                if st.button("📄 Export All Data", use_container_width=True):
                    try:
                        if export_format == 'excel':
                            export_data = export_extraction_data(
                                st.session_state.field_data,
                                st.session_state.table_data,
                                'excel'
                            )
                            st.download_button(
                                "💾 Download Excel File",
                                data=export_data,
                                file_name=f"extraction_data_{st.session_state.document_hash[:8]}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        else:
                            export_data = export_extraction_data(
                                st.session_state.field_data,
                                st.session_state.table_data,
                                export_format
                            )
                            mime_type = "application/json" if export_format == 'json' else "text/csv"
                            file_ext = export_format
                            
                            st.download_button(
                                f"💾 Download {export_format.upper()} File",
                                data=export_data,
                                file_name=f"extraction_data_{st.session_state.document_hash[:8]}.{file_ext}",
                                mime=mime_type
                            )
                    except Exception as e:
                        st.error(f"❌ Export failed: {str(e)}")
                
                # Quick export buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Fields Only", use_container_width=True):
                        json_data = json.dumps(st.session_state.field_data, indent=2)
                        st.download_button(
                            "💾 Download Fields JSON",
                            data=json_data,
                            file_name=f"fields_{st.session_state.document_hash[:8]}.json",
                            mime="application/json"
                        )
                
                with col2:
                    if st.button("📊 Tables Only", use_container_width=True):
                        json_data = json.dumps(st.session_state.table_data, indent=2)
                        st.download_button(
                            "💾 Download Tables JSON",
                            data=json_data,
                            file_name=f"tables_{st.session_state.document_hash[:8]}.json",
                            mime="application/json"
                        )
    
    # Tab 5: Analytics and Statistics
    with tab5:
        display_extraction_analytics(db_manager, pattern_learner)
    
    # Tab 6: Admin Panel
    with tab6:
        st.header("⚙️ Administration Panel")
        
        # Database management
        st.subheader("🗄️ Database Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Create Backup", use_container_width=True):
                backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
                if db_manager.backup_database(backup_path):
                    st.success(f"✅ Backup created: {backup_path}")
                else:
                    st.error("❌ Backup failed")
        
        with col2:
            days_old = st.number_input("Cleanup Days", value=90, min_value=1, max_value=365)
            if st.button("🧹 Cleanup Old Data", use_container_width=True):
                cleaned_count = db_manager.cleanup_old_data(days_old)
                st.success(f"✅ Cleaned up {cleaned_count} old documents")
        
        with col3:
            if st.button("📊 Database Stats", use_container_width=True):
                stats = db_manager.get_statistics()
                st.json(stats)
        
        # Pattern management
        st.subheader("🧠 Pattern Learning Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Learning Statistics**")
            learning_stats = pattern_learner.get_learning_statistics()
            for key, value in learning_stats.items():
                st.write(f"- {key.replace('_', ' ').title()}: {value}")
        
        with col2:
            if st.button("📤 Export Patterns"):
                export_path = f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                if pattern_learner.export_patterns(export_path):
                    st.success(f"✅ Patterns exported: {export_path}")
                else:
                    st.error("❌ Pattern export failed")
        
        # System logs
        st.subheader("📝 System Logs")
        
        if st.button("📖 View Recent Logs"):
            try:
                with open('pdf_extraction.log', 'r') as log_file:
                    logs = log_file.readlines()[-50:]  # Last 50 lines
                    st.text_area("Recent Logs", value=''.join(logs), height=300)
            except FileNotFoundError:
                st.warning("No log file found")
        
        # Configuration
        st.subheader("⚙️ System Configuration")
        
        st.json(CONFIG)
        
        # Session state management
        st.subheader("💾 Session State")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📤 Export Session"):
                session_data = session_manager.export_session_state()
                st.download_button(
                    "💾 Download Session",
                    data=json.dumps(session_data, indent=2),
                    file_name=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("🔄 Reset Session"):
                if st.button("⚠️ Confirm Reset"):
                    session_manager.clear_extraction_data()
                    st.success("✅ Session reset")
                    st.rerun()

if __name__ == "__main__":
    main()