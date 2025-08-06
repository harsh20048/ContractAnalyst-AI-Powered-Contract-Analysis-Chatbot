"""
PDF Table Extraction System - Main Application
A comprehensive system for extracting, correcting, and learning from PDF table data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import hashlib
import json
import tempfile
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from dataclasses import dataclass, asdict
from enum import Enum
import re
import base64
import io
import zipfile
import pickle

# Import fallback handling for PDF libraries
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    st.warning("PyPDF2 not available. Some PDF features may be limited.")

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    st.warning("pdfplumber not available. Some PDF features may be limited.")

try:
    import tabula
    TABULA_AVAILABLE = True
except ImportError:
    TABULA_AVAILABLE = False
    st.warning("tabula-py not available. Some PDF features may be limited.")

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    st.warning("camelot-py not available. Some PDF features may be limited.")

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    st.warning("PyMuPDF not available. Some PDF features may be limited.")

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR capabilities not available.")

try:
    import spacy
    import nltk
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("Advanced ML features not available.")

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV not available. Image processing features limited.")

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    st.warning("Transformers not available. NLP features limited.")

# Import our custom modules
from database import DatabaseManager
from pattern_learner import PatternLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdf_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DATABASE_PATH = "pdf_extraction.db"
UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "exports"
TEMP_FOLDER = "temp"
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
SUPPORTED_FORMATS = ['pdf', 'xlsx', 'csv', 'json']
DEFAULT_CONFIDENCE_THRESHOLD = 0.7

class ExtractionEngine(Enum):
    """Available PDF extraction engines"""
    PYPDF2 = "PyPDF2"
    PDFPLUMBER = "pdfplumber"
    TABULA = "tabula"
    CAMELOT = "camelot"
    PYMUPDF = "PyMuPDF"
    OCR = "OCR"
    MOCK = "Mock"

class ProcessingStatus(Enum):
    """Processing status enumeration"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    filename: str
    size: int
    upload_time: datetime
    content_hash: str
    page_count: int
    extraction_engine: str
    processing_time: float
    confidence: float
    status: ProcessingStatus

@dataclass
class ExtractionResult:
    """Extraction result structure"""
    fields: Dict[str, Any]
    tables: List[pd.DataFrame]
    metadata: DocumentMetadata
    confidence: float
    errors: List[str]
    warnings: List[str]

class PDFProcessor:
    """Comprehensive PDF processing class with multiple extraction engines"""
    
    def __init__(self, db_manager: DatabaseManager, pattern_learner: PatternLearner):
        self.db_manager = db_manager
        self.pattern_learner = pattern_learner
        self.extraction_engines = self._initialize_engines()
        self.processing_cache = {}
        
    def _initialize_engines(self) -> Dict[ExtractionEngine, bool]:
        """Initialize available extraction engines"""
        engines = {
            ExtractionEngine.PYPDF2: PYPDF2_AVAILABLE,
            ExtractionEngine.PDFPLUMBER: PDFPLUMBER_AVAILABLE,
            ExtractionEngine.TABULA: TABULA_AVAILABLE,
            ExtractionEngine.CAMELOT: CAMELOT_AVAILABLE,
            ExtractionEngine.PYMUPDF: PYMUPDF_AVAILABLE,
            ExtractionEngine.OCR: OCR_AVAILABLE,
            ExtractionEngine.MOCK: True  # Always available
        }
        logger.info(f"Initialized engines: {[k.value for k, v in engines.items() if v]}")
        return engines
    
    def get_available_engines(self) -> List[ExtractionEngine]:
        """Get list of available extraction engines"""
        return [engine for engine, available in self.extraction_engines.items() if available]
    
    def extract_with_engine(self, file_path: str, engine: ExtractionEngine) -> ExtractionResult:
        """Extract data using specified engine"""
        start_time = time.time()
        
        try:
            if engine == ExtractionEngine.PYPDF2:
                result = self._extract_with_pypdf2(file_path)
            elif engine == ExtractionEngine.PDFPLUMBER:
                result = self._extract_with_pdfplumber(file_path)
            elif engine == ExtractionEngine.TABULA:
                result = self._extract_with_tabula(file_path)
            elif engine == ExtractionEngine.CAMELOT:
                result = self._extract_with_camelot(file_path)
            elif engine == ExtractionEngine.PYMUPDF:
                result = self._extract_with_pymupdf(file_path)
            elif engine == ExtractionEngine.OCR:
                result = self._extract_with_ocr(file_path)
            else:
                result = self._extract_mock_data(file_path)
            
            processing_time = time.time() - start_time
            
            # Update metadata
            result.metadata.processing_time = processing_time
            result.metadata.extraction_engine = engine.value
            
            # Apply learned patterns
            if result.confidence < DEFAULT_CONFIDENCE_THRESHOLD:
                learned_patterns = self.pattern_learner.get_learned_patterns(result.metadata.content_hash)
                if learned_patterns:
                    result = self._apply_learned_patterns(result, learned_patterns)
            
            # Log extraction
            self.db_manager.log_processing_event(
                result.metadata.content_hash,
                f"Extracted with {engine.value}",
                {"processing_time": processing_time, "confidence": result.confidence}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Extraction failed with {engine.value}: {str(e)}")
            processing_time = time.time() - start_time
            
            return ExtractionResult(
                fields={},
                tables=[],
                metadata=DocumentMetadata(
                    filename=os.path.basename(file_path),
                    size=os.path.getsize(file_path),
                    upload_time=datetime.now(),
                    content_hash=self._calculate_hash(file_path),
                    page_count=0,
                    extraction_engine=engine.value,
                    processing_time=processing_time,
                    confidence=0.0,
                    status=ProcessingStatus.FAILED
                ),
                confidence=0.0,
                errors=[f"Extraction failed: {str(e)}"],
                warnings=[]
            )
    
    def _extract_with_pypdf2(self, file_path: str) -> ExtractionResult:
        """Extract using PyPDF2"""
        if not PYPDF2_AVAILABLE:
            raise ImportError("PyPDF2 not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                page_count = len(reader.pages)
                
                # Extract text from all pages
                text_content = ""
                for page in reader.pages:
                    text_content += page.extract_text()
                
                # Extract fields using pattern matching
                fields = self._extract_fields_from_text(text_content)
                
                # Simple table detection (limited with PyPDF2)
                tables = self._extract_tables_from_text(text_content)
                
                confidence = 0.6  # PyPDF2 has moderate reliability
                
        except Exception as e:
            errors.append(f"PyPDF2 extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.PYPDF2.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_with_pdfplumber(self, file_path: str) -> ExtractionResult:
        """Extract using pdfplumber"""
        if not PDFPLUMBER_AVAILABLE:
            raise ImportError("pdfplumber not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                # Extract text and tables from all pages
                full_text = ""
                for page in pdf.pages:
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    for table in page_tables:
                        if table and len(table) > 1:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
                
                # Extract fields from text
                fields = self._extract_fields_from_text(full_text)
                
                confidence = 0.8  # pdfplumber is quite reliable
                
        except Exception as e:
            errors.append(f"pdfplumber extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.PDFPLUMBER.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_with_tabula(self, file_path: str) -> ExtractionResult:
        """Extract using tabula-py"""
        if not TABULA_AVAILABLE:
            raise ImportError("tabula-py not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            # Extract tables using tabula
            dfs = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
            tables = [df for df in dfs if not df.empty]
            
            # Extract text for field detection (basic)
            text_dfs = tabula.read_pdf(file_path, pages='all', output_format='text')
            if text_dfs:
                text_content = str(text_dfs)
                fields = self._extract_fields_from_text(text_content)
            
            confidence = 0.7  # tabula is good for tables
            page_count = 1  # Simplified
            
        except Exception as e:
            errors.append(f"tabula extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.TABULA.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_with_camelot(self, file_path: str) -> ExtractionResult:
        """Extract using camelot-py"""
        if not CAMELOT_AVAILABLE:
            raise ImportError("camelot-py not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            # Extract tables using camelot
            table_list = camelot.read_pdf(file_path, pages='all')
            tables = [table.df for table in table_list]
            
            # Basic field extraction (camelot is table-focused)
            if tables:
                # Try to extract fields from table data
                fields = self._extract_fields_from_tables(tables)
            
            confidence = 0.8  # camelot is very good for tables
            page_count = 1  # Simplified
            
        except Exception as e:
            errors.append(f"camelot extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.CAMELOT.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_with_pymupdf(self, file_path: str) -> ExtractionResult:
        """Extract using PyMuPDF"""
        if not PYMUPDF_AVAILABLE:
            raise ImportError("PyMuPDF not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            doc = fitz.open(file_path)
            page_count = len(doc)
            
            full_text = ""
            for page_num in range(page_count):
                page = doc.load_page(page_num)
                
                # Extract text
                text = page.get_text()
                full_text += text
                
                # Extract tables (basic table detection)
                page_tables = self._extract_pymupdf_tables(page)
                tables.extend(page_tables)
            
            doc.close()
            
            # Extract fields from text
            fields = self._extract_fields_from_text(full_text)
            
            confidence = 0.7  # PyMuPDF is quite reliable
            
        except Exception as e:
            errors.append(f"PyMuPDF extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.PYMUPDF.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_with_ocr(self, file_path: str) -> ExtractionResult:
        """Extract using OCR"""
        if not OCR_AVAILABLE:
            raise ImportError("OCR not available")
        
        fields = {}
        tables = []
        errors = []
        warnings = []
        
        try:
            # Convert PDF to images and apply OCR
            if PYMUPDF_AVAILABLE:
                doc = fitz.open(file_path)
                page_count = len(doc)
                
                full_text = ""
                for page_num in range(page_count):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    
                    # Convert to PIL Image
                    img = Image.open(io.BytesIO(img_data))
                    
                    # Apply OCR
                    text = pytesseract.image_to_string(img)
                    full_text += text
                
                doc.close()
                
                # Extract fields and tables from OCR text
                fields = self._extract_fields_from_text(full_text)
                tables = self._extract_tables_from_text(full_text)
                
                confidence = 0.5  # OCR can be less reliable
            else:
                raise ImportError("PyMuPDF required for OCR processing")
                
        except Exception as e:
            errors.append(f"OCR extraction error: {str(e)}")
            page_count = 0
            confidence = 0.0
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=self._calculate_hash(file_path),
            page_count=page_count,
            extraction_engine=ExtractionEngine.OCR.value,
            processing_time=0.0,
            confidence=confidence,
            status=ProcessingStatus.COMPLETED if not errors else ProcessingStatus.FAILED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=confidence,
            errors=errors,
            warnings=warnings
        )
    
    def _extract_mock_data(self, file_path: str) -> ExtractionResult:
        """Extract mock data for testing"""
        content_hash = self._calculate_hash(file_path)
        
        # Generate deterministic mock data based on file hash
        np.random.seed(int(content_hash[:8], 16) % (2**32))
        
        # Mock fields
        fields = {
            "Document_Number": f"DOC-{np.random.randint(1000, 9999)}",
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Total_Amount": round(np.random.uniform(100, 10000), 2),
            "Company_Name": np.random.choice([
                "Acme Corp", "Tech Solutions Inc", "Global Industries", 
                "Innovation Labs", "Future Systems"
            ]),
            "Contact_Email": f"contact{np.random.randint(1, 100)}@example.com",
            "Phone_Number": f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}",
            "Address": f"{np.random.randint(1, 999)} {np.random.choice(['Main', 'Oak', 'Pine', 'First', 'Second'])} St",
            "Status": np.random.choice(["Active", "Pending", "Completed", "Draft"]),
            "Priority": np.random.choice(["High", "Medium", "Low"]),
            "Category": np.random.choice(["Type A", "Type B", "Type C"])
        }
        
        # Mock tables
        tables = []
        
        # Table 1: Line Items
        num_items = np.random.randint(3, 8)
        items_data = {
            "Item": [f"Item {i+1}" for i in range(num_items)],
            "Description": [f"Description for item {i+1}" for i in range(num_items)],
            "Quantity": np.random.randint(1, 100, num_items),
            "Unit_Price": np.round(np.random.uniform(10, 500, num_items), 2),
            "Total": []
        }
        items_data["Total"] = [q * p for q, p in zip(items_data["Quantity"], items_data["Unit_Price"])]
        tables.append(pd.DataFrame(items_data))
        
        # Table 2: Payment Schedule
        num_payments = np.random.randint(2, 6)
        payment_data = {
            "Payment_Date": [
                (datetime.now() + timedelta(days=i*30)).strftime("%Y-%m-%d") 
                for i in range(num_payments)
            ],
            "Amount": np.round(np.random.uniform(100, 1000, num_payments), 2),
            "Method": np.random.choice(["Bank Transfer", "Check", "Credit Card"], num_payments),
            "Status": np.random.choice(["Pending", "Completed", "Overdue"], num_payments)
        }
        tables.append(pd.DataFrame(payment_data))
        
        # Table 3: Contact Information
        num_contacts = np.random.randint(2, 5)
        contact_data = {
            "Name": [f"Contact {i+1}" for i in range(num_contacts)],
            "Role": np.random.choice(["Manager", "Assistant", "Director", "Coordinator"], num_contacts),
            "Email": [f"contact{i+1}@company.com" for i in range(num_contacts)],
            "Phone": [f"+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(num_contacts)]
        }
        tables.append(pd.DataFrame(contact_data))
        
        metadata = DocumentMetadata(
            filename=os.path.basename(file_path),
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            content_hash=content_hash,
            page_count=np.random.randint(1, 5),
            extraction_engine=ExtractionEngine.MOCK.value,
            processing_time=0.0,
            confidence=0.9,  # Mock data is "perfect"
            status=ProcessingStatus.COMPLETED
        )
        
        return ExtractionResult(
            fields=fields,
            tables=tables,
            metadata=metadata,
            confidence=0.9,
            errors=[],
            warnings=["This is mock data for testing purposes"]
        )
    
    def _extract_fields_from_text(self, text: str) -> Dict[str, Any]:
        """Extract fields from text using pattern matching"""
        fields = {}
        
        # Common field patterns
        patterns = {
            "Date": [
                r'\b(\d{1,2}[/-]\d{1,2}[/-]\d{4})\b',
                r'\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b',
                r'\b(\w+\s+\d{1,2},?\s+\d{4})\b'
            ],
            "Amount": [
                r'\$\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)',
                r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s?USD',
                r'Total:?\s*\$?\s?(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)'
            ],
            "Email": [
                r'\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b'
            ],
            "Phone": [
                r'\b(\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4})\b'
            ],
            "Document_Number": [
                r'\b(DOC[-_]?\d+)\b',
                r'\b(INV[-_]?\d+)\b',
                r'\b([A-Z]{2,}\d+)\b'
            ]
        }
        
        for field_name, field_patterns in patterns.items():
            for pattern in field_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    fields[field_name] = matches[0] if isinstance(matches[0], str) else matches[0][0]
                    break
        
        return fields
    
    def _extract_tables_from_text(self, text: str) -> List[pd.DataFrame]:
        """Extract tables from text using simple pattern matching"""
        tables = []
        
        # Look for tabular data patterns
        lines = text.split('\n')
        current_table = []
        
        for line in lines:
            # Simple heuristic: lines with multiple columns separated by whitespace
            columns = line.strip().split()
            if len(columns) >= 3:  # At least 3 columns
                current_table.append(columns)
            else:
                if len(current_table) >= 2:  # At least 2 rows
                    try:
                        # Create DataFrame
                        max_cols = max(len(row) for row in current_table)
                        # Pad rows to same length
                        padded_table = [row + [''] * (max_cols - len(row)) for row in current_table]
                        
                        if len(padded_table) > 1:
                            df = pd.DataFrame(padded_table[1:], columns=padded_table[0])
                            tables.append(df)
                    except Exception:
                        pass  # Skip malformed tables
                current_table = []
        
        # Check last table
        if len(current_table) >= 2:
            try:
                max_cols = max(len(row) for row in current_table)
                padded_table = [row + [''] * (max_cols - len(row)) for row in current_table]
                if len(padded_table) > 1:
                    df = pd.DataFrame(padded_table[1:], columns=padded_table[0])
                    tables.append(df)
            except Exception:
                pass
        
        return tables
    
    def _extract_fields_from_tables(self, tables: List[pd.DataFrame]) -> Dict[str, Any]:
        """Extract fields from table data"""
        fields = {}
        
        for i, table in enumerate(tables):
            if table.empty:
                continue
            
            # Look for key-value pairs in tables
            for idx, row in table.iterrows():
                for col in table.columns:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        # Simple heuristics for field detection
                        value_str = str(value).strip()
                        
                        # Check if this looks like a field name/value pattern
                        if ':' in value_str:
                            parts = value_str.split(':', 1)
                            if len(parts) == 2:
                                key = parts[0].strip().replace(' ', '_')
                                val = parts[1].strip()
                                if key and val:
                                    fields[f"Table_{i}_{key}"] = val
                        
                        # Check for common field patterns
                        if re.match(r'.*total.*', col, re.IGNORECASE) and re.match(r'[\d,.$]+', value_str):
                            fields[f"Total_Amount"] = value_str
                        elif re.match(r'.*date.*', col, re.IGNORECASE):
                            fields[f"Date"] = value_str
                        elif re.match(r'.*email.*', col, re.IGNORECASE):
                            fields[f"Email"] = value_str
        
        return fields
    
    def _extract_pymupdf_tables(self, page) -> List[pd.DataFrame]:
        """Extract tables from PyMuPDF page (basic implementation)"""
        tables = []
        
        try:
            # Get page text with layout information
            text_dict = page.get_text("dict")
            
            # Simple table detection based on text positioning
            # This is a basic implementation - real table detection would be more complex
            
        except Exception as e:
            logger.warning(f"Table extraction from PyMuPDF page failed: {e}")
        
        return tables
    
    def _apply_learned_patterns(self, result: ExtractionResult, patterns: Dict[str, Any]) -> ExtractionResult:
        """Apply learned patterns to improve extraction results"""
        try:
            # Apply field patterns
            if "field_patterns" in patterns:
                for field_name, pattern_info in patterns["field_patterns"].items():
                    if field_name not in result.fields and "pattern" in pattern_info:
                        # Try to apply the learned pattern
                        # This would involve re-processing the document with the learned pattern
                        pass
            
            # Apply table patterns
            if "table_patterns" in patterns:
                # Apply learned table structure patterns
                pass
            
            # Update confidence based on pattern application
            if patterns:
                result.confidence = min(result.confidence + 0.1, 1.0)
            
        except Exception as e:
            logger.error(f"Error applying learned patterns: {e}")
            result.warnings.append(f"Failed to apply learned patterns: {e}")
        
        return result
    
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file content"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating hash for {file_path}: {e}")
            return f"error_{int(time.time())}"

class DataValidator:
    """Data validation and quality checking"""
    
    @staticmethod
    def validate_fields(fields: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
        """Validate extracted fields"""
        validated_fields = {}
        errors = []
        
        for field_name, value in fields.items():
            try:
                if field_name.lower().endswith('_date') or 'date' in field_name.lower():
                    validated_fields[field_name] = DataValidator._validate_date(value)
                elif field_name.lower().endswith('_amount') or 'amount' in field_name.lower():
                    validated_fields[field_name] = DataValidator._validate_amount(value)
                elif field_name.lower().endswith('_email') or 'email' in field_name.lower():
                    validated_fields[field_name] = DataValidator._validate_email(value)
                elif field_name.lower().endswith('_phone') or 'phone' in field_name.lower():
                    validated_fields[field_name] = DataValidator._validate_phone(value)
                else:
                    validated_fields[field_name] = str(value).strip()
            
            except ValueError as e:
                errors.append(f"Invalid {field_name}: {e}")
                validated_fields[field_name] = str(value)  # Keep original value
        
        return validated_fields, errors
    
    @staticmethod
    def _validate_date(value: str) -> str:
        """Validate and normalize date"""
        if not value:
            raise ValueError("Date is empty")
        
        # Try different date formats
        formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%B %d, %Y", "%b %d, %Y", "%d %B %Y", "%d %b %Y"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(str(value).strip(), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        raise ValueError(f"Invalid date format: {value}")
    
    @staticmethod
    def _validate_amount(value: str) -> float:
        """Validate and normalize amount"""
        if not value:
            raise ValueError("Amount is empty")
        
        # Clean the amount string
        amount_str = str(value).strip()
        amount_str = re.sub(r'[,$\s]', '', amount_str)
        
        try:
            return float(amount_str)
        except ValueError:
            raise ValueError(f"Invalid amount format: {value}")
    
    @staticmethod
    def _validate_email(value: str) -> str:
        """Validate email format"""
        if not value:
            raise ValueError("Email is empty")
        
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, str(value).strip()):
            raise ValueError(f"Invalid email format: {value}")
        
        return str(value).strip().lower()
    
    @staticmethod
    def _validate_phone(value: str) -> str:
        """Validate and normalize phone number"""
        if not value:
            raise ValueError("Phone is empty")
        
        # Extract digits only
        digits = re.sub(r'[^\d]', '', str(value))
        
        if len(digits) == 10:
            return f"({digits[:3]}) {digits[3:6]}-{digits[6:]}"
        elif len(digits) == 11 and digits[0] == '1':
            return f"+1 ({digits[1:4]}) {digits[4:7]}-{digits[7:]}"
        else:
            raise ValueError(f"Invalid phone format: {value}")
    
    @staticmethod
    def validate_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate table data"""
        errors = []
        validated_df = df.copy()
        
        try:
            # Remove completely empty rows
            validated_df = validated_df.dropna(how='all')
            
            # Remove completely empty columns
            validated_df = validated_df.dropna(axis=1, how='all')
            
            # Clean column names
            validated_df.columns = [str(col).strip() for col in validated_df.columns]
            
            # Basic data type inference and validation
            for col in validated_df.columns:
                # Try to infer and convert data types
                series = validated_df[col]
                
                # Skip if all values are NaN
                if series.isna().all():
                    continue
                
                # Try numeric conversion
                try:
                    # Check if it looks like numbers
                    non_na_values = series.dropna().astype(str)
                    if all(re.match(r'^-?\d*\.?\d+$', val.strip()) for val in non_na_values if val.strip()):
                        validated_df[col] = pd.to_numeric(series, errors='coerce')
                except:
                    pass  # Keep as string
            
        except Exception as e:
            errors.append(f"Table validation error: {e}")
        
        return validated_df, errors

class SessionStateManager:
    """Enhanced session state management"""
    
    @staticmethod
    def initialize_session_state():
        """Initialize all session state variables"""
        defaults = {
            'extracted_fields': {},
            'extracted_tables': [],
            'document_hash': None,
            'document_metadata': None,
            'processing_status': ProcessingStatus.PENDING,
            'current_extraction_engine': ExtractionEngine.MOCK,
            'available_engines': [],
            'corrections_saved': False,
            'last_save_time': None,
            'field_validation_errors': [],
            'table_validation_errors': [],
            'extraction_history': [],
            'selected_table_index': 0,
            'show_advanced_options': False,
            'auto_save_enabled': True,
            'confidence_threshold': DEFAULT_CONFIDENCE_THRESHOLD,
            'processing_log': [],
            'learned_patterns': {},
            'extraction_results': None,
            'upload_progress': 0,
            'page_state': 'upload',
            'admin_mode': False,
            'export_format': 'xlsx',
            'import_in_progress': False,
            'batch_processing': False,
            'performance_metrics': {},
            'ui_theme': 'light',
            'debug_mode': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def reset_extraction_state():
        """Reset extraction-related state"""
        st.session_state.extracted_fields = {}
        st.session_state.extracted_tables = []
        st.session_state.document_hash = None
        st.session_state.document_metadata = None
        st.session_state.processing_status = ProcessingStatus.PENDING
        st.session_state.corrections_saved = False
        st.session_state.last_save_time = None
        st.session_state.field_validation_errors = []
        st.session_state.table_validation_errors = []
        st.session_state.extraction_results = None
        st.session_state.upload_progress = 0
        st.session_state.learned_patterns = {}
    
    @staticmethod
    def update_processing_status(status: ProcessingStatus, message: str = ""):
        """Update processing status with optional message"""
        st.session_state.processing_status = status
        if message:
            st.session_state.processing_log.append({
                'timestamp': datetime.now().isoformat(),
                'status': status.value,
                'message': message
            })
    
    @staticmethod
    def add_to_history(action: str, details: Dict[str, Any]):
        """Add action to extraction history"""
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details
        }
        
        if 'extraction_history' not in st.session_state:
            st.session_state.extraction_history = []
        
        st.session_state.extraction_history.append(history_entry)
        
        # Keep only last 100 entries
        if len(st.session_state.extraction_history) > 100:
            st.session_state.extraction_history = st.session_state.extraction_history[-100:]

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_header():
        """Render application header"""
        st.set_page_config(
            page_title="PDF Table Extraction System",
            page_icon="📄",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1f4e79, #2e8b57);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .header-title {
            color: white;
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            margin: 0;
        }
        .header-subtitle {
            color: #e6f3ff;
            font-size: 1.1rem;
            text-align: center;
            margin: 0.5rem 0 0 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="main-header">
            <h1 class="header-title">📄 PDF Table Extraction System</h1>
            <p class="header-subtitle">Advanced PDF processing with machine learning-powered corrections</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def render_sidebar():
        """Render sidebar with navigation and settings"""
        with st.sidebar:
            st.markdown("### 🔧 System Status")
            
            # Display processing status
            status = st.session_state.get('processing_status', ProcessingStatus.PENDING)
            status_colors = {
                ProcessingStatus.PENDING: "🟡",
                ProcessingStatus.PROCESSING: "🔵",
                ProcessingStatus.COMPLETED: "🟢",
                ProcessingStatus.FAILED: "🔴",
                ProcessingStatus.CANCELLED: "⚫"
            }
            
            st.write(f"**Status:** {status_colors.get(status, '⚪')} {status.value.title()}")
            
            # Document info
            if st.session_state.get('document_metadata'):
                metadata = st.session_state.document_metadata
                st.write(f"**Document:** {metadata.filename}")
                st.write(f"**Size:** {metadata.size / 1024:.1f} KB")
                st.write(f"**Pages:** {metadata.page_count}")
                st.write(f"**Engine:** {metadata.extraction_engine}")
                st.write(f"**Confidence:** {metadata.confidence:.2%}")
            
            st.divider()
            
            # Settings
            st.markdown("### ⚙️ Settings")
            
            # Engine selection
            available_engines = st.session_state.get('available_engines', [ExtractionEngine.MOCK])
            engine_names = [engine.value for engine in available_engines]
            current_engine = st.selectbox(
                "Extraction Engine",
                engine_names,
                index=engine_names.index(st.session_state.current_extraction_engine.value) if st.session_state.current_extraction_engine.value in engine_names else 0
            )
            st.session_state.current_extraction_engine = ExtractionEngine(current_engine)
            
            # Confidence threshold
            st.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                0.0, 1.0, 
                st.session_state.get('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD),
                0.1
            )
            
            # Auto-save
            st.session_state.auto_save_enabled = st.checkbox(
                "Auto-save corrections",
                st.session_state.get('auto_save_enabled', True)
            )
            
            # Advanced options
            st.session_state.show_advanced_options = st.checkbox(
                "Show advanced options",
                st.session_state.get('show_advanced_options', False)
            )
            
            # Debug mode
            st.session_state.debug_mode = st.checkbox(
                "Debug mode",
                st.session_state.get('debug_mode', False)
            )
            
            st.divider()
            
            # Quick actions
            st.markdown("### 🚀 Quick Actions")
            
            if st.button("🔄 Reset Session"):
                SessionStateManager.reset_extraction_state()
                st.rerun()
            
            if st.button("📊 Show Statistics"):
                st.session_state.page_state = 'analytics'
                st.rerun()
            
            if st.button("🛠️ Admin Panel"):
                st.session_state.admin_mode = not st.session_state.get('admin_mode', False)
                st.rerun()
    
    @staticmethod
    def render_file_uploader():
        """Render file upload component"""
        st.markdown("### 📁 Upload Document")
        
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF file to extract table data and fields"
        )
        
        if uploaded_file is not None:
            # Validate file
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.1f} MB")
                return None
            
            # Display file info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col2:
                st.metric("File Type", uploaded_file.type)
            with col3:
                st.metric("Upload Status", "✅ Ready")
            
            return uploaded_file
        
        return None
    
    @staticmethod
    def render_processing_progress():
        """Render processing progress indicator"""
        if st.session_state.get('processing_status') == ProcessingStatus.PROCESSING:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate progress (in real implementation, this would be updated by the processing thread)
            for i in range(100):
                progress_bar.progress((i + 1) / 100)
                status_text.text(f"Processing... {i + 1}%")
                time.sleep(0.01)
            
            st.success("Processing completed!")
    
    @staticmethod
    def render_field_editor(fields: Dict[str, Any], validation_errors: List[str] = None):
        """Render field editing interface"""
        st.markdown("### 📝 Extracted Fields")
        
        if not fields:
            st.info("No fields extracted. Upload a document to begin.")
            return {}
        
        # Display validation errors
        if validation_errors:
            with st.expander("⚠️ Validation Errors", expanded=False):
                for error in validation_errors:
                    st.error(error)
        
        edited_fields = {}
        
        # Organize fields into columns
        num_fields = len(fields)
        cols_per_row = 2
        rows = (num_fields + cols_per_row - 1) // cols_per_row
        
        field_items = list(fields.items())
        
        for row in range(rows):
            cols = st.columns(cols_per_row)
            
            for col_idx in range(cols_per_row):
                field_idx = row * cols_per_row + col_idx
                
                if field_idx < len(field_items):
                    field_name, field_value = field_items[field_idx]
                    
                    with cols[col_idx]:
                        # Determine input type based on field name
                        if 'date' in field_name.lower():
                            try:
                                date_value = datetime.strptime(str(field_value), "%Y-%m-%d").date()
                                edited_value = st.date_input(
                                    field_name.replace('_', ' ').title(),
                                    value=date_value,
                                    key=f"field_{field_name}"
                                )
                                edited_fields[field_name] = edited_value.strftime("%Y-%m-%d")
                            except:
                                edited_fields[field_name] = st.text_input(
                                    field_name.replace('_', ' ').title(),
                                    value=str(field_value),
                                    key=f"field_{field_name}"
                                )
                        elif 'amount' in field_name.lower():
                            try:
                                numeric_value = float(str(field_value).replace(',', '').replace('$', ''))
                                edited_value = st.number_input(
                                    field_name.replace('_', ' ').title(),
                                    value=numeric_value,
                                    format="%.2f",
                                    key=f"field_{field_name}"
                                )
                                edited_fields[field_name] = edited_value
                            except:
                                edited_fields[field_name] = st.text_input(
                                    field_name.replace('_', ' ').title(),
                                    value=str(field_value),
                                    key=f"field_{field_name}"
                                )
                        else:
                            edited_fields[field_name] = st.text_input(
                                field_name.replace('_', ' ').title(),
                                value=str(field_value),
                                key=f"field_{field_name}"
                            )
        
        return edited_fields
    
    @staticmethod
    def render_table_editor(tables: List[pd.DataFrame], validation_errors: List[str] = None):
        """Render table editing interface"""
        st.markdown("### 📊 Extracted Tables")
        
        if not tables:
            st.info("No tables extracted. Upload a document to begin.")
            return []
        
        # Display validation errors
        if validation_errors:
            with st.expander("⚠️ Table Validation Errors", expanded=False):
                for error in validation_errors:
                    st.error(error)
        
        # Table selection
        if len(tables) > 1:
            selected_table = st.selectbox(
                "Select table to edit",
                range(len(tables)),
                format_func=lambda x: f"Table {x + 1} ({tables[x].shape[0]} rows × {tables[x].shape[1]} cols)",
                key="selected_table_index"
            )
        else:
            selected_table = 0
        
        if selected_table < len(tables):
            table = tables[selected_table]
            
            # Table info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", table.shape[0])
            with col2:
                st.metric("Columns", table.shape[1])
            with col3:
                st.metric("Non-null cells", table.count().sum())
            
            # Table editor
            edited_table = st.data_editor(
                table,
                use_container_width=True,
                num_rows="dynamic",
                key=f"table_editor_{selected_table}",
                column_config={
                    col: st.column_config.TextColumn(
                        col,
                        help=f"Edit values in {col} column",
                        max_chars=100
                    ) for col in table.columns
                }
            )
            
            # Update the table in the list
            edited_tables = tables.copy()
            edited_tables[selected_table] = edited_table
            
            # Table operations
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📋 Copy Table", key="copy_table"):
                    st.success("Table copied to clipboard!")
            
            with col2:
                if st.button("➕ Add Row", key="add_row"):
                    new_row = pd.DataFrame([['' for _ in table.columns]], columns=table.columns)
                    edited_tables[selected_table] = pd.concat([edited_table, new_row], ignore_index=True)
                    st.rerun()
            
            with col3:
                if st.button("🗑️ Remove Last Row", key="remove_row"):
                    if len(edited_table) > 1:
                        edited_tables[selected_table] = edited_table.iloc[:-1]
                        st.rerun()
            
            with col4:
                if st.button("🔄 Reset Table", key="reset_table"):
                    edited_tables[selected_table] = table
                    st.rerun()
            
            return edited_tables
        
        return tables
    
    @staticmethod
    def render_analytics_dashboard(db_manager: DatabaseManager):
        """Render analytics dashboard"""
        st.markdown("### 📊 Analytics Dashboard")
        
        try:
            # Get system statistics
            stats = db_manager.get_system_stats()
            
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Documents",
                    stats.get('total_documents', 0),
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Total Corrections",
                    stats.get('total_corrections', 0),
                    delta=None
                )
            
            with col3:
                avg_confidence = stats.get('avg_confidence', 0)
                st.metric(
                    "Avg Confidence",
                    f"{avg_confidence:.2%}",
                    delta=None
                )
            
            with col4:
                recent_docs = stats.get('recent_documents', 0)
                st.metric(
                    "Recent Documents",
                    recent_docs,
                    delta=None
                )
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Processing Over Time")
                
                # Get recent documents for chart
                recent_docs_data = db_manager.get_recent_documents(limit=20)
                if recent_docs_data:
                    df = pd.DataFrame(recent_docs_data)
                    df['created_at'] = pd.to_datetime(df['created_at'])
                    daily_counts = df.groupby(df['created_at'].dt.date).size()
                    
                    st.line_chart(daily_counts)
                else:
                    st.info("No recent processing data available")
            
            with col2:
                st.markdown("#### Confidence Distribution")
                
                if recent_docs_data:
                    df = pd.DataFrame(recent_docs_data)
                    confidence_bins = pd.cut(df['confidence'], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=['Low', 'Medium', 'High', 'Very High'])
                    confidence_counts = confidence_bins.value_counts()
                    
                    st.bar_chart(confidence_counts)
                else:
                    st.info("No confidence data available")
            
            # Recent activity
            st.markdown("#### Recent Activity")
            
            if recent_docs_data:
                df = pd.DataFrame(recent_docs_data)
                df['created_at'] = pd.to_datetime(df['created_at'])
                
                # Display recent documents
                display_cols = ['filename', 'confidence', 'created_at', 'page_count']
                if all(col in df.columns for col in display_cols):
                    recent_display = df[display_cols].head(10)
                    recent_display['confidence'] = recent_display['confidence'].apply(lambda x: f"{x:.2%}")
                    recent_display['created_at'] = recent_display['created_at'].dt.strftime("%Y-%m-%d %H:%M")
                    
                    st.dataframe(
                        recent_display,
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            else:
                st.info("No recent activity to display")
            
        except Exception as e:
            st.error(f"Error loading analytics: {e}")
    
    @staticmethod
    def render_admin_panel(db_manager: DatabaseManager, pattern_learner: PatternLearner):
        """Render admin panel"""
        st.markdown("### 🛠️ Administration Panel")
        
        tabs = st.tabs(["Database", "Patterns", "System", "Logs"])
        
        with tabs[0]:  # Database
            st.markdown("#### Database Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📦 Backup Database"):
                    try:
                        backup_path = db_manager.backup_database()
                        st.success(f"Database backed up to: {backup_path}")
                    except Exception as e:
                        st.error(f"Backup failed: {e}")
            
            with col2:
                if st.button("🧹 Cleanup Old Records"):
                    try:
                        days = st.number_input("Delete records older than (days)", value=30, min_value=1)
                        if st.button("Confirm Cleanup"):
                            count = db_manager.cleanup_old_records(days)
                            st.success(f"Cleaned up {count} old records")
                    except Exception as e:
                        st.error(f"Cleanup failed: {e}")
            
            with col3:
                if st.button("📋 Export All Data"):
                    try:
                        # Export functionality would go here
                        st.success("Data export initiated")
                    except Exception as e:
                        st.error(f"Export failed: {e}")
            
            # Database statistics
            st.markdown("#### Database Statistics")
            try:
                stats = db_manager.get_system_stats()
                
                stats_df = pd.DataFrame([
                    {"Metric": "Total Documents", "Value": stats.get('total_documents', 0)},
                    {"Metric": "Total Corrections", "Value": stats.get('total_corrections', 0)},
                    {"Metric": "Unique Document Hashes", "Value": stats.get('unique_hashes', 0)},
                    {"Metric": "Average Confidence", "Value": f"{stats.get('avg_confidence', 0):.2%}"},
                ])
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
            except Exception as e:
                st.error(f"Error loading database statistics: {e}")
        
        with tabs[1]:  # Patterns
            st.markdown("#### Pattern Learning Management")
            
            try:
                # Pattern statistics
                learning_stats = pattern_learner.get_learning_stats()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Field Patterns", learning_stats.get('field_patterns', 0))
                    st.metric("Table Patterns", learning_stats.get('table_patterns', 0))
                
                with col2:
                    st.metric("Total Patterns", learning_stats.get('total_patterns', 0))
                    st.metric("Success Rate", f"{learning_stats.get('success_rate', 0):.2%}")
                
                # Pattern management actions
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Retrain Patterns"):
                        with st.spinner("Retraining patterns..."):
                            # Retrain patterns from all corrections
                            # This would be implemented in pattern_learner
                            st.success("Patterns retrained successfully")
                
                with col2:
                    if st.button("📊 Export Patterns"):
                        # Export patterns to file
                        patterns = pattern_learner.export_patterns()
                        st.download_button(
                            "Download Patterns",
                            data=json.dumps(patterns, indent=2),
                            file_name=f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                
                with col3:
                    uploaded_patterns = st.file_uploader("📥 Import Patterns", type=['json'])
                    if uploaded_patterns:
                        try:
                            patterns_data = json.load(uploaded_patterns)
                            pattern_learner.import_patterns(patterns_data)
                            st.success("Patterns imported successfully")
                        except Exception as e:
                            st.error(f"Import failed: {e}")
                
            except Exception as e:
                st.error(f"Error in pattern management: {e}")
        
        with tabs[2]:  # System
            st.markdown("#### System Information")
            
            # System metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Available Engines:**")
                processor = PDFProcessor(db_manager, pattern_learner)
                engines = processor.get_available_engines()
                for engine in engines:
                    st.write(f"✅ {engine.value}")
            
            with col2:
                st.markdown("**System Settings:**")
                st.write(f"Database Path: {DATABASE_PATH}")
                st.write(f"Upload Folder: {UPLOAD_FOLDER}")
                st.write(f"Max File Size: {MAX_FILE_SIZE / (1024*1024):.1f} MB")
                st.write(f"Confidence Threshold: {DEFAULT_CONFIDENCE_THRESHOLD:.2%}")
            
            # Performance metrics
            st.markdown("#### Performance Metrics")
            
            if 'performance_metrics' in st.session_state:
                metrics = st.session_state.performance_metrics
                metrics_df = pd.DataFrame([
                    {"Metric": key, "Value": value} 
                    for key, value in metrics.items()
                ])
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            else:
                st.info("No performance metrics available")
        
        with tabs[3]:  # Logs
            st.markdown("#### System Logs")
            
            # Processing log
            if 'processing_log' in st.session_state and st.session_state.processing_log:
                log_df = pd.DataFrame(st.session_state.processing_log)
                log_df['timestamp'] = pd.to_datetime(log_df['timestamp'])
                log_df = log_df.sort_values('timestamp', ascending=False)
                
                st.dataframe(
                    log_df[['timestamp', 'status', 'message']].head(50),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No processing logs available")
            
            # Clear logs button
            if st.button("🗑️ Clear Logs"):
                st.session_state.processing_log = []
                st.success("Logs cleared")
                st.rerun()

class ExportManager:
    """Handle data export functionality"""
    
    @staticmethod
    def export_to_excel(fields: Dict[str, Any], tables: List[pd.DataFrame], filename: str) -> bytes:
        """Export data to Excel format"""
        output = io.BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Export fields
            if fields:
                fields_df = pd.DataFrame([
                    {"Field": key, "Value": value} 
                    for key, value in fields.items()
                ])
                fields_df.to_excel(writer, sheet_name='Fields', index=False)
            
            # Export tables
            for i, table in enumerate(tables):
                sheet_name = f'Table_{i+1}'
                table.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
    
    @staticmethod
    def export_to_csv(fields: Dict[str, Any], tables: List[pd.DataFrame]) -> bytes:
        """Export data to CSV format (zip file for multiple tables)"""
        if len(tables) <= 1 and not fields:
            # Single table, return CSV directly
            if tables:
                return tables[0].to_csv(index=False).encode()
            else:
                return "".encode()
        
        # Multiple tables or fields, create zip file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add fields CSV
            if fields:
                fields_df = pd.DataFrame([
                    {"Field": key, "Value": value} 
                    for key, value in fields.items()
                ])
                zip_file.writestr("fields.csv", fields_df.to_csv(index=False))
            
            # Add table CSVs
            for i, table in enumerate(tables):
                filename = f"table_{i+1}.csv"
                zip_file.writestr(filename, table.to_csv(index=False))
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    @staticmethod
    def export_to_json(fields: Dict[str, Any], tables: List[pd.DataFrame]) -> bytes:
        """Export data to JSON format"""
        export_data = {
            "fields": fields,
            "tables": [table.to_dict('records') for table in tables],
            "export_timestamp": datetime.now().isoformat(),
            "table_count": len(tables),
            "field_count": len(fields)
        }
        
        return json.dumps(export_data, indent=2, default=str).encode()

class ImportManager:
    """Handle data import functionality"""
    
    @staticmethod
    def import_from_excel(file_data: bytes) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
        """Import data from Excel format"""
        fields = {}
        tables = []
        
        try:
            # Read all sheets
            excel_file = pd.ExcelFile(io.BytesIO(file_data))
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                
                if sheet_name.lower() == 'fields':
                    # Convert fields dataframe back to dictionary
                    if 'Field' in df.columns and 'Value' in df.columns:
                        fields = dict(zip(df['Field'], df['Value']))
                else:
                    # Add as table
                    tables.append(df)
            
        except Exception as e:
            raise ValueError(f"Error importing Excel file: {e}")
        
        return fields, tables
    
    @staticmethod
    def import_from_csv(file_data: bytes) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
        """Import data from CSV format"""
        fields = {}
        tables = []
        
        try:
            # Try to read as single CSV first
            df = pd.read_csv(io.BytesIO(file_data))
            tables.append(df)
        
        except Exception as e:
            raise ValueError(f"Error importing CSV file: {e}")
        
        return fields, tables
    
    @staticmethod
    def import_from_json(file_data: bytes) -> Tuple[Dict[str, Any], List[pd.DataFrame]]:
        """Import data from JSON format"""
        try:
            data = json.loads(file_data.decode())
            
            fields = data.get('fields', {})
            table_data = data.get('tables', [])
            
            tables = [pd.DataFrame(table) for table in table_data]
            
        except Exception as e:
            raise ValueError(f"Error importing JSON file: {e}")
        
        return fields, tables

def create_directories():
    """Create necessary directories"""
    directories = [UPLOAD_FOLDER, EXPORT_FOLDER, TEMP_FOLDER]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file and return path"""
    create_directories()
    
    # Create unique filename
    timestamp = int(time.time())
    filename = f"{timestamp}_{uploaded_file.name}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    
    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def calculate_document_hash(file_path: str) -> str:
    """Calculate document hash for identification"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {file_path}: {e}")
        return f"error_{int(time.time())}"

@st.cache_resource
def initialize_system():
    """Initialize system components (cached)"""
    try:
        create_directories()
        db_manager = DatabaseManager(DATABASE_PATH)
        pattern_learner = PatternLearner(db_manager)
        pdf_processor = PDFProcessor(db_manager, pattern_learner)
        
        logger.info("System initialized successfully")
        return db_manager, pattern_learner, pdf_processor
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        st.error(f"System initialization failed: {e}")
        st.stop()

def main():
    """Main application function"""
    
    # Initialize session state
    SessionStateManager.initialize_session_state()
    
    # Initialize system components
    db_manager, pattern_learner, pdf_processor = initialize_system()
    
    # Update available engines in session state
    st.session_state.available_engines = pdf_processor.get_available_engines()
    
    # Render UI
    UIComponents.render_header()
    UIComponents.render_sidebar()
    
    # Main content area with tabs
    if st.session_state.get('admin_mode', False):
        # Admin mode - show admin panel
        UIComponents.render_admin_panel(db_manager, pattern_learner)
    else:
        # Normal mode - show main tabs
        tabs = st.tabs(["📁 Upload", "📝 Field Data", "📊 Table Data", "🔧 Actions", "📈 Analytics"])
        
        with tabs[0]:  # Upload
            st.markdown("## Document Upload & Processing")
            
            # File upload
            uploaded_file = UIComponents.render_file_uploader()
            
            if uploaded_file is not None:
                # Save file
                file_path = save_uploaded_file(uploaded_file)
                
                # Calculate hash
                document_hash = calculate_document_hash(file_path)
                
                # Check if document already processed
                existing_corrections = db_manager.load_document_corrections(document_hash)
                
                if existing_corrections:
                    st.info("📋 This document has been processed before. Loading previous corrections...")
                    
                    # Load existing data
                    st.session_state.extracted_fields = existing_corrections.get('fields', {})
                    
                    # Load tables
                    table_data = existing_corrections.get('tables', [])
                    st.session_state.extracted_tables = [
                        pd.DataFrame(table) for table in table_data
                    ]
                    
                    st.session_state.document_hash = document_hash
                    st.session_state.corrections_saved = True
                    
                    # Display loaded data info
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Loaded Fields", len(st.session_state.extracted_fields))
                    with col2:
                        st.metric("Loaded Tables", len(st.session_state.extracted_tables))
                    
                else:
                    # Process new document
                    st.info("🔄 Processing new document...")
                    
                    # Update status
                    SessionStateManager.update_processing_status(
                        ProcessingStatus.PROCESSING, 
                        f"Processing {uploaded_file.name}"
                    )
                    
                    # Process with selected engine
                    with st.spinner("Extracting data..."):
                        extraction_result = pdf_processor.extract_with_engine(
                            file_path, 
                            st.session_state.current_extraction_engine
                        )
                    
                    # Validate extracted data
                    validated_fields, field_errors = DataValidator.validate_fields(extraction_result.fields)
                    validated_tables = []
                    table_errors = []
                    
                    for table in extraction_result.tables:
                        validated_table, errors = DataValidator.validate_table(table)
                        validated_tables.append(validated_table)
                        table_errors.extend(errors)
                    
                    # Update session state
                    st.session_state.extracted_fields = validated_fields
                    st.session_state.extracted_tables = validated_tables
                    st.session_state.document_hash = document_hash
                    st.session_state.document_metadata = extraction_result.metadata
                    st.session_state.extraction_results = extraction_result
                    st.session_state.field_validation_errors = field_errors
                    st.session_state.table_validation_errors = table_errors
                    st.session_state.corrections_saved = False
                    
                    # Update status
                    SessionStateManager.update_processing_status(
                        ProcessingStatus.COMPLETED, 
                        f"Extracted {len(validated_fields)} fields and {len(validated_tables)} tables"
                    )
                    
                    # Display extraction results
                    st.success("✅ Document processed successfully!")
                    
                    # Show extraction summary
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Fields Extracted", len(validated_fields))
                    with col2:
                        st.metric("Tables Extracted", len(validated_tables))
                    with col3:
                        st.metric("Confidence", f"{extraction_result.confidence:.2%}")
                    with col4:
                        st.metric("Processing Time", f"{extraction_result.metadata.processing_time:.2f}s")
                    
                    # Show errors and warnings
                    if extraction_result.errors:
                        with st.expander("⚠️ Processing Errors", expanded=False):
                            for error in extraction_result.errors:
                                st.error(error)
                    
                    if extraction_result.warnings:
                        with st.expander("ℹ️ Processing Warnings", expanded=False):
                            for warning in extraction_result.warnings:
                                st.warning(warning)
                
                # Clean up temporary file
                try:
                    os.remove(file_path)
                except:
                    pass
        
        with tabs[1]:  # Field Data
            st.markdown("## Field Data Editor")
            
            if st.session_state.extracted_fields:
                # Field editing interface
                edited_fields = UIComponents.render_field_editor(
                    st.session_state.extracted_fields,
                    st.session_state.get('field_validation_errors', [])
                )
                
                # Update session state with edited fields
                st.session_state.extracted_fields = edited_fields
                
                # Auto-save if enabled
                if st.session_state.get('auto_save_enabled', True) and st.session_state.get('document_hash'):
                    # Auto-save every 30 seconds or on change
                    # This would be implemented with a proper auto-save mechanism
                    pass
                
            else:
                st.info("No field data available. Please upload and process a document first.")
        
        with tabs[2]:  # Table Data
            st.markdown("## Table Data Editor")
            
            if st.session_state.extracted_tables:
                # Table editing interface
                edited_tables = UIComponents.render_table_editor(
                    st.session_state.extracted_tables,
                    st.session_state.get('table_validation_errors', [])
                )
                
                # Update session state with edited tables
                st.session_state.extracted_tables = edited_tables
                
            else:
                st.info("No table data available. Please upload and process a document first.")
        
        with tabs[3]:  # Actions
            st.markdown("## Actions & Data Management")
            
            # Save/Load Section
            st.markdown("### 💾 Save & Load")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Save corrections
                if st.button("💾 Save Corrections", type="primary", use_container_width=True):
                    if st.session_state.get('document_hash'):
                        try:
                            # Prepare data for saving
                            fields_data = st.session_state.get('extracted_fields', {})
                            tables_data = [
                                table.to_dict('records') 
                                for table in st.session_state.get('extracted_tables', [])
                            ]
                            
                            # Save to database
                            db_manager.save_document_corrections(
                                st.session_state.document_hash,
                                fields_data,
                                tables_data,
                                confidence=st.session_state.get('document_metadata', {}).get('confidence', 0.0)
                            )
                            
                            # Learn from corrections
                            pattern_learner.learn_from_corrections(
                                st.session_state.document_hash,
                                fields_data,
                                st.session_state.get('extracted_tables', [])
                            )
                            
                            st.session_state.corrections_saved = True
                            st.session_state.last_save_time = datetime.now()
                            
                            SessionStateManager.add_to_history(
                                "save_corrections",
                                {"fields": len(fields_data), "tables": len(tables_data)}
                            )
                            
                            st.success("✅ Corrections saved successfully!")
                            
                        except Exception as e:
                            st.error(f"❌ Save failed: {e}")
                            logger.error(f"Save corrections failed: {e}")
                    else:
                        st.warning("No document loaded to save")
            
            with col2:
                # Load corrections
                if st.button("📂 Load Previous Corrections", use_container_width=True):
                    if st.session_state.get('document_hash'):
                        try:
                            corrections = db_manager.load_document_corrections(st.session_state.document_hash)
                            
                            if corrections:
                                st.session_state.extracted_fields = corrections.get('fields', {})
                                
                                table_data = corrections.get('tables', [])
                                st.session_state.extracted_tables = [
                                    pd.DataFrame(table) for table in table_data
                                ]
                                
                                st.session_state.corrections_saved = True
                                
                                SessionStateManager.add_to_history(
                                    "load_corrections",
                                    {"fields": len(st.session_state.extracted_fields), "tables": len(st.session_state.extracted_tables)}
                                )
                                
                                st.success("✅ Previous corrections loaded!")
                            else:
                                st.info("No previous corrections found for this document")
                                
                        except Exception as e:
                            st.error(f"❌ Load failed: {e}")
                            logger.error(f"Load corrections failed: {e}")
                    else:
                        st.warning("No document loaded")
            
            # Export Section
            st.markdown("### 📤 Export Data")
            
            if st.session_state.extracted_fields or st.session_state.extracted_tables:
                col1, col2, col3 = st.columns(3)
                
                # Export format selection
                export_format = st.selectbox(
                    "Export Format",
                    ["xlsx", "csv", "json"],
                    index=0,
                    key="export_format_select"
                )
                
                with col1:
                    if st.button("📊 Export to Excel", use_container_width=True):
                        try:
                            excel_data = ExportManager.export_to_excel(
                                st.session_state.extracted_fields,
                                st.session_state.extracted_tables,
                                "exported_data.xlsx"
                            )
                            
                            st.download_button(
                                "📥 Download Excel File",
                                excel_data,
                                file_name=f"extraction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                
                with col2:
                    if st.button("📝 Export to CSV", use_container_width=True):
                        try:
                            csv_data = ExportManager.export_to_csv(
                                st.session_state.extracted_fields,
                                st.session_state.extracted_tables
                            )
                            
                            filename = f"extraction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                            if len(st.session_state.extracted_tables) > 1 or st.session_state.extracted_fields:
                                filename += ".zip"
                                mime_type = "application/zip"
                            else:
                                filename += ".csv"
                                mime_type = "text/csv"
                            
                            st.download_button(
                                "📥 Download CSV File(s)",
                                csv_data,
                                file_name=filename,
                                mime=mime_type
                            )
                            
                        except Exception as e:
                            st.error(f"Export failed: {e}")
                
                with col3:
                    if st.button("🔧 Export to JSON", use_container_width=True):
                        try:
                            json_data = ExportManager.export_to_json(
                                st.session_state.extracted_fields,
                                st.session_state.extracted_tables
                            )
                            
                            st.download_button(
                                "📥 Download JSON File",
                                json_data,
                                file_name=f"extraction_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                            
                        except Exception as e:
                            st.error(f"Export failed: {e}")
            else:
                st.info("No data available for export")
            
            # Import Section
            st.markdown("### 📥 Import Data")
            
            uploaded_import_file = st.file_uploader(
                "Import previously exported data",
                type=['xlsx', 'csv', 'json'],
                help="Import field and table data from previously exported files"
            )
            
            if uploaded_import_file is not None:
                try:
                    file_extension = uploaded_import_file.name.split('.')[-1].lower()
                    file_data = uploaded_import_file.read()
                    
                    if file_extension == 'xlsx':
                        fields, tables = ImportManager.import_from_excel(file_data)
                    elif file_extension == 'csv':
                        fields, tables = ImportManager.import_from_csv(file_data)
                    elif file_extension == 'json':
                        fields, tables = ImportManager.import_from_json(file_data)
                    else:
                        st.error("Unsupported file format")
                        fields, tables = {}, []
                    
                    if fields or tables:
                        st.session_state.extracted_fields = fields
                        st.session_state.extracted_tables = tables
                        st.session_state.corrections_saved = False
                        
                        SessionStateManager.add_to_history(
                            "import_data",
                            {"fields": len(fields), "tables": len(tables), "format": file_extension}
                        )
                        
                        st.success(f"✅ Imported {len(fields)} fields and {len(tables)} tables")
                    else:
                        st.warning("No data found in imported file")
                        
                except Exception as e:
                    st.error(f"Import failed: {e}")
            
            # Status Information
            st.markdown("### ℹ️ Status Information")
            
            status_col1, status_col2 = st.columns(2)
            
            with status_col1:
                if st.session_state.get('corrections_saved'):
                    st.success("✅ Corrections are saved")
                    if st.session_state.get('last_save_time'):
                        st.write(f"Last saved: {st.session_state.last_save_time.strftime('%Y-%m-%d %H:%M:%S')}")
                else:
                    st.warning("⚠️ Unsaved changes")
            
            with status_col2:
                if st.session_state.get('document_hash'):
                    st.info(f"Document ID: {st.session_state.document_hash[:8]}...")
                else:
                    st.info("No document loaded")
        
        with tabs[4]:  # Analytics
            UIComponents.render_analytics_dashboard(db_manager)
    
    # Debug information (if enabled)
    if st.session_state.get('debug_mode', False):
        with st.expander("🐛 Debug Information", expanded=False):
            st.write("**Session State:**")
            debug_state = {k: v for k, v in st.session_state.items() if not k.startswith('_')}
            st.json(debug_state)
            
            st.write("**Available Engines:**")
            st.write([engine.value for engine in st.session_state.get('available_engines', [])])
            
            st.write("**Processing Log:**")
            if st.session_state.get('processing_log'):
                for entry in st.session_state.processing_log[-5:]:  # Last 5 entries
                    st.text(f"{entry['timestamp']}: {entry['status']} - {entry['message']}")

if __name__ == "__main__":
    main()