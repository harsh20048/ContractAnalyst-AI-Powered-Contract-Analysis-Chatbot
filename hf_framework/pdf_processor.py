"""
PDF Processing module for extracting text, metadata, and structure from PDF documents.
Supports multiple PDF libraries for robust extraction.
"""

import os
import re
import logging
import tempfile
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

import fitz  # PyMuPDF
import pdfplumber
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
import pandas as pd
import numpy as np
from collections import Counter

from .config import settings

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
    
try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)
    
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    """Metadata extracted from PDF document."""
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    creator: Optional[str] = None
    producer: Optional[str] = None
    creation_date: Optional[str] = None
    modification_date: Optional[str] = None
    page_count: int = 0
    file_size: int = 0
    file_hash: Optional[str] = None


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""
    content: str
    page_number: int
    chunk_index: int
    bbox: Optional[Tuple[float, float, float, float]] = None  # x0, y0, x1, y1
    font_size: Optional[float] = None
    font_name: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    confidence_score: float = 1.0


@dataclass 
class ExtractedTable:
    """Represents a table extracted from PDF."""
    data: List[List[str]]
    page_number: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    headers: Optional[List[str]] = None


@dataclass
class ProcessingResult:
    """Complete processing result for a PDF document."""
    success: bool
    text_chunks: List[TextChunk]
    tables: List[ExtractedTable]
    metadata: DocumentMetadata
    error_message: Optional[str] = None
    processing_time: float = 0.0
    word_count: int = 0
    sentence_count: int = 0
    language: Optional[str] = None


class PDFProcessor:
    """
    Comprehensive PDF processor supporting multiple extraction methods.
    
    Uses PyMuPDF, pdfplumber, and PyPDF2 for robust text extraction,
    with fallback strategies for difficult documents.
    """
    
    def __init__(self):
        """Initialize the PDF processor."""
        self.stop_words = set(stopwords.words('english'))
        self.extraction_methods = ['pymupdf', 'pdfplumber', 'pypdf2']
        
    def process_pdf(self, file_path: str) -> ProcessingResult:
        """
        Process a PDF file and extract all relevant information.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            ProcessingResult containing extracted data
        """
        import time
        start_time = time.time()
        
        try:
            # Validate file
            if not self._validate_pdf(file_path):
                return ProcessingResult(
                    success=False,
                    text_chunks=[],
                    tables=[],
                    metadata=DocumentMetadata(),
                    error_message="Invalid PDF file"
                )
            
            # Extract metadata
            metadata = self._extract_metadata(file_path)
            
            # Try different extraction methods
            text_chunks = []
            tables = []
            
            for method in self.extraction_methods:
                try:
                    if method == 'pymupdf':
                        chunks, extracted_tables = self._extract_with_pymupdf(file_path)
                    elif method == 'pdfplumber':
                        chunks, extracted_tables = self._extract_with_pdfplumber(file_path)
                    elif method == 'pypdf2':
                        chunks, extracted_tables = self._extract_with_pypdf2(file_path)
                    
                    if chunks:  # If we got some text, use this method
                        text_chunks = chunks
                        tables = extracted_tables
                        logger.info(f"Successfully extracted text using {method}")
                        break
                        
                except Exception as e:
                    logger.warning(f"Failed to extract with {method}: {str(e)}")
                    continue
            
            if not text_chunks:
                return ProcessingResult(
                    success=False,
                    text_chunks=[],
                    tables=[],
                    metadata=metadata,
                    error_message="Failed to extract text with any method"
                )
            
            # Post-process text chunks
            text_chunks = self._post_process_chunks(text_chunks)
            
            # Calculate statistics
            all_text = " ".join([chunk.content for chunk in text_chunks])
            word_count = len(word_tokenize(all_text))
            sentence_count = len(sent_tokenize(all_text))
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                success=True,
                text_chunks=text_chunks,
                tables=tables,
                metadata=metadata,
                processing_time=processing_time,
                word_count=word_count,
                sentence_count=sentence_count,
                language=self._detect_language(all_text)
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return ProcessingResult(
                success=False,
                text_chunks=[],
                tables=[],
                metadata=DocumentMetadata(),
                error_message=str(e),
                processing_time=time.time() - start_time
            )
    
    def _validate_pdf(self, file_path: str) -> bool:
        """Validate that the file is a proper PDF."""
        try:
            if not os.path.exists(file_path):
                return False
                
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > settings.max_file_size:
                logger.error(f"File too large: {file_size} bytes")
                return False
                
            # Check if it's a PDF by reading header
            with open(file_path, 'rb') as f:
                header = f.read(5)
                if header != b'%PDF-':
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF: {str(e)}")
            return False
    
    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from PDF using PyPDF2."""
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                
                metadata = DocumentMetadata()
                metadata.page_count = len(reader.pages)
                metadata.file_size = os.path.getsize(file_path)
                
                # Calculate file hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    metadata.file_hash = file_hash
                
                # Extract document info
                if reader.metadata:
                    metadata.title = reader.metadata.get('/Title')
                    metadata.author = reader.metadata.get('/Author')
                    metadata.subject = reader.metadata.get('/Subject')
                    metadata.creator = reader.metadata.get('/Creator')
                    metadata.producer = reader.metadata.get('/Producer')
                    metadata.creation_date = str(reader.metadata.get('/CreationDate'))
                    metadata.modification_date = str(reader.metadata.get('/ModDate'))
                
                return metadata
                
        except Exception as e:
            logger.warning(f"Failed to extract metadata: {str(e)}")
            return DocumentMetadata(
                page_count=0,
                file_size=os.path.getsize(file_path) if os.path.exists(file_path) else 0
            )
    
    def _extract_with_pymupdf(self, file_path: str) -> Tuple[List[TextChunk], List[ExtractedTable]]:
        """Extract text and tables using PyMuPDF (fitz)."""
        doc = fitz.open(file_path)
        text_chunks = []
        tables = []
        
        for page_num, page in enumerate(doc):
            # Extract text blocks with formatting
            blocks = page.get_text("dict")
            
            chunk_index = 0
            for block in blocks.get("blocks", []):
                if block.get("type") == 0:  # Text block
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                chunk = TextChunk(
                                    content=text,
                                    page_number=page_num + 1,
                                    chunk_index=chunk_index,
                                    bbox=(span.get("bbox")),
                                    font_size=span.get("size"),
                                    font_name=span.get("font"),
                                    is_bold="Bold" in span.get("font", ""),
                                    is_italic="Italic" in span.get("font", "")
                                )
                                text_chunks.append(chunk)
                                chunk_index += 1
            
            # Extract tables
            try:
                page_tables = page.find_tables()
                for i, table in enumerate(page_tables):
                    table_data = table.extract()
                    if table_data:
                        extracted_table = ExtractedTable(
                            data=table_data,
                            page_number=page_num + 1,
                            bbox=table.bbox,
                            headers=table_data[0] if table_data else None
                        )
                        tables.append(extracted_table)
            except Exception as e:
                logger.warning(f"Failed to extract tables from page {page_num + 1}: {str(e)}")
        
        doc.close()
        return text_chunks, tables
    
    def _extract_with_pdfplumber(self, file_path: str) -> Tuple[List[TextChunk], List[ExtractedTable]]:
        """Extract text and tables using pdfplumber."""
        text_chunks = []
        tables = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    # Split into sentences and create chunks
                    sentences = sent_tokenize(text)
                    for i, sentence in enumerate(sentences):
                        if sentence.strip():
                            chunk = TextChunk(
                                content=sentence.strip(),
                                page_number=page_num + 1,
                                chunk_index=i
                            )
                            text_chunks.append(chunk)
                
                # Extract tables
                page_tables = page.extract_tables()
                for table_data in page_tables:
                    if table_data:
                        extracted_table = ExtractedTable(
                            data=table_data,
                            page_number=page_num + 1,
                            headers=table_data[0] if table_data else None
                        )
                        tables.append(extracted_table)
        
        return text_chunks, tables
    
    def _extract_with_pypdf2(self, file_path: str) -> Tuple[List[TextChunk], List[ExtractedTable]]:
        """Extract text using PyPDF2 (fallback method)."""
        text_chunks = []
        tables = []  # PyPDF2 doesn't extract tables well
        
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    # Split into paragraphs and sentences
                    paragraphs = text.split('\n\n')
                    chunk_index = 0
                    
                    for paragraph in paragraphs:
                        paragraph = paragraph.strip()
                        if paragraph:
                            sentences = sent_tokenize(paragraph)
                            for sentence in sentences:
                                if sentence.strip():
                                    chunk = TextChunk(
                                        content=sentence.strip(),
                                        page_number=page_num + 1,
                                        chunk_index=chunk_index
                                    )
                                    text_chunks.append(chunk)
                                    chunk_index += 1
        
        return text_chunks, tables
    
    def _post_process_chunks(self, chunks: List[TextChunk]) -> List[TextChunk]:
        """Post-process text chunks to improve quality."""
        processed_chunks = []
        
        for chunk in chunks:
            # Clean text
            text = chunk.content
            
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove very short chunks (likely noise)
            if len(text.strip()) < 10:
                continue
                
            # Update chunk
            chunk.content = text.strip()
            processed_chunks.append(chunk)
        
        return processed_chunks
    
    def _detect_language(self, text: str) -> Optional[str]:
        """Simple language detection based on common words."""
        if not text:
            return None
            
        # Simple heuristic - count English stop words
        words = word_tokenize(text.lower())
        english_count = sum(1 for word in words if word in self.stop_words)
        
        if english_count / len(words) > 0.1:  # At least 10% stop words
            return "english"
        else:
            return "unknown"
    
    def chunk_text(self, text: str, chunk_size: int = None, chunk_overlap: int = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk (default from settings)
            chunk_overlap: Overlap between chunks (default from settings)
            
        Returns:
            List of text chunks
        """
        chunk_size = chunk_size or settings.chunk_size
        chunk_overlap = chunk_overlap or settings.chunk_overlap
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence ending within the overlap region
                for i in range(end - chunk_overlap, end):
                    if i < len(text) and text[i] in '.!?':
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def extract_key_phrases(self, text: str, top_k: int = 10) -> List[str]:
        """Extract key phrases from text using NLP techniques."""
        try:
            # Tokenize and tag
            tokens = word_tokenize(text.lower())
            tagged = pos_tag(tokens)
            
            # Extract noun phrases
            phrases = []
            current_phrase = []
            
            for word, tag in tagged:
                if tag.startswith('NN') or tag.startswith('JJ'):  # Nouns and adjectives
                    current_phrase.append(word)
                else:
                    if len(current_phrase) >= 2:  # Multi-word phrases
                        phrases.append(' '.join(current_phrase))
                    current_phrase = []
            
            # Add final phrase if exists
            if len(current_phrase) >= 2:
                phrases.append(' '.join(current_phrase))
            
            # Count and return top phrases
            phrase_counts = Counter(phrases)
            return [phrase for phrase, _ in phrase_counts.most_common(top_k)]
            
        except Exception as e:
            logger.warning(f"Failed to extract key phrases: {str(e)}")
            return []
    
    def extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        try:
            tokens = word_tokenize(text)
            tagged = pos_tag(tokens)
            entities = ne_chunk(tagged)
            
            entity_dict = {}
            
            for chunk in entities:
                if hasattr(chunk, 'label'):
                    entity_type = chunk.label()
                    entity_text = ' '.join([token for token, pos in chunk.leaves()])
                    
                    if entity_type not in entity_dict:
                        entity_dict[entity_type] = []
                    entity_dict[entity_type].append(entity_text)
            
            # Remove duplicates
            for entity_type in entity_dict:
                entity_dict[entity_type] = list(set(entity_dict[entity_type]))
            
            return entity_dict
            
        except Exception as e:
            logger.warning(f"Failed to extract named entities: {str(e)}")
            return {}