"""
Text extraction module for PDF documents.
"""

import logging
from typing import List, Dict, Any, Tuple
from PyPDF2 import PdfReader
import re

logger = logging.getLogger(__name__)

class TextExtractor:
    """Extract and process text from PDF documents."""
    
    def __init__(self):
        """Initialize text extractor."""
        self.page_separator = "\n" + "="*50 + "\n"
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text from PDF with metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        try:
            pdf = PdfReader(pdf_path)
            
            # Extract metadata
            metadata = {
                'num_pages': len(pdf.pages),
                'title': pdf.metadata.title if pdf.metadata and pdf.metadata.title else '',
                'author': pdf.metadata.author if pdf.metadata and pdf.metadata.author else '',
                'creator': pdf.metadata.creator if pdf.metadata and pdf.metadata.creator else '',
                'producer': pdf.metadata.producer if pdf.metadata and pdf.metadata.producer else '',
            }
            
            # Extract text from all pages
            pages = []
            full_text = ""
            
            for i, page in enumerate(pdf.pages):
                page_text = self._clean_text(page.extract_text())
                pages.append({
                    'page_number': i + 1,
                    'text': page_text,
                    'word_count': len(page_text.split()),
                    'char_count': len(page_text)
                })
                full_text += page_text + self.page_separator
            
            return {
                'metadata': metadata,
                'pages': pages,
                'full_text': full_text.strip(),
                'total_words': sum(page['word_count'] for page in pages),
                'total_chars': sum(page['char_count'] for page in pages)
            }
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return {
                'metadata': {},
                'pages': [],
                'full_text': '',
                'total_words': 0,
                'total_chars': 0,
                'error': str(e)
            }
    
    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """
        Extract text from each page separately.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text strings, one per page
        """
        try:
            pdf = PdfReader(pdf_path)
            return [self._clean_text(page.extract_text()) for page in pdf.pages]
        except Exception as e:
            logger.error(f"Error extracting text by page: {e}")
            return []
    
    def search_text(self, pdf_path: str, search_term: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Search for specific text in PDF.
        
        Args:
            pdf_path: Path to PDF file
            search_term: Text to search for
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matches with page numbers and context
        """
        try:
            extraction_result = self.extract_text_from_pdf(pdf_path)
            pages = extraction_result['pages']
            matches = []
            
            flags = 0 if case_sensitive else re.IGNORECASE
            
            for page in pages:
                page_text = page['text']
                page_num = page['page_number']
                
                # Find all matches in this page
                for match in re.finditer(re.escape(search_term), page_text, flags):
                    context_start = max(0, match.start() - 50)
                    context_end = min(len(page_text), match.end() + 50)
                    context = page_text[context_start:context_end].strip()
                    
                    matches.append({
                        'page_number': page_num,
                        'match_text': match.group(),
                        'context': context,
                        'position': match.start()
                    })
            
            return matches
            
        except Exception as e:
            logger.error(f"Error searching text: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text by removing unwanted characters and normalizing whitespace.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove special characters and normalize whitespace
        text = text.replace("\f", " ").replace("\r", " ").replace("\u0000", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def get_text_statistics(self, pdf_path: str) -> Dict[str, Any]:
        """
        Get comprehensive text statistics for PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with text statistics
        """
        try:
            extraction_result = self.extract_text_from_pdf(pdf_path)
            full_text = extraction_result['full_text']
            pages = extraction_result['pages']
            
            # Calculate statistics
            words = full_text.split()
            sentences = re.split(r'[.!?]+', full_text)
            paragraphs = re.split(r'\n\s*\n', full_text)
            
            # Find most common words (excluding common stop words)
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an', 'is', 'are', 'was', 'were'}
            word_freq = {}
            for word in words:
                word_clean = re.sub(r'[^\w]', '', word.lower())
                if word_clean and word_clean not in stop_words and len(word_clean) > 2:
                    word_freq[word_clean] = word_freq.get(word_clean, 0) + 1
            
            most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'total_pages': extraction_result['metadata']['num_pages'],
                'total_words': len(words),
                'total_sentences': len([s for s in sentences if s.strip()]),
                'total_paragraphs': len([p for p in paragraphs if p.strip()]),
                'average_words_per_page': len(words) / max(len(pages), 1),
                'most_common_words': most_common,
                'pages_statistics': [
                    {
                        'page': page['page_number'],
                        'words': page['word_count'],
                        'characters': page['char_count']
                    }
                    for page in pages
                ]
            }
            
        except Exception as e:
            logger.error(f"Error calculating text statistics: {e}")
            return {}