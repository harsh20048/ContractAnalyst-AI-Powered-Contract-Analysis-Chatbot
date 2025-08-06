import hashlib
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from PyPDF2 import PdfReader
from database import LearningDatabase
from pattern import PatternLearner

logger = logging.getLogger(__name__)

@dataclass
class ExtractedField:
    """Data class for extracted field information"""
    value: str
    confidence: float
    source: str  # 'regex', 'ai', 'user_correction', 'learned_pattern'
    context: str = ""
    page_number: int = 0

class FieldExtractor:
    """Advanced field extraction system with learning capabilities"""
    
    def __init__(self, db_path: str = "learning.db"):
        """Initialize field extractor with database and pattern learner"""
        self.db = LearningDatabase(db_path)
        self.pattern_learner = PatternLearner(db_path)
        
        # Define extraction patterns for different field types
        self.field_patterns = {
            'date': [
                r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})\b',
                r'\b(\d{2,4}[./\-]\d{1,2}[./\-]\d{1,2})\b',
                r'\b(\d{1,2}\.\d{1,2}\.\d{2,4})\b'
            ],
            'angebot_number': [
                r'(?:angebot|quote|offer|nr\.?|no\.?|number|ref\.?)[\s:]*([A-Z0-9\-_/\.#]{3,20})\b',
                r'\b([A-Z0-9\-_/\.#]{3,20})\b(?=\s*(?:angebot|quote|offer))',
                r'(?:A|Q|O)[\-_]?(\d{4,})',
                r'\b([A-Z]{2,4}\d{4,})\b'
            ],
            'company_name': [
                r'(?:company|firma|unternehmen)[\s:]+([A-Z][A-Za-z\s&,.-]{2,50})',
                r'\b([A-Z][A-Za-z\s&,.-]{2,50})\s+(?:GmbH|AG|Ltd|Inc|Corp)',
                r'(?:von|from|für|for)[\s:]+([A-Z][A-Za-z\s&,.-]{2,50})'
            ],
            'amount': [
                r'(?:€|EUR|\$|USD)\s*(\d+(?:[.,]\d{1,2})?)',
                r'(\d+(?:[.,]\d{1,2})?)\s*(?:€|EUR|\$|USD)',
                r'(?:betrag|amount|summe|total)[\s:]*(\d+(?:[.,]\d{1,2})?)',
                r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?)\s*(?:€|EUR)'
            ],
            'delivery_date': [
                r'(?:delivery|lieferung|liefertermin)[\s:]*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
                r'(?:bis|until|by)[\s:]*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})',
                r'(?:deadline|frist)[\s:]*(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})'
            ],
            'contact_person': [
                r'(?:contact|ansprechpartner|kontakt)[\s:]*([A-Z][a-z]+\s+[A-Z][a-z]+)',
                r'(?:mr\.?|mrs\.?|herr|frau)[\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)',
                r'(?:bearbeiter|sachbearbeiter)[\s:]*([A-Z][a-z]+\s+[A-Z][a-z]+)'
            ]
        }
        
        # Field importance weights for prioritizing extraction
        self.field_weights = {
            'angebot_number': 1.0,
            'date': 0.9,
            'company_name': 0.8,
            'amount': 0.9,
            'delivery_date': 0.7,
            'contact_person': 0.6
        }
    
    def generate_document_hash(self, pdf_path: str) -> str:
        """Generate a consistent hash for the PDF document based on content"""
        try:
            # Read PDF content
            pdf = PdfReader(pdf_path)
            content = ""
            
            # Extract text from all pages
            for page in pdf.pages:
                content += page.extract_text()
            
            # Clean and normalize content for consistent hashing
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.strip().lower()
            
            # Generate hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            return content_hash
            
        except Exception as e:
            logger.error(f"Error generating document hash: {e}")
            # Fallback to file-based hash if content extraction fails
            with open(pdf_path, 'rb') as f:
                file_content = f.read()
                return hashlib.sha256(file_content).hexdigest()
    
    def extract_fields_from_pdf(self, pdf_path: str) -> Tuple[str, Dict[str, ExtractedField]]:
        """Extract fields from PDF and return document hash and extracted fields"""
        try:
            # Generate document hash
            document_hash = self.generate_document_hash(pdf_path)
            pdf_name = pdf_path.split('/')[-1]
            
            # Save document metadata
            import os
            file_size = os.path.getsize(pdf_path)
            self.db.save_document_metadata(document_hash, pdf_name, file_size)
            
            # Check if we have existing corrections for this document
            if self.db.has_corrections_for_document(document_hash):
                logger.info(f"Found existing corrections for document {document_hash}")
                existing_corrections = self.db.get_corrections_for_document(document_hash)
                
                # Convert corrections to ExtractedField objects
                extracted_fields = {}
                for field_name, corrected_value in existing_corrections.items():
                    extracted_fields[field_name] = ExtractedField(
                        value=corrected_value,
                        confidence=0.95,
                        source='user_correction',
                        context='From saved corrections'
                    )
                
                return document_hash, extracted_fields
            
            # Extract text from PDF
            pdf = PdfReader(pdf_path)
            full_text = ""
            page_texts = []
            
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                page_texts.append((page_text, i + 1))
                full_text += page_text + "\n"
            
            # Extract fields using patterns and learned patterns
            extracted_fields = {}
            
            for field_type in self.field_patterns.keys():
                field_result = self._extract_field_with_learning(
                    field_type, full_text, page_texts, document_hash
                )
                if field_result:
                    extracted_fields[field_type] = field_result
            
            logger.info(f"Extracted {len(extracted_fields)} fields from document")
            return document_hash, extracted_fields
            
        except Exception as e:
            logger.error(f"Error extracting fields from PDF: {e}")
            return "", {}
    
    def _extract_field_with_learning(self, field_type: str, full_text: str, 
                                   page_texts: List[Tuple[str, int]], 
                                   document_hash: str) -> Optional[ExtractedField]:
        """Extract a specific field using patterns and learned patterns"""
        best_match = None
        best_confidence = 0.0
        
        # Try learned patterns first (highest priority)
        learned_patterns = self.pattern_learner.get_learned_patterns(field_type)
        for pattern_info in learned_patterns:
            pattern = pattern_info['pattern']
            confidence = pattern_info['confidence']
            
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if match.group(1):  # Ensure we have a captured group
                    value = match.group(1).strip()
                    if self._validate_field_value(field_type, value):
                        if confidence > best_confidence:
                            context = self._extract_context(full_text, match.start(), match.end())
                            page_num = self._find_page_number(page_texts, match.start())
                            
                            best_match = ExtractedField(
                                value=value,
                                confidence=confidence,
                                source='learned_pattern',
                                context=context,
                                page_number=page_num
                            )
                            best_confidence = confidence
                            
                            # Update pattern performance
                            self.pattern_learner.update_pattern_performance(
                                field_type, pattern, True
                            )
        
        # Try built-in patterns if no learned pattern found or confidence is low
        if best_confidence < 0.7:
            patterns = self.field_patterns.get(field_type, [])
            for pattern in patterns:
                matches = re.finditer(pattern, full_text, re.IGNORECASE)
                for match in matches:
                    if match.group(1):
                        value = match.group(1).strip()
                        if self._validate_field_value(field_type, value):
                            # Calculate confidence based on context and pattern quality
                            confidence = self._calculate_confidence(field_type, value, match, full_text)
                            
                            if confidence > best_confidence:
                                context = self._extract_context(full_text, match.start(), match.end())
                                page_num = self._find_page_number(page_texts, match.start())
                                
                                best_match = ExtractedField(
                                    value=value,
                                    confidence=confidence,
                                    source='regex',
                                    context=context,
                                    page_number=page_num
                                )
                                best_confidence = confidence
        
        return best_match
    
    def _validate_field_value(self, field_type: str, value: str) -> bool:
        """Validate extracted field value based on field type"""
        if not value or len(value.strip()) < 1:
            return False
        
        if field_type == 'date':
            # Check if it looks like a valid date
            return bool(re.match(r'\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4}', value))
        elif field_type == 'angebot_number':
            # Check if it looks like a reasonable ID
            return len(value) >= 3 and bool(re.match(r'^[A-Z0-9\-_/\.#]+$', value, re.IGNORECASE))
        elif field_type == 'amount':
            # Check if it looks like a monetary amount
            return bool(re.match(r'^\d+(?:[.,]\d{1,2})?$', value))
        elif field_type == 'company_name':
            # Check if it looks like a company name
            return len(value) > 2 and any(c.isupper() for c in value)
        
        return True
    
    def _calculate_confidence(self, field_type: str, value: str, match, full_text: str) -> float:
        """Calculate confidence score for extracted field"""
        base_confidence = 0.6
        
        # Apply field weight
        weight = self.field_weights.get(field_type, 0.5)
        confidence = base_confidence * weight
        
        # Boost confidence for exact keyword matches
        context_before = full_text[max(0, match.start() - 50):match.start()].lower()
        if field_type == 'angebot_number' and any(kw in context_before for kw in ['angebot', 'quote', 'offer']):
            confidence += 0.2
        elif field_type == 'amount' and any(kw in context_before for kw in ['betrag', 'amount', 'summe']):
            confidence += 0.2
        elif field_type == 'date' and any(kw in context_before for kw in ['datum', 'date']):
            confidence += 0.15
        
        # Boost confidence for values that follow expected patterns
        if field_type == 'angebot_number' and re.match(r'^[A-Z]{2,4}\d{4,}$', value):
            confidence += 0.15
        elif field_type == 'amount' and ',' in value or '.' in value:
            confidence += 0.1
        
        return min(0.95, confidence)
    
    def _extract_context(self, text: str, start: int, end: int, context_length: int = 100) -> str:
        """Extract context around matched text"""
        context_start = max(0, start - context_length // 2)
        context_end = min(len(text), end + context_length // 2)
        return text[context_start:context_end].strip()
    
    def _find_page_number(self, page_texts: List[Tuple[str, int]], char_position: int) -> int:
        """Find which page a character position belongs to"""
        current_pos = 0
        for page_text, page_num in page_texts:
            if current_pos <= char_position < current_pos + len(page_text):
                return page_num
            current_pos += len(page_text) + 1  # +1 for newline
        return 1  # Default to page 1
    
    def save_field_corrections(self, document_hash: str, pdf_name: str, 
                             corrections: Dict[str, str], 
                             original_fields: Dict[str, ExtractedField]) -> bool:
        """Save field corrections and learn from them"""
        try:
            saved_count = 0
            for field_name, corrected_value in corrections.items():
                if corrected_value and corrected_value.strip():
                    # Get original value for learning
                    original_value = ""
                    context = ""
                    if field_name in original_fields:
                        original_value = original_fields[field_name].value
                        context = original_fields[field_name].context
                    
                    # Save correction to database
                    success = self.db.save_correction(
                        document_hash=document_hash,
                        pdf_name=pdf_name,
                        field_name=field_name,
                        corrected_value=corrected_value,
                        original_value=original_value,
                        confidence=0.95
                    )
                    
                    if success:
                        saved_count += 1
                        
                        # Learn from correction if we have context
                        if context and original_value != corrected_value:
                            self.pattern_learner.learn_from_feedback(
                                field_type=field_name,
                                correct_value=corrected_value,
                                context=context
                            )
                            
                            # Record feedback for adaptive learning
                            self.db.record_feedback(
                                pdf_name=pdf_name,
                                field_type=field_name,
                                extracted=original_value,
                                correct=corrected_value,
                                confidence=0.95,
                                context=context,
                                document_hash=document_hash
                            )
            
            logger.info(f"Saved {saved_count} field corrections for document {document_hash}")
            return saved_count > 0
            
        except Exception as e:
            logger.error(f"Error saving field corrections: {e}")
            return False
    
    def get_extraction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive extraction statistics"""
        db_stats = self.db.get_correction_statistics()
        pattern_stats = self.pattern_learner.get_pattern_statistics()
        
        return {
            'database_stats': db_stats,
            'pattern_stats': pattern_stats,
            'supported_fields': list(self.field_patterns.keys())
        }