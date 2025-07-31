"""
Training Data Extractor for specific parameters extraction.

This module handles extraction of 5 specific parameters:
1. Date
2. Company Name  
3. Company Address
4. Tables
5. Angebot (Offer/Quote information)

Designed for training data preparation from PDF documents.
"""

import logging
import re
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from pathlib import Path

from .pdf_processor import PDFProcessor, ProcessingResult, TextChunk, ExtractedTable
from .models import ModelManager
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class TrainingDataPoint:
    """Single training data point with extracted parameters."""
    file_name: str
    date: Optional[str] = None
    company_name: Optional[str] = None
    company_address: Optional[str] = None
    tables: Optional[List[Dict[str, Any]]] = None
    angebot: Optional[str] = None
    
    # Additional metadata
    extraction_confidence: Dict[str, float] = None
    raw_text: Optional[str] = None
    page_count: int = 0
    processing_time: float = 0.0
    errors: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return asdict(self)
    
    def to_excel_row(self) -> Dict[str, Any]:
        """Convert to Excel-compatible row format."""
        return {
            'file_name': self.file_name,
            'date': self.date,
            'company_name': self.company_name,
            'company_address': self.company_address,
            'tables_count': len(self.tables) if self.tables else 0,
            'tables_data': json.dumps(self.tables) if self.tables else '',
            'angebot': self.angebot,
            'page_count': self.page_count,
            'processing_time': self.processing_time,
            'errors': '; '.join(self.errors) if self.errors else ''
        }


class TrainingDataExtractor:
    """
    Specialized extractor for training data preparation.
    
    Extracts specific parameters needed for model training:
    - Date
    - Company Name
    - Company Address  
    - Tables
    - Angebot
    """
    
    def __init__(self):
        """Initialize the training data extractor."""
        self.pdf_processor = PDFProcessor()
        self.model_manager = ModelManager()
        
        # Date patterns (German and English formats)
        self.date_patterns = [
            r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',  # DD/MM/YYYY or DD.MM.YYYY
            r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',  # YYYY/MM/DD or YYYY.MM.DD
            r'\b\d{1,2}\.\s*\w+\s*\d{4}\b',     # DD. Month YYYY (German)
            r'\b\w+\s+\d{1,2},?\s+\d{4}\b',     # Month DD, YYYY (English)
            r'\b\d{1,2}[./]\d{1,2}[./]\d{2}\b', # DD/MM/YY
        ]
        
        # Company indicators (German/English)
        self.company_indicators = [
            r'\b(GmbH|AG|KG|OHG|UG|e\.V\.)\b',
            r'\b(Inc|LLC|Ltd|Corp|Corporation|Company)\b',
            r'\b(Firma|Unternehmen|Betrieb|Gesellschaft)\b',
        ]
        
        # Address patterns
        self.address_patterns = [
            r'\b\d{5}\s+[A-Za-zäöüÄÖÜß\s]+\b',  # German postal code + city
            r'\b[A-Za-zäöüÄÖÜß\s]+str\.\s*\d+[a-z]?\b',  # Street with number
            r'\b[A-Za-zäöüÄÖÜß\s]+straße\s*\d+[a-z]?\b',  # Straße variant
            r'\b[A-Za-zäöüÄÖÜß\s]+weg\s*\d+[a-z]?\b',     # Weg variant
        ]
        
        # Angebot/Quote indicators
        self.angebot_indicators = [
            r'\b(Angebot|Kostenvoranschlag|Offerte)\b',
            r'\b(Quote|Quotation|Proposal|Estimate)\b',
            r'\b(Angebotsnummer|Quote\s*Number|Proposal\s*ID)\b',
        ]
        
    def extract_from_pdf(self, pdf_path: str) -> TrainingDataPoint:
        """
        Extract training parameters from a single PDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            TrainingDataPoint with extracted parameters
        """
        import time
        start_time = time.time()
        
        file_name = Path(pdf_path).name
        
        # Initialize result
        result = TrainingDataPoint(
            file_name=file_name,
            extraction_confidence={},
            errors=[]
        )
        
        try:
            # Process PDF
            processing_result = self.pdf_processor.process_pdf(pdf_path)
            
            if not processing_result.success:
                result.errors.append(f"PDF processing failed: {processing_result.error_message}")
                return result
            
            # Get full text
            all_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
            result.raw_text = all_text
            result.page_count = processing_result.metadata.page_count
            
            # Extract each parameter
            result.date = self._extract_date(all_text, processing_result.text_chunks)
            result.company_name = self._extract_company_name(all_text, processing_result.text_chunks)
            result.company_address = self._extract_company_address(all_text, processing_result.text_chunks)
            result.tables = self._extract_tables_data(processing_result.tables)
            result.angebot = self._extract_angebot(all_text, processing_result.text_chunks)
            
            result.processing_time = time.time() - start_time
            
            logger.info(f"Extracted training data from {file_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from {file_name}: {str(e)}")
            result.errors.append(str(e))
            result.processing_time = time.time() - start_time
            return result
    
    def _extract_date(self, text: str, chunks: List[TextChunk]) -> Optional[str]:
        """Extract date from document."""
        try:
            dates_found = []
            
            # Try multiple date patterns
            for pattern in self.date_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                dates_found.extend(matches)
            
            if dates_found:
                # Return the first valid date found
                for date_str in dates_found:
                    # Try to parse and validate the date
                    try:
                        # Simple validation - if it contains reasonable numbers
                        if any(char.isdigit() for char in date_str):
                            return date_str.strip()
                    except:
                        continue
            
            # Alternative: Look for date-like patterns near document header
            header_text = " ".join([chunk.content for chunk in chunks[:10]])  # First 10 chunks
            for pattern in self.date_patterns:
                matches = re.findall(pattern, header_text, re.IGNORECASE)
                if matches:
                    return matches[0].strip()
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting date: {str(e)}")
            return None
    
    def _extract_company_name(self, text: str, chunks: List[TextChunk]) -> Optional[str]:
        """Extract company name from document."""
        try:
            # Strategy 1: Look for company legal forms
            for pattern in self.company_indicators:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    # Extract potential company name from context
                    lines = context.split('\n')
                    for line in lines:
                        if pattern.lower() in line.lower():
                            # Clean and return the line containing company indicator
                            cleaned = re.sub(r'\s+', ' ', line).strip()
                            if len(cleaned) > 5 and len(cleaned) < 100:
                                return cleaned
            
            # Strategy 2: Look in document header/footer
            header_chunks = chunks[:5]  # First 5 text chunks
            for chunk in header_chunks:
                text_lines = chunk.content.split('\n')
                for line in text_lines:
                    # Look for capitalized text that might be company name
                    if len(line.strip()) > 5 and len(line.strip()) < 80:
                        # Check if line has typical company name characteristics
                        if (line.isupper() or 
                            any(indicator.lower() in line.lower() for indicator in ['gmbh', 'ag', 'inc', 'ltd', 'corp'])):
                            return line.strip()
            
            # Strategy 3: Use NER model if available
            try:
                ner_result = self.model_manager.extract_entities(text[:1000])  # First 1000 chars
                if ner_result.result and 'ORG' in ner_result.result:
                    organizations = ner_result.result['ORG']
                    if organizations:
                        # Return the longest organization name found
                        return max(organizations, key=lambda x: len(x['text']))['text']
            except Exception as e:
                logger.warning(f"NER extraction failed: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting company name: {str(e)}")
            return None
    
    def _extract_company_address(self, text: str, chunks: List[TextChunk]) -> Optional[str]:
        """Extract company address from document."""
        try:
            addresses_found = []
            
            # Strategy 1: Look for address patterns
            for pattern in self.address_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                addresses_found.extend(matches)
            
            if addresses_found:
                # Return the most complete looking address
                longest_address = max(addresses_found, key=len)
                return longest_address.strip()
            
            # Strategy 2: Look for postal code + city combinations
            postal_pattern = r'\b\d{5}\s+[A-Za-zäöüÄÖÜß\s]+\b'
            postal_matches = re.findall(postal_pattern, text)
            
            if postal_matches:
                # Try to find street address near postal code
                for postal in postal_matches:
                    postal_index = text.find(postal)
                    if postal_index > 0:
                        # Look backwards for street address
                        preceding_text = text[max(0, postal_index-200):postal_index]
                        street_patterns = [
                            r'[A-Za-zäöüÄÖÜß\s]+str\.\s*\d+[a-z]?',
                            r'[A-Za-zäöüÄÖÜß\s]+straße\s*\d+[a-z]?',
                            r'[A-Za-zäöüÄÖÜß\s]+weg\s*\d+[a-z]?'
                        ]
                        
                        for street_pattern in street_patterns:
                            street_matches = re.findall(street_pattern, preceding_text, re.IGNORECASE)
                            if street_matches:
                                full_address = f"{street_matches[-1]}, {postal}"
                                return full_address.strip()
                
                # If no street found, return just postal + city
                return postal_matches[0].strip()
            
            # Strategy 3: Use NER for location entities
            try:
                ner_result = self.model_manager.extract_entities(text[:1500])
                if ner_result.result and 'LOC' in ner_result.result:
                    locations = ner_result.result['LOC']
                    if locations:
                        # Combine location entities that might form an address
                        location_texts = [loc['text'] for loc in locations]
                        if len(location_texts) >= 2:
                            return ', '.join(location_texts[:2])
                        else:
                            return location_texts[0]
            except Exception as e:
                logger.warning(f"NER location extraction failed: {str(e)}")
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting address: {str(e)}")
            return None
    
    def _extract_tables_data(self, tables: List[ExtractedTable]) -> Optional[List[Dict[str, Any]]]:
        """Extract and structure table data."""
        try:
            if not tables:
                return None
            
            structured_tables = []
            
            for i, table in enumerate(tables):
                table_info = {
                    'table_id': i + 1,
                    'page_number': table.page_number,
                    'rows': len(table.data),
                    'columns': len(table.data[0]) if table.data else 0,
                    'headers': table.headers,
                    'data': table.data,
                    'bbox': table.bbox
                }
                
                # Try to identify table type based on content
                table_type = self._identify_table_type(table.data)
                table_info['table_type'] = table_type
                
                # Extract key-value pairs if it's a simple table
                if table_type == 'key_value':
                    table_info['key_value_pairs'] = self._extract_key_value_pairs(table.data)
                
                structured_tables.append(table_info)
            
            return structured_tables
            
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
            return None
    
    def _identify_table_type(self, table_data: List[List[str]]) -> str:
        """Identify the type of table based on its content."""
        if not table_data or len(table_data) < 2:
            return 'unknown'
        
        # Check if it's a key-value table (2 columns, descriptive first column)
        if len(table_data[0]) == 2:
            return 'key_value'
        
        # Check if it's a price/items table
        headers = table_data[0] if table_data else []
        price_indicators = ['preis', 'price', 'betrag', 'amount', 'kosten', 'cost', 'summe', 'total']
        item_indicators = ['artikel', 'item', 'position', 'beschreibung', 'description']
        
        header_text = ' '.join(headers).lower()
        
        if any(indicator in header_text for indicator in price_indicators):
            if any(indicator in header_text for indicator in item_indicators):
                return 'items_pricing'
            else:
                return 'pricing'
        
        return 'data'
    
    def _extract_key_value_pairs(self, table_data: List[List[str]]) -> Dict[str, str]:
        """Extract key-value pairs from a 2-column table."""
        pairs = {}
        
        for row in table_data[1:]:  # Skip header
            if len(row) >= 2:
                key = row[0].strip()
                value = row[1].strip()
                if key and value:
                    pairs[key] = value
        
        return pairs
    
    def _extract_angebot(self, text: str, chunks: List[TextChunk]) -> Optional[str]:
        """Extract Angebot (quote/offer) information."""
        try:
            angebot_info = {}
            
            # Strategy 1: Look for Angebot indicators and extract nearby information
            for pattern in self.angebot_indicators:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 200)
                    context = text[start:end]
                    
                    # Look for numbers (could be quote numbers, prices, etc.)
                    numbers = re.findall(r'\b\d+[.,]?\d*\b', context)
                    if numbers:
                        angebot_info['numbers_found'] = numbers
                    
                    # Extract the line containing the angebot reference
                    lines = context.split('\n')
                    for line in lines:
                        if any(indicator.lower() in line.lower() for indicator in ['angebot', 'quote', 'quotation']):
                            angebot_info['reference_line'] = line.strip()
                            break
            
            # Strategy 2: Look for quote/offer numbers
            quote_number_patterns = [
                r'(Angebotsnummer|Quote\s*Number|Angebot\s*Nr\.?)\s*:?\s*([A-Za-z0-9\-_]+)',
                r'(Angebot|Quote)\s*:?\s*([A-Za-z0-9\-_]+)',
                r'(Nr\.?|Number|#)\s*:?\s*([A-Za-z0-9\-_]+)'
            ]
            
            for pattern in quote_number_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    angebot_info['quote_numbers'] = [match[1] for match in matches]
                    break
            
            # Strategy 3: Look for pricing information in the document
            price_patterns = [
                r'(\d+[.,]\d{2})\s*(€|EUR|Dollar|\$)',
                r'(€|EUR|\$)\s*(\d+[.,]\d{2})',
                r'(Summe|Total|Gesamt)\s*:?\s*(\d+[.,]\d{2})'
            ]
            
            prices_found = []
            for pattern in price_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                prices_found.extend(matches)
            
            if prices_found:
                angebot_info['prices'] = prices_found
            
            # Strategy 4: Use summarization to get quote summary
            try:
                # Get a summary of the document focusing on offer/quote aspects
                quote_context = ""
                for chunk in chunks:
                    if any(indicator.lower() in chunk.content.lower() 
                          for indicator in ['angebot', 'quote', 'offer', 'proposal']):
                        quote_context += chunk.content + " "
                
                if quote_context and len(quote_context) > 50:
                    summary_result = self.model_manager.summarize_text(
                        quote_context[:800], 
                        max_length=150, 
                        min_length=30
                    )
                    if summary_result.result:
                        angebot_info['summary'] = summary_result.result
            except Exception as e:
                logger.warning(f"Angebot summarization failed: {str(e)}")
            
            # Return structured angebot information as JSON string
            if angebot_info:
                return json.dumps(angebot_info, ensure_ascii=False)
            
            return None
            
        except Exception as e:
            logger.warning(f"Error extracting angebot: {str(e)}")
            return None
    
    def extract_batch(self, pdf_directory: str, excel_output_path: str = None) -> List[TrainingDataPoint]:
        """
        Extract training data from multiple PDFs in a directory.
        
        Args:
            pdf_directory: Directory containing PDF files
            excel_output_path: Optional path to save Excel file with results
            
        Returns:
            List of TrainingDataPoint objects
        """
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory {pdf_directory} not found")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_directory}")
        
        results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.extract_from_pdf(str(pdf_file))
                results.append(result)
                
                # Log progress
                if i % 10 == 0:
                    logger.info(f"Completed {i}/{len(pdf_files)} files")
                    
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {str(e)}")
                # Add error record
                error_result = TrainingDataPoint(
                    file_name=pdf_file.name,
                    errors=[str(e)]
                )
                results.append(error_result)
        
        # Save to Excel if path provided
        if excel_output_path:
            self.save_to_excel(results, excel_output_path)
        
        logger.info(f"Batch extraction completed. Processed {len(results)} files.")
        return results
    
    def save_to_excel(self, results: List[TrainingDataPoint], output_path: str):
        """Save extraction results to Excel file."""
        try:
            # Convert to DataFrame
            excel_data = [result.to_excel_row() for result in results]
            df = pd.DataFrame(excel_data)
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Extracted_Data', index=False)
                
                # Summary sheet
                summary_data = {
                    'Total Files': len(results),
                    'Successful Extractions': len([r for r in results if not r.errors]),
                    'Failed Extractions': len([r for r in results if r.errors]),
                    'Date Extracted': len([r for r in results if r.date]),
                    'Company Name Extracted': len([r for r in results if r.company_name]),
                    'Address Extracted': len([r for r in results if r.company_address]),
                    'Tables Found': len([r for r in results if r.tables]),
                    'Angebot Found': len([r for r in results if r.angebot])
                }
                
                summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Count'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Detailed tables sheet (if any tables found)
                tables_data = []
                for result in results:
                    if result.tables:
                        for table in result.tables:
                            tables_data.append({
                                'file_name': result.file_name,
                                'table_id': table['table_id'],
                                'page_number': table['page_number'],
                                'table_type': table['table_type'],
                                'rows': table['rows'],
                                'columns': table['columns'],
                                'data': json.dumps(table['data'])
                            })
                
                if tables_data:
                    tables_df = pd.DataFrame(tables_data)
                    tables_df.to_excel(writer, sheet_name='Tables_Detail', index=False)
            
            logger.info(f"Results saved to Excel: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving to Excel: {str(e)}")
            raise
    
    def load_labeled_data(self, excel_path: str) -> pd.DataFrame:
        """
        Load labeled training data from Excel file.
        
        Args:
            excel_path: Path to Excel file with labeled data
            
        Returns:
            DataFrame with labeled training data
        """
        try:
            df = pd.read_excel(excel_path, sheet_name='Extracted_Data')
            logger.info(f"Loaded {len(df)} labeled examples from {excel_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading labeled data: {str(e)}")
            raise