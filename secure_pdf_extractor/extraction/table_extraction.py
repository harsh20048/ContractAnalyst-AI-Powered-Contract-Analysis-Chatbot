"""
Table extraction module for PDF documents.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from PyPDF2 import PdfReader

logger = logging.getLogger(__name__)

class TableExtractor:
    """Extract and process tables from PDF documents."""
    
    def __init__(self):
        """Initialize table extractor."""
        # Common table indicators
        self.table_indicators = [
            r'\b(?:table|tabelle|tab\.)\s*\d+',
            r'\|\s*[^|]+\s*\|',  # Pipe-separated values
            r'[-─]{3,}',  # Horizontal lines
            r'\s{3,}',  # Multiple spaces (column separation)
        ]
        
        # Column separators
        self.column_separators = [
            r'\s{2,}',  # Multiple spaces
            r'\t+',     # Tabs
            r'\|',      # Pipes
            r';',       # Semicolons
        ]
    
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF document.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of extracted tables with metadata
        """
        try:
            pdf = PdfReader(pdf_path)
            tables = []
            
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                page_tables = self._extract_tables_from_text(page_text, page_num)
                tables.extend(page_tables)
            
            return tables
            
        except Exception as e:
            logger.error(f"Error extracting tables from PDF: {e}")
            return []
    
    def _extract_tables_from_text(self, text: str, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables from text content.
        
        Args:
            text: Text content from PDF page
            page_num: Page number
            
        Returns:
            List of extracted tables
        """
        tables = []
        lines = text.split('\n')
        
        # Look for table patterns
        table_start = None
        potential_table_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Check if line looks like a table row
            if self._is_table_row(line):
                if table_start is None:
                    table_start = i
                potential_table_lines.append((i, line))
            else:
                # End of potential table
                if len(potential_table_lines) >= 2:  # Minimum 2 rows for a table
                    table = self._parse_table(potential_table_lines, page_num)
                    if table:
                        tables.append(table)
                
                # Reset
                table_start = None
                potential_table_lines = []
        
        # Check for table at end of page
        if len(potential_table_lines) >= 2:
            table = self._parse_table(potential_table_lines, page_num)
            if table:
                tables.append(table)
        
        return tables
    
    def _is_table_row(self, line: str) -> bool:
        """
        Check if a line looks like a table row.
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be a table row
        """
        # Check for common table patterns
        indicators = [
            len(re.findall(r'\s{3,}', line)) >= 2,  # Multiple column separations
            '|' in line and len(line.split('|')) >= 3,  # Pipe-separated
            len(re.findall(r'\t', line)) >= 2,  # Tab-separated
            re.match(r'^[-─=]{10,}$', line.strip()),  # Horizontal line
        ]
        
        return any(indicators)
    
    def _parse_table(self, table_lines: List[Tuple[int, str]], page_num: int) -> Optional[Dict[str, Any]]:
        """
        Parse table lines into structured data.
        
        Args:
            table_lines: List of (line_number, line_text) tuples
            page_num: Page number
            
        Returns:
            Parsed table data or None
        """
        try:
            rows = []
            headers = []
            
            # Remove horizontal separator lines
            data_lines = [line for _, line in table_lines if not re.match(r'^[-─=\s|]+$', line.strip())]
            
            if len(data_lines) < 2:
                return None
            
            # Try to identify the best column separator
            separator = self._identify_separator(data_lines)
            
            for i, line in enumerate(data_lines):
                # Split line into columns
                if separator == 'spaces':
                    columns = re.split(r'\s{2,}', line.strip())
                elif separator == 'pipes':
                    columns = [col.strip() for col in line.split('|') if col.strip()]
                elif separator == 'tabs':
                    columns = [col.strip() for col in line.split('\t') if col.strip()]
                else:
                    columns = re.split(r'\s{2,}', line.strip())
                
                # Clean columns
                columns = [col.strip() for col in columns if col.strip()]
                
                if i == 0:
                    headers = columns
                else:
                    rows.append(columns)
            
            # Validate table structure
            if not headers or not rows:
                return None
            
            # Ensure all rows have the same number of columns as headers
            max_cols = len(headers)
            normalized_rows = []
            for row in rows:
                # Pad or truncate row to match header length
                normalized_row = row[:max_cols] + [''] * (max_cols - len(row))
                normalized_rows.append(normalized_row)
            
            return {
                'page_number': page_num,
                'headers': headers,
                'rows': normalized_rows,
                'row_count': len(normalized_rows),
                'column_count': len(headers),
                'raw_text': '\n'.join([line for _, line in table_lines])
            }
            
        except Exception as e:
            logger.error(f"Error parsing table: {e}")
            return None
    
    def _identify_separator(self, lines: List[str]) -> str:
        """
        Identify the most likely column separator.
        
        Args:
            lines: List of table lines
            
        Returns:
            Separator type ('spaces', 'pipes', 'tabs', or 'unknown')
        """
        # Count separator occurrences
        separator_counts = {
            'spaces': 0,
            'pipes': 0,
            'tabs': 0
        }
        
        for line in lines:
            separator_counts['spaces'] += len(re.findall(r'\s{2,}', line))
            separator_counts['pipes'] += line.count('|')
            separator_counts['tabs'] += line.count('\t')
        
        # Return the most common separator
        if separator_counts['pipes'] > 0:
            return 'pipes'
        elif separator_counts['tabs'] > 0:
            return 'tabs'
        elif separator_counts['spaces'] > 0:
            return 'spaces'
        else:
            return 'spaces'  # Default fallback
    
    def export_table_to_dict(self, table: Dict[str, Any]) -> Dict[str, List]:
        """
        Export table to dictionary format.
        
        Args:
            table: Table data
            
        Returns:
            Dictionary with column names as keys
        """
        if not table or 'headers' not in table or 'rows' not in table:
            return {}
        
        result = {}
        headers = table['headers']
        rows = table['rows']
        
        for i, header in enumerate(headers):
            column_data = []
            for row in rows:
                if i < len(row):
                    column_data.append(row[i])
                else:
                    column_data.append('')
            result[header] = column_data
        
        return result
    
    def search_table_content(self, tables: List[Dict[str, Any]], search_term: str) -> List[Dict[str, Any]]:
        """
        Search for specific content within tables.
        
        Args:
            tables: List of table data
            search_term: Term to search for
            
        Returns:
            List of matching tables with highlighted content
        """
        matches = []
        
        for table in tables:
            found_matches = []
            
            # Search headers
            for i, header in enumerate(table.get('headers', [])):
                if search_term.lower() in header.lower():
                    found_matches.append({
                        'type': 'header',
                        'column': i,
                        'content': header
                    })
            
            # Search rows
            for row_idx, row in enumerate(table.get('rows', [])):
                for col_idx, cell in enumerate(row):
                    if search_term.lower() in cell.lower():
                        found_matches.append({
                            'type': 'cell',
                            'row': row_idx,
                            'column': col_idx,
                            'content': cell
                        })
            
            if found_matches:
                table_copy = table.copy()
                table_copy['matches'] = found_matches
                matches.append(table_copy)
        
        return matches
    
    def get_table_statistics(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about extracted tables.
        
        Args:
            tables: List of table data
            
        Returns:
            Dictionary with table statistics
        """
        if not tables:
            return {
                'total_tables': 0,
                'total_rows': 0,
                'total_columns': 0,
                'pages_with_tables': 0
            }
        
        total_rows = sum(table.get('row_count', 0) for table in tables)
        total_columns = sum(table.get('column_count', 0) for table in tables)
        pages_with_tables = len(set(table.get('page_number', 0) for table in tables))
        
        # Find largest table
        largest_table = max(tables, key=lambda t: t.get('row_count', 0) * t.get('column_count', 0))
        
        return {
            'total_tables': len(tables),
            'total_rows': total_rows,
            'total_columns': total_columns,
            'pages_with_tables': pages_with_tables,
            'average_rows_per_table': total_rows / len(tables),
            'average_columns_per_table': total_columns / len(tables),
            'largest_table': {
                'page': largest_table.get('page_number', 0),
                'rows': largest_table.get('row_count', 0),
                'columns': largest_table.get('column_count', 0)
            }
        }