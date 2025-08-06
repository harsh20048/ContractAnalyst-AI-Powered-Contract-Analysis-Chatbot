"""
Pattern Learning Module for PDF Table Extraction System
Implements machine learning capabilities to improve extraction accuracy based on user corrections.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, Counter
import re
import sqlite3
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ExtractionPattern:
    """Represents a learned extraction pattern."""
    pattern_id: str
    pattern_type: str  # 'field' or 'table'
    pattern_name: str
    pattern_data: Dict[str, Any]
    confidence_score: float
    usage_count: int
    success_rate: float
    created_date: str
    last_used: str

@dataclass
class FieldPattern:
    """Represents a learned field extraction pattern."""
    field_name: str
    extraction_rules: List[str]
    validation_rules: List[str]
    common_values: List[str]
    confidence_threshold: float
    data_type: str

@dataclass
class TablePattern:
    """Represents a learned table extraction pattern."""
    table_type: str
    header_patterns: List[str]
    structure_rules: Dict[str, Any]
    validation_rules: List[str]
    common_formats: List[Dict[str, Any]]

class PatternLearner:
    """Learns patterns from user corrections to improve future extractions."""
    
    def __init__(self, database_manager):
        """
        Initialize the pattern learner.
        
        Args:
            database_manager: DatabaseManager instance for pattern storage
        """
        self.db_manager = database_manager
        self.learned_patterns = {}
        self.field_patterns = {}
        self.table_patterns = {}
        self.confidence_threshold = 0.7
        self.min_usage_count = 3
        self.load_existing_patterns()
    
    def load_existing_patterns(self):
        """Load existing patterns from the database."""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT pattern_type, pattern_name, pattern_data, usage_count, 
                       success_rate, created_date, last_used
                FROM pattern_learning 
                WHERE is_active = 1
                ORDER BY success_rate DESC, usage_count DESC
            """)
            
            for row in cursor.fetchall():
                pattern_type, pattern_name, pattern_data, usage_count, success_rate, created_date, last_used = row
                
                pattern_id = self._generate_pattern_id(pattern_type, pattern_name)
                pattern = ExtractionPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_type,
                    pattern_name=pattern_name,
                    pattern_data=json.loads(pattern_data),
                    confidence_score=success_rate,
                    usage_count=usage_count,
                    success_rate=success_rate,
                    created_date=created_date,
                    last_used=last_used
                )
                
                self.learned_patterns[pattern_id] = pattern
                
                if pattern_type == 'field':
                    self.field_patterns[pattern_name] = pattern
                elif pattern_type == 'table':
                    self.table_patterns[pattern_name] = pattern
            
            logger.info(f"Loaded {len(self.learned_patterns)} existing patterns")
            
        except Exception as e:
            logger.error(f"Error loading existing patterns: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def learn_from_corrections(self, document_hash: str, field_data: Dict[str, Any], 
                             table_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Learn new patterns from user corrections.
        
        Args:
            document_hash: Unique document identifier
            field_data: Corrected field data
            table_data: Corrected table data
            
        Returns:
            Dictionary with learning results and confidence scores
        """
        learning_results = {
            'fields_learned': 0,
            'tables_learned': 0,
            'patterns_updated': 0,
            'avg_confidence': 0.0
        }
        
        try:
            # Learn field patterns
            if field_data:
                field_results = self._learn_field_patterns(document_hash, field_data)
                learning_results['fields_learned'] = field_results['patterns_learned']
                learning_results['patterns_updated'] += field_results['patterns_updated']
            
            # Learn table patterns
            if table_data:
                table_results = self._learn_table_patterns(document_hash, table_data)
                learning_results['tables_learned'] = table_results['patterns_learned']
                learning_results['patterns_updated'] += table_results['patterns_updated']
            
            # Calculate average confidence
            if self.learned_patterns:
                total_confidence = sum(p.confidence_score for p in self.learned_patterns.values())
                learning_results['avg_confidence'] = total_confidence / len(self.learned_patterns)
            
            # Save patterns to database
            self._save_patterns_to_db()
            
            logger.info(f"Learning completed: {learning_results}")
            return learning_results
            
        except Exception as e:
            logger.error(f"Error in pattern learning: {e}")
            return learning_results
    
    def _learn_field_patterns(self, document_hash: str, field_data: Dict[str, Any]) -> Dict[str, int]:
        """Learn patterns from field corrections."""
        results = {'patterns_learned': 0, 'patterns_updated': 0}
        
        for field_name, field_value in field_data.items():
            try:
                # Skip internal fields
                if field_name.startswith('_') or field_name == 'confidence_score':
                    continue
                
                # Analyze field pattern
                field_pattern = self._analyze_field_pattern(field_name, field_value)
                
                if field_name in self.field_patterns:
                    # Update existing pattern
                    existing_pattern = self.field_patterns[field_name]
                    updated_pattern = self._merge_field_patterns(existing_pattern, field_pattern)
                    self.field_patterns[field_name] = updated_pattern
                    self.learned_patterns[updated_pattern.pattern_id] = updated_pattern
                    results['patterns_updated'] += 1
                else:
                    # Create new pattern
                    pattern_id = self._generate_pattern_id('field', field_name)
                    new_pattern = ExtractionPattern(
                        pattern_id=pattern_id,
                        pattern_type='field',
                        pattern_name=field_name,
                        pattern_data=field_pattern.pattern_data,
                        confidence_score=0.8,  # Initial confidence
                        usage_count=1,
                        success_rate=0.8,
                        created_date=datetime.now().isoformat(),
                        last_used=datetime.now().isoformat()
                    )
                    
                    self.field_patterns[field_name] = new_pattern
                    self.learned_patterns[pattern_id] = new_pattern
                    results['patterns_learned'] += 1
                
            except Exception as e:
                logger.error(f"Error learning field pattern for {field_name}: {e}")
        
        return results
    
    def _learn_table_patterns(self, document_hash: str, table_data: Dict[str, Any]) -> Dict[str, int]:
        """Learn patterns from table corrections."""
        results = {'patterns_learned': 0, 'patterns_updated': 0}
        
        for table_name, table_content in table_data.items():
            try:
                if not isinstance(table_content, dict) or 'headers' not in table_content:
                    continue
                
                # Analyze table pattern
                table_pattern = self._analyze_table_pattern(table_name, table_content)
                
                table_type = table_content.get('metadata', {}).get('table_type', table_name)
                
                if table_type in self.table_patterns:
                    # Update existing pattern
                    existing_pattern = self.table_patterns[table_type]
                    updated_pattern = self._merge_table_patterns(existing_pattern, table_pattern)
                    self.table_patterns[table_type] = updated_pattern
                    self.learned_patterns[updated_pattern.pattern_id] = updated_pattern
                    results['patterns_updated'] += 1
                else:
                    # Create new pattern
                    pattern_id = self._generate_pattern_id('table', table_type)
                    new_pattern = ExtractionPattern(
                        pattern_id=pattern_id,
                        pattern_type='table',
                        pattern_name=table_type,
                        pattern_data=table_pattern.pattern_data,
                        confidence_score=0.8,  # Initial confidence
                        usage_count=1,
                        success_rate=0.8,
                        created_date=datetime.now().isoformat(),
                        last_used=datetime.now().isoformat()
                    )
                    
                    self.table_patterns[table_type] = new_pattern
                    self.learned_patterns[pattern_id] = new_pattern
                    results['patterns_learned'] += 1
                
            except Exception as e:
                logger.error(f"Error learning table pattern for {table_name}: {e}")
        
        return results
    
    def _analyze_field_pattern(self, field_name: str, field_value: Any) -> FieldPattern:
        """Analyze a field to extract learning patterns."""
        field_str = str(field_value)
        
        # Determine data type
        data_type = self._determine_data_type(field_value)
        
        # Extract patterns based on field type
        extraction_rules = []
        validation_rules = []
        
        if data_type == 'date':
            extraction_rules.append('date_pattern')
            validation_rules.append('date_format_check')
        elif data_type == 'currency':
            extraction_rules.append('currency_pattern')
            validation_rules.append('currency_format_check')
        elif data_type == 'number':
            extraction_rules.append('number_pattern')
            validation_rules.append('number_range_check')
        elif data_type == 'email':
            extraction_rules.append('email_pattern')
            validation_rules.append('email_format_check')
        else:
            extraction_rules.append('text_pattern')
            validation_rules.append('text_length_check')
        
        # Add field-specific rules based on field name
        if 'phone' in field_name.lower():
            extraction_rules.append('phone_pattern')
            validation_rules.append('phone_format_check')
        elif 'address' in field_name.lower():
            extraction_rules.append('address_pattern')
            validation_rules.append('address_completeness_check')
        
        return FieldPattern(
            field_name=field_name,
            extraction_rules=extraction_rules,
            validation_rules=validation_rules,
            common_values=[field_str],
            confidence_threshold=0.7,
            data_type=data_type
        )
    
    def _analyze_table_pattern(self, table_name: str, table_content: Dict[str, Any]) -> TablePattern:
        """Analyze a table to extract learning patterns."""
        headers = table_content.get('headers', [])
        rows = table_content.get('rows', [])
        metadata = table_content.get('metadata', {})
        
        # Analyze header patterns
        header_patterns = []
        for header in headers:
            header_patterns.append(self._normalize_header_name(header))
        
        # Analyze structure
        structure_rules = {
            'min_columns': len(headers),
            'max_columns': len(headers),
            'expected_headers': header_patterns,
            'data_types': {},
            'column_order': headers
        }
        
        # Analyze data types for each column
        if rows:
            for col_idx, header in enumerate(headers):
                column_values = []
                for row in rows[:5]:  # Sample first 5 rows
                    if col_idx < len(row):
                        column_values.append(row[col_idx])
                
                if column_values:
                    structure_rules['data_types'][header] = self._analyze_column_type(column_values)
        
        # Generate validation rules
        validation_rules = [
            'header_presence_check',
            'column_count_check',
            'data_type_consistency_check'
        ]
        
        # Add specific validation based on table type
        table_type = metadata.get('table_type', 'unknown')
        if 'financial' in table_type.lower():
            validation_rules.append('financial_data_validation')
        elif 'employee' in table_type.lower():
            validation_rules.append('employee_data_validation')
        
        return TablePattern(
            table_type=table_type,
            header_patterns=header_patterns,
            structure_rules=structure_rules,
            validation_rules=validation_rules,
            common_formats=[{
                'headers': headers,
                'column_count': len(headers),
                'sample_data': rows[:3] if rows else []
            }]
        )
    
    def get_learned_patterns(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """
        Get learned patterns that might apply to a document.
        
        Args:
            document_hash: Document identifier
            
        Returns:
            Dictionary containing applicable patterns or None
        """
        try:
            # For now, return all high-confidence patterns
            # In a real implementation, this would analyze the document to determine applicable patterns
            
            applicable_patterns = {
                'field_patterns': {},
                'table_patterns': {}
            }
            
            # Get high-confidence field patterns
            for field_name, pattern in self.field_patterns.items():
                if (pattern.confidence_score >= self.confidence_threshold and 
                    pattern.usage_count >= self.min_usage_count):
                    applicable_patterns['field_patterns'][field_name] = pattern.pattern_data
            
            # Get high-confidence table patterns
            for table_type, pattern in self.table_patterns.items():
                if (pattern.confidence_score >= self.confidence_threshold and 
                    pattern.usage_count >= self.min_usage_count):
                    applicable_patterns['table_patterns'][table_type] = pattern.pattern_data
            
            if applicable_patterns['field_patterns'] or applicable_patterns['table_patterns']:
                return applicable_patterns
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting learned patterns: {e}")
            return None
    
    def update_patterns(self, document_hash: str, field_data: Dict[str, Any], table_data: Dict[str, Any]):
        """
        Update existing patterns with new usage data.
        
        Args:
            document_hash: Document identifier
            field_data: Field data
            table_data: Table data
        """
        try:
            # Update field pattern usage
            for field_name in field_data.keys():
                if field_name in self.field_patterns:
                    pattern = self.field_patterns[field_name]
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now().isoformat()
                    # Update success rate (simplified calculation)
                    pattern.success_rate = min(0.95, pattern.success_rate + 0.01)
                    pattern.confidence_score = pattern.success_rate
            
            # Update table pattern usage
            for table_name, table_content in table_data.items():
                table_type = table_content.get('metadata', {}).get('table_type', table_name)
                if table_type in self.table_patterns:
                    pattern = self.table_patterns[table_type]
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now().isoformat()
                    # Update success rate (simplified calculation)
                    pattern.success_rate = min(0.95, pattern.success_rate + 0.01)
                    pattern.confidence_score = pattern.success_rate
            
            # Save updated patterns
            self._save_patterns_to_db()
            
        except Exception as e:
            logger.error(f"Error updating patterns: {e}")
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the learning system.
        
        Returns:
            Dictionary containing learning statistics
        """
        try:
            stats = {
                'total_patterns': len(self.learned_patterns),
                'field_patterns': len(self.field_patterns),
                'table_patterns': len(self.table_patterns),
                'avg_confidence': 0.0,
                'high_confidence_patterns': 0,
                'pattern_count': len(self.learned_patterns),
                'most_used_patterns': [],
                'recent_patterns': []
            }
            
            if self.learned_patterns:
                # Calculate average confidence
                total_confidence = sum(p.confidence_score for p in self.learned_patterns.values())
                stats['avg_confidence'] = total_confidence / len(self.learned_patterns)
                
                # Count high confidence patterns
                stats['high_confidence_patterns'] = sum(
                    1 for p in self.learned_patterns.values() 
                    if p.confidence_score >= self.confidence_threshold
                )
                
                # Get most used patterns
                sorted_patterns = sorted(
                    self.learned_patterns.values(), 
                    key=lambda p: p.usage_count, 
                    reverse=True
                )
                stats['most_used_patterns'] = [
                    {'name': p.pattern_name, 'usage_count': p.usage_count, 'type': p.pattern_type}
                    for p in sorted_patterns[:5]
                ]
                
                # Get recent patterns
                recent_patterns = sorted(
                    self.learned_patterns.values(),
                    key=lambda p: p.created_date,
                    reverse=True
                )
                stats['recent_patterns'] = [
                    {'name': p.pattern_name, 'created_date': p.created_date, 'type': p.pattern_type}
                    for p in recent_patterns[:5]
                ]
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {}
    
    def suggest_improvements(self, field_data: Dict[str, Any], table_data: Dict[str, Any]) -> List[str]:
        """
        Suggest improvements based on learned patterns.
        
        Args:
            field_data: Current field data
            table_data: Current table data
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            # Check field data against learned patterns
            for field_name, field_value in field_data.items():
                if field_name in self.field_patterns:
                    pattern = self.field_patterns[field_name]
                    # Simple validation against learned pattern
                    if not self._validate_field_against_pattern(field_value, pattern):
                        suggestions.append(f"Field '{field_name}' might need correction based on learned patterns")
            
            # Check table data against learned patterns
            for table_name, table_content in table_data.items():
                table_type = table_content.get('metadata', {}).get('table_type', table_name)
                if table_type in self.table_patterns:
                    pattern = self.table_patterns[table_type]
                    # Simple validation against learned pattern
                    if not self._validate_table_against_pattern(table_content, pattern):
                        suggestions.append(f"Table '{table_name}' structure differs from learned patterns")
            
        except Exception as e:
            logger.error(f"Error generating suggestions: {e}")
        
        return suggestions
    
    def _save_patterns_to_db(self):
        """Save all patterns to the database."""
        try:
            conn = sqlite3.connect(self.db_manager.db_path)
            cursor = conn.cursor()
            
            for pattern in self.learned_patterns.values():
                cursor.execute("""
                    INSERT OR REPLACE INTO pattern_learning 
                    (pattern_type, pattern_name, pattern_data, usage_count, success_rate, 
                     created_date, last_used, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.pattern_type,
                    pattern.pattern_name,
                    json.dumps(pattern.pattern_data),
                    pattern.usage_count,
                    pattern.success_rate,
                    pattern.created_date,
                    pattern.last_used,
                    1  # is_active
                ))
            
            conn.commit()
            logger.debug("Patterns saved to database")
            
        except Exception as e:
            logger.error(f"Error saving patterns to database: {e}")
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _generate_pattern_id(self, pattern_type: str, pattern_name: str) -> str:
        """Generate a unique pattern ID."""
        content = f"{pattern_type}_{pattern_name}_{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _determine_data_type(self, value: Any) -> str:
        """Determine the data type of a field value."""
        if value is None:
            return 'null'
        
        value_str = str(value).strip()
        
        # Check for currency
        if re.match(r'^\$?[\d,]+\.?\d*$', value_str) or 'USD' in value_str.upper():
            return 'currency'
        
        # Check for date
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}'
        ]
        for pattern in date_patterns:
            if re.match(pattern, value_str):
                return 'date'
        
        # Check for email
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_str):
            return 'email'
        
        # Check for number
        try:
            float(value_str.replace(',', ''))
            return 'number'
        except ValueError:
            pass
        
        return 'text'
    
    def _normalize_header_name(self, header: str) -> str:
        """Normalize header name for pattern matching."""
        return re.sub(r'[^\w\s]', '', header.lower().strip()).replace(' ', '_')
    
    def _analyze_column_type(self, values: List[Any]) -> str:
        """Analyze the data type of a column based on sample values."""
        types = [self._determine_data_type(val) for val in values if val is not None]
        if not types:
            return 'text'
        
        # Return most common type
        type_counts = Counter(types)
        return type_counts.most_common(1)[0][0]
    
    def _merge_field_patterns(self, existing: ExtractionPattern, new: FieldPattern) -> ExtractionPattern:
        """Merge an existing field pattern with new data."""
        # Simple merge - in practice, this would be more sophisticated
        existing.usage_count += 1
        existing.last_used = datetime.now().isoformat()
        
        # Update pattern data
        if 'common_values' in existing.pattern_data:
            existing.pattern_data['common_values'].extend(new.common_values)
            # Keep only unique values, limited to recent ones
            existing.pattern_data['common_values'] = list(set(existing.pattern_data['common_values']))[-10:]
        
        return existing
    
    def _merge_table_patterns(self, existing: ExtractionPattern, new: TablePattern) -> ExtractionPattern:
        """Merge an existing table pattern with new data."""
        # Simple merge - in practice, this would be more sophisticated
        existing.usage_count += 1
        existing.last_used = datetime.now().isoformat()
        
        # Update pattern data
        if 'common_formats' in existing.pattern_data:
            existing.pattern_data['common_formats'].extend(new.common_formats)
            # Keep only recent formats
            existing.pattern_data['common_formats'] = existing.pattern_data['common_formats'][-5:]
        
        return existing
    
    def _validate_field_against_pattern(self, field_value: Any, pattern: ExtractionPattern) -> bool:
        """Validate a field value against a learned pattern."""
        # Simple validation - in practice, this would be more comprehensive
        try:
            pattern_data = pattern.pattern_data
            if 'data_type' in pattern_data:
                expected_type = pattern_data['data_type']
                actual_type = self._determine_data_type(field_value)
                return expected_type == actual_type
            return True
        except Exception:
            return True
    
    def _validate_table_against_pattern(self, table_content: Dict[str, Any], pattern: ExtractionPattern) -> bool:
        """Validate a table against a learned pattern."""
        # Simple validation - in practice, this would be more comprehensive
        try:
            headers = table_content.get('headers', [])
            pattern_data = pattern.pattern_data
            
            if 'expected_headers' in pattern_data:
                expected_headers = set(pattern_data['expected_headers'])
                actual_headers = set(self._normalize_header_name(h) for h in headers)
                # Check if at least 70% of expected headers are present
                overlap = len(expected_headers.intersection(actual_headers))
                return overlap / len(expected_headers) >= 0.7 if expected_headers else True
            
            return True
        except Exception:
            return True
    
    def export_patterns(self, export_path: str) -> bool:
        """
        Export learned patterns to a file.
        
        Args:
            export_path: Path to export file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'total_patterns': len(self.learned_patterns),
                'patterns': {}
            }
            
            for pattern_id, pattern in self.learned_patterns.items():
                export_data['patterns'][pattern_id] = {
                    'pattern_type': pattern.pattern_type,
                    'pattern_name': pattern.pattern_name,
                    'pattern_data': pattern.pattern_data,
                    'confidence_score': pattern.confidence_score,
                    'usage_count': pattern.usage_count,
                    'success_rate': pattern.success_rate,
                    'created_date': pattern.created_date,
                    'last_used': pattern.last_used
                }
            
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.info(f"Patterns exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting patterns: {e}")
            return False
    
    def import_patterns(self, import_path: str) -> bool:
        """
        Import patterns from a file.
        
        Args:
            import_path: Path to import file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(import_path, 'r') as f:
                import_data = json.load(f)
            
            patterns_imported = 0
            for pattern_id, pattern_data in import_data.get('patterns', {}).items():
                pattern = ExtractionPattern(
                    pattern_id=pattern_id,
                    pattern_type=pattern_data['pattern_type'],
                    pattern_name=pattern_data['pattern_name'],
                    pattern_data=pattern_data['pattern_data'],
                    confidence_score=pattern_data['confidence_score'],
                    usage_count=pattern_data['usage_count'],
                    success_rate=pattern_data['success_rate'],
                    created_date=pattern_data['created_date'],
                    last_used=pattern_data['last_used']
                )
                
                self.learned_patterns[pattern_id] = pattern
                
                if pattern.pattern_type == 'field':
                    self.field_patterns[pattern.pattern_name] = pattern
                elif pattern.pattern_type == 'table':
                    self.table_patterns[pattern.pattern_name] = pattern
                
                patterns_imported += 1
            
            # Save imported patterns to database
            self._save_patterns_to_db()
            
            logger.info(f"Imported {patterns_imported} patterns from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error importing patterns: {e}")
            return False