"""
Pattern Learning Module for PDF Table Extraction System
======================================================

Comprehensive machine learning module for adaptive pattern learning including:
- Field pattern recognition and learning
- Table structure analysis and pattern storage
- Confidence scoring and validation
- Pattern application and suggestion
- Performance tracking and optimization
- Smart fallback mechanisms

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import json
import logging
import re
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FieldPattern:
    """Data class for field patterns"""
    pattern_id: str
    field_name: str
    data_type: str
    regex_pattern: str
    validation_rules: List[str]
    extraction_confidence: float
    usage_count: int
    success_rate: float
    examples: List[str]
    created_date: datetime
    last_used: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TablePattern:
    """Data class for table patterns"""
    pattern_id: str
    table_type: str
    column_structure: List[str]
    header_patterns: List[str]
    data_patterns: Dict[str, str]
    validation_rules: List[str]
    extraction_confidence: float
    usage_count: int
    success_rate: float
    examples: List[Dict[str, Any]]
    created_date: datetime
    last_used: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LearningResult:
    """Data class for learning operation results"""
    patterns_learned: int
    patterns_updated: int
    field_patterns: int
    table_patterns: int
    success: bool
    processing_time: float
    errors: List[str]
    metadata: Dict[str, Any]

class PatternLearnerError(Exception):
    """Custom exception for pattern learning operations"""
    pass

class PatternLearner:
    """Advanced pattern learning system for PDF extraction optimization"""
    
    def __init__(self, database_manager, min_pattern_confidence: float = 0.5,
                 max_patterns_per_type: int = 1000):
        """
        Initialize pattern learner
        
        Args:
            database_manager: Database manager instance
            min_pattern_confidence: Minimum confidence for pattern acceptance
            max_patterns_per_type: Maximum patterns to store per type
        """
        self.db_manager = database_manager
        self.min_pattern_confidence = min_pattern_confidence
        self.max_patterns_per_type = max_patterns_per_type
        
        # Pattern storage
        self.field_patterns: Dict[str, FieldPattern] = {}
        self.table_patterns: Dict[str, TablePattern] = {}
        
        # Learning statistics
        self.learning_stats = {
            'total_corrections_analyzed': 0,
            'patterns_created': 0,
            'patterns_updated': 0,
            'field_patterns_count': 0,
            'table_patterns_count': 0,
            'avg_pattern_confidence': 0.0,
            'last_learning_session': None
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Load existing patterns
        self._load_existing_patterns()
        
        logger.info("PatternLearner initialized")
    
    def _load_existing_patterns(self):
        """Load existing patterns from database"""
        try:
            with self.lock:
                # Load field patterns
                field_patterns = self.db_manager.load_patterns(pattern_type='field')
                for pattern_data in field_patterns:
                    pattern = self._deserialize_field_pattern(pattern_data)
                    if pattern:
                        self.field_patterns[pattern.pattern_id] = pattern
                
                # Load table patterns
                table_patterns = self.db_manager.load_patterns(pattern_type='table')
                for pattern_data in table_patterns:
                    pattern = self._deserialize_table_pattern(pattern_data)
                    if pattern:
                        self.table_patterns[pattern.pattern_id] = pattern
                
                logger.info(f"Loaded {len(self.field_patterns)} field patterns and {len(self.table_patterns)} table patterns")
                
        except Exception as e:
            logger.error(f"Failed to load existing patterns: {e}")
    
    def learn_from_corrections(self, document_hash: str, fields: Dict[str, Any], 
                              tables: List[pd.DataFrame], metadata: Dict[str, Any] = None) -> LearningResult:
        """
        Learn patterns from user corrections
        
        Args:
            document_hash: Document identifier
            fields: Corrected field data
            tables: Corrected table data
            metadata: Additional metadata
            
        Returns:
            LearningResult: Results of the learning operation
        """
        start_time = datetime.now()
        errors = []
        patterns_learned = 0
        patterns_updated = 0
        
        try:
            with self.lock:
                logger.info(f"Learning from corrections for document {document_hash[:8]}...")
                
                # Learn field patterns
                field_results = self._learn_field_patterns(fields, document_hash, metadata)
                patterns_learned += field_results['created']
                patterns_updated += field_results['updated']
                errors.extend(field_results['errors'])
                
                # Learn table patterns
                table_results = self._learn_table_patterns(tables, document_hash, metadata)
                patterns_learned += table_results['created']
                patterns_updated += table_results['updated']
                errors.extend(table_results['errors'])
                
                # Update learning statistics
                self.learning_stats['total_corrections_analyzed'] += 1
                self.learning_stats['patterns_created'] += patterns_learned
                self.learning_stats['patterns_updated'] += patterns_updated
                self.learning_stats['field_patterns_count'] = len(self.field_patterns)
                self.learning_stats['table_patterns_count'] = len(self.table_patterns)
                self.learning_stats['last_learning_session'] = datetime.now()
                
                # Calculate average confidence
                all_patterns = list(self.field_patterns.values()) + list(self.table_patterns.values())
                if all_patterns:
                    avg_confidence = sum(p.extraction_confidence for p in all_patterns) / len(all_patterns)
                    self.learning_stats['avg_pattern_confidence'] = avg_confidence
                
                # Save updated patterns to database
                self._save_patterns_to_database()
                
                processing_time = (datetime.now() - start_time).total_seconds()
                
                return LearningResult(
                    patterns_learned=patterns_learned,
                    patterns_updated=patterns_updated,
                    field_patterns=field_results['created'] + field_results['updated'],
                    table_patterns=table_results['created'] + table_results['updated'],
                    success=True,
                    processing_time=processing_time,
                    errors=errors,
                    metadata={
                        'document_hash': document_hash,
                        'field_count': len(fields),
                        'table_count': len(tables)
                    }
                )
                
        except Exception as e:
            logger.error(f"Learning failed for document {document_hash}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return LearningResult(
                patterns_learned=0,
                patterns_updated=0,
                field_patterns=0,
                table_patterns=0,
                success=False,
                processing_time=processing_time,
                errors=[str(e)],
                metadata={'document_hash': document_hash}
            )
    
    def _learn_field_patterns(self, fields: Dict[str, Any], document_hash: str,
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn patterns from field data"""
        results = {'created': 0, 'updated': 0, 'errors': []}
        
        try:
            for field_name, field_value in fields.items():
                if field_value is None or str(field_value).strip() == '':
                    continue
                
                try:
                    # Analyze field characteristics
                    analysis = self._analyze_field(field_name, field_value)
                    
                    # Generate pattern ID
                    pattern_id = self._generate_field_pattern_id(field_name, analysis['data_type'])
                    
                    if pattern_id in self.field_patterns:
                        # Update existing pattern
                        pattern = self.field_patterns[pattern_id]
                        pattern.usage_count += 1
                        pattern.last_used = datetime.now()
                        
                        # Update examples (keep only most recent 10)
                        pattern.examples.append(str(field_value))
                        if len(pattern.examples) > 10:
                            pattern.examples = pattern.examples[-10:]
                        
                        # Recalculate confidence based on consistency
                        pattern.extraction_confidence = self._calculate_field_confidence(pattern)
                        
                        results['updated'] += 1
                        logger.debug(f"Updated field pattern {pattern_id}")
                        
                    else:
                        # Create new pattern
                        pattern = FieldPattern(
                            pattern_id=pattern_id,
                            field_name=field_name,
                            data_type=analysis['data_type'],
                            regex_pattern=analysis['regex_pattern'],
                            validation_rules=analysis['validation_rules'],
                            extraction_confidence=0.7,  # Initial confidence
                            usage_count=1,
                            success_rate=1.0,
                            examples=[str(field_value)],
                            created_date=datetime.now(),
                            last_used=datetime.now(),
                            metadata={
                                'source_document': document_hash,
                                'analysis': analysis
                            }
                        )
                        
                        self.field_patterns[pattern_id] = pattern
                        results['created'] += 1
                        logger.debug(f"Created new field pattern {pattern_id}")
                
                except Exception as e:
                    error_msg = f"Error learning field pattern for {field_name}: {e}"
                    results['errors'].append(error_msg)
                    logger.warning(error_msg)
            
            return results
            
        except Exception as e:
            results['errors'].append(f"Field pattern learning failed: {e}")
            return results
    
    def _learn_table_patterns(self, tables: List[pd.DataFrame], document_hash: str,
                             metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Learn patterns from table data"""
        results = {'created': 0, 'updated': 0, 'errors': []}
        
        try:
            for i, table in enumerate(tables):
                if table.empty:
                    continue
                
                try:
                    # Analyze table characteristics
                    analysis = self._analyze_table(table, i)
                    
                    # Generate pattern ID
                    pattern_id = self._generate_table_pattern_id(analysis['table_type'], analysis['structure_hash'])
                    
                    if pattern_id in self.table_patterns:
                        # Update existing pattern
                        pattern = self.table_patterns[pattern_id]
                        pattern.usage_count += 1
                        pattern.last_used = datetime.now()
                        
                        # Update examples (keep only most recent 5)
                        table_example = table.head(3).to_dict('records')
                        pattern.examples.append(table_example)
                        if len(pattern.examples) > 5:
                            pattern.examples = pattern.examples[-5:]
                        
                        # Recalculate confidence
                        pattern.extraction_confidence = self._calculate_table_confidence(pattern, table)
                        
                        results['updated'] += 1
                        logger.debug(f"Updated table pattern {pattern_id}")
                        
                    else:
                        # Create new pattern
                        pattern = TablePattern(
                            pattern_id=pattern_id,
                            table_type=analysis['table_type'],
                            column_structure=analysis['column_structure'],
                            header_patterns=analysis['header_patterns'],
                            data_patterns=analysis['data_patterns'],
                            validation_rules=analysis['validation_rules'],
                            extraction_confidence=0.7,  # Initial confidence
                            usage_count=1,
                            success_rate=1.0,
                            examples=[table.head(3).to_dict('records')],
                            created_date=datetime.now(),
                            last_used=datetime.now(),
                            metadata={
                                'source_document': document_hash,
                                'table_index': i,
                                'analysis': analysis
                            }
                        )
                        
                        self.table_patterns[pattern_id] = pattern
                        results['created'] += 1
                        logger.debug(f"Created new table pattern {pattern_id}")
                
                except Exception as e:
                    error_msg = f"Error learning table pattern for table {i}: {e}"
                    results['errors'].append(error_msg)
                    logger.warning(error_msg)
            
            return results
            
        except Exception as e:
            results['errors'].append(f"Table pattern learning failed: {e}")
            return results
    
    def _analyze_field(self, field_name: str, field_value: Any) -> Dict[str, Any]:
        """Analyze field characteristics for pattern creation"""
        value_str = str(field_value).strip()
        
        analysis = {
            'data_type': self._determine_data_type(value_str),
            'regex_pattern': self._generate_regex_pattern(value_str),
            'validation_rules': self._generate_validation_rules(field_name, value_str),
            'length': len(value_str),
            'has_digits': bool(re.search(r'\d', value_str)),
            'has_letters': bool(re.search(r'[a-zA-Z]', value_str)),
            'has_special_chars': bool(re.search(r'[^a-zA-Z0-9\s]', value_str)),
            'normalized_name': self._normalize_field_name(field_name)
        }
        
        return analysis
    
    def _analyze_table(self, table: pd.DataFrame, table_index: int) -> Dict[str, Any]:
        """Analyze table characteristics for pattern creation"""
        columns = table.columns.tolist()
        
        # Determine table type based on column names and content
        table_type = self._classify_table_type(columns, table)
        
        # Analyze column structure
        column_structure = []
        data_patterns = {}
        
        for col in columns:
            col_analysis = self._analyze_column(table[col])
            column_structure.append(col_analysis['data_type'])
            data_patterns[col] = col_analysis['pattern']
        
        # Generate header patterns
        header_patterns = [self._normalize_header(col) for col in columns]
        
        # Create structure hash for similarity detection
        structure_hash = hashlib.md5(
            json.dumps(sorted(header_patterns)).encode()
        ).hexdigest()[:8]
        
        analysis = {
            'table_type': table_type,
            'column_structure': column_structure,
            'header_patterns': header_patterns,
            'data_patterns': data_patterns,
            'validation_rules': self._generate_table_validation_rules(table),
            'structure_hash': structure_hash,
            'row_count': len(table),
            'column_count': len(columns),
            'density': table.count().sum() / (len(table) * len(columns)) if len(table) > 0 else 0
        }
        
        return analysis
    
    def _determine_data_type(self, value: str) -> str:
        """Determine data type of a field value"""
        value = value.strip()
        
        # Date patterns
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b'  # Month DD, YYYY
        ]
        
        for pattern in date_patterns:
            if re.match(pattern, value, re.IGNORECASE):
                return 'date'
        
        # Currency
        if re.match(r'^\$?[\d,]+\.?\d*$', value) or re.search(r'[$€£¥₹]', value):
            return 'currency'
        
        # Percentage
        if re.match(r'^\d+\.?\d*%$', value):
            return 'percentage'
        
        # Email
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value):
            return 'email'
        
        # Phone
        phone_cleaned = re.sub(r'[\s\-\(\)\+\.]', '', value)
        if len(phone_cleaned) >= 10 and phone_cleaned.isdigit():
            return 'phone'
        
        # URL
        if re.match(r'^https?://', value, re.IGNORECASE):
            return 'url'
        
        # Integer
        if re.match(r'^-?\d+$', value):
            return 'integer'
        
        # Float
        if re.match(r'^-?\d+\.\d+$', value):
            return 'float'
        
        # Boolean
        if value.lower() in ['true', 'false', 'yes', 'no', '1', '0']:
            return 'boolean'
        
        # Default to string
        return 'string'
    
    def _generate_regex_pattern(self, value: str) -> str:
        """Generate regex pattern for a field value"""
        data_type = self._determine_data_type(value)
        
        patterns = {
            'date': r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}',
            'currency': r'\$?[\d,]+\.?\d*',
            'percentage': r'\d+\.?\d*%',
            'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
            'phone': r'[\d\s\-\(\)\+\.]{10,}',
            'url': r'https?://[^\s]+',
            'integer': r'-?\d+',
            'float': r'-?\d+\.\d+',
            'boolean': r'true|false|yes|no|1|0',
            'string': r'.+'
        }
        
        return patterns.get(data_type, r'.+')
    
    def _generate_validation_rules(self, field_name: str, value: str) -> List[str]:
        """Generate validation rules for a field"""
        rules = []
        data_type = self._determine_data_type(value)
        
        # Required field rule
        if value.strip():
            rules.append('required')
        
        # Type-specific rules
        if data_type == 'email':
            rules.append('valid_email')
        elif data_type == 'phone':
            rules.append('valid_phone')
        elif data_type == 'url':
            rules.append('valid_url')
        elif data_type in ['integer', 'float', 'currency']:
            rules.append('numeric')
            if data_type == 'currency':
                rules.append('positive')
        elif data_type == 'percentage':
            rules.append('percentage_range')
        
        # Field name-based rules
        field_lower = field_name.lower()
        if 'amount' in field_lower or 'total' in field_lower or 'price' in field_lower:
            rules.append('positive_number')
        elif 'date' in field_lower:
            rules.append('valid_date')
        elif 'email' in field_lower:
            rules.append('valid_email')
        
        return rules
    
    def _classify_table_type(self, columns: List[str], table: pd.DataFrame) -> str:
        """Classify table type based on columns and content"""
        column_names = [col.lower() for col in columns]
        
        # Financial/Invoice table
        financial_keywords = ['amount', 'price', 'total', 'cost', 'invoice', 'payment']
        if any(keyword in ' '.join(column_names) for keyword in financial_keywords):
            return 'financial'
        
        # Employee/Contact table
        contact_keywords = ['name', 'email', 'phone', 'address', 'employee']
        if any(keyword in ' '.join(column_names) for keyword in contact_keywords):
            return 'contact'
        
        # Inventory/Product table
        inventory_keywords = ['product', 'item', 'quantity', 'stock', 'inventory']
        if any(keyword in ' '.join(column_names) for keyword in inventory_keywords):
            return 'inventory'
        
        # Schedule/Time table
        time_keywords = ['date', 'time', 'schedule', 'appointment', 'meeting']
        if any(keyword in ' '.join(column_names) for keyword in time_keywords):
            return 'schedule'
        
        # Default to generic
        return 'generic'
    
    def _analyze_column(self, column: pd.Series) -> Dict[str, Any]:
        """Analyze column characteristics"""
        non_null_values = column.dropna().astype(str)
        
        if len(non_null_values) == 0:
            return {'data_type': 'string', 'pattern': r'.*'}
        
        # Sample a few values to determine type
        sample_values = non_null_values.head(5).tolist()
        
        # Determine most common data type
        type_counts = Counter()
        for value in sample_values:
            data_type = self._determine_data_type(value)
            type_counts[data_type] += 1
        
        most_common_type = type_counts.most_common(1)[0][0]
        pattern = self._generate_regex_pattern(sample_values[0])
        
        return {
            'data_type': most_common_type,
            'pattern': pattern,
            'sample_values': sample_values
        }
    
    def _generate_table_validation_rules(self, table: pd.DataFrame) -> List[str]:
        """Generate validation rules for table"""
        rules = []
        
        # Basic structure rules
        rules.append(f'min_rows:{max(1, len(table) // 2)}')
        rules.append(f'max_rows:{len(table) * 2}')
        rules.append(f'column_count:{len(table.columns)}')
        
        # Content rules
        numeric_columns = table.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            rules.append('has_numeric_data')
        
        # Check for required columns based on table type
        column_names = [col.lower() for col in table.columns]
        if any('total' in col or 'amount' in col for col in column_names):
            rules.append('has_totals')
        
        return rules
    
    def _normalize_field_name(self, field_name: str) -> str:
        """Normalize field name for pattern matching"""
        return re.sub(r'[^a-zA-Z0-9]', '_', field_name.lower()).strip('_')
    
    def _normalize_header(self, header: str) -> str:
        """Normalize table header for pattern matching"""
        return re.sub(r'[^a-zA-Z0-9]', '_', header.lower()).strip('_')
    
    def _generate_field_pattern_id(self, field_name: str, data_type: str) -> str:
        """Generate unique pattern ID for field"""
        normalized_name = self._normalize_field_name(field_name)
        return f"field_{normalized_name}_{data_type}"
    
    def _generate_table_pattern_id(self, table_type: str, structure_hash: str) -> str:
        """Generate unique pattern ID for table"""
        return f"table_{table_type}_{structure_hash}"
    
    def _calculate_field_confidence(self, pattern: FieldPattern) -> float:
        """Calculate confidence score for field pattern"""
        base_confidence = 0.5
        
        # Usage factor (more usage = higher confidence)
        usage_factor = min(0.3, pattern.usage_count * 0.01)
        
        # Success rate factor
        success_factor = pattern.success_rate * 0.2
        
        # Consistency factor (based on examples)
        consistency_factor = self._calculate_field_consistency(pattern) * 0.2
        
        confidence = base_confidence + usage_factor + success_factor + consistency_factor
        return min(1.0, max(0.0, confidence))
    
    def _calculate_table_confidence(self, pattern: TablePattern, table: pd.DataFrame) -> float:
        """Calculate confidence score for table pattern"""
        base_confidence = 0.5
        
        # Usage factor
        usage_factor = min(0.3, pattern.usage_count * 0.01)
        
        # Success rate factor
        success_factor = pattern.success_rate * 0.2
        
        # Structure consistency factor
        structure_factor = self._calculate_table_consistency(pattern, table) * 0.3
        
        confidence = base_confidence + usage_factor + success_factor + structure_factor
        return min(1.0, max(0.0, confidence))
    
    def _calculate_field_consistency(self, pattern: FieldPattern) -> float:
        """Calculate consistency score for field pattern examples"""
        if len(pattern.examples) < 2:
            return 0.5
        
        # Check data type consistency
        consistent_types = 0
        for example in pattern.examples:
            if self._determine_data_type(example) == pattern.data_type:
                consistent_types += 1
        
        return consistent_types / len(pattern.examples)
    
    def _calculate_table_consistency(self, pattern: TablePattern, table: pd.DataFrame) -> float:
        """Calculate consistency score for table pattern"""
        consistency_score = 0.0
        factors = 0
        
        # Column count consistency
        if len(table.columns) == len(pattern.column_structure):
            consistency_score += 0.5
        factors += 1
        
        # Header pattern consistency
        table_headers = [self._normalize_header(col) for col in table.columns]
        matching_headers = sum(1 for h1, h2 in zip(table_headers, pattern.header_patterns) if h1 == h2)
        if len(pattern.header_patterns) > 0:
            consistency_score += (matching_headers / len(pattern.header_patterns)) * 0.5
            factors += 1
        
        return consistency_score / factors if factors > 0 else 0.0
    
    def get_learned_patterns(self, document_hash: str = None, 
                           pattern_type: str = None) -> Dict[str, Any]:
        """
        Get learned patterns applicable to a document
        
        Args:
            document_hash: Document identifier (for context-specific patterns)
            pattern_type: Filter by pattern type ('field' or 'table')
            
        Returns:
            Dictionary of applicable patterns
        """
        try:
            with self.lock:
                result = {
                    'field_patterns': {},
                    'table_patterns': {},
                    'metadata': {
                        'total_patterns': len(self.field_patterns) + len(self.table_patterns),
                        'field_count': len(self.field_patterns),
                        'table_count': len(self.table_patterns),
                        'min_confidence_threshold': self.min_pattern_confidence
                    }
                }
                
                # Get field patterns
                if pattern_type is None or pattern_type == 'field':
                    for pattern_id, pattern in self.field_patterns.items():
                        if pattern.extraction_confidence >= self.min_pattern_confidence:
                            result['field_patterns'][pattern_id] = self._serialize_field_pattern(pattern)
                
                # Get table patterns
                if pattern_type is None or pattern_type == 'table':
                    for pattern_id, pattern in self.table_patterns.items():
                        if pattern.extraction_confidence >= self.min_pattern_confidence:
                            result['table_patterns'][pattern_id] = self._serialize_table_pattern(pattern)
                
                return result
                
        except Exception as e:
            logger.error(f"Failed to get learned patterns: {e}")
            return {'field_patterns': {}, 'table_patterns': {}, 'metadata': {}}
    
    def suggest_improvements(self, fields: Dict[str, Any], 
                           tables: List[pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Suggest improvements based on learned patterns
        
        Args:
            fields: Current field data
            tables: Current table data
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        try:
            with self.lock:
                # Field suggestions
                for field_name, field_value in fields.items():
                    field_suggestions = self._get_field_suggestions(field_name, field_value)
                    suggestions.extend(field_suggestions)
                
                # Table suggestions
                for i, table in enumerate(tables):
                    table_suggestions = self._get_table_suggestions(table, i)
                    suggestions.extend(table_suggestions)
                
                return suggestions
                
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    def _get_field_suggestions(self, field_name: str, field_value: Any) -> List[Dict[str, Any]]:
        """Get suggestions for field improvements"""
        suggestions = []
        value_str = str(field_value).strip()
        
        # Find similar patterns
        similar_patterns = self._find_similar_field_patterns(field_name, value_str)
        
        for pattern in similar_patterns:
            # Validation suggestions
            if not self._validate_field_against_pattern(value_str, pattern):
                suggestions.append({
                    'type': 'field_validation',
                    'field_name': field_name,
                    'current_value': value_str,
                    'suggested_pattern': pattern.regex_pattern,
                    'confidence': pattern.extraction_confidence,
                    'reason': f"Value doesn't match learned pattern for {pattern.data_type} fields"
                })
            
            # Format suggestions
            if pattern.data_type in ['date', 'currency', 'percentage']:
                formatted_value = self._suggest_format_improvement(value_str, pattern.data_type)
                if formatted_value != value_str:
                    suggestions.append({
                        'type': 'field_format',
                        'field_name': field_name,
                        'current_value': value_str,
                        'suggested_value': formatted_value,
                        'confidence': pattern.extraction_confidence,
                        'reason': f"Improved formatting for {pattern.data_type}"
                    })
        
        return suggestions
    
    def _get_table_suggestions(self, table: pd.DataFrame, table_index: int) -> List[Dict[str, Any]]:
        """Get suggestions for table improvements"""
        suggestions = []
        
        # Find similar table patterns
        similar_patterns = self._find_similar_table_patterns(table)
        
        for pattern in similar_patterns:
            # Column structure suggestions
            if len(table.columns) != len(pattern.column_structure):
                suggestions.append({
                    'type': 'table_structure',
                    'table_index': table_index,
                    'current_columns': len(table.columns),
                    'suggested_columns': len(pattern.column_structure),
                    'confidence': pattern.extraction_confidence,
                    'reason': f"Column count mismatch with {pattern.table_type} pattern"
                })
            
            # Header suggestions
            table_headers = [self._normalize_header(col) for col in table.columns]
            for i, (current, expected) in enumerate(zip(table_headers, pattern.header_patterns)):
                if current != expected and pattern.extraction_confidence > 0.8:
                    suggestions.append({
                        'type': 'table_header',
                        'table_index': table_index,
                        'column_index': i,
                        'current_header': table.columns[i],
                        'suggested_header': expected,
                        'confidence': pattern.extraction_confidence,
                        'reason': f"Header doesn't match {pattern.table_type} pattern"
                    })
        
        return suggestions
    
    def _find_similar_field_patterns(self, field_name: str, field_value: str) -> List[FieldPattern]:
        """Find field patterns similar to current field"""
        similar_patterns = []
        normalized_name = self._normalize_field_name(field_name)
        data_type = self._determine_data_type(field_value)
        
        for pattern in self.field_patterns.values():
            # Exact match on normalized name and data type
            if (pattern.metadata and 
                pattern.metadata.get('analysis', {}).get('normalized_name') == normalized_name and
                pattern.data_type == data_type):
                similar_patterns.append(pattern)
            
            # Partial match on name similarity
            elif (self._calculate_name_similarity(normalized_name, pattern.field_name) > 0.7 and
                  pattern.data_type == data_type):
                similar_patterns.append(pattern)
        
        # Sort by confidence
        return sorted(similar_patterns, key=lambda p: p.extraction_confidence, reverse=True)
    
    def _find_similar_table_patterns(self, table: pd.DataFrame) -> List[TablePattern]:
        """Find table patterns similar to current table"""
        similar_patterns = []
        table_headers = [self._normalize_header(col) for col in table.columns]
        
        for pattern in self.table_patterns.values():
            # Calculate header similarity
            similarity = self._calculate_header_similarity(table_headers, pattern.header_patterns)
            
            if similarity > 0.6:  # Minimum 60% similarity
                similar_patterns.append(pattern)
        
        # Sort by confidence and similarity
        return sorted(similar_patterns, key=lambda p: p.extraction_confidence, reverse=True)
    
    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two field names"""
        if not name1 or not name2:
            return 0.0
        
        # Simple Jaccard similarity on character sets
        set1 = set(name1.lower())
        set2 = set(name2.lower())
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_header_similarity(self, headers1: List[str], headers2: List[str]) -> float:
        """Calculate similarity between two header lists"""
        if not headers1 or not headers2:
            return 0.0
        
        matches = 0
        for h1 in headers1:
            for h2 in headers2:
                if self._calculate_name_similarity(h1, h2) > 0.8:
                    matches += 1
                    break
        
        return matches / max(len(headers1), len(headers2))
    
    def _validate_field_against_pattern(self, value: str, pattern: FieldPattern) -> bool:
        """Validate field value against pattern"""
        try:
            return bool(re.match(pattern.regex_pattern, value))
        except:
            return False
    
    def _suggest_format_improvement(self, value: str, data_type: str) -> str:
        """Suggest format improvements for field value"""
        if data_type == 'date':
            # Try to standardize date format
            date_patterns = [
                (r'(\d{2})/(\d{2})/(\d{4})', r'\3-\1-\2'),  # MM/DD/YYYY -> YYYY-MM-DD
                (r'(\d{1,2})/(\d{1,2})/(\d{4})', r'\3-\1-\2'),  # M/D/YYYY -> YYYY-M-D
            ]
            
            for pattern, replacement in date_patterns:
                if re.match(pattern, value):
                    return re.sub(pattern, replacement, value)
        
        elif data_type == 'currency':
            # Standardize currency format
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned:
                return f"${float(cleaned):.2f}"
        
        elif data_type == 'percentage':
            # Standardize percentage format
            cleaned = re.sub(r'[^\d.]', '', value)
            if cleaned:
                return f"{float(cleaned):.1f}%"
        
        return value
    
    def update_pattern_performance(self, pattern_id: str, success: bool) -> bool:
        """
        Update pattern performance metrics
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the pattern application was successful
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                pattern = None
                
                # Find pattern in field patterns
                if pattern_id in self.field_patterns:
                    pattern = self.field_patterns[pattern_id]
                # Find pattern in table patterns
                elif pattern_id in self.table_patterns:
                    pattern = self.table_patterns[pattern_id]
                
                if pattern:
                    # Update usage count
                    pattern.usage_count += 1
                    pattern.last_used = datetime.now()
                    
                    # Update success rate
                    current_successes = pattern.success_rate * (pattern.usage_count - 1)
                    if success:
                        current_successes += 1
                    
                    pattern.success_rate = current_successes / pattern.usage_count
                    
                    # Recalculate confidence
                    if isinstance(pattern, FieldPattern):
                        pattern.extraction_confidence = self._calculate_field_confidence(pattern)
                    else:
                        # For table patterns, we need the original table (not available here)
                        # So we use a simplified confidence calculation
                        pattern.extraction_confidence = (
                            pattern.extraction_confidence * 0.9 + pattern.success_rate * 0.1
                        )
                    
                    # Update in database
                    self.db_manager.update_pattern_usage(pattern_id, success)
                    
                    logger.debug(f"Updated pattern {pattern_id} performance: success={success}")
                    return True
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to update pattern performance: {e}")
            return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """
        Get learning statistics and performance metrics
        
        Returns:
            Dictionary of learning statistics
        """
        try:
            with self.lock:
                stats = self.learning_stats.copy()
                
                # Add current pattern counts
                stats['current_field_patterns'] = len(self.field_patterns)
                stats['current_table_patterns'] = len(self.table_patterns)
                
                # Calculate pattern usage statistics
                if self.field_patterns:
                    field_usage = [p.usage_count for p in self.field_patterns.values()]
                    stats['avg_field_pattern_usage'] = sum(field_usage) / len(field_usage)
                    stats['max_field_pattern_usage'] = max(field_usage)
                
                if self.table_patterns:
                    table_usage = [p.usage_count for p in self.table_patterns.values()]
                    stats['avg_table_pattern_usage'] = sum(table_usage) / len(table_usage)
                    stats['max_table_pattern_usage'] = max(table_usage)
                
                # Pattern age analysis
                now = datetime.now()
                recent_patterns = 0
                for pattern in list(self.field_patterns.values()) + list(self.table_patterns.values()):
                    if (now - pattern.created_date).days < 7:
                        recent_patterns += 1
                
                stats['recent_patterns_count'] = recent_patterns
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get learning statistics: {e}")
            return {}
    
    def _save_patterns_to_database(self):
        """Save current patterns to database"""
        try:
            # Save field patterns
            for pattern in self.field_patterns.values():
                pattern_data = self._serialize_field_pattern(pattern)
                self.db_manager.save_pattern(
                    pattern_id=pattern.pattern_id,
                    pattern_type='field',
                    pattern_name=pattern.field_name,
                    pattern_data=pattern_data,
                    confidence=pattern.extraction_confidence,
                    metadata=pattern.metadata
                )
            
            # Save table patterns
            for pattern in self.table_patterns.values():
                pattern_data = self._serialize_table_pattern(pattern)
                self.db_manager.save_pattern(
                    pattern_id=pattern.pattern_id,
                    pattern_type='table',
                    pattern_name=pattern.table_type,
                    pattern_data=pattern_data,
                    confidence=pattern.extraction_confidence,
                    metadata=pattern.metadata
                )
            
            logger.debug("Patterns saved to database")
            
        except Exception as e:
            logger.error(f"Failed to save patterns to database: {e}")
    
    def _serialize_field_pattern(self, pattern: FieldPattern) -> Dict[str, Any]:
        """Serialize field pattern to dictionary"""
        return {
            'pattern_id': pattern.pattern_id,
            'field_name': pattern.field_name,
            'data_type': pattern.data_type,
            'regex_pattern': pattern.regex_pattern,
            'validation_rules': pattern.validation_rules,
            'extraction_confidence': pattern.extraction_confidence,
            'usage_count': pattern.usage_count,
            'success_rate': pattern.success_rate,
            'examples': pattern.examples,
            'created_date': pattern.created_date.isoformat(),
            'last_used': pattern.last_used.isoformat() if pattern.last_used else None,
            'metadata': pattern.metadata
        }
    
    def _serialize_table_pattern(self, pattern: TablePattern) -> Dict[str, Any]:
        """Serialize table pattern to dictionary"""
        return {
            'pattern_id': pattern.pattern_id,
            'table_type': pattern.table_type,
            'column_structure': pattern.column_structure,
            'header_patterns': pattern.header_patterns,
            'data_patterns': pattern.data_patterns,
            'validation_rules': pattern.validation_rules,
            'extraction_confidence': pattern.extraction_confidence,
            'usage_count': pattern.usage_count,
            'success_rate': pattern.success_rate,
            'examples': pattern.examples,
            'created_date': pattern.created_date.isoformat(),
            'last_used': pattern.last_used.isoformat() if pattern.last_used else None,
            'metadata': pattern.metadata
        }
    
    def _deserialize_field_pattern(self, data: Dict[str, Any]) -> Optional[FieldPattern]:
        """Deserialize field pattern from dictionary"""
        try:
            pattern_data = data.get('pattern_data', {})
            if isinstance(pattern_data, str):
                pattern_data = json.loads(pattern_data)
            
            return FieldPattern(
                pattern_id=pattern_data.get('pattern_id'),
                field_name=pattern_data.get('field_name'),
                data_type=pattern_data.get('data_type'),
                regex_pattern=pattern_data.get('regex_pattern'),
                validation_rules=pattern_data.get('validation_rules', []),
                extraction_confidence=pattern_data.get('extraction_confidence', 0.0),
                usage_count=pattern_data.get('usage_count', 0),
                success_rate=pattern_data.get('success_rate', 0.0),
                examples=pattern_data.get('examples', []),
                created_date=datetime.fromisoformat(pattern_data.get('created_date')),
                last_used=datetime.fromisoformat(pattern_data.get('last_used')) if pattern_data.get('last_used') else None,
                metadata=pattern_data.get('metadata')
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize field pattern: {e}")
            return None
    
    def _deserialize_table_pattern(self, data: Dict[str, Any]) -> Optional[TablePattern]:
        """Deserialize table pattern from dictionary"""
        try:
            pattern_data = data.get('pattern_data', {})
            if isinstance(pattern_data, str):
                pattern_data = json.loads(pattern_data)
            
            return TablePattern(
                pattern_id=pattern_data.get('pattern_id'),
                table_type=pattern_data.get('table_type'),
                column_structure=pattern_data.get('column_structure', []),
                header_patterns=pattern_data.get('header_patterns', []),
                data_patterns=pattern_data.get('data_patterns', {}),
                validation_rules=pattern_data.get('validation_rules', []),
                extraction_confidence=pattern_data.get('extraction_confidence', 0.0),
                usage_count=pattern_data.get('usage_count', 0),
                success_rate=pattern_data.get('success_rate', 0.0),
                examples=pattern_data.get('examples', []),
                created_date=datetime.fromisoformat(pattern_data.get('created_date')),
                last_used=datetime.fromisoformat(pattern_data.get('last_used')) if pattern_data.get('last_used') else None,
                metadata=pattern_data.get('metadata')
            )
            
        except Exception as e:
            logger.error(f"Failed to deserialize table pattern: {e}")
            return None
    
    def cleanup_patterns(self, min_usage_count: int = 5, max_age_days: int = 90) -> int:
        """
        Clean up unused or old patterns
        
        Args:
            min_usage_count: Minimum usage count to keep pattern
            max_age_days: Maximum age in days to keep pattern
            
        Returns:
            int: Number of patterns removed
        """
        try:
            with self.lock:
                removed_count = 0
                cutoff_date = datetime.now() - timedelta(days=max_age_days)
                
                # Clean field patterns
                patterns_to_remove = []
                for pattern_id, pattern in self.field_patterns.items():
                    if (pattern.usage_count < min_usage_count and 
                        pattern.created_date < cutoff_date):
                        patterns_to_remove.append(pattern_id)
                
                for pattern_id in patterns_to_remove:
                    del self.field_patterns[pattern_id]
                    removed_count += 1
                
                # Clean table patterns
                patterns_to_remove = []
                for pattern_id, pattern in self.table_patterns.items():
                    if (pattern.usage_count < min_usage_count and 
                        pattern.created_date < cutoff_date):
                        patterns_to_remove.append(pattern_id)
                
                for pattern_id in patterns_to_remove:
                    del self.table_patterns[pattern_id]
                    removed_count += 1
                
                logger.info(f"Cleaned up {removed_count} unused patterns")
                return removed_count
                
        except Exception as e:
            logger.error(f"Pattern cleanup failed: {e}")
            return 0
    
    def export_patterns(self, export_path: str) -> bool:
        """
        Export patterns to file
        
        Args:
            export_path: Path to export file
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                export_data = {
                    'field_patterns': [self._serialize_field_pattern(p) for p in self.field_patterns.values()],
                    'table_patterns': [self._serialize_table_pattern(p) for p in self.table_patterns.values()],
                    'learning_stats': self.learning_stats,
                    'export_timestamp': datetime.now().isoformat(),
                    'export_version': '2.0.0'
                }
                
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
                
                logger.info(f"Patterns exported to {export_path}")
                return True
                
        except Exception as e:
            logger.error(f"Pattern export failed: {e}")
            return False
    
    def import_patterns(self, import_path: str) -> bool:
        """
        Import patterns from file
        
        Args:
            import_path: Path to import file
            
        Returns:
            bool: Success status
        """
        try:
            with self.lock:
                with open(import_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
                
                imported_count = 0
                
                # Import field patterns
                for pattern_data in import_data.get('field_patterns', []):
                    try:
                        # Convert back to proper format for deserialization
                        mock_data = {'pattern_data': pattern_data}
                        pattern = self._deserialize_field_pattern(mock_data)
                        if pattern:
                            self.field_patterns[pattern.pattern_id] = pattern
                            imported_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import field pattern: {e}")
                
                # Import table patterns
                for pattern_data in import_data.get('table_patterns', []):
                    try:
                        # Convert back to proper format for deserialization
                        mock_data = {'pattern_data': pattern_data}
                        pattern = self._deserialize_table_pattern(mock_data)
                        if pattern:
                            self.table_patterns[pattern.pattern_id] = pattern
                            imported_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to import table pattern: {e}")
                
                # Update learning stats if available
                if 'learning_stats' in import_data:
                    self.learning_stats.update(import_data['learning_stats'])
                
                logger.info(f"Imported {imported_count} patterns from {import_path}")
                return True
                
        except Exception as e:
            logger.error(f"Pattern import failed: {e}")
            return False