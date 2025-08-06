#!/usr/bin/env python3
"""
Pattern Learning Module for PDF Table Extraction System
=======================================================

Advanced machine learning module for adaptive pattern learning and extraction improvement.
Learns from user corrections to enhance future extraction accuracy.

Author: AI Assistant
Version: 3.0.0
License: MIT
"""

import json
import logging
import re
import hashlib
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict, Counter
import math
import statistics
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

# Constants
PATTERN_VERSION = "3.0.0"
MIN_PATTERN_CONFIDENCE = 0.3
MIN_PATTERN_USAGE = 3
LEARNING_RATE = 0.1
DECAY_FACTOR = 0.95
MAX_PATTERNS_PER_TYPE = 100

class PatternType(Enum):
    """Types of patterns that can be learned"""
    FIELD_PATTERN = "field_pattern"
    TABLE_PATTERN = "table_pattern"
    DOCUMENT_STRUCTURE = "document_structure"
    VALUE_FORMAT = "value_format"
    EXTRACTION_HINT = "extraction_hint"
    VALIDATION_RULE = "validation_rule"
    SEMANTIC_PATTERN = "semantic_pattern"

class LearningMode(Enum):
    """Learning modes for pattern extraction"""
    CONSERVATIVE = "conservative"  # Only learn from high-confidence corrections
    BALANCED = "balanced"         # Balance between learning and stability
    AGGRESSIVE = "aggressive"     # Learn from all corrections

@dataclass
class FieldPattern:
    """Field pattern data structure"""
    field_name: str
    pattern_type: str
    regex_pattern: Optional[str]
    position_hints: Dict[str, Any]
    value_type: str
    validation_rules: List[str]
    confidence: float
    source_documents: List[str]
    usage_count: int
    success_count: int

@dataclass
class TablePattern:
    """Table pattern data structure"""
    table_signature: str
    column_patterns: Dict[str, Any]
    row_patterns: Dict[str, Any]
    structural_hints: Dict[str, Any]
    extraction_rules: List[str]
    confidence: float
    source_documents: List[str]
    usage_count: int
    success_count: int

@dataclass
class LearningInsight:
    """Learning insight data structure"""
    insight_type: str
    description: str
    confidence: float
    supporting_evidence: List[str]
    suggested_actions: List[str]
    created_at: datetime

class PatternLearner:
    """Advanced pattern learning and extraction optimization"""
    
    def __init__(self, database_manager, learning_mode: LearningMode = LearningMode.BALANCED):
        """
        Initialize pattern learner
        
        Args:
            database_manager: DatabaseManager instance
            learning_mode: Learning mode (conservative, balanced, aggressive)
        """
        self.db_manager = database_manager
        self.learning_mode = learning_mode
        
        # Pattern storage
        self.field_patterns: Dict[str, FieldPattern] = {}
        self.table_patterns: Dict[str, TablePattern] = {}
        self.document_patterns: Dict[str, Any] = {}
        
        # Learning statistics
        self.learning_stats = {
            'patterns_learned': 0,
            'patterns_applied': 0,
            'successful_applications': 0,
            'last_learning_session': None,
            'confidence_improvements': []
        }
        
        # Load existing patterns
        self._load_existing_patterns()
        
        logger.info(f"PatternLearner initialized in {learning_mode.value} mode")
    
    def learn_from_corrections(self, document_hash: str, field_data: Dict[str, Any], 
                             table_data: List[Dict[str, Any]], 
                             original_extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Learn patterns from user corrections
        
        Args:
            document_hash: Document identifier
            field_data: Corrected field data
            table_data: Corrected table data
            original_extraction: Original extraction before corrections
            
        Returns:
            Dictionary with learning results and statistics
        """
        try:
            learning_results = {
                'field_patterns_learned': 0,
                'table_patterns_learned': 0,
                'patterns_updated': 0,
                'insights_generated': 0,
                'confidence_improvement': 0.0
            }
            
            # Learn field patterns
            field_learning = self._learn_field_patterns(document_hash, field_data, original_extraction)
            learning_results['field_patterns_learned'] = field_learning['patterns_learned']
            learning_results['patterns_updated'] += field_learning['patterns_updated']
            
            # Learn table patterns
            table_learning = self._learn_table_patterns(document_hash, table_data, original_extraction)
            learning_results['table_patterns_learned'] = table_learning['patterns_learned']
            learning_results['patterns_updated'] += table_learning['patterns_updated']
            
            # Generate insights
            insights = self._generate_learning_insights(document_hash, field_data, table_data)
            learning_results['insights_generated'] = len(insights)
            
            # Update learning statistics
            self._update_learning_stats(learning_results)
            
            # Save patterns to database
            self._save_patterns_to_database()
            
            logger.info(f"Learning completed for document {document_hash}: {learning_results}")
            return learning_results
            
        except Exception as e:
            logger.error(f"Error learning from corrections: {str(e)}")
            return {'error': str(e)}
    
    def _learn_field_patterns(self, document_hash: str, field_data: Dict[str, Any], 
                            original_extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn patterns from field corrections"""
        results = {'patterns_learned': 0, 'patterns_updated': 0}
        
        for field_name, corrected_value in field_data.items():
            try:
                # Analyze field characteristics
                field_analysis = self._analyze_field_value(field_name, corrected_value)
                
                # Check if pattern exists
                pattern_id = self._generate_field_pattern_id(field_name, field_analysis)
                
                if pattern_id in self.field_patterns:
                    # Update existing pattern
                    pattern = self.field_patterns[pattern_id]
                    self._update_field_pattern(pattern, corrected_value, document_hash)
                    results['patterns_updated'] += 1
                else:
                    # Create new pattern
                    pattern = self._create_field_pattern(field_name, corrected_value, document_hash, field_analysis)
                    if pattern:
                        self.field_patterns[pattern_id] = pattern
                        results['patterns_learned'] += 1
                
                # Learn positional patterns if original extraction available
                if original_extraction:
                    self._learn_field_position_patterns(field_name, corrected_value, original_extraction)
                
            except Exception as e:
                logger.warning(f"Error learning pattern for field {field_name}: {str(e)}")
        
        return results
    
    def _learn_table_patterns(self, document_hash: str, table_data: List[Dict[str, Any]], 
                            original_extraction: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Learn patterns from table corrections"""
        results = {'patterns_learned': 0, 'patterns_updated': 0}
        
        if not table_data:
            return results
        
        try:
            # Analyze table structure
            table_analysis = self._analyze_table_structure(table_data)
            
            # Generate table signature
            table_signature = self._generate_table_signature(table_analysis)
            
            if table_signature in self.table_patterns:
                # Update existing pattern
                pattern = self.table_patterns[table_signature]
                self._update_table_pattern(pattern, table_data, document_hash)
                results['patterns_updated'] += 1
            else:
                # Create new pattern
                pattern = self._create_table_pattern(table_signature, table_data, document_hash, table_analysis)
                if pattern:
                    self.table_patterns[table_signature] = pattern
                    results['patterns_learned'] += 1
            
            # Learn column patterns
            column_patterns = self._learn_column_patterns(table_data)
            results['patterns_learned'] += len(column_patterns)
            
        except Exception as e:
            logger.warning(f"Error learning table patterns: {str(e)}")
        
        return results
    
    def _analyze_field_value(self, field_name: str, value: Any) -> Dict[str, Any]:
        """Analyze field value to extract patterns"""
        analysis = {
            'data_type': self._determine_data_type(value),
            'length': len(str(value)) if value is not None else 0,
            'format_pattern': self._extract_format_pattern(value),
            'semantic_type': self._determine_semantic_type(field_name, value),
            'validation_rules': self._generate_validation_rules(field_name, value),
            'regex_pattern': self._generate_regex_pattern(value),
            'normalization_hints': self._generate_normalization_hints(value)
        }
        
        return analysis
    
    def _analyze_table_structure(self, table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze table structure to extract patterns"""
        if not table_data:
            return {}
        
        # Get column information
        columns = list(table_data[0].keys()) if table_data else []
        
        analysis = {
            'column_count': len(columns),
            'row_count': len(table_data),
            'columns': columns,
            'column_types': {},
            'column_patterns': {},
            'structural_hints': {}
        }
        
        # Analyze each column
        for col in columns:
            col_values = [row.get(col) for row in table_data if row.get(col) is not None]
            if col_values:
                analysis['column_types'][col] = self._determine_column_type(col_values)
                analysis['column_patterns'][col] = self._extract_column_patterns(col, col_values)
        
        # Detect structural patterns
        analysis['structural_hints'] = self._detect_structural_patterns(table_data)
        
        return analysis
    
    def _determine_data_type(self, value: Any) -> str:
        """Determine the data type of a value"""
        if value is None:
            return 'null'
        elif isinstance(value, bool):
            return 'boolean'
        elif isinstance(value, int):
            return 'integer'
        elif isinstance(value, float):
            return 'float'
        elif isinstance(value, str):
            # Check for special string types
            value_str = value.strip().lower()
            
            if re.match(r'^\d{4}-\d{2}-\d{2}', value_str):
                return 'date'
            elif re.match(r'^[\+\-]?\$?[\d,]+\.?\d*$', value_str):
                return 'currency'
            elif re.match(r'^\d+\.?\d*%$', value_str):
                return 'percentage'
            elif re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', value_str):
                return 'email'
            elif re.match(r'^[\+]?[\d\s\-\(\)]{10,}$', value_str):
                return 'phone'
            elif re.match(r'^https?://', value_str):
                return 'url'
            else:
                return 'text'
        else:
            return 'unknown'
    
    def _determine_column_type(self, values: List[Any]) -> str:
        """Determine the most common data type in a column"""
        type_counts = Counter()
        
        for value in values:
            data_type = self._determine_data_type(value)
            type_counts[data_type] += 1
        
        # Return most common type
        return type_counts.most_common(1)[0][0] if type_counts else 'unknown'
    
    def _extract_format_pattern(self, value: Any) -> str:
        """Extract format pattern from value"""
        if value is None:
            return ''
        
        value_str = str(value)
        
        # Replace digits with 'N', letters with 'A', preserve special characters
        pattern = ''
        for char in value_str:
            if char.isdigit():
                pattern += 'N'
            elif char.isalpha():
                pattern += 'A'
            else:
                pattern += char
        
        return pattern
    
    def _determine_semantic_type(self, field_name: str, value: Any) -> str:
        """Determine semantic type based on field name and value"""
        field_name_lower = field_name.lower()
        
        # Common semantic patterns
        semantic_patterns = {
            'id': ['id', 'identifier', 'number'],
            'name': ['name', 'title', 'label'],
            'address': ['address', 'location', 'street'],
            'amount': ['amount', 'price', 'cost', 'total', 'sum'],
            'date': ['date', 'time', 'created', 'updated'],
            'contact': ['email', 'phone', 'contact'],
            'status': ['status', 'state', 'condition'],
            'category': ['category', 'type', 'class', 'group'],
            'description': ['description', 'notes', 'comment', 'details']
        }
        
        for semantic_type, keywords in semantic_patterns.items():
            if any(keyword in field_name_lower for keyword in keywords):
                return semantic_type
        
        return 'generic'
    
    def _generate_validation_rules(self, field_name: str, value: Any) -> List[str]:
        """Generate validation rules for a field"""
        rules = []
        
        if value is None:
            return rules
        
        data_type = self._determine_data_type(value)
        semantic_type = self._determine_semantic_type(field_name, value)
        
        # Add type-specific validation rules
        if data_type == 'integer':
            rules.append(f"type:integer")
            if isinstance(value, int) and value >= 0:
                rules.append("min:0")
        elif data_type == 'float':
            rules.append(f"type:float")
        elif data_type == 'currency':
            rules.append("type:currency")
            rules.append("format:currency")
        elif data_type == 'email':
            rules.append("type:email")
            rules.append("format:email")
        elif data_type == 'date':
            rules.append("type:date")
            rules.append("format:date")
        
        # Add semantic validation rules
        if semantic_type == 'amount':
            rules.append("semantic:amount")
            rules.append("min:0")
        elif semantic_type == 'id':
            rules.append("semantic:identifier")
            rules.append("required:true")
        
        # Add length constraints
        if isinstance(value, str):
            length = len(value)
            if length > 0:
                rules.append(f"max_length:{min(length * 2, 255)}")  # Allow some flexibility
        
        return rules
    
    def _generate_regex_pattern(self, value: Any) -> str:
        """Generate regex pattern for value format"""
        if value is None:
            return ''
        
        value_str = str(value).strip()
        
        # Generate regex based on data type
        data_type = self._determine_data_type(value)
        
        if data_type == 'integer':
            return r'^\d+$'
        elif data_type == 'float':
            return r'^\d+\.?\d*$'
        elif data_type == 'currency':
            return r'^\$?[\d,]+\.?\d*$'
        elif data_type == 'percentage':
            return r'^\d+\.?\d*%$'
        elif data_type == 'email':
            return r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        elif data_type == 'phone':
            return r'^[\+]?[\d\s\-\(\)]{10,}$'
        elif data_type == 'date':
            return r'^\d{4}-\d{2}-\d{2}$'
        elif data_type == 'url':
            return r'^https?://[^\s/$.?#].[^\s]*$'
        else:
            # Generate pattern based on actual value structure
            pattern = ''
            for char in value_str:
                if char.isdigit():
                    pattern += r'\d'
                elif char.isalpha():
                    pattern += r'[a-zA-Z]'
                elif char in '.,;:!?':
                    pattern += re.escape(char)
                elif char.isspace():
                    pattern += r'\s'
                else:
                    pattern += re.escape(char)
            return f'^{pattern}$' if pattern else ''
    
    def _generate_normalization_hints(self, value: Any) -> List[str]:
        """Generate normalization hints for value processing"""
        hints = []
        
        if value is None:
            return hints
        
        value_str = str(value)
        
        # Common normalization patterns
        if value_str != value_str.strip():
            hints.append("trim_whitespace")
        
        if any(char.isupper() for char in value_str):
            hints.append("preserve_case")
        
        if ',' in value_str and any(char.isdigit() for char in value_str):
            hints.append("remove_number_commas")
        
        if '$' in value_str:
            hints.append("extract_currency_symbol")
        
        if '%' in value_str:
            hints.append("extract_percentage")
        
        return hints
    
    def _generate_field_pattern_id(self, field_name: str, analysis: Dict[str, Any]) -> str:
        """Generate unique pattern ID for field"""
        pattern_data = {
            'field_name': field_name,
            'data_type': analysis['data_type'],
            'semantic_type': analysis['semantic_type'],
            'format_pattern': analysis['format_pattern']
        }
        
        pattern_string = json.dumps(pattern_data, sort_keys=True)
        return hashlib.md5(pattern_string.encode()).hexdigest()
    
    def _generate_table_signature(self, analysis: Dict[str, Any]) -> str:
        """Generate unique signature for table structure"""
        signature_data = {
            'column_count': analysis['column_count'],
            'columns': sorted(analysis['columns']),
            'column_types': analysis['column_types']
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.md5(signature_string.encode()).hexdigest()
    
    def _create_field_pattern(self, field_name: str, value: Any, document_hash: str, 
                            analysis: Dict[str, Any]) -> Optional[FieldPattern]:
        """Create new field pattern"""
        try:
            # Only create pattern if confidence is sufficient
            if self.learning_mode == LearningMode.CONSERVATIVE and analysis.get('confidence', 0.5) < 0.7:
                return None
            
            pattern = FieldPattern(
                field_name=field_name,
                pattern_type=analysis['data_type'],
                regex_pattern=analysis['regex_pattern'],
                position_hints={},
                value_type=analysis['semantic_type'],
                validation_rules=analysis['validation_rules'],
                confidence=0.8,  # Initial confidence
                source_documents=[document_hash],
                usage_count=1,
                success_count=1
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Error creating field pattern: {str(e)}")
            return None
    
    def _create_table_pattern(self, signature: str, table_data: List[Dict[str, Any]], 
                            document_hash: str, analysis: Dict[str, Any]) -> Optional[TablePattern]:
        """Create new table pattern"""
        try:
            pattern = TablePattern(
                table_signature=signature,
                column_patterns=analysis['column_patterns'],
                row_patterns={},
                structural_hints=analysis['structural_hints'],
                extraction_rules=[],
                confidence=0.7,  # Initial confidence
                source_documents=[document_hash],
                usage_count=1,
                success_count=1
            )
            
            return pattern
            
        except Exception as e:
            logger.warning(f"Error creating table pattern: {str(e)}")
            return None
    
    def _update_field_pattern(self, pattern: FieldPattern, value: Any, document_hash: str):
        """Update existing field pattern with new data"""
        pattern.usage_count += 1
        pattern.success_count += 1
        
        if document_hash not in pattern.source_documents:
            pattern.source_documents.append(document_hash)
        
        # Update confidence based on usage
        pattern.confidence = min(pattern.confidence + LEARNING_RATE * (1 - pattern.confidence), 0.95)
        
        # Update validation rules if needed
        new_rules = self._generate_validation_rules(pattern.field_name, value)
        for rule in new_rules:
            if rule not in pattern.validation_rules:
                pattern.validation_rules.append(rule)
    
    def _update_table_pattern(self, pattern: TablePattern, table_data: List[Dict[str, Any]], 
                            document_hash: str):
        """Update existing table pattern with new data"""
        pattern.usage_count += 1
        pattern.success_count += 1
        
        if document_hash not in pattern.source_documents:
            pattern.source_documents.append(document_hash)
        
        # Update confidence
        pattern.confidence = min(pattern.confidence + LEARNING_RATE * (1 - pattern.confidence), 0.95)
        
        # Update column patterns
        for col in table_data[0].keys() if table_data else []:
            col_values = [row.get(col) for row in table_data if row.get(col) is not None]
            if col_values:
                new_patterns = self._extract_column_patterns(col, col_values)
                if col in pattern.column_patterns:
                    pattern.column_patterns[col].update(new_patterns)
                else:
                    pattern.column_patterns[col] = new_patterns
    
    def _extract_column_patterns(self, column_name: str, values: List[Any]) -> Dict[str, Any]:
        """Extract patterns from column values"""
        patterns = {
            'data_type': self._determine_column_type(values),
            'value_patterns': [],
            'length_distribution': {},
            'common_values': [],
            'null_percentage': 0.0
        }
        
        # Calculate null percentage
        null_count = sum(1 for v in values if v is None or str(v).strip() == '')
        patterns['null_percentage'] = null_count / len(values) if values else 0
        
        # Get length distribution
        lengths = [len(str(v)) for v in values if v is not None]
        if lengths:
            patterns['length_distribution'] = {
                'min': min(lengths),
                'max': max(lengths),
                'avg': statistics.mean(lengths),
                'std': statistics.stdev(lengths) if len(lengths) > 1 else 0
            }
        
        # Get common values
        value_counts = Counter(str(v) for v in values if v is not None)
        patterns['common_values'] = value_counts.most_common(5)
        
        # Extract format patterns
        format_patterns = [self._extract_format_pattern(v) for v in values if v is not None]
        pattern_counts = Counter(format_patterns)
        patterns['value_patterns'] = pattern_counts.most_common(3)
        
        return patterns
    
    def _detect_structural_patterns(self, table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect structural patterns in table data"""
        patterns = {}
        
        if not table_data:
            return patterns
        
        # Detect header patterns
        columns = list(table_data[0].keys())
        patterns['header_style'] = self._analyze_header_style(columns)
        
        # Detect row grouping patterns
        patterns['grouping_patterns'] = self._detect_grouping_patterns(table_data)
        
        # Detect totals/summary rows
        patterns['summary_patterns'] = self._detect_summary_patterns(table_data)
        
        return patterns
    
    def _analyze_header_style(self, headers: List[str]) -> Dict[str, Any]:
        """Analyze header naming style"""
        style = {
            'case_style': 'mixed',
            'separator_style': 'space',
            'prefix_patterns': [],
            'suffix_patterns': []
        }
        
        # Analyze case style
        if all(h.islower() for h in headers):
            style['case_style'] = 'lowercase'
        elif all(h.isupper() for h in headers):
            style['case_style'] = 'uppercase'
        elif all(h.istitle() for h in headers):
            style['case_style'] = 'title'
        
        # Analyze separator style
        if all('_' in h for h in headers):
            style['separator_style'] = 'underscore'
        elif all('-' in h for h in headers):
            style['separator_style'] = 'hyphen'
        elif all(' ' in h for h in headers):
            style['separator_style'] = 'space'
        
        return style
    
    def _detect_grouping_patterns(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect row grouping patterns"""
        patterns = []
        
        # Look for common grouping indicators
        for col in table_data[0].keys() if table_data else []:
            col_values = [row.get(col) for row in table_data]
            
            # Check for repeated values (potential grouping)
            value_positions = defaultdict(list)
            for i, value in enumerate(col_values):
                if value is not None:
                    value_positions[str(value)].append(i)
            
            # If values repeat in clusters, it might indicate grouping
            for value, positions in value_positions.items():
                if len(positions) > 1:
                    # Check if positions are clustered
                    gaps = [positions[i+1] - positions[i] for i in range(len(positions)-1)]
                    if gaps and statistics.mean(gaps) < len(table_data) / len(positions):
                        patterns.append({
                            'type': 'grouping',
                            'column': col,
                            'value': value,
                            'positions': positions
                        })
        
        return patterns
    
    def _detect_summary_patterns(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect summary/total row patterns"""
        patterns = []
        
        if len(table_data) < 2:
            return patterns
        
        # Check last few rows for summary indicators
        for i, row in enumerate(table_data[-3:], len(table_data)-3):
            for col, value in row.items():
                if value and isinstance(value, str):
                    value_lower = value.lower()
                    if any(keyword in value_lower for keyword in ['total', 'sum', 'summary', 'subtotal']):
                        patterns.append({
                            'type': 'summary',
                            'row_index': i,
                            'column': col,
                            'indicator': value
                        })
        
        return patterns
    
    def _learn_field_position_patterns(self, field_name: str, value: Any, 
                                     original_extraction: Dict[str, Any]):
        """Learn positional patterns for field extraction"""
        # This would analyze where in the document the field was found
        # and learn patterns about typical positions for similar fields
        pass
    
    def _learn_column_patterns(self, table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Learn patterns from table columns"""
        patterns = []
        
        if not table_data:
            return patterns
        
        for col in table_data[0].keys():
            col_values = [row.get(col) for row in table_data if row.get(col) is not None]
            if col_values:
                col_pattern = self._extract_column_patterns(col, col_values)
                patterns.append({
                    'column_name': col,
                    'pattern': col_pattern
                })
        
        return patterns
    
    def get_learned_patterns(self, document_hash: str = None, pattern_type: str = None) -> Dict[str, Any]:
        """
        Get learned patterns applicable to a document
        
        Args:
            document_hash: Document identifier (optional)
            pattern_type: Type of patterns to retrieve (optional)
            
        Returns:
            Dictionary with applicable patterns
        """
        try:
            patterns = {
                'field_patterns': [],
                'table_patterns': [],
                'confidence_threshold': MIN_PATTERN_CONFIDENCE,
                'pattern_count': 0
            }
            
            # Get field patterns
            for pattern_id, pattern in self.field_patterns.items():
                if pattern.confidence >= MIN_PATTERN_CONFIDENCE:
                    if pattern_type is None or pattern_type == 'field':
                        patterns['field_patterns'].append({
                            'pattern_id': pattern_id,
                            'field_name': pattern.field_name,
                            'pattern_type': pattern.pattern_type,
                            'regex_pattern': pattern.regex_pattern,
                            'validation_rules': pattern.validation_rules,
                            'confidence': pattern.confidence,
                            'usage_count': pattern.usage_count
                        })
            
            # Get table patterns
            for signature, pattern in self.table_patterns.items():
                if pattern.confidence >= MIN_PATTERN_CONFIDENCE:
                    if pattern_type is None or pattern_type == 'table':
                        patterns['table_patterns'].append({
                            'signature': signature,
                            'column_patterns': pattern.column_patterns,
                            'structural_hints': pattern.structural_hints,
                            'confidence': pattern.confidence,
                            'usage_count': pattern.usage_count
                        })
            
            patterns['pattern_count'] = len(patterns['field_patterns']) + len(patterns['table_patterns'])
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error getting learned patterns: {str(e)}")
            return {'error': str(e)}
    
    def apply_patterns(self, field_data: Dict[str, Any], table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply learned patterns to improve extraction
        
        Args:
            field_data: Field data to improve
            table_data: Table data to improve
            
        Returns:
            Dictionary with improved data and application results
        """
        try:
            results = {
                'improved_field_data': field_data.copy(),
                'improved_table_data': [row.copy() for row in table_data],
                'patterns_applied': 0,
                'improvements_made': 0,
                'confidence_boost': 0.0
            }
            
            # Apply field patterns
            field_improvements = self._apply_field_patterns(results['improved_field_data'])
            results['patterns_applied'] += field_improvements['patterns_applied']
            results['improvements_made'] += field_improvements['improvements_made']
            
            # Apply table patterns
            table_improvements = self._apply_table_patterns(results['improved_table_data'])
            results['patterns_applied'] += table_improvements['patterns_applied']
            results['improvements_made'] += table_improvements['improvements_made']
            
            # Calculate confidence boost
            if results['patterns_applied'] > 0:
                results['confidence_boost'] = min(0.1 * results['improvements_made'], 0.3)
            
            return results
            
        except Exception as e:
            logger.error(f"Error applying patterns: {str(e)}")
            return {'error': str(e)}
    
    def _apply_field_patterns(self, field_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field patterns to improve field data"""
        results = {'patterns_applied': 0, 'improvements_made': 0}
        
        for field_name, value in field_data.items():
            # Find applicable patterns
            applicable_patterns = self._find_applicable_field_patterns(field_name, value)
            
            for pattern in applicable_patterns:
                try:
                    # Apply pattern improvements
                    improved_value = self._apply_field_pattern(value, pattern)
                    if improved_value != value:
                        field_data[field_name] = improved_value
                        results['improvements_made'] += 1
                    
                    results['patterns_applied'] += 1
                    
                    # Update pattern usage
                    self._update_pattern_usage(pattern, success=True)
                    
                except Exception as e:
                    logger.warning(f"Error applying field pattern: {str(e)}")
                    self._update_pattern_usage(pattern, success=False)
        
        return results
    
    def _apply_table_patterns(self, table_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply table patterns to improve table data"""
        results = {'patterns_applied': 0, 'improvements_made': 0}
        
        if not table_data:
            return results
        
        # Find applicable table patterns
        table_analysis = self._analyze_table_structure(table_data)
        applicable_patterns = self._find_applicable_table_patterns(table_analysis)
        
        for pattern in applicable_patterns:
            try:
                # Apply structural improvements
                improvements = self._apply_table_pattern(table_data, pattern)
                results['improvements_made'] += improvements
                results['patterns_applied'] += 1
                
                # Update pattern usage
                self._update_pattern_usage(pattern, success=improvements > 0)
                
            except Exception as e:
                logger.warning(f"Error applying table pattern: {str(e)}")
                self._update_pattern_usage(pattern, success=False)
        
        return results
    
    def _find_applicable_field_patterns(self, field_name: str, value: Any) -> List[FieldPattern]:
        """Find field patterns applicable to a specific field"""
        applicable = []
        
        for pattern in self.field_patterns.values():
            if pattern.confidence >= MIN_PATTERN_CONFIDENCE:
                # Check field name match
                if pattern.field_name == field_name:
                    applicable.append(pattern)
                # Check semantic type match
                elif pattern.value_type == self._determine_semantic_type(field_name, value):
                    applicable.append(pattern)
        
        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)
        return applicable[:3]  # Return top 3 patterns
    
    def _find_applicable_table_patterns(self, table_analysis: Dict[str, Any]) -> List[TablePattern]:
        """Find table patterns applicable to a table structure"""
        applicable = []
        
        for pattern in self.table_patterns.values():
            if pattern.confidence >= MIN_PATTERN_CONFIDENCE:
                # Check structural similarity
                similarity = self._calculate_table_similarity(table_analysis, pattern)
                if similarity > 0.6:  # 60% similarity threshold
                    applicable.append(pattern)
        
        # Sort by confidence
        applicable.sort(key=lambda p: p.confidence, reverse=True)
        return applicable[:2]  # Return top 2 patterns
    
    def _apply_field_pattern(self, value: Any, pattern: FieldPattern) -> Any:
        """Apply field pattern to improve a value"""
        if value is None:
            return value
        
        improved_value = value
        
        # Apply normalization hints
        for hint in pattern.validation_rules:
            if hint == "trim_whitespace":
                improved_value = str(improved_value).strip()
            elif hint == "remove_number_commas" and isinstance(improved_value, str):
                improved_value = improved_value.replace(',', '')
            elif hint.startswith("type:"):
                # Apply type conversion
                target_type = hint.split(':')[1]
                improved_value = self._convert_to_type(improved_value, target_type)
        
        return improved_value
    
    def _apply_table_pattern(self, table_data: List[Dict[str, Any]], pattern: TablePattern) -> int:
        """Apply table pattern to improve table data"""
        improvements = 0
        
        # Apply column pattern improvements
        for row in table_data:
            for col, value in row.items():
                if col in pattern.column_patterns:
                    col_pattern = pattern.column_patterns[col]
                    improved_value = self._apply_column_pattern(value, col_pattern)
                    if improved_value != value:
                        row[col] = improved_value
                        improvements += 1
        
        return improvements
    
    def _apply_column_pattern(self, value: Any, pattern: Dict[str, Any]) -> Any:
        """Apply column pattern to improve a value"""
        if value is None:
            return value
        
        improved_value = value
        
        # Apply data type conversion
        target_type = pattern.get('data_type')
        if target_type:
            improved_value = self._convert_to_type(improved_value, target_type)
        
        return improved_value
    
    def _convert_to_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type"""
        try:
            if target_type == 'integer':
                # Remove common formatting
                clean_value = str(value).replace(',', '').replace('$', '').strip()
                return int(float(clean_value))
            elif target_type == 'float':
                clean_value = str(value).replace(',', '').replace('$', '').strip()
                return float(clean_value)
            elif target_type == 'currency':
                clean_value = str(value).replace(',', '').replace('$', '').strip()
                return float(clean_value)
            else:
                return value
        except (ValueError, TypeError):
            return value
    
    def _calculate_table_similarity(self, analysis1: Dict[str, Any], pattern: TablePattern) -> float:
        """Calculate similarity between table analysis and pattern"""
        similarity = 0.0
        
        # Compare column count
        if analysis1.get('column_count') == len(pattern.column_patterns):
            similarity += 0.3
        
        # Compare column names
        columns1 = set(analysis1.get('columns', []))
        columns2 = set(pattern.column_patterns.keys())
        if columns1 and columns2:
            overlap = len(columns1.intersection(columns2))
            similarity += 0.4 * (overlap / max(len(columns1), len(columns2)))
        
        # Compare column types
        types1 = analysis1.get('column_types', {})
        if types1:
            type_matches = 0
            for col, type1 in types1.items():
                if col in pattern.column_patterns:
                    col_pattern = pattern.column_patterns[col]
                    if col_pattern.get('data_type') == type1:
                        type_matches += 1
            similarity += 0.3 * (type_matches / len(types1))
        
        return similarity
    
    def _update_pattern_usage(self, pattern: Union[FieldPattern, TablePattern], success: bool):
        """Update pattern usage statistics"""
        pattern.usage_count += 1
        if success:
            pattern.success_count += 1
        
        # Recalculate confidence
        success_rate = pattern.success_count / pattern.usage_count
        pattern.confidence = pattern.confidence * DECAY_FACTOR + success_rate * (1 - DECAY_FACTOR)
    
    def _generate_learning_insights(self, document_hash: str, field_data: Dict[str, Any], 
                                  table_data: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Generate insights from learning session"""
        insights = []
        
        # Analyze field patterns
        field_insights = self._analyze_field_learning_insights(field_data)
        insights.extend(field_insights)
        
        # Analyze table patterns
        table_insights = self._analyze_table_learning_insights(table_data)
        insights.extend(table_insights)
        
        return insights
    
    def _analyze_field_learning_insights(self, field_data: Dict[str, Any]) -> List[LearningInsight]:
        """Analyze field learning for insights"""
        insights = []
        
        # Group fields by data type
        type_groups = defaultdict(list)
        for field_name, value in field_data.items():
            data_type = self._determine_data_type(value)
            type_groups[data_type].append(field_name)
        
        # Generate insights for common patterns
        for data_type, fields in type_groups.items():
            if len(fields) > 2:  # Multiple fields of same type
                insight = LearningInsight(
                    insight_type="field_pattern",
                    description=f"Multiple {data_type} fields detected: {', '.join(fields)}",
                    confidence=0.7,
                    supporting_evidence=fields,
                    suggested_actions=[f"Create validation rules for {data_type} fields"],
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        return insights
    
    def _analyze_table_learning_insights(self, table_data: List[Dict[str, Any]]) -> List[LearningInsight]:
        """Analyze table learning for insights"""
        insights = []
        
        if not table_data:
            return insights
        
        # Analyze column consistency
        columns = list(table_data[0].keys())
        for col in columns:
            col_values = [row.get(col) for row in table_data if row.get(col) is not None]
            if col_values:
                data_types = [self._determine_data_type(v) for v in col_values]
                type_consistency = len(set(data_types)) / len(data_types)
                
                if type_consistency > 0.8:  # High consistency
                    insight = LearningInsight(
                        insight_type="table_pattern",
                        description=f"Column '{col}' shows high type consistency ({type_consistency:.1%})",
                        confidence=0.8,
                        supporting_evidence=[f"Data type: {max(set(data_types), key=data_types.count)}"],
                        suggested_actions=[f"Apply strict validation for column '{col}'"],
                        created_at=datetime.now()
                    )
                    insights.append(insight)
        
        return insights
    
    def _update_learning_stats(self, results: Dict[str, Any]):
        """Update learning statistics"""
        self.learning_stats['patterns_learned'] += (
            results.get('field_patterns_learned', 0) + 
            results.get('table_patterns_learned', 0)
        )
        self.learning_stats['last_learning_session'] = datetime.now()
        
        if 'confidence_improvement' in results:
            self.learning_stats['confidence_improvements'].append(results['confidence_improvement'])
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics"""
        stats = self.learning_stats.copy()
        
        # Add pattern counts
        stats['total_field_patterns'] = len(self.field_patterns)
        stats['total_table_patterns'] = len(self.table_patterns)
        stats['active_field_patterns'] = sum(1 for p in self.field_patterns.values() 
                                           if p.confidence >= MIN_PATTERN_CONFIDENCE)
        stats['active_table_patterns'] = sum(1 for p in self.table_patterns.values() 
                                           if p.confidence >= MIN_PATTERN_CONFIDENCE)
        
        # Calculate average confidence
        if self.field_patterns:
            stats['avg_field_pattern_confidence'] = statistics.mean(
                p.confidence for p in self.field_patterns.values()
            )
        
        if self.table_patterns:
            stats['avg_table_pattern_confidence'] = statistics.mean(
                p.confidence for p in self.table_patterns.values()
            )
        
        # Learning effectiveness
        if self.learning_stats['patterns_applied'] > 0:
            stats['learning_effectiveness'] = (
                self.learning_stats['successful_applications'] / 
                self.learning_stats['patterns_applied']
            )
        
        return stats
    
    def suggest_improvements(self, field_data: Dict[str, Any], 
                           table_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Suggest improvements based on learned patterns"""
        suggestions = []
        
        # Analyze field data for suggestions
        for field_name, value in field_data.items():
            applicable_patterns = self._find_applicable_field_patterns(field_name, value)
            for pattern in applicable_patterns:
                if pattern.confidence > 0.8:
                    suggestion = {
                        'type': 'field_improvement',
                        'field': field_name,
                        'current_value': value,
                        'suggested_improvement': self._apply_field_pattern(value, pattern),
                        'confidence': pattern.confidence,
                        'reason': f"Based on pattern learned from {len(pattern.source_documents)} documents"
                    }
                    suggestions.append(suggestion)
        
        # Analyze table data for suggestions
        if table_data:
            table_analysis = self._analyze_table_structure(table_data)
            applicable_patterns = self._find_applicable_table_patterns(table_analysis)
            
            for pattern in applicable_patterns:
                if pattern.confidence > 0.8:
                    suggestion = {
                        'type': 'table_improvement',
                        'description': 'Apply learned table structure pattern',
                        'confidence': pattern.confidence,
                        'improvements': len(pattern.column_patterns),
                        'reason': f"Based on pattern learned from {len(pattern.source_documents)} documents"
                    }
                    suggestions.append(suggestion)
        
        return suggestions
    
    def _load_existing_patterns(self):
        """Load existing patterns from database"""
        try:
            # Load field patterns
            field_patterns = self.db_manager.get_patterns(pattern_type='field_pattern')
            for pattern_data in field_patterns:
                pattern_id = pattern_data['pattern_id']
                data = pattern_data['pattern_data']
                
                field_pattern = FieldPattern(
                    field_name=data.get('field_name', ''),
                    pattern_type=data.get('pattern_type', ''),
                    regex_pattern=data.get('regex_pattern'),
                    position_hints=data.get('position_hints', {}),
                    value_type=data.get('value_type', ''),
                    validation_rules=data.get('validation_rules', []),
                    confidence=pattern_data['confidence'],
                    source_documents=data.get('source_documents', []),
                    usage_count=pattern_data['usage_count'],
                    success_count=pattern_data['success_count']
                )
                self.field_patterns[pattern_id] = field_pattern
            
            # Load table patterns
            table_patterns = self.db_manager.get_patterns(pattern_type='table_pattern')
            for pattern_data in table_patterns:
                signature = pattern_data['pattern_id']
                data = pattern_data['pattern_data']
                
                table_pattern = TablePattern(
                    table_signature=signature,
                    column_patterns=data.get('column_patterns', {}),
                    row_patterns=data.get('row_patterns', {}),
                    structural_hints=data.get('structural_hints', {}),
                    extraction_rules=data.get('extraction_rules', []),
                    confidence=pattern_data['confidence'],
                    source_documents=data.get('source_documents', []),
                    usage_count=pattern_data['usage_count'],
                    success_count=pattern_data['success_count']
                )
                self.table_patterns[signature] = table_pattern
            
            logger.info(f"Loaded {len(self.field_patterns)} field patterns and {len(self.table_patterns)} table patterns")
            
        except Exception as e:
            logger.warning(f"Error loading existing patterns: {str(e)}")
    
    def _save_patterns_to_database(self):
        """Save patterns to database"""
        try:
            # Save field patterns
            for pattern_id, pattern in self.field_patterns.items():
                pattern_data = {
                    'field_name': pattern.field_name,
                    'pattern_type': pattern.pattern_type,
                    'regex_pattern': pattern.regex_pattern,
                    'position_hints': pattern.position_hints,
                    'value_type': pattern.value_type,
                    'validation_rules': pattern.validation_rules,
                    'source_documents': pattern.source_documents
                }
                
                self.db_manager.save_pattern(
                    pattern_id=pattern_id,
                    pattern_type='field_pattern',
                    pattern_data=pattern_data,
                    confidence=pattern.confidence
                )
                
                # Update usage statistics
                self.db_manager.update_pattern_usage(pattern_id, success=True)
            
            # Save table patterns
            for signature, pattern in self.table_patterns.items():
                pattern_data = {
                    'column_patterns': pattern.column_patterns,
                    'row_patterns': pattern.row_patterns,
                    'structural_hints': pattern.structural_hints,
                    'extraction_rules': pattern.extraction_rules,
                    'source_documents': pattern.source_documents
                }
                
                self.db_manager.save_pattern(
                    pattern_id=signature,
                    pattern_type='table_pattern',
                    pattern_data=pattern_data,
                    confidence=pattern.confidence
                )
                
                # Update usage statistics
                self.db_manager.update_pattern_usage(signature, success=True)
            
        except Exception as e:
            logger.error(f"Error saving patterns to database: {str(e)}")

# Export main class
__all__ = ['PatternLearner', 'PatternType', 'LearningMode', 'FieldPattern', 'TablePattern', 'LearningInsight']