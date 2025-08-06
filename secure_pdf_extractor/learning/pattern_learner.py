"""
Pattern learner for adaptive field extraction.
"""

import re
import sqlite3
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class PatternLearner:
    """Learns new extraction patterns from user feedback."""

    def __init__(self, db_path: str = "learning.db"):
        """Initialize pattern learner with database setup."""
        self.db_path = str(Path(db_path))
        
        # Create directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database tables
        self._init_database()

    def _init_database(self):
        """Create required database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    field_type TEXT NOT NULL,
                    pattern TEXT NOT NULL,
                    confidence REAL DEFAULT 0.6,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    created_timestamp TEXT NOT NULL,
                    last_used_timestamp TEXT,
                    UNIQUE(field_type, pattern)
                )
            """)
            conn.commit()
        logger.info(f"Pattern learner database initialized at {self.db_path}")

    def learn_from_feedback(self, field_type: str, correct_value: str, context: str):
        """
        Learn new patterns from user feedback.

        Args:
            field_type: Type of field (e.g., 'date', 'angebot', 'company_name')
            correct_value: The correct value provided by user
            context: Text context around the value
        """
        pattern = self._generate_regex_from_context(context, correct_value)
        if pattern and self._validate_pattern(pattern):
            self.learn_new_pattern(field_type, pattern, confidence=0.7)
            logger.info(f"Learned new pattern for {field_type}: {pattern}")

    def _generate_regex_from_context(self, context: str, correct_value: str) -> Optional[str]:
        """
        Generate regex pattern from context and correct value.

        Args:
            context: Text context containing the value
            correct_value: The correct value to extract

        Returns:
            Generated regex pattern or None
        """
        if not correct_value or not context:
            return None

        # Find the value in context
        value_index = context.lower().find(correct_value.lower())
        if value_index == -1:
            return None

        # Extract surrounding context (15 chars before and after)
        start_ctx = max(0, value_index - 15)
        end_ctx = min(len(context), value_index + len(correct_value) + 15)

        before_text = context[start_ctx:value_index]
        after_text = context[value_index + len(correct_value):end_ctx]

        # Generate pattern based on field type and context
        if self._is_date(correct_value):
            return self._generate_date_pattern(before_text, correct_value, after_text)
        elif self._is_angebot_number(correct_value):
            return self._generate_angebot_pattern(before_text, correct_value, after_text)
        elif self._is_amount(correct_value):
            return self._generate_amount_pattern(before_text, correct_value, after_text)
        else:
            return self._generate_generic_pattern(before_text, correct_value, after_text)

    def _is_angebot_number(self, value: str) -> bool:
        """Check if value looks like an angebot/quote number."""
        return bool(re.match(r'^[A-Z0-9\-_/\.#]{3,}$', value.upper()))

    def _is_amount(self, value: str) -> bool:
        """Check if value looks like an amount."""
        return bool(re.match(r'^\d+(?:[.,]\d{1,2})?$', value))

    def _is_date(self, value: str) -> bool:
        """Check if value looks like a date."""
        date_patterns = [
            r'\d{1,2}[./]\d{1,2}[./]\d{2,4}',
            r'\d{2,4}-\d{1,2}-\d{1,2}',
            r'\d{1,2}\.\d{1,2}\.\d{2,4}'
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)

    def _generate_angebot_pattern(self, before: str, value: str, after: str) -> str:
        """Generate pattern for angebot/quote numbers."""
        before_clean = before.strip().lower()
        
        # Look for common German/English prefixes
        angebot_keywords = ['angebot', 'quote', 'offer', 'nr', 'no', 'number', 'ref']
        
        if any(keyword in before_clean for keyword in angebot_keywords):
            prefix_pattern = r'(?:angebot|quote|offer|nr\.?|no\.?|number|ref\.?)[\s:]*'
            value_pattern = r'([A-Z0-9\-_/\.#]{3,20})'
            return f'{prefix_pattern}{value_pattern}\\b'
        
        # Generic alphanumeric pattern for angebot numbers
        return r'\\b([A-Z0-9\-_/\.#]{3,20})\\b'

    def _generate_amount_pattern(self, before: str, value: str, after: str) -> str:
        """Generate pattern for monetary amounts."""
        before_clean = before.strip().lower()
        after_clean = after.strip().lower()

        currencies = ['€', '$', '£', 'eur', 'usd', 'gbp', 'euro']
        
        currency_before = any(curr in before_clean for curr in currencies)
        currency_after = any(curr in after_clean for curr in currencies)

        if currency_before:
            return r'[€$£]?\s*(\d+(?:[.,]\d{1,2})?)'
        elif currency_after:
            return r'(\d+(?:[.,]\d{1,2})?)\s*[€$£]?'
        else:
            return r'(\d+(?:[.,]\d{1,2})?)'

    def _generate_date_pattern(self, before: str, value: str, after: str) -> str:
        """Generate pattern for dates."""
        # Detect date format from the value and add word boundaries
        if re.match(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}', value):
            return r'(\d{1,2}[./]\d{1,2}[./]\d{2,4})(?!\d)'
        elif re.match(r'\d{2,4}-\d{1,2}-\d{1,2}', value):
            return r'(\d{2,4}-\d{1,2}-\d{1,2})(?!\d)'
        elif re.match(r'\d{1,2}\.\d{1,2}\.\d{2,4}', value):
            return r'(\d{1,2}\.\d{1,2}\.\d{2,4})(?!\d)'
        else:
            return r'(\d{1,2}[./\-]\d{1,2}[./\-]\d{2,4})(?!\d)'

    def _generate_generic_pattern(self, before: str, value: str, after: str) -> str:
        """Generate generic pattern for other field types."""
        # Enhanced generic pattern generation
        if len(value) < 3:
            return f'({re.escape(value)})'
        
        if value.isalpha():
            return r'([A-Za-zÄÖÜäöüßÀ-ÿ\s]+)'  # Include German and accented characters
        elif value.isdigit():
            return r'(\d+)'
        elif value.isalnum():
            return r'([A-Za-z0-9ÄÖÜäöüß]+)'
        else:
            # For complex values, escape the entire value for safety
            return f'({re.escape(value)})'

    def _validate_pattern(self, pattern: str) -> bool:
        """Validate that a regex pattern is syntactically correct."""
        try:
            re.compile(pattern)
            return True
        except re.error:
            logger.error(f"Invalid regex pattern: {pattern}")
            return False

    def learn_new_pattern(self, field_type: str, pattern: str, confidence: float = 0.6):
        """
        Store a new learned pattern in the database.

        Args:
            field_type: Type of field the pattern extracts
            pattern: Regex pattern
            confidence: Initial confidence score
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if pattern already exists
                cursor = conn.execute(
                    "SELECT id, confidence FROM learned_patterns WHERE field_type = ? AND pattern = ?",
                    (field_type, pattern)
                )
                
                existing = cursor.fetchone()
                
                if existing:
                    # Pattern exists, update confidence if new confidence is higher
                    existing_id, existing_confidence = existing
                    if confidence > existing_confidence:
                        conn.execute(
                            "UPDATE learned_patterns SET confidence = ?, last_used_timestamp = ? WHERE id = ?",
                            (confidence, datetime.now().isoformat(), existing_id)
                        )
                else:
                    # New pattern, insert
                    conn.execute(
                        """INSERT INTO learned_patterns 
                           (field_type, pattern, confidence, created_timestamp) 
                           VALUES (?, ?, ?, ?)""",
                        (field_type, pattern, confidence, datetime.now().isoformat())
                    )

                conn.commit()
            
            logger.info(f"Learned pattern for {field_type}: {pattern}")
            return True
            
        except Exception as e:
            logger.error(f"Error learning pattern: {e}")
            return False

    def get_learned_patterns(self, field_type: str) -> List[Dict[str, Any]]:
        """Get all learned patterns for a field type."""
        patterns = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern, confidence, success_count, total_count 
                    FROM learned_patterns 
                    WHERE field_type = ? AND confidence >= 0.5
                    ORDER BY confidence DESC, success_count DESC
                """, (field_type,))

                for row in cursor.fetchall():
                    patterns.append({
                        'pattern': row[0],
                        'confidence': row[1],
                        'success_count': row[2] or 0,
                        'total_count': row[3] or 0
                    })
        except Exception as e:
            logger.error(f"Error getting learned patterns: {e}")

        return patterns

    def update_pattern_performance(self, field_type: str, pattern: str, success: bool):
        """
        Update pattern performance metrics.

        Args:
            field_type: Type of field
            pattern: The regex pattern
            success: Whether the pattern successfully extracted the correct value
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update counts
                if success:
                    conn.execute("""
                        UPDATE learned_patterns 
                        SET success_count = success_count + 1, 
                            total_count = total_count + 1, 
                            last_used_timestamp = ?
                        WHERE field_type = ? AND pattern = ?
                    """, (datetime.now().isoformat(), field_type, pattern))
                else:
                    conn.execute("""
                        UPDATE learned_patterns 
                        SET total_count = total_count + 1, 
                            last_used_timestamp = ?
                        WHERE field_type = ? AND pattern = ?
                    """, (datetime.now().isoformat(), field_type, pattern))

                # Recalculate confidence based on success rate
                conn.execute("""
                    UPDATE learned_patterns 
                    SET confidence = CASE 
                        WHEN total_count > 0 THEN 
                            LEAST(0.95, (success_count * 1.0 / total_count) * 0.8 + 0.2)
                        ELSE confidence 
                    END
                    WHERE field_type = ? AND pattern = ?
                """, (field_type, pattern))

                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating pattern performance: {e}")

    def get_top_patterns(self, field_type: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top performing patterns for a field type."""
        patterns = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern, confidence, success_count, total_count, 
                           CASE WHEN total_count > 0 THEN (success_count * 1.0 / total_count) ELSE 0 END as success_rate
                    FROM learned_patterns 
                    WHERE field_type = ? AND total_count > 0
                    ORDER BY success_rate DESC, confidence DESC, success_count DESC
                    LIMIT ?
                """, (field_type, limit))

                for row in cursor.fetchall():
                    patterns.append({
                        'pattern': row[0],
                        'confidence': row[1],
                        'success_count': row[2],
                        'total_count': row[3],
                        'success_rate': row[4]
                    })
        except Exception as e:
            logger.error(f"Error getting top patterns: {e}")

        return patterns

    def clear_learned_patterns(self, field_type: Optional[str] = None):
        """Clear learned patterns, optionally for a specific field type."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if field_type:
                    conn.execute("DELETE FROM learned_patterns WHERE field_type = ?", (field_type,))
                    logger.info(f"Cleared learned patterns for {field_type}")
                else:
                    conn.execute("DELETE FROM learned_patterns")
                    logger.info("Cleared all learned patterns")
                conn.commit()
        except Exception as e:
            logger.error(f"Error clearing patterns: {e}")

    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get overall statistics about learned patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total patterns
                cursor.execute("SELECT COUNT(*) FROM learned_patterns")
                total_patterns = cursor.fetchone()[0]
                
                # Patterns by field type
                cursor.execute("""
                    SELECT field_type, COUNT(*) as count, AVG(confidence) as avg_confidence
                    FROM learned_patterns 
                    GROUP BY field_type
                    ORDER BY count DESC
                """)
                patterns_by_type = cursor.fetchall()
                
                # Most successful patterns
                cursor.execute("""
                    SELECT field_type, pattern, success_count, total_count,
                           CASE WHEN total_count > 0 THEN (success_count * 1.0 / total_count) ELSE 0 END as success_rate
                    FROM learned_patterns 
                    WHERE total_count > 0
                    ORDER BY success_rate DESC, success_count DESC
                    LIMIT 5
                """)
                top_patterns = cursor.fetchall()
                
                return {
                    'total_patterns': total_patterns,
                    'patterns_by_type': patterns_by_type,
                    'top_patterns': top_patterns
                }
                
        except Exception as e:
            logger.error(f"Error getting pattern statistics: {e}")
            return {
                'total_patterns': 0,
                'patterns_by_type': [],
                'top_patterns': []
            }