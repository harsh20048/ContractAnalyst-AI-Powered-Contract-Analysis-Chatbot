import sqlite3
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class LearningDatabase:
    """Database for storing learning data and user corrections with content-based identification."""

    def __init__(self, db_path: str = "learning.db"):
        """Initialize learning database with document hash support."""
        self.db_path = db_path
        
        # Create directory if it doesn't exist
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        self.init_database()
        self.init_table_learning_tables()

    def init_database(self):
        """Initialize database tables with document hash support."""
        with sqlite3.connect(self.db_path) as conn:
            # Corrections table with document hash key
            conn.execute("""
                CREATE TABLE IF NOT EXISTS corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    pdf_name TEXT,
                    field_name TEXT NOT NULL,
                    original_value TEXT,
                    corrected_value TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.95,
                    UNIQUE(document_hash, field_name)
                )
            """)

            # Document metadata table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS document_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT UNIQUE NOT NULL,
                    pdf_name TEXT,
                    file_size INTEGER,
                    first_processed TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_processed TEXT DEFAULT CURRENT_TIMESTAMP,
                    processing_count INTEGER DEFAULT 1
                )
            """)

            # Create indexes for faster lookups
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_hash ON corrections(document_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corrections_field ON corrections(field_name)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_metadata_hash ON document_metadata(document_hash)")

            # Feedback table for user feedback and adaptive learning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY,
                    document_hash TEXT,
                    pdf_name TEXT,
                    field_type TEXT,
                    extracted_value TEXT,
                    correct_value TEXT,
                    confidence REAL,
                    timestamp TEXT,
                    context_snippet TEXT
                )
            """)

            # Learned patterns for regex pattern learning
            conn.execute("""
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY,
                    field_type TEXT,
                    pattern TEXT,
                    confidence REAL,
                    success_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    created_timestamp TEXT,
                    last_used_timestamp TEXT
                )
            """)

            conn.commit()
        logger.info(f"Learning database initialized at {self.db_path}")

    def init_table_learning_tables(self):
        """Initialize tables for table learning functionality."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS table_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT,
                    original_table_data TEXT,
                    corrected_table_data TEXT,
                    confidence_improvement REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    file_name TEXT,
                    page_number INTEGER
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_table_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT,
                    original_value TEXT,
                    corrected_value TEXT,
                    position_info TEXT,
                    confidence_boost REAL,
                    usage_count INTEGER DEFAULT 1,
                    success_rate REAL DEFAULT 1.0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')

            conn.commit()

    def save_document_metadata(self, document_hash: str, pdf_name: str, file_size: int = 0):
        """Save or update document metadata."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT processing_count FROM document_metadata WHERE document_hash = ?", (document_hash,))
            existing = cursor.fetchone()

            if existing:
                cursor.execute("""
                    UPDATE document_metadata
                    SET pdf_name = ?, last_processed = CURRENT_TIMESTAMP, processing_count = processing_count + 1
                    WHERE document_hash = ?
                """, (pdf_name, document_hash))
            else:
                cursor.execute("""
                    INSERT INTO document_metadata (document_hash, pdf_name, file_size)
                    VALUES (?, ?, ?)
                """, (document_hash, pdf_name, file_size))

            conn.commit()

    def save_correction(self, document_hash: str, pdf_name: str, field_name: str, 
                       corrected_value: str, confidence: float = 0.95, original_value: str = ""):
        """Save user field correction to database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO corrections 
                    (document_hash, pdf_name, field_name, original_value, corrected_value, timestamp, confidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (document_hash, pdf_name, field_name, original_value, corrected_value, 
                      datetime.now().isoformat(), confidence))
                conn.commit()
            logger.info(f"Saved field correction for {field_name}: {corrected_value}")
            return True
        except Exception as e:
            logger.error(f"Error saving field correction: {e}")
            return False

    def save_table_correction(self, document_hash: str, file_name: str, page_number: int,
                             original_table_data: str, corrected_table_data: str, 
                             confidence_improvement: float = 0.0):
        """Save table correction data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO table_corrections 
                    (document_hash, file_name, page_number, original_table_data, 
                     corrected_table_data, confidence_improvement, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (document_hash, file_name, page_number, original_table_data, 
                      corrected_table_data, confidence_improvement, datetime.now().isoformat()))
                conn.commit()
            logger.info(f"Saved table correction for {file_name} page {page_number}")
            return True
        except Exception as e:
            logger.error(f"Error saving table correction: {e}")
            return False

    def get_corrections_for_document(self, document_hash: str) -> Dict[str, str]:
        """Retrieve all corrections for a specific document by its hash."""
        corrections = {}
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT field_name, corrected_value
                    FROM corrections
                    WHERE document_hash = ?
                    ORDER BY timestamp DESC
                """, (document_hash,))
                for row in cursor.fetchall():
                    corrections[row[0]] = row[1]
            return corrections
        except Exception as e:
            logger.error(f"Error retrieving corrections for document {document_hash}: {e}")
            return {}

    def has_corrections_for_document(self, document_hash: str) -> bool:
        """Check if a document has any saved corrections."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM corrections WHERE document_hash = ?", (document_hash,))
            count = cursor.fetchone()[0]
        return count > 0

    def get_document_info(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Get document metadata and correction count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT dm.pdf_name, dm.first_processed, dm.last_processed, dm.processing_count,
                       COUNT(c.id) as correction_count
                FROM document_metadata dm
                LEFT JOIN corrections c ON dm.document_hash = c.document_hash
                WHERE dm.document_hash = ?
                GROUP BY dm.document_hash
            """, (document_hash,))

            row = cursor.fetchone()
            if row:
                return {
                    'pdf_name': row[0],
                    'first_processed': row[1],
                    'last_processed': row[2],
                    'processing_count': row[3],
                    'correction_count': row[4]
                }
        return None

    def apply_corrections_to_fields(self, document_hash: str, extracted_fields: Dict[str, Any]) -> Dict[str, Any]:
        """Apply saved corrections to extracted fields."""
        corrections = self.get_corrections_for_document(document_hash)
        applied_count = 0

        for field_name, corrected_value in corrections.items():
            if field_name in extracted_fields:
                if isinstance(extracted_fields[field_name], dict):
                    extracted_fields[field_name]['value'] = corrected_value
                    extracted_fields[field_name]['confidence'] = 0.95
                    extracted_fields[field_name]['source'] = 'user_correction'
                else:
                    extracted_fields[field_name] = corrected_value
                applied_count += 1

        logger.info(f"Applied {applied_count} corrections to extracted fields")
        return extracted_fields

    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about corrections and learning."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM corrections")
            total_corrections = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT document_hash) FROM corrections")
            unique_documents = cursor.fetchone()[0]

            cursor.execute("""
                SELECT field_name, COUNT(*) as count
                FROM corrections
                GROUP BY field_name
                ORDER BY count DESC
                LIMIT 5
            """)
            top_corrected_fields = cursor.fetchall()

            cursor.execute("""
                SELECT pdf_name, field_name, corrected_value, timestamp
                FROM corrections
                ORDER BY timestamp DESC
                LIMIT 10
            """)
            recent_corrections = cursor.fetchall()

            return {
                'total_corrections': total_corrections,
                'unique_documents': unique_documents,
                'top_corrected_fields': top_corrected_fields,
                'recent_corrections': recent_corrections
            }

    def get_table_correction_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about table corrections and learned patterns."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) FROM table_corrections")
                total_table_corrections = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT document_hash) FROM table_corrections")
                documents_with_table_corrections = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT AVG(confidence_improvement) 
                    FROM table_corrections
                    WHERE confidence_improvement IS NOT NULL
                """)
                avg_improvement_result = cursor.fetchone()[0]
                average_confidence_improvement = avg_improvement_result if avg_improvement_result is not None else 0.0
                
                cursor.execute("""
                    SELECT file_name, page_number, confidence_improvement, timestamp
                    FROM table_corrections
                    ORDER BY timestamp DESC
                    LIMIT 5
                """)
                recent_table_corrections = cursor.fetchall()
                
                cursor.execute("""
                    SELECT COUNT(*), AVG(success_rate)
                    FROM learned_table_patterns
                """)
                pattern_stats = cursor.fetchone()
                learned_pattern_count = pattern_stats[0] if pattern_stats else 0
                average_pattern_success_rate = pattern_stats[1] if pattern_stats and pattern_stats[1] is not None else 0.0
                
                average_corrections_per_document = (
                    total_table_corrections / max(documents_with_table_corrections, 1)
                )
                
                return {
                    'total_table_corrections': total_table_corrections,
                    'documents_with_table_corrections': documents_with_table_corrections,
                    'average_confidence_improvement': average_confidence_improvement,
                    'average_corrections_per_document': average_corrections_per_document,
                    'recent_table_corrections': recent_table_corrections,
                    'learned_pattern_count': learned_pattern_count,
                    'average_pattern_success_rate': average_pattern_success_rate
                }
                
        except Exception as e:
            logger.error(f"Error getting table correction stats: {e}")
            return self._get_empty_table_stats()

    def _get_empty_table_stats(self) -> Dict[str, Any]:
        """Return empty statistics when database query fails."""
        return {
            'total_table_corrections': 0,
            'documents_with_table_corrections': 0,
            'average_confidence_improvement': 0.0,
            'average_corrections_per_document': 0,
            'recent_table_corrections': [],
            'learned_pattern_count': 0,
            'average_pattern_success_rate': 0.0
        }

    def get_learned_table_patterns(self, limit: int = 10) -> List[Dict]:
        """Get learned table patterns for display."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT pattern_type, original_value, corrected_value, 
                           confidence_boost, usage_count, success_rate, timestamp
                    FROM learned_table_patterns
                    ORDER BY usage_count DESC, success_rate DESC
                    LIMIT ?
                """, (limit,))
                
                patterns = []
                for row in cursor.fetchall():
                    patterns.append({
                        'pattern_type': row[0],
                        'original_value': row[1],
                        'corrected_value': row[2],
                        'confidence_boost': row[3],
                        'usage_count': row[4],
                        'success_rate': row[5],
                        'timestamp': row[6]
                    })
                return patterns
        except Exception as e:
            logger.error(f"Error getting learned patterns: {e}")
            return []

    def save_learned_table_pattern(self, pattern_type: str, original_value: str, 
                                  corrected_value: str, position_info: str = "",
                                  confidence_boost: float = 0.1):
        """Save learned table pattern."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, usage_count, success_rate 
                    FROM learned_table_patterns 
                    WHERE pattern_type = ? AND original_value = ? AND corrected_value = ?
                """, (pattern_type, original_value, corrected_value))
                
                existing = cursor.fetchone()
                
                if existing:
                    pattern_id, usage_count, success_rate = existing
                    new_usage_count = usage_count + 1
                    new_success_rate = min(1.0, success_rate + 0.1)
                    
                    cursor.execute("""
                        UPDATE learned_table_patterns 
                        SET usage_count = ?, success_rate = ?, timestamp = ?
                        WHERE id = ?
                    """, (new_usage_count, new_success_rate, datetime.now().isoformat(), pattern_id))
                else:
                    cursor.execute("""
                        INSERT INTO learned_table_patterns 
                        (pattern_type, original_value, corrected_value, position_info, 
                         confidence_boost, usage_count, success_rate, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (pattern_type, original_value, corrected_value, position_info,
                          confidence_boost, 1, 1.0, datetime.now().isoformat()))
                
                conn.commit()
            logger.info(f"Saved learned table pattern: {pattern_type}")
        except Exception as e:
            logger.error(f"Error saving learned table pattern: {e}")

    def clear_all_corrections(self):
        """Clear all corrections and learning data - use with caution."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM corrections")
            cursor.execute("DELETE FROM document_metadata")
            cursor.execute("DELETE FROM feedback")
            cursor.execute("DELETE FROM learned_patterns")
            cursor.execute("DELETE FROM table_corrections")
            cursor.execute("DELETE FROM learned_table_patterns")
            conn.commit()
        logger.warning("All corrections and learning data cleared!")

    def get_corrections(self, pdf_name: str) -> Dict[str, str]:
        """Legacy method - get corrections by filename (less reliable)."""
        logger.warning("Using legacy filename-based correction retrieval - consider using document hash")
        corrections = {}
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT field_name, corrected_value FROM corrections WHERE pdf_name = ?",
                (pdf_name,)
            )
            for row in cursor.fetchall():
                corrections[row[0]] = row[1]
        return corrections

    def record_feedback(self, pdf_name: str, field_type: str, extracted: str,
                        correct: str, confidence: float, context: str, document_hash: str = ""):
        """Record user feedback for adaptive learning."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO feedback
                (document_hash, pdf_name, field_type, extracted_value, correct_value, confidence, timestamp, context_snippet)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (document_hash, pdf_name, field_type, extracted, correct, confidence, datetime.now().isoformat(), context))
            conn.commit()

        logger.info(f"Recorded feedback for {field_type}: {extracted} -> {correct}")

    def extract_context_snippet(self, full_text: str, value: str, context_length: int = 100) -> str:
        """Extract context snippet around a value for pattern learning."""
        try:
            if not full_text or not value:
                return ""
            
            pos = full_text.lower().find(value.lower())
            if pos == -1:
                return full_text[:context_length]
            
            start = max(0, pos - context_length // 2)
            end = min(len(full_text), pos + len(value) + context_length // 2)
            
            return full_text[start:end].strip()
            
        except Exception as e:
            logger.error(f"Error extracting context snippet: {e}")
            return ""