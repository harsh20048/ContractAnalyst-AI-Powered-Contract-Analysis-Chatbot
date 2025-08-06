"""
Database Manager for PDF Table Extraction System
"""

import sqlite3
import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for PDF extraction system"""
    
    def __init__(self, db_path: str = "pdf_extraction.db"):
        self.db_path = db_path
        self.connection_lock = threading.Lock()
        self._initialize_database()
        logger.info(f"Database manager initialized: {db_path}")
    
    def _initialize_database(self):
        """Initialize database tables"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Document corrections table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT UNIQUE NOT NULL,
                    filename TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    confidence REAL DEFAULT 0.0,
                    page_count INTEGER DEFAULT 0,
                    extraction_engine TEXT,
                    processing_time REAL DEFAULT 0.0,
                    metadata TEXT
                )
                """)
                
                # Field corrections table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    corrected_value TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash)
                )
                """)
                
                # Table corrections table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    table_index INTEGER NOT NULL,
                    corrected_data TEXT,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash)
                )
                """)
                
                # Pattern learning table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    success_count INTEGER DEFAULT 0,
                    usage_count INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
                """)
                
                # Processing logs table
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT,
                    log_level TEXT DEFAULT 'info',
                    message TEXT NOT NULL,
                    details TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_hash ON document_corrections (document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_document_hash ON field_corrections (document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_table_document_hash ON table_corrections (document_hash)")
                
                conn.commit()
                logger.info("Database tables initialized")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @contextmanager
    def _get_connection(self):
        """Get database connection"""
        conn = None
        try:
            with self.connection_lock:
                conn = sqlite3.connect(self.db_path, timeout=30.0)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_document_corrections(self, document_hash: str, fields: Dict[str, Any], 
                                tables: List[Dict[str, Any]], confidence: float = 0.0,
                                filename: str = None, metadata: Dict[str, Any] = None) -> bool:
        """Save document corrections"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Insert or update document
                cursor.execute("""
                INSERT OR REPLACE INTO document_corrections 
                (document_hash, filename, confidence, updated_at, metadata)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)
                """, (document_hash, filename, confidence, json.dumps(metadata or {})))
                
                # Clear existing corrections
                cursor.execute("DELETE FROM field_corrections WHERE document_hash = ?", (document_hash,))
                cursor.execute("DELETE FROM table_corrections WHERE document_hash = ?", (document_hash,))
                
                # Insert field corrections
                for field_name, field_value in fields.items():
                    cursor.execute("""
                    INSERT INTO field_corrections 
                    (document_hash, field_name, corrected_value)
                    VALUES (?, ?, ?)
                    """, (document_hash, field_name, str(field_value)))
                
                # Insert table corrections
                for table_index, table_data in enumerate(tables):
                    table_json = json.dumps(table_data)
                    row_count = len(table_data) if isinstance(table_data, list) else 0
                    
                    cursor.execute("""
                    INSERT INTO table_corrections 
                    (document_hash, table_index, corrected_data, row_count)
                    VALUES (?, ?, ?, ?)
                    """, (document_hash, table_index, table_json, row_count))
                
                conn.commit()
                
                self.log_processing_event(document_hash, "Corrections saved")
                logger.info(f"Saved corrections for document {document_hash[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
            return False
    
    def load_document_corrections(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Load document corrections"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if document exists
                cursor.execute("""
                SELECT filename, confidence, created_at, updated_at
                FROM document_corrections 
                WHERE document_hash = ?
                """, (document_hash,))
                
                doc_row = cursor.fetchone()
                if not doc_row:
                    return None
                
                # Load fields
                cursor.execute("""
                SELECT field_name, corrected_value
                FROM field_corrections 
                WHERE document_hash = ?
                """, (document_hash,))
                
                fields = {}
                for row in cursor.fetchall():
                    fields[row['field_name']] = row['corrected_value']
                
                # Load tables
                cursor.execute("""
                SELECT corrected_data
                FROM table_corrections 
                WHERE document_hash = ?
                ORDER BY table_index
                """, (document_hash,))
                
                tables = []
                for row in cursor.fetchall():
                    try:
                        table_data = json.loads(row['corrected_data'])
                        tables.append(table_data)
                    except json.JSONDecodeError:
                        continue
                
                return {
                    'fields': fields,
                    'tables': tables
                }
                
        except Exception as e:
            logger.error(f"Failed to load corrections: {e}")
            return None
    
    def document_exists(self, document_hash: str) -> bool:
        """Check if document exists"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1 FROM document_corrections WHERE document_hash = ?", (document_hash,))
                return cursor.fetchone() is not None
        except Exception as e:
            logger.error(f"Failed to check document existence: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT COUNT(*) as count FROM document_corrections")
                total_documents = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM field_corrections")
                total_field_corrections = cursor.fetchone()['count']
                
                cursor.execute("SELECT COUNT(*) as count FROM table_corrections")
                total_table_corrections = cursor.fetchone()['count']
                
                cursor.execute("SELECT AVG(confidence) as avg_conf FROM document_corrections WHERE confidence > 0")
                avg_confidence_row = cursor.fetchone()
                avg_confidence = avg_confidence_row['avg_conf'] if avg_confidence_row['avg_conf'] else 0.0
                
                thirty_days_ago = (datetime.now() - timedelta(days=30)).isoformat()
                cursor.execute("SELECT COUNT(*) as count FROM document_corrections WHERE created_at >= ?", (thirty_days_ago,))
                recent_documents = cursor.fetchone()['count']
                
                return {
                    'total_documents': total_documents,
                    'total_corrections': total_field_corrections + total_table_corrections,
                    'avg_confidence': avg_confidence,
                    'recent_documents': recent_documents
                }
                
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent documents"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                SELECT document_hash, filename, confidence, created_at
                FROM document_corrections 
                ORDER BY created_at DESC
                LIMIT ?
                """, (limit,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
            return []
    
    def save_pattern(self, pattern_type: str, pattern_name: str, pattern_data: Dict[str, Any]) -> bool:
        """Save pattern"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT OR REPLACE INTO pattern_learning 
                (pattern_type, pattern_name, pattern_data, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                """, (pattern_type, pattern_name, json.dumps(pattern_data)))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
            return False
    
    def load_patterns(self, pattern_type: str = None) -> List[Dict[str, Any]]:
        """Load patterns"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if pattern_type:
                    cursor.execute("""
                    SELECT pattern_name, pattern_data, confidence
                    FROM pattern_learning 
                    WHERE pattern_type = ? AND is_active = 1
                    ORDER BY confidence DESC
                    """, (pattern_type,))
                else:
                    cursor.execute("""
                    SELECT pattern_type, pattern_name, pattern_data, confidence
                    FROM pattern_learning 
                    WHERE is_active = 1
                    ORDER BY confidence DESC
                    """)
                
                patterns = []
                for row in cursor.fetchall():
                    pattern = dict(row)
                    try:
                        pattern['pattern_data'] = json.loads(pattern['pattern_data'])
                        patterns.append(pattern)
                    except json.JSONDecodeError:
                        continue
                
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return []
    
    def log_processing_event(self, document_hash: str, message: str, 
                           details: Dict[str, Any] = None, log_level: str = "info") -> bool:
        """Log processing event"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                INSERT INTO processing_logs 
                (document_hash, log_level, message, details)
                VALUES (?, ?, ?, ?)
                """, (document_hash, log_level, message, json.dumps(details or {})))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to log event: {e}")
            return False
    
    def cleanup_old_records(self, days: int = 30) -> int:
        """Clean up old records"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            deleted_count = 0
            
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT document_hash FROM document_corrections WHERE created_at < ?", (cutoff_date,))
                old_hashes = [row['document_hash'] for row in cursor.fetchall()]
                
                for doc_hash in old_hashes:
                    cursor.execute("DELETE FROM field_corrections WHERE document_hash = ?", (doc_hash,))
                    cursor.execute("DELETE FROM table_corrections WHERE document_hash = ?", (doc_hash,))
                    cursor.execute("DELETE FROM processing_logs WHERE document_hash = ?", (doc_hash,))
                    deleted_count += cursor.rowcount
                
                cursor.execute("DELETE FROM document_corrections WHERE created_at < ?", (cutoff_date,))
                deleted_count += cursor.rowcount
                
                conn.commit()
                return deleted_count
                
        except Exception as e:
            logger.error(f"Failed to cleanup records: {e}")
            return 0
    
    def backup_database(self, backup_path: str = None) -> str:
        """Create database backup"""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.db_path}.backup_{timestamp}"
            
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Database backed up to: {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def close(self):
        """Close database manager"""
        logger.info("Database manager closed")
