#!/usr/bin/env python3
"""
Database Management Module for PDF Table Extraction System
===========================================================

Comprehensive database operations for storing document corrections,
learned patterns, system statistics, and processing logs.

Author: AI Assistant
Version: 3.0.0
License: MIT
"""

import sqlite3
import json
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import threading
import os
import shutil
from contextlib import contextmanager
import time
import uuid
from dataclasses import dataclass, asdict
from enum import Enum

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DATABASE_VERSION = "3.0.0"
BACKUP_RETENTION_DAYS = 30
MAX_LOG_ENTRIES = 10000
CLEANUP_INTERVAL_DAYS = 7

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

class BackupError(Exception):
    """Custom exception for backup operations"""
    pass

@dataclass
class DocumentCorrection:
    """Document correction data structure"""
    hash: str
    filename: str
    field_data: Dict[str, Any]
    table_data: List[Dict[str, Any]]
    engine: str
    confidence: float
    created_at: datetime
    updated_at: datetime
    version: int = 1
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class PatternData:
    """Pattern learning data structure"""
    pattern_id: str
    pattern_type: str
    pattern_data: Dict[str, Any]
    success_count: int
    usage_count: int
    confidence: float
    created_at: datetime
    updated_at: datetime

@dataclass
class ProcessingLog:
    """Processing log data structure"""
    log_id: str
    document_hash: str
    operation: str
    status: str
    engine: str
    processing_time: float
    error_message: Optional[str]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class DatabaseManager:
    """Comprehensive database management with advanced features"""
    
    def __init__(self, db_path: str = "pdf_extraction.db", backup_dir: str = "backups"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
            backup_dir: Directory for database backups
        """
        self.db_path = db_path
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        self._connection_pool = {}
        
        # Initialize database
        self.init_database()
        
        # Setup automatic cleanup
        self._setup_cleanup_schedule()
        
        logger.info(f"DatabaseManager initialized with database: {db_path}")
    
    def init_database(self):
        """Initialize database with all required tables and indexes"""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Create tables
                self._create_document_corrections_table(cursor)
                self._create_field_corrections_table(cursor)
                self._create_table_corrections_table(cursor)
                self._create_pattern_learning_table(cursor)
                self._create_system_stats_table(cursor)
                self._create_processing_logs_table(cursor)
                self._create_backup_metadata_table(cursor)
                self._create_user_sessions_table(cursor)
                
                # Create indexes for performance
                self._create_indexes(cursor)
                
                # Create triggers for automatic updates
                self._create_triggers(cursor)
                
                # Insert initial system data
                self._insert_initial_data(cursor)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {str(e)}")
            raise DatabaseError(f"Failed to initialize database: {str(e)}")
    
    def _create_document_corrections_table(self, cursor):
        """Create document corrections table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT UNIQUE NOT NULL,
                filename TEXT NOT NULL,
                original_field_data TEXT,
                corrected_field_data TEXT,
                original_table_data TEXT,
                corrected_table_data TEXT,
                extraction_engine TEXT,
                extraction_confidence REAL,
                correction_count INTEGER DEFAULT 0,
                validation_status TEXT DEFAULT 'pending',
                processing_time REAL,
                file_size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                metadata TEXT,
                user_id TEXT,
                tags TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
    
    def _create_field_corrections_table(self, cursor):
        """Create field corrections table for detailed tracking"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS field_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT NOT NULL,
                field_name TEXT NOT NULL,
                original_value TEXT,
                corrected_value TEXT,
                correction_type TEXT,
                confidence REAL,
                validation_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash)
            )
        """)
    
    def _create_table_corrections_table(self, cursor):
        """Create table corrections table for detailed tracking"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS table_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_hash TEXT NOT NULL,
                table_index INTEGER,
                row_index INTEGER,
                column_name TEXT,
                original_value TEXT,
                corrected_value TEXT,
                correction_type TEXT,
                confidence REAL,
                validation_result TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash)
            )
        """)
    
    def _create_pattern_learning_table(self, cursor):
        """Create pattern learning table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_id TEXT UNIQUE NOT NULL,
                pattern_type TEXT NOT NULL,
                pattern_name TEXT,
                pattern_data TEXT NOT NULL,
                field_patterns TEXT,
                table_patterns TEXT,
                success_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0.0,
                accuracy REAL DEFAULT 0.0,
                last_used TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                version INTEGER DEFAULT 1,
                metadata TEXT,
                is_active BOOLEAN DEFAULT 1,
                source_documents TEXT
            )
        """)
    
    def _create_system_stats_table(self, cursor):
        """Create system statistics table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stat_name TEXT UNIQUE NOT NULL,
                stat_value TEXT NOT NULL,
                stat_type TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
    
    def _create_processing_logs_table(self, cursor):
        """Create processing logs table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                log_id TEXT UNIQUE NOT NULL,
                document_hash TEXT,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                engine TEXT,
                processing_time REAL,
                memory_usage INTEGER,
                cpu_usage REAL,
                error_message TEXT,
                error_code TEXT,
                stack_trace TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                user_id TEXT,
                session_id TEXT
            )
        """)
    
    def _create_backup_metadata_table(self, cursor):
        """Create backup metadata table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS backup_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backup_id TEXT UNIQUE NOT NULL,
                backup_path TEXT NOT NULL,
                backup_size INTEGER,
                backup_type TEXT,
                compression_ratio REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                metadata TEXT
            )
        """)
    
    def _create_user_sessions_table(self, cursor):
        """Create user sessions table"""
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                user_id TEXT,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                end_time TIMESTAMP,
                documents_processed INTEGER DEFAULT 0,
                corrections_made INTEGER DEFAULT 0,
                session_data TEXT,
                ip_address TEXT,
                user_agent TEXT
            )
        """)
    
    def _create_indexes(self, cursor):
        """Create database indexes for performance optimization"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_document_hash ON document_corrections (document_hash)",
            "CREATE INDEX IF NOT EXISTS idx_document_created_at ON document_corrections (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_document_filename ON document_corrections (filename)",
            "CREATE INDEX IF NOT EXISTS idx_document_status ON document_corrections (status)",
            
            "CREATE INDEX IF NOT EXISTS idx_field_document_hash ON field_corrections (document_hash)",
            "CREATE INDEX IF NOT EXISTS idx_field_name ON field_corrections (field_name)",
            "CREATE INDEX IF NOT EXISTS idx_field_created_at ON field_corrections (created_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_table_document_hash ON table_corrections (document_hash)",
            "CREATE INDEX IF NOT EXISTS idx_table_created_at ON table_corrections (created_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_learning (pattern_type)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_active ON pattern_learning (is_active)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_confidence ON pattern_learning (confidence)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_last_used ON pattern_learning (last_used)",
            
            "CREATE INDEX IF NOT EXISTS idx_logs_document_hash ON processing_logs (document_hash)",
            "CREATE INDEX IF NOT EXISTS idx_logs_operation ON processing_logs (operation)",
            "CREATE INDEX IF NOT EXISTS idx_logs_status ON processing_logs (status)",
            "CREATE INDEX IF NOT EXISTS idx_logs_created_at ON processing_logs (created_at)",
            
            "CREATE INDEX IF NOT EXISTS idx_stats_name ON system_stats (stat_name)",
            "CREATE INDEX IF NOT EXISTS idx_backup_created_at ON backup_metadata (created_at)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON user_sessions (start_time)",
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
    
    def _create_triggers(self, cursor):
        """Create database triggers for automatic updates"""
        # Update timestamp trigger for document_corrections
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_document_corrections_timestamp
            AFTER UPDATE ON document_corrections
            BEGIN
                UPDATE document_corrections SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
        
        # Update timestamp trigger for pattern_learning
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_pattern_learning_timestamp
            AFTER UPDATE ON pattern_learning
            BEGIN
                UPDATE pattern_learning SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
        
        # Update system stats trigger
        cursor.execute("""
            CREATE TRIGGER IF NOT EXISTS update_system_stats_timestamp
            AFTER UPDATE ON system_stats
            BEGIN
                UPDATE system_stats SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
            END
        """)
    
    def _insert_initial_data(self, cursor):
        """Insert initial system data and statistics"""
        initial_stats = [
            ('total_documents', '0', 'integer', 'Total number of processed documents'),
            ('total_corrections', '0', 'integer', 'Total number of corrections made'),
            ('avg_confidence', '0.0', 'float', 'Average extraction confidence'),
            ('success_rate', '0.0', 'float', 'Overall success rate'),
            ('database_version', DATABASE_VERSION, 'string', 'Database schema version'),
            ('last_cleanup', datetime.now().isoformat(), 'datetime', 'Last cleanup timestamp'),
            ('system_initialized', datetime.now().isoformat(), 'datetime', 'System initialization timestamp'),
        ]
        
        for stat_name, stat_value, stat_type, description in initial_stats:
            cursor.execute("""
                INSERT OR IGNORE INTO system_stats (stat_name, stat_value, stat_type, description)
                VALUES (?, ?, ?, ?)
            """, (stat_name, stat_value, stat_type, description))
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with proper resource management"""
        thread_id = threading.current_thread().ident
        
        with self._lock:
            if thread_id not in self._connection_pool:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA cache_size = 10000")
                self._connection_pool[thread_id] = conn
            
            conn = self._connection_pool[thread_id]
        
        try:
            yield conn
        except Exception as e:
            conn.rollback()
            raise
        finally:
            pass  # Keep connection in pool for reuse
    
    def save_document_corrections(self, document_hash: str, filename: str,
                                field_data: Dict[str, Any], table_data: List[Dict[str, Any]],
                                engine: str = "unknown", confidence: float = 0.0,
                                processing_time: float = 0.0, file_size: int = 0,
                                user_id: str = None, metadata: Dict[str, Any] = None) -> bool:
        """
        Save document corrections with comprehensive tracking
        
        Args:
            document_hash: Unique document identifier
            filename: Original filename
            field_data: Extracted/corrected field data
            table_data: Extracted/corrected table data
            engine: Extraction engine used
            confidence: Extraction confidence score
            processing_time: Time taken for processing
            file_size: File size in bytes
            user_id: User identifier
            metadata: Additional metadata
        
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if document already exists
                cursor.execute(
                    "SELECT id, version FROM document_corrections WHERE document_hash = ?",
                    (document_hash,)
                )
                existing = cursor.fetchone()
                
                current_time = datetime.now()
                metadata_json = json.dumps(metadata) if metadata else None
                
                if existing:
                    # Update existing document
                    new_version = existing['version'] + 1
                    cursor.execute("""
                        UPDATE document_corrections SET
                            filename = ?, corrected_field_data = ?, corrected_table_data = ?,
                            extraction_engine = ?, extraction_confidence = ?,
                            correction_count = correction_count + 1,
                            processing_time = ?, file_size = ?, updated_at = ?,
                            version = ?, metadata = ?, user_id = ?
                        WHERE document_hash = ?
                    """, (
                        filename, json.dumps(field_data), json.dumps(table_data),
                        engine, confidence, processing_time, file_size, current_time,
                        new_version, metadata_json, user_id, document_hash
                    ))
                    
                    # Log individual field corrections
                    self._save_field_corrections(cursor, document_hash, field_data, user_id)
                    
                    # Log individual table corrections
                    self._save_table_corrections(cursor, document_hash, table_data, user_id)
                    
                else:
                    # Insert new document
                    cursor.execute("""
                        INSERT INTO document_corrections (
                            document_hash, filename, original_field_data, corrected_field_data,
                            original_table_data, corrected_table_data, extraction_engine,
                            extraction_confidence, processing_time, file_size,
                            created_at, updated_at, metadata, user_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        document_hash, filename, json.dumps(field_data), json.dumps(field_data),
                        json.dumps(table_data), json.dumps(table_data), engine, confidence,
                        processing_time, file_size, current_time, current_time,
                        metadata_json, user_id
                    ))
                
                # Update system statistics
                self._update_system_stats(cursor)
                
                # Log the operation
                self.log_processing_event(
                    document_hash=document_hash,
                    operation="save_corrections",
                    status="success",
                    engine=engine,
                    processing_time=processing_time,
                    user_id=user_id
                )
                
                conn.commit()
                logger.info(f"Successfully saved corrections for document {document_hash}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save corrections for {document_hash}: {str(e)}")
            self.log_processing_event(
                document_hash=document_hash,
                operation="save_corrections",
                status="error",
                error_message=str(e),
                user_id=user_id
            )
            return False
    
    def _save_field_corrections(self, cursor, document_hash: str, field_data: Dict[str, Any], user_id: str = None):
        """Save individual field corrections for detailed tracking"""
        for field_name, corrected_value in field_data.items():
            # Get original value if it exists
            cursor.execute("""
                SELECT original_field_data FROM document_corrections WHERE document_hash = ?
            """, (document_hash,))
            result = cursor.fetchone()
            
            original_value = None
            if result and result['original_field_data']:
                try:
                    original_data = json.loads(result['original_field_data'])
                    original_value = original_data.get(field_name)
                except:
                    pass
            
            # Determine correction type
            correction_type = "new" if original_value is None else "modified" if original_value != corrected_value else "confirmed"
            
            cursor.execute("""
                INSERT INTO field_corrections (
                    document_hash, field_name, original_value, corrected_value,
                    correction_type, confidence, created_at, user_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_hash, field_name, str(original_value) if original_value is not None else None,
                str(corrected_value), correction_type, 1.0, datetime.now(), user_id
            ))
    
    def _save_table_corrections(self, cursor, document_hash: str, table_data: List[Dict[str, Any]], user_id: str = None):
        """Save individual table corrections for detailed tracking"""
        for table_index, table_row in enumerate(table_data):
            for row_index, (column_name, corrected_value) in enumerate(table_row.items()):
                # Get original value if it exists
                cursor.execute("""
                    SELECT original_table_data FROM document_corrections WHERE document_hash = ?
                """, (document_hash,))
                result = cursor.fetchone()
                
                original_value = None
                if result and result['original_table_data']:
                    try:
                        original_data = json.loads(result['original_table_data'])
                        if table_index < len(original_data) and column_name in original_data[table_index]:
                            original_value = original_data[table_index][column_name]
                    except:
                        pass
                
                # Determine correction type
                correction_type = "new" if original_value is None else "modified" if original_value != corrected_value else "confirmed"
                
                cursor.execute("""
                    INSERT INTO table_corrections (
                        document_hash, table_index, row_index, column_name,
                        original_value, corrected_value, correction_type,
                        confidence, created_at, user_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    document_hash, table_index, row_index, column_name,
                    str(original_value) if original_value is not None else None,
                    str(corrected_value), correction_type, 1.0, datetime.now(), user_id
                ))
    
    def load_document_corrections(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """
        Load document corrections from database
        
        Args:
            document_hash: Document identifier
            
        Returns:
            Dict with correction data or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM document_corrections WHERE document_hash = ?
                """, (document_hash,))
                
                result = cursor.fetchone()
                if not result:
                    return None
                
                # Parse JSON data
                field_data = json.loads(result['corrected_field_data']) if result['corrected_field_data'] else {}
                table_data = json.loads(result['corrected_table_data']) if result['corrected_table_data'] else []
                metadata = json.loads(result['metadata']) if result['metadata'] else {}
                
                return {
                    'document_hash': result['document_hash'],
                    'filename': result['filename'],
                    'field_data': field_data,
                    'table_data': table_data,
                    'engine': result['extraction_engine'],
                    'confidence': result['extraction_confidence'],
                    'last_updated': result['updated_at'],
                    'version': result['version'],
                    'metadata': metadata,
                    'correction_count': result['correction_count']
                }
                
        except Exception as e:
            logger.error(f"Failed to load corrections for {document_hash}: {str(e)}")
            return None
    
    def search_documents(self, query: str = None, limit: int = 100, 
                        start_date: datetime = None, end_date: datetime = None,
                        engine: str = None, min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Search documents with advanced filtering
        
        Args:
            query: Search query for filename or content
            limit: Maximum number of results
            start_date: Filter by creation date (start)
            end_date: Filter by creation date (end)
            engine: Filter by extraction engine
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching documents
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Build dynamic query
                where_conditions = []
                params = []
                
                if query:
                    where_conditions.append("(filename LIKE ? OR corrected_field_data LIKE ?)")
                    params.extend([f"%{query}%", f"%{query}%"])
                
                if start_date:
                    where_conditions.append("created_at >= ?")
                    params.append(start_date.isoformat())
                
                if end_date:
                    where_conditions.append("created_at <= ?")
                    params.append(end_date.isoformat())
                
                if engine:
                    where_conditions.append("extraction_engine = ?")
                    params.append(engine)
                
                if min_confidence is not None:
                    where_conditions.append("extraction_confidence >= ?")
                    params.append(min_confidence)
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                sql = f"""
                    SELECT document_hash, filename, extraction_engine, extraction_confidence,
                           created_at, updated_at, correction_count, version
                    FROM document_corrections
                    WHERE {where_clause}
                    ORDER BY updated_at DESC
                    LIMIT ?
                """
                
                params.append(limit)
                cursor.execute(sql, params)
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
                
        except Exception as e:
            logger.error(f"Document search failed: {str(e)}")
            return []
    
    def delete_document(self, document_hash: str) -> bool:
        """
        Delete document and all related data
        
        Args:
            document_hash: Document identifier
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete in order to respect foreign key constraints
                tables = [
                    'field_corrections',
                    'table_corrections', 
                    'document_corrections'
                ]
                
                for table in tables:
                    cursor.execute(f"DELETE FROM {table} WHERE document_hash = ?", (document_hash,))
                
                # Also delete related processing logs
                cursor.execute("DELETE FROM processing_logs WHERE document_hash = ?", (document_hash,))
                
                # Update system statistics
                self._update_system_stats(cursor)
                
                conn.commit()
                logger.info(f"Successfully deleted document {document_hash}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document {document_hash}: {str(e)}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        
        Returns:
            Dictionary with system statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Get stored statistics
                cursor.execute("SELECT stat_name, stat_value, stat_type FROM system_stats")
                for row in cursor.fetchall():
                    value = row['stat_value']
                    if row['stat_type'] == 'integer':
                        value = int(value)
                    elif row['stat_type'] == 'float':
                        value = float(value)
                    stats[row['stat_name']] = value
                
                # Calculate real-time statistics
                cursor.execute("SELECT COUNT(*) as count FROM document_corrections")
                stats['total_documents'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT SUM(correction_count) as count FROM document_corrections")
                result = cursor.fetchone()
                stats['total_corrections'] = result['count'] or 0
                
                cursor.execute("SELECT AVG(extraction_confidence) as avg FROM document_corrections")
                result = cursor.fetchone()
                stats['avg_confidence'] = result['avg'] or 0.0
                
                # Calculate success rate (documents with confidence > 0.7)
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN extraction_confidence > 0.7 THEN 1 ELSE 0 END) as successful
                    FROM document_corrections
                """)
                result = cursor.fetchone()
                if result['total'] > 0:
                    stats['success_rate'] = result['successful'] / result['total']
                else:
                    stats['success_rate'] = 0.0
                
                # Engine usage statistics
                cursor.execute("""
                    SELECT extraction_engine, COUNT(*) as count
                    FROM document_corrections
                    GROUP BY extraction_engine
                """)
                engine_stats = {}
                for row in cursor.fetchall():
                    engine_stats[row['extraction_engine']] = row['count']
                stats['engine_usage'] = engine_stats
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) as count FROM document_corrections
                    WHERE created_at > datetime('now', '-7 days')
                """)
                stats['documents_last_week'] = cursor.fetchone()['count']
                
                cursor.execute("""
                    SELECT COUNT(*) as count FROM processing_logs
                    WHERE created_at > datetime('now', '-24 hours') AND status = 'success'
                """)
                stats['successful_operations_today'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get system stats: {str(e)}")
            return {}
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently processed documents
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of recent documents
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT document_hash, filename, extraction_engine, extraction_confidence,
                           created_at, updated_at, correction_count
                    FROM document_corrections
                    ORDER BY updated_at DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append(dict(row))
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get recent documents: {str(e)}")
            return []
    
    def get_confidence_distribution(self) -> List[Dict[str, Any]]:
        """
        Get confidence score distribution for analytics
        
        Returns:
            List of confidence data points
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT extraction_confidence as confidence, COUNT(*) as count
                    FROM document_corrections
                    WHERE extraction_confidence IS NOT NULL
                    GROUP BY ROUND(extraction_confidence, 1)
                    ORDER BY confidence
                """)
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'confidence': row['confidence'],
                        'count': row['count']
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get confidence distribution: {str(e)}")
            return []
    
    def get_processing_logs(self, days: int = 7, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get processing logs for analytics
        
        Args:
            days: Number of days to look back
            limit: Maximum number of logs to return
            
        Returns:
            List of processing logs
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT log_id, document_hash, operation, status, engine,
                           processing_time, created_at, metadata
                    FROM processing_logs
                    WHERE created_at > datetime('now', '-{} days')
                    ORDER BY created_at DESC
                    LIMIT ?
                """.format(days), (limit,))
                
                results = []
                for row in cursor.fetchall():
                    log_entry = dict(row)
                    if log_entry['metadata']:
                        try:
                            log_entry['metadata'] = json.loads(log_entry['metadata'])
                        except:
                            pass
                    results.append(log_entry)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get processing logs: {str(e)}")
            return []
    
    def log_processing_event(self, document_hash: str = None, operation: str = "unknown",
                           status: str = "unknown", engine: str = None,
                           processing_time: float = 0.0, error_message: str = None,
                           metadata: Dict[str, Any] = None, user_id: str = None,
                           session_id: str = None) -> bool:
        """
        Log processing event for monitoring and analytics
        
        Args:
            document_hash: Document identifier
            operation: Operation type
            status: Operation status
            engine: Engine used
            processing_time: Time taken
            error_message: Error message if failed
            metadata: Additional metadata
            user_id: User identifier
            session_id: Session identifier
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                log_id = str(uuid.uuid4())
                metadata_json = json.dumps(metadata) if metadata else None
                
                cursor.execute("""
                    INSERT INTO processing_logs (
                        log_id, document_hash, operation, status, engine,
                        processing_time, error_message, created_at, metadata,
                        user_id, session_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id, document_hash, operation, status, engine,
                    processing_time, error_message, datetime.now(), metadata_json,
                    user_id, session_id
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to log processing event: {str(e)}")
            return False
    
    def save_pattern(self, pattern_id: str, pattern_type: str, pattern_data: Dict[str, Any],
                    confidence: float = 0.0, metadata: Dict[str, Any] = None) -> bool:
        """
        Save learned pattern to database
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of pattern
            pattern_data: Pattern data
            confidence: Pattern confidence
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata) if metadata else None
                pattern_data_json = json.dumps(pattern_data)
                
                cursor.execute("""
                    INSERT OR REPLACE INTO pattern_learning (
                        pattern_id, pattern_type, pattern_data, confidence,
                        created_at, updated_at, metadata, usage_count, success_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0)
                """, (
                    pattern_id, pattern_type, pattern_data_json, confidence,
                    datetime.now(), datetime.now(), metadata_json
                ))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save pattern {pattern_id}: {str(e)}")
            return False
    
    def get_patterns(self, pattern_type: str = None, min_confidence: float = 0.0,
                    active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Get learned patterns from database
        
        Args:
            pattern_type: Filter by pattern type
            min_confidence: Minimum confidence threshold
            active_only: Only return active patterns
            
        Returns:
            List of patterns
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                where_conditions = []
                params = []
                
                if pattern_type:
                    where_conditions.append("pattern_type = ?")
                    params.append(pattern_type)
                
                if min_confidence > 0:
                    where_conditions.append("confidence >= ?")
                    params.append(min_confidence)
                
                if active_only:
                    where_conditions.append("is_active = 1")
                
                where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
                
                sql = f"""
                    SELECT * FROM pattern_learning
                    WHERE {where_clause}
                    ORDER BY confidence DESC, usage_count DESC
                """
                
                cursor.execute(sql, params)
                
                results = []
                for row in cursor.fetchall():
                    pattern = dict(row)
                    if pattern['pattern_data']:
                        pattern['pattern_data'] = json.loads(pattern['pattern_data'])
                    if pattern['metadata']:
                        try:
                            pattern['metadata'] = json.loads(pattern['metadata'])
                        except:
                            pass
                    results.append(pattern)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get patterns: {str(e)}")
            return []
    
    def update_pattern_usage(self, pattern_id: str, success: bool = True) -> bool:
        """
        Update pattern usage statistics
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the pattern was successful
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                if success:
                    cursor.execute("""
                        UPDATE pattern_learning SET
                            usage_count = usage_count + 1,
                            success_count = success_count + 1,
                            last_used = ?,
                            updated_at = ?
                        WHERE pattern_id = ?
                    """, (datetime.now(), datetime.now(), pattern_id))
                else:
                    cursor.execute("""
                        UPDATE pattern_learning SET
                            usage_count = usage_count + 1,
                            last_used = ?,
                            updated_at = ?
                        WHERE pattern_id = ?
                    """, (datetime.now(), datetime.now(), pattern_id))
                
                # Recalculate confidence based on success rate
                cursor.execute("""
                    UPDATE pattern_learning SET
                        confidence = CASE 
                            WHEN usage_count > 0 THEN CAST(success_count AS FLOAT) / usage_count
                            ELSE 0.0
                        END
                    WHERE pattern_id = ?
                """, (pattern_id,))
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to update pattern usage for {pattern_id}: {str(e)}")
            return False
    
    def backup_database(self, backup_type: str = "manual") -> str:
        """
        Create database backup
        
        Args:
            backup_type: Type of backup (manual, automatic, scheduled)
            
        Returns:
            str: Backup file path
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"pdf_extraction_backup_{timestamp}.db"
            backup_path = self.backup_dir / backup_filename
            
            # Create backup
            shutil.copy2(self.db_path, backup_path)
            
            # Record backup metadata
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                backup_id = str(uuid.uuid4())
                file_size = os.path.getsize(backup_path)
                expires_at = datetime.now() + timedelta(days=BACKUP_RETENTION_DAYS)
                
                cursor.execute("""
                    INSERT INTO backup_metadata (
                        backup_id, backup_path, backup_size, backup_type,
                        created_at, expires_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    backup_id, str(backup_path), file_size, backup_type,
                    datetime.now(), expires_at
                ))
                
                conn.commit()
            
            logger.info(f"Database backed up to {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Database backup failed: {str(e)}")
            raise BackupError(f"Backup failed: {str(e)}")
    
    def cleanup_old_data(self, days: int = 90) -> bool:
        """
        Clean up old data to maintain database performance
        
        Args:
            days: Number of days to retain data
            
        Returns:
            bool: Success status
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cutoff_date = datetime.now() - timedelta(days=days)
                
                # Clean up old processing logs
                cursor.execute("""
                    DELETE FROM processing_logs 
                    WHERE created_at < ? AND status != 'error'
                """, (cutoff_date,))
                logs_deleted = cursor.rowcount
                
                # Clean up expired backups
                cursor.execute("""
                    SELECT backup_path FROM backup_metadata 
                    WHERE expires_at < ?
                """, (datetime.now(),))
                
                expired_backups = cursor.fetchall()
                for backup in expired_backups:
                    try:
                        backup_path = Path(backup['backup_path'])
                        if backup_path.exists():
                            backup_path.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete backup file: {e}")
                
                cursor.execute("""
                    DELETE FROM backup_metadata WHERE expires_at < ?
                """, (datetime.now(),))
                backups_deleted = cursor.rowcount
                
                # Update last cleanup timestamp
                cursor.execute("""
                    UPDATE system_stats SET stat_value = ?, updated_at = ?
                    WHERE stat_name = 'last_cleanup'
                """, (datetime.now().isoformat(), datetime.now()))
                
                conn.commit()
                
                logger.info(f"Cleanup completed: {logs_deleted} logs deleted, {backups_deleted} backups removed")
                return True
                
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            return False
    
    def _update_system_stats(self, cursor):
        """Update system statistics"""
        try:
            # Get current counts
            cursor.execute("SELECT COUNT(*) as count FROM document_corrections")
            total_docs = cursor.fetchone()['count']
            
            cursor.execute("SELECT SUM(correction_count) as count FROM document_corrections")
            result = cursor.fetchone()
            total_corrections = result['count'] or 0
            
            cursor.execute("SELECT AVG(extraction_confidence) as avg FROM document_corrections")
            result = cursor.fetchone()
            avg_confidence = result['avg'] or 0.0
            
            # Update stats
            stats_updates = [
                ('total_documents', str(total_docs)),
                ('total_corrections', str(total_corrections)),
                ('avg_confidence', str(avg_confidence))
            ]
            
            for stat_name, stat_value in stats_updates:
                cursor.execute("""
                    UPDATE system_stats SET stat_value = ?, updated_at = ?
                    WHERE stat_name = ?
                """, (stat_value, datetime.now(), stat_name))
                
        except Exception as e:
            logger.warning(f"Failed to update system stats: {str(e)}")
    
    def _setup_cleanup_schedule(self):
        """Setup automatic cleanup schedule"""
        def cleanup_worker():
            while True:
                try:
                    time.sleep(86400)  # 24 hours
                    self.cleanup_old_data()
                except Exception as e:
                    logger.error(f"Scheduled cleanup failed: {str(e)}")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def close(self):
        """Close all database connections"""
        with self._lock:
            for conn in self._connection_pool.values():
                try:
                    conn.close()
                except:
                    pass
            self._connection_pool.clear()
        
        logger.info("Database connections closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Export main class
__all__ = ['DatabaseManager', 'DatabaseError', 'BackupError']