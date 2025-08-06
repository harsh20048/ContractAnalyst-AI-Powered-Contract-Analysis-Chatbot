"""
Database Manager Module for PDF Table Extraction System
======================================================

Comprehensive database management module for handling:
- Document corrections storage and retrieval
- Pattern learning data management
- System statistics and analytics
- Database maintenance and optimization
- Backup and recovery operations

Author: AI Assistant
Version: 2.0.0
License: MIT
"""

import sqlite3
import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import hashlib
import pickle
import gzip
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class DocumentRecord:
    """Data class for document records"""
    document_hash: str
    filename: str
    file_size: int
    upload_date: datetime
    last_modified: datetime
    fields: Dict[str, Any]
    tables: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    extraction_confidence: float = 0.0
    processing_time: float = 0.0
    correction_count: int = 0

@dataclass
class PatternRecord:
    """Data class for pattern learning records"""
    pattern_id: str
    pattern_type: str  # 'field' or 'table'
    pattern_data: Dict[str, Any]
    confidence: float
    usage_count: int
    success_rate: float
    created_date: datetime
    last_used: datetime
    metadata: Dict[str, Any]

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

class DatabaseManager:
    """Comprehensive database manager for PDF extraction system"""
    
    def __init__(self, db_path: str = "pdf_extraction.db"):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.connection_pool = {}
        self.lock = threading.RLock()
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize database with required tables"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Document corrections table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS document_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_hash TEXT UNIQUE NOT NULL,
                        filename TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        upload_date TIMESTAMP NOT NULL,
                        last_modified TIMESTAMP NOT NULL,
                        fields TEXT NOT NULL,
                        tables TEXT NOT NULL,
                        metadata TEXT,
                        extraction_confidence REAL DEFAULT 0.0,
                        processing_time REAL DEFAULT 0.0,
                        correction_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Field corrections table for detailed tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS field_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_hash TEXT NOT NULL,
                        field_name TEXT NOT NULL,
                        original_value TEXT,
                        corrected_value TEXT NOT NULL,
                        correction_type TEXT DEFAULT 'manual',
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_hash) REFERENCES document_corrections(document_hash)
                    )
                """)
                
                # Table corrections table for detailed tracking
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS table_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_hash TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        table_index INTEGER NOT NULL,
                        original_data TEXT,
                        corrected_data TEXT NOT NULL,
                        correction_type TEXT DEFAULT 'manual',
                        confidence REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (document_hash) REFERENCES document_corrections(document_hash)
                    )
                """)
                
                # Pattern learning table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS pattern_learning (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        pattern_id TEXT UNIQUE NOT NULL,
                        pattern_type TEXT NOT NULL CHECK (pattern_type IN ('field', 'table')),
                        pattern_name TEXT NOT NULL,
                        pattern_data TEXT NOT NULL,
                        confidence REAL DEFAULT 0.0,
                        usage_count INTEGER DEFAULT 0,
                        success_rate REAL DEFAULT 0.0,
                        created_date TIMESTAMP NOT NULL,
                        last_used TIMESTAMP,
                        metadata TEXT,
                        is_active BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # System statistics table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS system_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stat_name TEXT UNIQUE NOT NULL,
                        stat_value TEXT NOT NULL,
                        stat_type TEXT DEFAULT 'counter',
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Processing logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS processing_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        document_hash TEXT,
                        operation_type TEXT NOT NULL,
                        operation_status TEXT NOT NULL,
                        processing_time REAL,
                        error_message TEXT,
                        metadata TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_hash ON document_corrections(document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_filename ON document_corrections(filename)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_date ON document_corrections(upload_date)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_corrections_hash ON field_corrections(document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_table_corrections_hash ON table_corrections(document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_learning(pattern_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_active ON pattern_learning(is_active)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_logs_hash ON processing_logs(document_hash)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_logs_type ON processing_logs(operation_type)")
                
                # Initialize system stats
                self._initialize_system_stats(cursor)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    def _initialize_system_stats(self, cursor):
        """Initialize system statistics"""
        default_stats = [
            ('total_documents', '0', 'counter'),
            ('total_corrections', '0', 'counter'),
            ('total_patterns', '0', 'counter'),
            ('database_version', '2.0.0', 'string'),
            ('last_backup', '', 'timestamp'),
            ('system_uptime', '', 'timestamp')
        ]
        
        for stat_name, stat_value, stat_type in default_stats:
            cursor.execute("""
                INSERT OR IGNORE INTO system_stats (stat_name, stat_value, stat_type)
                VALUES (?, ?, ?)
            """, (stat_name, stat_value, stat_type))
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper context management"""
        conn = None
        try:
            with self.lock:
                conn = sqlite3.connect(
                    self.db_path,
                    timeout=30.0,
                    isolation_level=None,
                    check_same_thread=False
                )
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA temp_store = MEMORY")
                conn.execute("PRAGMA mmap_size = 268435456")  # 256MB
                
            yield conn
            
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database connection error: {e}")
            raise DatabaseError(f"Database connection failed: {e}")
        finally:
            if conn:
                conn.close()
    
    def save_document_corrections(self, document_hash: str, fields: Dict[str, Any], 
                                tables: List[Dict[str, Any]], metadata: Dict[str, Any] = None) -> bool:
        """
        Save document corrections to database
        
        Args:
            document_hash: Unique document identifier
            fields: Field corrections data
            tables: Table corrections data
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Convert complex data to JSON
                fields_json = json.dumps(fields, default=str)
                tables_json = json.dumps(tables, default=str)
                metadata_json = json.dumps(metadata or {}, default=str)
                
                # Check if document already exists
                cursor.execute(
                    "SELECT id FROM document_corrections WHERE document_hash = ?",
                    (document_hash,)
                )
                existing = cursor.fetchone()
                
                current_time = datetime.now()
                
                if existing:
                    # Update existing record
                    cursor.execute("""
                        UPDATE document_corrections 
                        SET fields = ?, tables = ?, metadata = ?, 
                            last_modified = ?, correction_count = correction_count + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE document_hash = ?
                    """, (fields_json, tables_json, metadata_json, current_time, document_hash))
                    
                    logger.info(f"Updated corrections for document {document_hash[:8]}...")
                else:
                    # Insert new record
                    filename = metadata.get('filename', 'unknown') if metadata else 'unknown'
                    file_size = metadata.get('file_size', 0) if metadata else 0
                    
                    cursor.execute("""
                        INSERT INTO document_corrections 
                        (document_hash, filename, file_size, upload_date, last_modified, 
                         fields, tables, metadata, correction_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """, (document_hash, filename, file_size, current_time, current_time,
                          fields_json, tables_json, metadata_json))
                    
                    logger.info(f"Saved new corrections for document {document_hash[:8]}...")
                
                # Save detailed field corrections
                for field_name, field_value in fields.items():
                    cursor.execute("""
                        INSERT INTO field_corrections 
                        (document_hash, field_name, corrected_value)
                        VALUES (?, ?, ?)
                    """, (document_hash, field_name, str(field_value)))
                
                # Save detailed table corrections
                for i, table_data in enumerate(tables):
                    table_json = json.dumps(table_data, default=str)
                    cursor.execute("""
                        INSERT INTO table_corrections 
                        (document_hash, table_name, table_index, corrected_data)
                        VALUES (?, ?, ?, ?)
                    """, (document_hash, f"table_{i+1}", i, table_json))
                
                # Update system statistics
                self._update_stat(cursor, 'total_corrections', 1, 'increment')
                if not existing:
                    self._update_stat(cursor, 'total_documents', 1, 'increment')
                
                # Log the operation
                self._log_operation(cursor, document_hash, 'save_corrections', 'success', 0.0)
                
                conn.commit()
                return True
                
        except Exception as e:
            logger.error(f"Failed to save corrections: {e}")
            self._log_operation(None, document_hash, 'save_corrections', 'failed', 0.0, str(e))
            return False
    
    def load_document_corrections(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """
        Load document corrections from database
        
        Args:
            document_hash: Document identifier
            
        Returns:
            Dict containing fields and tables data, or None if not found
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT fields, tables, metadata, filename, upload_date, 
                           correction_count, extraction_confidence, processing_time
                    FROM document_corrections 
                    WHERE document_hash = ?
                """, (document_hash,))
                
                result = cursor.fetchone()
                
                if result:
                    fields = json.loads(result['fields'])
                    tables = json.loads(result['tables'])
                    metadata = json.loads(result['metadata'] or '{}')
                    
                    # Convert table data back to DataFrames if needed
                    if isinstance(tables, list):
                        processed_tables = []
                        for table_data in tables:
                            if isinstance(table_data, dict) and 'records' in table_data:
                                # Table was stored as records
                                df = pd.DataFrame(table_data['records'])
                                processed_tables.append(df)
                            elif isinstance(table_data, list):
                                # Table was stored as list of records
                                df = pd.DataFrame(table_data)
                                processed_tables.append(df)
                            else:
                                processed_tables.append(table_data)
                        tables = processed_tables
                    
                    corrections_data = {
                        'fields': fields,
                        'tables': tables,
                        'metadata': metadata,
                        'document_info': {
                            'filename': result['filename'],
                            'upload_date': result['upload_date'],
                            'correction_count': result['correction_count'],
                            'extraction_confidence': result['extraction_confidence'],
                            'processing_time': result['processing_time']
                        }
                    }
                    
                    logger.info(f"Loaded corrections for document {document_hash[:8]}...")
                    return corrections_data
                else:
                    logger.info(f"No corrections found for document {document_hash[:8]}...")
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to load corrections: {e}")
            return None
    
    def search_documents(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Search documents by filename or content
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Search by filename, fields content, or metadata
                cursor.execute("""
                    SELECT document_hash, filename, upload_date, correction_count,
                           extraction_confidence, file_size
                    FROM document_corrections 
                    WHERE filename LIKE ? OR fields LIKE ? OR metadata LIKE ?
                    ORDER BY upload_date DESC
                    LIMIT ?
                """, (f"%{query}%", f"%{query}%", f"%{query}%", limit))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'document_hash': row['document_hash'],
                        'filename': row['filename'],
                        'upload_date': row['upload_date'],
                        'correction_count': row['correction_count'],
                        'extraction_confidence': row['extraction_confidence'],
                        'file_size': row['file_size']
                    })
                
                logger.info(f"Found {len(results)} documents matching query: {query}")
                return results
                
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recently processed documents
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of recent documents
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT document_hash, filename, upload_date, last_modified,
                           correction_count, extraction_confidence, file_size,
                           processing_time
                    FROM document_corrections 
                    ORDER BY last_modified DESC
                    LIMIT ?
                """, (limit,))
                
                results = []
                for row in cursor.fetchall():
                    results.append({
                        'document_hash': row['document_hash'][:8] + '...',
                        'filename': row['filename'],
                        'upload_date': row['upload_date'],
                        'last_modified': row['last_modified'],
                        'correction_count': row['correction_count'],
                        'confidence': f"{row['extraction_confidence']:.1%}",
                        'file_size_kb': f"{row['file_size'] / 1024:.1f}",
                        'processing_time': f"{row['processing_time']:.2f}s"
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to get recent documents: {e}")
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
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Delete from all related tables
                cursor.execute("DELETE FROM field_corrections WHERE document_hash = ?", (document_hash,))
                cursor.execute("DELETE FROM table_corrections WHERE document_hash = ?", (document_hash,))
                cursor.execute("DELETE FROM processing_logs WHERE document_hash = ?", (document_hash,))
                cursor.execute("DELETE FROM document_corrections WHERE document_hash = ?", (document_hash,))
                
                # Update statistics
                if cursor.rowcount > 0:
                    self._update_stat(cursor, 'total_documents', -1, 'increment')
                    
                conn.commit()
                logger.info(f"Deleted document {document_hash[:8]}...")
                return True
                
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def save_pattern(self, pattern_id: str, pattern_type: str, pattern_name: str,
                    pattern_data: Dict[str, Any], confidence: float = 0.0,
                    metadata: Dict[str, Any] = None) -> bool:
        """
        Save a learned pattern to database
        
        Args:
            pattern_id: Unique pattern identifier
            pattern_type: Type of pattern ('field' or 'table')
            pattern_name: Human-readable pattern name
            pattern_data: Pattern data
            confidence: Pattern confidence score
            metadata: Additional metadata
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                pattern_json = json.dumps(pattern_data, default=str)
                metadata_json = json.dumps(metadata or {}, default=str)
                current_time = datetime.now()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO pattern_learning
                    (pattern_id, pattern_type, pattern_name, pattern_data, confidence,
                     created_date, metadata, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1)
                """, (pattern_id, pattern_type, pattern_name, pattern_json, 
                      confidence, current_time, metadata_json))
                
                # Update statistics
                self._update_stat(cursor, 'total_patterns', 1, 'increment')
                
                conn.commit()
                logger.info(f"Saved pattern {pattern_id}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to save pattern: {e}")
            return False
    
    def load_patterns(self, pattern_type: str = None, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        Load patterns from database
        
        Args:
            pattern_type: Filter by pattern type ('field' or 'table')
            active_only: Only return active patterns
            
        Returns:
            List of patterns
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM pattern_learning WHERE 1=1"
                params = []
                
                if pattern_type:
                    query += " AND pattern_type = ?"
                    params.append(pattern_type)
                
                if active_only:
                    query += " AND is_active = 1"
                
                query += " ORDER BY confidence DESC, usage_count DESC"
                
                cursor.execute(query, params)
                
                patterns = []
                for row in cursor.fetchall():
                    pattern_data = json.loads(row['pattern_data'])
                    metadata = json.loads(row['metadata'] or '{}')
                    
                    patterns.append({
                        'pattern_id': row['pattern_id'],
                        'pattern_type': row['pattern_type'],
                        'pattern_name': row['pattern_name'],
                        'pattern_data': pattern_data,
                        'confidence': row['confidence'],
                        'usage_count': row['usage_count'],
                        'success_rate': row['success_rate'],
                        'created_date': row['created_date'],
                        'last_used': row['last_used'],
                        'metadata': metadata,
                        'is_active': bool(row['is_active'])
                    })
                
                logger.info(f"Loaded {len(patterns)} patterns")
                return patterns
                
        except Exception as e:
            logger.error(f"Failed to load patterns: {e}")
            return []
    
    def update_pattern_usage(self, pattern_id: str, success: bool = True) -> bool:
        """
        Update pattern usage statistics
        
        Args:
            pattern_id: Pattern identifier
            success: Whether the pattern was successfully applied
            
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get current stats
                cursor.execute("""
                    SELECT usage_count, success_rate FROM pattern_learning 
                    WHERE pattern_id = ?
                """, (pattern_id,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                current_usage = result['usage_count']
                current_success_rate = result['success_rate']
                
                # Calculate new success rate
                total_attempts = current_usage + 1
                successful_attempts = int(current_success_rate * current_usage)
                if success:
                    successful_attempts += 1
                
                new_success_rate = successful_attempts / total_attempts if total_attempts > 0 else 0.0
                
                # Update pattern
                cursor.execute("""
                    UPDATE pattern_learning 
                    SET usage_count = ?, success_rate = ?, last_used = CURRENT_TIMESTAMP,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE pattern_id = ?
                """, (total_attempts, new_success_rate, pattern_id))
                
                conn.commit()
                logger.debug(f"Updated pattern {pattern_id} usage stats")
                return True
                
        except Exception as e:
            logger.error(f"Failed to update pattern usage: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics
        
        Returns:
            Dictionary of system statistics
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Get basic stats from system_stats table
                cursor.execute("SELECT stat_name, stat_value, stat_type FROM system_stats")
                for row in cursor.fetchall():
                    stats[row['stat_name']] = row['stat_value']
                
                # Calculate additional statistics
                
                # Document statistics
                cursor.execute("SELECT COUNT(*) as count FROM document_corrections")
                stats['total_documents'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT AVG(extraction_confidence) as avg_conf FROM document_corrections")
                result = cursor.fetchone()
                stats['avg_extraction_confidence'] = result['avg_conf'] or 0.0
                
                cursor.execute("SELECT AVG(processing_time) as avg_time FROM document_corrections")
                result = cursor.fetchone()
                stats['avg_processing_time'] = result['avg_time'] or 0.0
                
                # Pattern statistics
                cursor.execute("SELECT COUNT(*) as count FROM pattern_learning WHERE is_active = 1")
                stats['active_patterns'] = cursor.fetchone()['count']
                
                cursor.execute("SELECT AVG(success_rate) as avg_success FROM pattern_learning WHERE is_active = 1")
                result = cursor.fetchone()
                stats['avg_pattern_success_rate'] = result['avg_success'] or 0.0
                
                # Database size
                try:
                    db_size = os.path.getsize(self.db_path)
                    stats['database_size'] = db_size
                    stats['database_size_mb'] = db_size / (1024 * 1024)
                except:
                    stats['database_size'] = 0
                    stats['database_size_mb'] = 0.0
                
                # Recent activity
                cursor.execute("""
                    SELECT COUNT(*) as count FROM document_corrections 
                    WHERE upload_date > date('now', '-7 days')
                """)
                stats['documents_last_week'] = cursor.fetchone()['count']
                
                cursor.execute("""
                    SELECT COUNT(*) as count FROM processing_logs 
                    WHERE created_at > date('now', '-1 day')
                """)
                stats['operations_last_day'] = cursor.fetchone()['count']
                
                # Error statistics
                cursor.execute("""
                    SELECT COUNT(*) as count FROM processing_logs 
                    WHERE operation_status = 'failed'
                """)
                stats['total_errors'] = cursor.fetchone()['count']
                
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def get_analytics_data(self, days: int = 30) -> Dict[str, Any]:
        """
        Get analytics data for dashboard
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Analytics data dictionary
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                since_date = datetime.now() - timedelta(days=days)
                analytics = {}
                
                # Document processing over time
                cursor.execute("""
                    SELECT DATE(upload_date) as date, COUNT(*) as count
                    FROM document_corrections 
                    WHERE upload_date > ?
                    GROUP BY DATE(upload_date)
                    ORDER BY date
                """, (since_date,))
                
                processing_trend = []
                for row in cursor.fetchall():
                    processing_trend.append({
                        'date': row['date'],
                        'count': row['count']
                    })
                analytics['processing_trend'] = processing_trend
                
                # Confidence distribution
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN extraction_confidence >= 0.9 THEN 'High (90%+)'
                            WHEN extraction_confidence >= 0.7 THEN 'Medium (70-90%)'
                            WHEN extraction_confidence >= 0.5 THEN 'Low (50-70%)'
                            ELSE 'Very Low (<50%)'
                        END as confidence_range,
                        COUNT(*) as count
                    FROM document_corrections
                    WHERE upload_date > ?
                    GROUP BY confidence_range
                """, (since_date,))
                
                confidence_dist = []
                for row in cursor.fetchall():
                    confidence_dist.append({
                        'range': row['confidence_range'],
                        'count': row['count']
                    })
                analytics['confidence_distribution'] = confidence_dist
                
                # Top file types
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN filename LIKE '%.pdf' THEN 'PDF'
                            WHEN filename LIKE '%.doc%' THEN 'Word'
                            WHEN filename LIKE '%.xls%' THEN 'Excel'
                            ELSE 'Other'
                        END as file_type,
                        COUNT(*) as count
                    FROM document_corrections
                    WHERE upload_date > ?
                    GROUP BY file_type
                    ORDER BY count DESC
                """, (since_date,))
                
                file_types = []
                for row in cursor.fetchall():
                    file_types.append({
                        'type': row['file_type'],
                        'count': row['count']
                    })
                analytics['file_types'] = file_types
                
                # Processing performance
                cursor.execute("""
                    SELECT 
                        AVG(processing_time) as avg_time,
                        MIN(processing_time) as min_time,
                        MAX(processing_time) as max_time,
                        COUNT(*) as total_processed
                    FROM document_corrections
                    WHERE upload_date > ? AND processing_time > 0
                """, (since_date,))
                
                perf_data = cursor.fetchone()
                analytics['performance'] = {
                    'avg_processing_time': perf_data['avg_time'] or 0.0,
                    'min_processing_time': perf_data['min_time'] or 0.0,
                    'max_processing_time': perf_data['max_time'] or 0.0,
                    'total_processed': perf_data['total_processed'] or 0
                }
                
                # Error analysis
                cursor.execute("""
                    SELECT operation_type, COUNT(*) as error_count
                    FROM processing_logs
                    WHERE operation_status = 'failed' AND created_at > ?
                    GROUP BY operation_type
                    ORDER BY error_count DESC
                """, (since_date,))
                
                error_analysis = []
                for row in cursor.fetchall():
                    error_analysis.append({
                        'operation': row['operation_type'],
                        'count': row['error_count']
                    })
                analytics['error_analysis'] = error_analysis
                
                return analytics
                
        except Exception as e:
            logger.error(f"Failed to get analytics data: {e}")
            return {}
    
    def backup_database(self, backup_path: str = None) -> str:
        """
        Create database backup
        
        Args:
            backup_path: Custom backup path
            
        Returns:
            str: Path to backup file
        """
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"backup_pdf_extraction_{timestamp}.db"
            
            # Create backup using file copy
            shutil.copy2(self.db_path, backup_path)
            
            # Update backup timestamp in stats
            with self.get_connection() as conn:
                cursor = conn.cursor()
                self._update_stat(cursor, 'last_backup', datetime.now().isoformat(), 'set')
                conn.commit()
            
            logger.info(f"Database backed up to {backup_path}")
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup failed: {e}")
            raise DatabaseError(f"Failed to create backup: {e}")
    
    def cleanup_old_records(self, days: int = 90) -> int:
        """
        Clean up old records from database
        
        Args:
            days: Number of days to keep
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            deleted_count = 0
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Clean up old processing logs
                cursor.execute("""
                    DELETE FROM processing_logs 
                    WHERE created_at < ? AND operation_status != 'failed'
                """, (cutoff_date,))
                deleted_count += cursor.rowcount
                
                # Clean up old field corrections (keep document corrections)
                cursor.execute("""
                    DELETE FROM field_corrections 
                    WHERE created_at < ? AND document_hash NOT IN (
                        SELECT document_hash FROM document_corrections 
                        WHERE last_modified > ?
                    )
                """, (cutoff_date, cutoff_date))
                deleted_count += cursor.rowcount
                
                # Clean up old table corrections
                cursor.execute("""
                    DELETE FROM table_corrections 
                    WHERE created_at < ? AND document_hash NOT IN (
                        SELECT document_hash FROM document_corrections 
                        WHERE last_modified > ?
                    )
                """, (cutoff_date, cutoff_date))
                deleted_count += cursor.rowcount
                
                # Clean up unused patterns
                cursor.execute("""
                    UPDATE pattern_learning 
                    SET is_active = 0 
                    WHERE last_used < ? AND usage_count < 5
                """, (cutoff_date,))
                
                conn.commit()
                logger.info(f"Cleaned up {deleted_count} old records")
                return deleted_count
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
            return 0
    
    def optimize_database(self) -> bool:
        """
        Optimize database performance
        
        Returns:
            bool: Success status
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Analyze tables for query optimization
                cursor.execute("ANALYZE")
                
                # Vacuum to reclaim space and defragment
                cursor.execute("VACUUM")
                
                # Update table statistics
                cursor.execute("PRAGMA optimize")
                
                conn.commit()
                logger.info("Database optimization completed")
                return True
                
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            return False
    
    def export_data(self, export_path: str, include_patterns: bool = True) -> bool:
        """
        Export all data to JSON file
        
        Args:
            export_path: Path to export file
            include_patterns: Whether to include pattern data
            
        Returns:
            bool: Success status
        """
        try:
            export_data = {}
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Export document corrections
                cursor.execute("SELECT * FROM document_corrections")
                documents = []
                for row in cursor.fetchall():
                    doc_data = dict(row)
                    # Parse JSON fields
                    doc_data['fields'] = json.loads(doc_data['fields'])
                    doc_data['tables'] = json.loads(doc_data['tables'])
                    doc_data['metadata'] = json.loads(doc_data['metadata'] or '{}')
                    documents.append(doc_data)
                
                export_data['documents'] = documents
                
                # Export patterns if requested
                if include_patterns:
                    cursor.execute("SELECT * FROM pattern_learning")
                    patterns = []
                    for row in cursor.fetchall():
                        pattern_data = dict(row)
                        pattern_data['pattern_data'] = json.loads(pattern_data['pattern_data'])
                        pattern_data['metadata'] = json.loads(pattern_data['metadata'] or '{}')
                        patterns.append(pattern_data)
                    
                    export_data['patterns'] = patterns
                
                # Export system stats
                export_data['system_stats'] = self.get_system_stats()
                export_data['export_timestamp'] = datetime.now().isoformat()
                export_data['export_version'] = '2.0.0'
            
            # Write to file (compressed if large)
            if len(json.dumps(export_data)) > 1024 * 1024:  # > 1MB
                with gzip.open(export_path, 'wt', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            else:
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Data exported to {export_path}")
            return True
            
        except Exception as e:
            logger.error(f"Data export failed: {e}")
            return False
    
    def import_data(self, import_path: str) -> bool:
        """
        Import data from JSON file
        
        Args:
            import_path: Path to import file
            
        Returns:
            bool: Success status
        """
        try:
            # Read data from file
            if import_path.endswith('.gz'):
                with gzip.open(import_path, 'rt', encoding='utf-8') as f:
                    import_data = json.load(f)
            else:
                with open(import_path, 'r', encoding='utf-8') as f:
                    import_data = json.load(f)
            
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Import documents
                if 'documents' in import_data:
                    for doc in import_data['documents']:
                        self.save_document_corrections(
                            doc['document_hash'],
                            doc['fields'],
                            doc['tables'],
                            doc.get('metadata', {})
                        )
                
                # Import patterns
                if 'patterns' in import_data:
                    for pattern in import_data['patterns']:
                        self.save_pattern(
                            pattern['pattern_id'],
                            pattern['pattern_type'],
                            pattern['pattern_name'],
                            pattern['pattern_data'],
                            pattern.get('confidence', 0.0),
                            pattern.get('metadata', {})
                        )
                
                conn.commit()
            
            logger.info(f"Data imported from {import_path}")
            return True
            
        except Exception as e:
            logger.error(f"Data import failed: {e}")
            return False
    
    def calculate_extraction_confidence(self, fields: Dict[str, Any], 
                                      tables: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score for extraction
        
        Args:
            fields: Extracted fields
            tables: Extracted tables
            
        Returns:
            float: Confidence score between 0 and 1
        """
        try:
            total_score = 0.0
            total_weight = 0.0
            
            # Field confidence scoring
            for field_name, field_value in fields.items():
                weight = 1.0
                score = 0.0
                
                if field_value is not None and str(field_value).strip():
                    score = 0.7  # Base score for non-empty field
                    
                    # Bonus for fields with expected patterns
                    if field_name.lower() in ['date', 'amount', 'total', 'email', 'phone']:
                        score += 0.2
                    
                    # Penalty for very short values
                    if len(str(field_value).strip()) < 3:
                        score -= 0.2
                
                total_score += score * weight
                total_weight += weight
            
            # Table confidence scoring
            for table_data in tables:
                weight = 2.0  # Tables are weighted more heavily
                score = 0.0
                
                if isinstance(table_data, dict) and 'records' in table_data:
                    records = table_data['records']
                elif isinstance(table_data, list):
                    records = table_data
                else:
                    continue
                
                if records and len(records) > 1:
                    score = 0.6  # Base score for valid table
                    
                    # Bonus for consistent columns
                    if all(isinstance(record, dict) for record in records):
                        first_keys = set(records[0].keys()) if records else set()
                        if all(set(record.keys()) == first_keys for record in records[1:]):
                            score += 0.3
                
                total_score += score * weight
                total_weight += weight
            
            # Calculate final confidence
            confidence = total_score / total_weight if total_weight > 0 else 0.0
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            logger.error(f"Confidence calculation failed: {e}")
            return 0.0
    
    def _update_stat(self, cursor, stat_name: str, value: Union[int, float, str], 
                    operation: str = 'set'):
        """Update system statistic"""
        try:
            if operation == 'increment':
                cursor.execute("""
                    UPDATE system_stats 
                    SET stat_value = CAST(stat_value AS INTEGER) + ?, 
                        last_updated = CURRENT_TIMESTAMP
                    WHERE stat_name = ?
                """, (value, stat_name))
            elif operation == 'set':
                cursor.execute("""
                    UPDATE system_stats 
                    SET stat_value = ?, last_updated = CURRENT_TIMESTAMP
                    WHERE stat_name = ?
                """, (str(value), stat_name))
                
        except Exception as e:
            logger.error(f"Failed to update stat {stat_name}: {e}")
    
    def _log_operation(self, cursor, document_hash: str, operation_type: str,
                      operation_status: str, processing_time: float,
                      error_message: str = None, metadata: Dict[str, Any] = None):
        """Log an operation to the processing logs"""
        try:
            if cursor is None:
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    self._log_operation_internal(cursor, document_hash, operation_type,
                                               operation_status, processing_time,
                                               error_message, metadata)
                    conn.commit()
            else:
                self._log_operation_internal(cursor, document_hash, operation_type,
                                           operation_status, processing_time,
                                           error_message, metadata)
                
        except Exception as e:
            logger.error(f"Failed to log operation: {e}")
    
    def _log_operation_internal(self, cursor, document_hash: str, operation_type: str,
                               operation_status: str, processing_time: float,
                               error_message: str = None, metadata: Dict[str, Any] = None):
        """Internal method to log operation"""
        metadata_json = json.dumps(metadata or {}, default=str)
        
        cursor.execute("""
            INSERT INTO processing_logs 
            (document_hash, operation_type, operation_status, processing_time,
             error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (document_hash, operation_type, operation_status, processing_time,
              error_message, metadata_json))
    
    def close(self):
        """Close database manager and cleanup resources"""
        try:
            with self.lock:
                for conn in self.connection_pool.values():
                    if conn:
                        conn.close()
                self.connection_pool.clear()
            
            logger.info("Database manager closed")
            
        except Exception as e:
            logger.error(f"Error closing database manager: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()