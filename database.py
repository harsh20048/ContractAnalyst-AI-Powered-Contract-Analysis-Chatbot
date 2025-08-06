"""
Database Management Module for PDF Table Extraction System
Handles all database operations including corrections storage, retrieval, and analytics.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the PDF extraction system."""
    
    def __init__(self, db_path: str = "pdf_corrections.db"):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the SQLite database with required tables."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Enable foreign key constraints
            cursor.execute("PRAGMA foreign_keys = ON")
            
            # Document corrections table - main storage for corrections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS document_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    corrections_data TEXT NOT NULL,
                    table_data TEXT,
                    field_data TEXT,
                    extraction_confidence REAL DEFAULT 0.0,
                    processing_time_ms INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'processed',
                    version INTEGER DEFAULT 1
                )
            """)
            
            # Individual field corrections tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS field_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    original_value TEXT,
                    corrected_value TEXT NOT NULL,
                    correction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    correction_type TEXT DEFAULT 'manual',
                    confidence_score REAL DEFAULT 1.0,
                    user_id TEXT DEFAULT 'default',
                    FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash) ON DELETE CASCADE
                )
            """)
            
            # Table corrections tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS table_corrections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    table_name TEXT NOT NULL,
                    table_type TEXT,
                    row_count INTEGER DEFAULT 0,
                    column_count INTEGER DEFAULT 0,
                    corrections_count INTEGER DEFAULT 0,
                    extraction_confidence REAL DEFAULT 0.0,
                    last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash) ON DELETE CASCADE
                )
            """)
            
            # Pattern learning data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_learning (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pattern_type TEXT NOT NULL,
                    pattern_name TEXT NOT NULL,
                    pattern_data TEXT NOT NULL,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1
                )
            """)
            
            # System statistics and metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stat_name TEXT NOT NULL UNIQUE,
                    stat_value TEXT NOT NULL,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Document processing logs
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS processing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_hash TEXT NOT NULL,
                    log_level TEXT NOT NULL,
                    log_message TEXT NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    component TEXT DEFAULT 'system',
                    FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash) ON DELETE CASCADE
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_hash ON document_corrections(document_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_upload_date ON document_corrections(upload_date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_field_doc_hash ON field_corrections(document_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_table_doc_hash ON table_corrections(document_hash)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_learning(pattern_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_processing_logs_hash ON processing_logs(document_hash)")
            
            conn.commit()
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
        finally:
            conn.close()
    
    def save_corrections(self, document_hash: str, filename: str, corrections: Dict,
                        table_data: Optional[Dict] = None, field_data: Optional[Dict] = None,
                        file_size: Optional[int] = None, processing_time_ms: int = 0) -> bool:
        """
        Save corrections to the database.
        
        Args:
            document_hash: Unique document identifier
            filename: Original filename
            corrections: General correction metadata
            table_data: Table corrections data
            field_data: Field corrections data
            file_size: Size of the original file
            processing_time_ms: Processing time in milliseconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Calculate extraction confidence
            extraction_confidence = self._calculate_extraction_confidence(field_data, table_data)
            
            # Insert or update document corrections
            cursor.execute("""
                INSERT OR REPLACE INTO document_corrections 
                (document_hash, filename, file_size, corrections_data, table_data, field_data, 
                 extraction_confidence, processing_time_ms, last_modified)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                document_hash,
                filename,
                file_size,
                json.dumps(corrections),
                json.dumps(table_data) if table_data else None,
                json.dumps(field_data) if field_data else None,
                extraction_confidence,
                processing_time_ms,
                datetime.now().isoformat()
            ))
            
            # Clear existing field corrections for this document
            cursor.execute("DELETE FROM field_corrections WHERE document_hash = ?", (document_hash,))
            
            # Save individual field corrections
            if field_data:
                for field_name, field_value in field_data.items():
                    cursor.execute("""
                        INSERT INTO field_corrections 
                        (document_hash, field_name, corrected_value, confidence_score)
                        VALUES (?, ?, ?, ?)
                    """, (
                        document_hash, 
                        field_name, 
                        str(field_value),
                        self._get_field_confidence(field_name, field_value)
                    ))
            
            # Clear existing table corrections for this document
            cursor.execute("DELETE FROM table_corrections WHERE document_hash = ?", (document_hash,))
            
            # Save table corrections metadata
            if table_data:
                for table_name, table_content in table_data.items():
                    if isinstance(table_content, dict) and 'rows' in table_content:
                        rows = table_content.get('rows', [])
                        headers = table_content.get('headers', [])
                        metadata = table_content.get('metadata', {})
                        
                        cursor.execute("""
                            INSERT INTO table_corrections 
                            (document_hash, table_name, table_type, row_count, column_count, 
                             extraction_confidence, last_modified)
                            VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            document_hash,
                            table_name,
                            metadata.get('table_type', 'unknown'),
                            len(rows),
                            len(headers),
                            metadata.get('extraction_confidence', 0.0),
                            datetime.now().isoformat()
                        ))
            
            # Log the correction save
            self._log_processing_event(cursor, document_hash, "INFO", "Corrections saved successfully")
            
            # Update system statistics
            self._update_system_stats(cursor)
            
            conn.commit()
            logger.info(f"Corrections saved successfully for document {document_hash[:8]}...")
            return True
            
        except Exception as e:
            logger.error(f"Error saving corrections: {e}")
            if 'conn' in locals():
                conn.rollback()
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def load_corrections(self, document_hash: str) -> Optional[Dict]:
        """
        Load previously saved corrections from the database.
        
        Args:
            document_hash: Unique document identifier
            
        Returns:
            Dict containing corrections data or None if not found
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT corrections_data, table_data, field_data, filename, upload_date, 
                       last_modified, extraction_confidence, processing_time_ms, status, version
                FROM document_corrections 
                WHERE document_hash = ?
            """, (document_hash,))
            
            result = cursor.fetchone()
            if result:
                (corrections_data, table_data, field_data, filename, upload_date, 
                 last_modified, extraction_confidence, processing_time_ms, status, version) = result
                
                return {
                    'corrections': json.loads(corrections_data) if corrections_data else {},
                    'table_data': json.loads(table_data) if table_data else {},
                    'field_data': json.loads(field_data) if field_data else {},
                    'filename': filename,
                    'upload_date': upload_date,
                    'last_modified': last_modified,
                    'extraction_confidence': extraction_confidence,
                    'processing_time_ms': processing_time_ms,
                    'status': status,
                    'version': version
                }
            return None
            
        except Exception as e:
            logger.error(f"Error loading corrections: {e}")
            return None
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get system statistics and metrics.
        
        Returns:
            Dictionary containing various system statistics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            stats = {}
            
            # Total documents processed
            cursor.execute("SELECT COUNT(*) FROM document_corrections")
            stats['total_documents'] = cursor.fetchone()[0]
            
            # Total corrections made
            cursor.execute("SELECT COUNT(*) FROM field_corrections")
            field_corrections = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM table_corrections")
            table_corrections = cursor.fetchone()[0]
            stats['total_corrections'] = field_corrections + table_corrections
            
            # Average extraction confidence
            cursor.execute("SELECT AVG(extraction_confidence) FROM document_corrections")
            avg_confidence = cursor.fetchone()[0]
            stats['avg_extraction_confidence'] = avg_confidence if avg_confidence else 0.0
            
            # Documents processed in last 7 days
            week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            cursor.execute("SELECT COUNT(*) FROM document_corrections WHERE upload_date >= ?", (week_ago,))
            stats['documents_last_week'] = cursor.fetchone()[0]
            
            # Most common field types
            cursor.execute("""
                SELECT field_name, COUNT(*) as count 
                FROM field_corrections 
                GROUP BY field_name 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats['common_fields'] = [{'field': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Most common table types
            cursor.execute("""
                SELECT table_type, COUNT(*) as count 
                FROM table_corrections 
                WHERE table_type IS NOT NULL 
                GROUP BY table_type 
                ORDER BY count DESC 
                LIMIT 5
            """)
            stats['common_table_types'] = [{'type': row[0], 'count': row[1]} for row in cursor.fetchall()]
            
            # Average processing time
            cursor.execute("SELECT AVG(processing_time_ms) FROM document_corrections WHERE processing_time_ms > 0")
            avg_processing_time = cursor.fetchone()[0]
            stats['avg_processing_time_ms'] = avg_processing_time if avg_processing_time else 0
            
            # Database size
            stats['database_size_mb'] = round(os.path.getsize(self.db_path) / (1024 * 1024), 2)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_recent_documents(self, limit: int = 10) -> List[Dict]:
        """
        Get recently processed documents.
        
        Args:
            limit: Maximum number of documents to return
            
        Returns:
            List of recent document information
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT document_hash, filename, upload_date, last_modified, 
                       extraction_confidence, status
                FROM document_corrections 
                ORDER BY last_modified DESC 
                LIMIT ?
            """, (limit,))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'hash': row[0][:12] + '...',
                    'filename': row[1],
                    'uploaded': row[2],
                    'modified': row[3],
                    'confidence': f"{row[4]:.1%}" if row[4] else "N/A",
                    'status': row[5]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting recent documents: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def search_documents(self, query: str, search_type: str = "filename") -> List[Dict]:
        """
        Search for documents based on various criteria.
        
        Args:
            query: Search query
            search_type: Type of search ('filename', 'field_value', 'hash')
            
        Returns:
            List of matching documents
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if search_type == "filename":
                cursor.execute("""
                    SELECT document_hash, filename, upload_date, extraction_confidence
                    FROM document_corrections 
                    WHERE filename LIKE ? 
                    ORDER BY upload_date DESC
                """, (f"%{query}%",))
            
            elif search_type == "field_value":
                cursor.execute("""
                    SELECT dc.document_hash, dc.filename, dc.upload_date, dc.extraction_confidence
                    FROM document_corrections dc
                    JOIN field_corrections fc ON dc.document_hash = fc.document_hash
                    WHERE fc.corrected_value LIKE ?
                    ORDER BY dc.upload_date DESC
                """, (f"%{query}%",))
            
            elif search_type == "hash":
                cursor.execute("""
                    SELECT document_hash, filename, upload_date, extraction_confidence
                    FROM document_corrections 
                    WHERE document_hash LIKE ? 
                    ORDER BY upload_date DESC
                """, (f"{query}%",))
            
            documents = []
            for row in cursor.fetchall():
                documents.append({
                    'document_hash': row[0],
                    'filename': row[1],
                    'upload_date': row[2],
                    'extraction_confidence': row[3]
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def delete_document(self, document_hash: str) -> bool:
        """
        Delete a document and all associated data.
        
        Args:
            document_hash: Unique document identifier
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete from main table (cascading will handle related tables)
            cursor.execute("DELETE FROM document_corrections WHERE document_hash = ?", (document_hash,))
            
            if cursor.rowcount > 0:
                conn.commit()
                logger.info(f"Document {document_hash[:8]}... deleted successfully")
                return True
            else:
                logger.warning(f"Document {document_hash[:8]}... not found")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
        finally:
            if 'conn' in locals():
                conn.close()
    
    def backup_database(self, backup_path: str) -> bool:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path for the backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create backup directory if it doesn't exist
            backup_dir = Path(backup_path).parent
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy database file
            import shutil
            shutil.copy2(self.db_path, backup_path)
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error backing up database: {e}")
            return False
    
    def get_document_analytics(self, document_hash: str) -> Dict[str, Any]:
        """
        Get detailed analytics for a specific document.
        
        Args:
            document_hash: Unique document identifier
            
        Returns:
            Dictionary containing document analytics
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            analytics = {}
            
            # Basic document info
            cursor.execute("""
                SELECT filename, upload_date, last_modified, extraction_confidence, 
                       processing_time_ms, status, version
                FROM document_corrections 
                WHERE document_hash = ?
            """, (document_hash,))
            
            doc_info = cursor.fetchone()
            if doc_info:
                analytics['document_info'] = {
                    'filename': doc_info[0],
                    'upload_date': doc_info[1],
                    'last_modified': doc_info[2],
                    'extraction_confidence': doc_info[3],
                    'processing_time_ms': doc_info[4],
                    'status': doc_info[5],
                    'version': doc_info[6]
                }
            
            # Field corrections count
            cursor.execute("SELECT COUNT(*) FROM field_corrections WHERE document_hash = ?", (document_hash,))
            analytics['field_corrections_count'] = cursor.fetchone()[0]
            
            # Table corrections count
            cursor.execute("SELECT COUNT(*) FROM table_corrections WHERE document_hash = ?", (document_hash,))
            analytics['table_corrections_count'] = cursor.fetchone()[0]
            
            # Processing logs
            cursor.execute("""
                SELECT log_level, log_message, timestamp, component
                FROM processing_logs 
                WHERE document_hash = ? 
                ORDER BY timestamp DESC 
                LIMIT 10
            """, (document_hash,))
            
            analytics['recent_logs'] = [
                {
                    'level': row[0],
                    'message': row[1],
                    'timestamp': row[2],
                    'component': row[3]
                }
                for row in cursor.fetchall()
            ]
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting document analytics: {e}")
            return {}
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _calculate_extraction_confidence(self, field_data: Optional[Dict], table_data: Optional[Dict]) -> float:
        """Calculate overall extraction confidence score."""
        confidences = []
        
        if field_data and 'confidence_score' in field_data:
            confidences.append(field_data['confidence_score'])
        
        if table_data:
            for table_content in table_data.values():
                if isinstance(table_content, dict) and 'metadata' in table_content:
                    metadata = table_content['metadata']
                    if 'extraction_confidence' in metadata:
                        confidences.append(metadata['extraction_confidence'])
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _get_field_confidence(self, field_name: str, field_value: Any) -> float:
        """Calculate confidence score for individual field."""
        # Simple heuristic - in real implementation, this would be more sophisticated
        if field_value is None or str(field_value).strip() == '':
            return 0.0
        elif field_name in ['confidence_score'] and isinstance(field_value, (int, float)):
            return float(field_value)
        else:
            return 1.0
    
    def _log_processing_event(self, cursor, document_hash: str, level: str, message: str, component: str = "database"):
        """Log a processing event to the database."""
        cursor.execute("""
            INSERT INTO processing_logs (document_hash, log_level, log_message, component)
            VALUES (?, ?, ?, ?)
        """, (document_hash, level, message, component))
    
    def _update_system_stats(self, cursor):
        """Update system-wide statistics."""
        # This could be expanded to track more detailed metrics
        timestamp = datetime.now().isoformat()
        cursor.execute("""
            INSERT OR REPLACE INTO system_stats (stat_name, stat_value, last_updated)
            VALUES ('last_correction_save', ?, ?)
        """, (timestamp, timestamp))
    
    def get_field_correction_history(self, field_name: str, limit: int = 10) -> List[Dict]:
        """
        Get correction history for a specific field across all documents.
        
        Args:
            field_name: Name of the field
            limit: Maximum number of records to return
            
        Returns:
            List of correction history records
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT fc.document_hash, dc.filename, fc.original_value, fc.corrected_value, 
                       fc.correction_date, fc.confidence_score
                FROM field_corrections fc
                JOIN document_corrections dc ON fc.document_hash = dc.document_hash
                WHERE fc.field_name = ?
                ORDER BY fc.correction_date DESC
                LIMIT ?
            """, (field_name, limit))
            
            history = []
            for row in cursor.fetchall():
                history.append({
                    'document_hash': row[0][:12] + '...',
                    'filename': row[1],
                    'original_value': row[2],
                    'corrected_value': row[3],
                    'correction_date': row[4],
                    'confidence_score': row[5]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting field correction history: {e}")
            return []
        finally:
            if 'conn' in locals():
                conn.close()
    
    def cleanup_old_data(self, days_old: int = 90) -> int:
        """
        Clean up old document data to manage database size.
        
        Args:
            days_old: Number of days after which data should be cleaned up
            
        Returns:
            Number of documents cleaned up
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = (datetime.now() - timedelta(days=days_old)).isoformat()
            
            # Get count before deletion
            cursor.execute("""
                SELECT COUNT(*) FROM document_corrections 
                WHERE upload_date < ? AND status != 'archived'
            """, (cutoff_date,))
            count_to_delete = cursor.fetchone()[0]
            
            # Delete old documents (cascading will handle related tables)
            cursor.execute("""
                DELETE FROM document_corrections 
                WHERE upload_date < ? AND status != 'archived'
            """, (cutoff_date,))
            
            conn.commit()
            logger.info(f"Cleaned up {count_to_delete} old documents")
            return count_to_delete
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
        finally:
            if 'conn' in locals():
                conn.close()