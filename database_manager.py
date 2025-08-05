"""
Database Manager for Adaptive PDF Table Extraction System
Handles all database operations for learning data, template storage, and statistics.
"""

import sqlite3
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages all database operations for the learning system"""
    
    def __init__(self, db_path: str = "/workspace/learning/learning.db"):
        """Initialize the database manager with the specified database path"""
        self.db_path = db_path
        self.lock = threading.Lock()
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize the database
        self._initialize_database()
        logger.info(f"Database manager initialized with path: {self.db_path}")
    
    def _initialize_database(self):
        """Create all necessary tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Table for storing learned templates
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS learned_templates (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        template_hash TEXT UNIQUE NOT NULL,
                        template_data TEXT NOT NULL,
                        headers TEXT,
                        row_count INTEGER,
                        column_count INTEGER,
                        confidence_score REAL,
                        usage_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Table for storing header corrections
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS header_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_header TEXT NOT NULL,
                        corrected_header TEXT NOT NULL,
                        correction_count INTEGER DEFAULT 1,
                        confidence_score REAL DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Table for extraction statistics
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS extraction_stats (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        file_name TEXT,
                        file_hash TEXT,
                        extraction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        tables_extracted INTEGER DEFAULT 0,
                        extraction_method TEXT,
                        success_rate REAL DEFAULT 0.0,
                        processing_time REAL DEFAULT 0.0
                    )
                ''')
                
                # Table for user feedback
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        extraction_id INTEGER,
                        table_index INTEGER,
                        feedback_type TEXT,
                        original_data TEXT,
                        corrected_data TEXT,
                        user_notes TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (extraction_id) REFERENCES extraction_stats(id)
                    )
                ''')
                
                # Create indexes for better performance
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_template_hash ON learned_templates(template_hash)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_header_original ON header_corrections(original_header)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_extraction_date ON extraction_stats(extraction_date)')
                
                conn.commit()
                logger.info("Database tables initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def save_learned_template(self, template_data: Dict[str, Any]) -> bool:
        """Save a learned template to the database"""
        try:
            with self.lock:
                # Generate hash for the template
                template_str = json.dumps(template_data, sort_keys=True)
                template_hash = hashlib.md5(template_str.encode()).hexdigest()
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check if template already exists
                    cursor.execute('SELECT id, usage_count FROM learned_templates WHERE template_hash = ?', 
                                 (template_hash,))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update usage count
                        cursor.execute('''
                            UPDATE learned_templates 
                            SET usage_count = usage_count + 1, updated_at = CURRENT_TIMESTAMP
                            WHERE template_hash = ?
                        ''', (template_hash,))
                        logger.info(f"Updated existing template usage count: {template_hash}")
                    else:
                        # Insert new template
                        cursor.execute('''
                            INSERT INTO learned_templates 
                            (template_hash, template_data, headers, row_count, column_count, confidence_score)
                            VALUES (?, ?, ?, ?, ?, ?)
                        ''', (
                            template_hash,
                            template_str,
                            json.dumps(template_data.get('headers', [])),
                            template_data.get('row_count', 0),
                            template_data.get('column_count', 0),
                            template_data.get('confidence_score', 1.0)
                        ))
                        logger.info(f"Saved new learned template: {template_hash}")
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            logger.error(f"Error saving learned template: {e}")
            return False
    
    def save_header_correction(self, original_header: str, corrected_header: str) -> bool:
        """Save a header correction to the database"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Check if correction already exists
                    cursor.execute('''
                        SELECT id, correction_count FROM header_corrections 
                        WHERE original_header = ? AND corrected_header = ?
                    ''', (original_header, corrected_header))
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update correction count
                        cursor.execute('''
                            UPDATE header_corrections 
                            SET correction_count = correction_count + 1, updated_at = CURRENT_TIMESTAMP
                            WHERE id = ?
                        ''', (existing[0],))
                    else:
                        # Insert new correction
                        cursor.execute('''
                            INSERT INTO header_corrections (original_header, corrected_header)
                            VALUES (?, ?)
                        ''', (original_header, corrected_header))
                    
                    conn.commit()
                    logger.info(f"Saved header correction: '{original_header}' -> '{corrected_header}'")
                    return True
                    
        except Exception as e:
            logger.error(f"Error saving header correction: {e}")
            return False
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get learning statistics from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get template count
                cursor.execute('SELECT COUNT(*) FROM learned_templates')
                templates_count = cursor.fetchone()[0]
                
                # Get header corrections count
                cursor.execute('SELECT COUNT(*) FROM header_corrections')
                corrections_count = cursor.fetchone()[0]
                
                # Get total extractions
                cursor.execute('SELECT COUNT(*) FROM extraction_stats')
                extractions_count = cursor.fetchone()[0]
                
                # Get average success rate
                cursor.execute('SELECT AVG(success_rate) FROM extraction_stats WHERE success_rate > 0')
                avg_success_rate = cursor.fetchone()[0] or 0.0
                
                # Get most recent extraction
                cursor.execute('SELECT MAX(extraction_date) FROM extraction_stats')
                last_extraction = cursor.fetchone()[0]
                
                stats = {
                    'templates_learned': templates_count,
                    'header_corrections': corrections_count,
                    'total_extractions': extractions_count,
                    'average_success_rate': round(avg_success_rate, 2),
                    'last_extraction': last_extraction,
                    'database_path': self.db_path,
                    'database_size': self._get_database_size()
                }
                
                logger.info(f"Retrieved learning statistics: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Error getting learning statistics: {e}")
            return {
                'templates_learned': 0,
                'header_corrections': 0,
                'total_extractions': 0,
                'average_success_rate': 0.0,
                'last_extraction': None,
                'database_path': self.db_path,
                'database_size': 0
            }
    
    def get_learned_templates(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get learned templates from the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT template_hash, template_data, headers, usage_count, confidence_score, created_at
                    FROM learned_templates
                    ORDER BY usage_count DESC, confidence_score DESC
                    LIMIT ?
                ''', (limit,))
                
                templates = []
                for row in cursor.fetchall():
                    template = {
                        'hash': row[0],
                        'data': json.loads(row[1]),
                        'headers': json.loads(row[2]) if row[2] else [],
                        'usage_count': row[3],
                        'confidence_score': row[4],
                        'created_at': row[5]
                    }
                    templates.append(template)
                
                return templates
                
        except Exception as e:
            logger.error(f"Error getting learned templates: {e}")
            return []
    
    def save_extraction_stats(self, file_name: str, file_hash: str, 
                            tables_extracted: int, extraction_method: str,
                            success_rate: float, processing_time: float) -> int:
        """Save extraction statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO extraction_stats 
                        (file_name, file_hash, tables_extracted, extraction_method, success_rate, processing_time)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (file_name, file_hash, tables_extracted, extraction_method, success_rate, processing_time))
                    
                    extraction_id = cursor.lastrowid
                    conn.commit()
                    
                    logger.info(f"Saved extraction stats for {file_name}: ID {extraction_id}")
                    return extraction_id
                    
        except Exception as e:
            logger.error(f"Error saving extraction stats: {e}")
            return -1
    
    def _get_database_size(self) -> int:
        """Get the size of the database file in bytes"""
        try:
            return os.path.getsize(self.db_path)
        except:
            return 0
    
    def cleanup_old_data(self, days_old: int = 30) -> int:
        """Clean up old data from the database"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Delete old extraction stats
                    cursor.execute('''
                        DELETE FROM extraction_stats 
                        WHERE extraction_date < datetime('now', '-{} days')
                    '''.format(days_old))
                    
                    deleted_count = cursor.rowcount
                    conn.commit()
                    
                    logger.info(f"Cleaned up {deleted_count} old records")
                    return deleted_count
                    
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def export_learning_data(self) -> Dict[str, Any]:
        """Export all learning data for backup"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all templates
                cursor.execute('SELECT * FROM learned_templates')
                templates = [dict(zip([col[0] for col in cursor.description], row)) 
                           for row in cursor.fetchall()]
                
                # Get all corrections
                cursor.execute('SELECT * FROM header_corrections')
                corrections = [dict(zip([col[0] for col in cursor.description], row)) 
                             for row in cursor.fetchall()]
                
                # Get all stats
                cursor.execute('SELECT * FROM extraction_stats')
                stats = [dict(zip([col[0] for col in cursor.description], row)) 
                        for row in cursor.fetchall()]
                
                export_data = {
                    'export_date': datetime.now().isoformat(),
                    'templates': templates,
                    'corrections': corrections,
                    'stats': stats
                }
                
                return export_data
                
        except Exception as e:
            logger.error(f"Error exporting learning data: {e}")
            return {}
    
    def reset_database(self) -> bool:
        """Reset the database by dropping and recreating all tables"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    # Drop all tables
                    cursor.execute('DROP TABLE IF EXISTS user_feedback')
                    cursor.execute('DROP TABLE IF EXISTS extraction_stats')
                    cursor.execute('DROP TABLE IF EXISTS header_corrections')
                    cursor.execute('DROP TABLE IF EXISTS learned_templates')
                    
                    conn.commit()
                
                # Reinitialize the database
                self._initialize_database()
                logger.info("Database reset successfully")
                return True
                
        except Exception as e:
            logger.error(f"Error resetting database: {e}")
            return False

# Global database manager instance
db_manager = None

def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance"""
    global db_manager
    if db_manager is None:
        db_manager = DatabaseManager()
    return db_manager

def initialize_database() -> DatabaseManager:
    """Initialize the database and return the manager"""
    return get_database_manager()