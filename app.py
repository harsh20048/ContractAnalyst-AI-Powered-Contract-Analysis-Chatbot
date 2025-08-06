"""
Adaptive PDF Table Extraction Application with Data Persistence
Fixed version addressing field reset, correction saving, and fallback logic issues.
"""

import sys
import copy
import os
import tempfile
import sqlite3
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Adaptive PDF Data Extraction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database initialization
DB_FILE = "pdf_corrections.db"

def init_database():
    """Initialize the SQLite database for storing corrections."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables for storing corrections and document metadata
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_hash TEXT NOT NULL,
            filename TEXT NOT NULL,
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            corrections_data TEXT NOT NULL,
            table_data TEXT,
            field_data TEXT,
            UNIQUE(document_hash)
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_corrections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_hash TEXT NOT NULL,
            field_name TEXT NOT NULL,
            original_value TEXT,
            corrected_value TEXT NOT NULL,
            correction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_hash) REFERENCES document_corrections (document_hash)
        )
    """)
    
    conn.commit()
    conn.close()

def get_document_hash(file_content: bytes) -> str:
    """Generate a unique hash for the document content."""
    return hashlib.sha256(file_content).hexdigest()

def save_corrections_to_db(document_hash: str, filename: str, corrections: Dict, 
                          table_data: Optional[Dict] = None, field_data: Optional[Dict] = None):
    """Save corrections to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        # Insert or update document corrections
        cursor.execute("""
            INSERT OR REPLACE INTO document_corrections 
            (document_hash, filename, corrections_data, table_data, field_data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            document_hash,
            filename,
            json.dumps(corrections),
            json.dumps(table_data) if table_data else None,
            json.dumps(field_data) if field_data else None
        ))
        
        # Save individual field corrections
        if field_data:
            # Clear existing field corrections for this document
            cursor.execute("DELETE FROM field_corrections WHERE document_hash = ?", (document_hash,))
            
            # Insert new field corrections
            for field_name, field_value in field_data.items():
                cursor.execute("""
                    INSERT INTO field_corrections 
                    (document_hash, field_name, corrected_value)
                    VALUES (?, ?, ?)
                """, (document_hash, field_name, str(field_value)))
        
        conn.commit()
        logger.info(f"Corrections saved for document {document_hash}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving corrections: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def load_corrections_from_db(document_hash: str) -> Optional[Dict]:
    """Load previously saved corrections from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT corrections_data, table_data, field_data, filename, upload_date
            FROM document_corrections 
            WHERE document_hash = ?
        """, (document_hash,))
        
        result = cursor.fetchone()
        if result:
            corrections_data, table_data, field_data, filename, upload_date = result
            return {
                'corrections': json.loads(corrections_data) if corrections_data else {},
                'table_data': json.loads(table_data) if table_data else {},
                'field_data': json.loads(field_data) if field_data else {},
                'filename': filename,
                'upload_date': upload_date
            }
        return None
        
    except Exception as e:
        logger.error(f"Error loading corrections: {e}")
        return None
    finally:
        conn.close()

def extract_mock_table_data(file_content: bytes) -> Dict:
    """Mock table extraction - replace with actual extraction logic."""
    # This is a placeholder for actual table extraction
    return {
        'table_1': {
            'headers': ['Name', 'Amount', 'Date', 'Category'],
            'rows': [
                ['John Doe', '1500.00', '2024-01-15', 'Salary'],
                ['Jane Smith', '2000.00', '2024-01-20', 'Consulting'],
                ['Bob Wilson', '750.00', '2024-01-25', 'Expenses']
            ]
        }
    }

def extract_mock_field_data(file_content: bytes) -> Dict:
    """Mock field extraction - replace with actual field extraction logic."""
    # This is a placeholder for actual field extraction
    return {
        'total_amount': 4250.00,
        'document_date': '2024-01-31',
        'document_type': 'Financial Report',
        'vendor_name': 'ABC Corporation',
        'invoice_number': 'INV-2024-001'
    }

# Initialize database
init_database()

# Initialize session state with persistence
def init_session_state():
    """Initialize session state variables with proper defaults."""
    defaults = {
        'uploaded_file': None,
        'document_hash': None,
        'table_data': {},
        'field_data': {},
        'corrections_applied': False,
        'document_processed': False,
        'show_save_success': False,
        'previously_saved': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()

st.title("📊 Adaptive PDF Data Extraction System")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload PDF Document", type=['pdf'])
    
    if uploaded_file is not None:
        # Get file content and hash
        file_content = uploaded_file.getvalue()
        document_hash = get_document_hash(file_content)
        
        # Check if this is a new document or file changed
        if (st.session_state.document_hash != document_hash or 
            not st.session_state.document_processed):
            
            st.session_state.document_hash = document_hash
            st.session_state.uploaded_file = uploaded_file
            
            # Try to load existing corrections
            existing_corrections = load_corrections_from_db(document_hash)
            
            if existing_corrections:
                # Document has been processed before
                st.session_state.table_data = existing_corrections['table_data']
                st.session_state.field_data = existing_corrections['field_data']
                st.session_state.previously_saved = True
                st.session_state.document_processed = True
                st.success(f"✅ Loaded previous corrections for: {existing_corrections['filename']}")
                st.info(f"Last saved: {existing_corrections['upload_date']}")
            else:
                # New document - use extraction algorithms
                st.session_state.previously_saved = False
                with st.spinner("Processing document..."):
                    # Extract data using algorithms (fallback logic)
                    st.session_state.table_data = extract_mock_table_data(file_content)
                    st.session_state.field_data = extract_mock_field_data(file_content)
                    st.session_state.document_processed = True
                st.success("✅ Document processed using extraction algorithms")
    
    # Show document status
    if st.session_state.document_processed:
        if st.session_state.previously_saved:
            st.info("📋 Using previously saved corrections")
        else:
            st.warning("🔧 Using fallback extraction algorithms")

# Main content area
if st.session_state.document_processed and st.session_state.uploaded_file:
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Extracted Data")
        
        # Field Data Section
        st.subheader("📝 Field Data")
        
        # Create editable fields that persist their values
        field_data_copy = st.session_state.field_data.copy()
        
        for field_name, field_value in field_data_copy.items():
            # Use the field name as the key to maintain state
            new_value = st.text_input(
                f"{field_name.replace('_', ' ').title()}:",
                value=str(field_value),
                key=f"field_{field_name}_{st.session_state.document_hash}"
            )
            # Update the session state with new value
            st.session_state.field_data[field_name] = new_value
        
        # Table Data Section
        st.subheader("📊 Table Data")
        
        if st.session_state.table_data:
            for table_name, table_content in st.session_state.table_data.items():
                st.write(f"**{table_name.replace('_', ' ').title()}**")
                
                # Convert to DataFrame for editing
                df = pd.DataFrame(table_content['rows'], columns=table_content['headers'])
                
                # Use data_editor for inline editing
                edited_df = st.data_editor(
                    df,
                    key=f"table_{table_name}_{st.session_state.document_hash}",
                    use_container_width=True
                )
                
                # Update session state with edited data
                st.session_state.table_data[table_name]['rows'] = edited_df.values.tolist()
    
    with col2:
        st.header("💾 Actions")
        
        # Save Corrections Button
        if st.button("💾 Save Corrections", type="primary", use_container_width=True):
            success = save_corrections_to_db(
                document_hash=st.session_state.document_hash,
                filename=st.session_state.uploaded_file.name,
                corrections={
                    'timestamp': datetime.now().isoformat(),
                    'corrections_applied': True
                },
                table_data=st.session_state.table_data,
                field_data=st.session_state.field_data
            )
            
            if success:
                st.session_state.show_save_success = True
                st.session_state.previously_saved = True
                st.session_state.corrections_applied = True
                st.success("✅ Corrections saved successfully!")
            else:
                st.error("❌ Failed to save corrections")
        
        # Show save status
        if st.session_state.show_save_success:
            st.success("✅ Last save successful")
        
        if st.session_state.previously_saved:
            st.info("📋 Document has saved corrections")
        
        # Export options
        st.subheader("📤 Export")
        
        # Export field data as JSON
        if st.button("📋 Export Field Data", use_container_width=True):
            json_data = json.dumps(st.session_state.field_data, indent=2)
            st.download_button(
                label="💾 Download Field Data JSON",
                data=json_data,
                file_name=f"field_data_{st.session_state.document_hash[:8]}.json",
                mime="application/json",
                use_container_width=True
            )
        
        # Export table data as CSV
        if st.button("📊 Export Table Data", use_container_width=True) and st.session_state.table_data:
            for table_name, table_content in st.session_state.table_data.items():
                df = pd.DataFrame(table_content['rows'], columns=table_content['headers'])
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label=f"💾 Download {table_name.replace('_', ' ').title()} CSV",
                    data=csv_data,
                    file_name=f"{table_name}_{st.session_state.document_hash[:8]}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Debug information
        with st.expander("🔍 Debug Info"):
            st.write(f"**Document Hash:** {st.session_state.document_hash[:16]}...")
            st.write(f"**Previously Saved:** {st.session_state.previously_saved}")
            st.write(f"**Document Processed:** {st.session_state.document_processed}")
            st.write(f"**Corrections Applied:** {st.session_state.corrections_applied}")

else:
    # Welcome screen
    st.info("👈 Please upload a PDF document to begin extraction")
    
    st.markdown("""
    ### 🔍 Features
    
    This adaptive PDF data extraction system provides:
    - **Smart Field Extraction**: Automatically extracts key data fields
    - **Table Recognition**: Identifies and extracts tabular data
    - **Correction Memory**: Saves your corrections for future uploads
    - **Persistent State**: Field values are preserved during UI refresh
    - **Fallback Logic**: Uses algorithms only for new documents
    
    ### 🚀 How it Works
    
    1. **Upload**: Upload your PDF document
    2. **Extract**: System extracts data using AI algorithms or loads previous corrections
    3. **Correct**: Edit any incorrect values in the form fields or tables
    4. **Save**: Click "Save Corrections" to store your changes
    5. **Reuse**: Re-upload the same document to load your saved corrections
    """)

# Reset save success message after displaying
if st.session_state.show_save_success:
    # Use a small delay to show the success message
    time.sleep(0.1)
    st.session_state.show_save_success = False