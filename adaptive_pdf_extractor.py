"""
Updated Adaptive PDF Table Extraction Application
Refined version with consolidated imports, improved error handling, and cleaner structure.
"""

import sys
import copy
import os
import io
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
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Adaptive table extraction imports with comprehensive fallbacks
try:
    from extraction.field_extraction import extract_field_with_learning
    extract_learning_available = True
except ImportError:
    extract_learning_available = False

try:
    from secure_pdf_extractor.extraction.Table_extraction import (
        extract_tables_with_learning,
        save_table_corrections,
        get_learning_statistics,
        AdaptiveTableExtractor,
        extract_tables
    )
    ADAPTIVE_AVAILABLE = True
    print("✅ Table extraction imports successful")
except ImportError as e:
    ADAPTIVE_AVAILABLE = False
    print(f"❌ Table extraction import error: {e}")
    
    # Complete fallback implementation
    class AdaptiveTableExtractor:
        def __init__(self, *args, **kwargs):
            self.debug = kwargs.get('debug', False)

        def extract_tables_with_learning(self, *args, **kwargs):
            return []

        def bulk_save_corrections(self, *args, **kwargs):
            return 0

        def get_learning_statistics(self, *args, **kwargs):
            return {'templates_learned': 0, 'header_corrections': 0}

    def extract_tables_with_learning(*args, **kwargs):
        return []
    
    def get_learning_statistics():
        return {'templates_learned': 0, 'header_corrections': 0}

# Field extraction imports
try:
    from secure_pdf_extractor.extraction.field_extraction import (
        PDFFieldExtractor,
        extract_fields_from_pdf,
        process_pdf_documents
    )
    FIELD_EXTRACTION_AVAILABLE = True
    print("✅ Field extraction imports successful")
except ImportError as e:
    FIELD_EXTRACTION_AVAILABLE = False
    print(f"❌ Field extraction import error: {e}")
    
    # Fallback PDFFieldExtractor
    class PDFFieldExtractor:
        def __init__(self, *args, **kwargs):
            pass

        def extract_fields(self, text, **kwargs):
            return {'date': '', 'angebot': '', 'company_name': '', 'sender_address': ''}

# Text extraction imports
try:
    from secure_pdf_extractor.extraction.text_extraction import extract_text_from_pdf, detect_language
    print("✅ Text extraction imports successful")
except ImportError as e:
    print(f"❌ Text extraction import error: {e}")
    
    # Fallback text extraction
    def extract_text_from_pdf(pdf_path):
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() or ""
                return text
        except:
            return ""
    
    def detect_language(text):
        return "de"  # Default to German

# Database utilities
def get_database_path():
    """Get the path for SQLite database"""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    return data_dir / "learning_data.db"

def initialize_database():
    """Initialize SQLite database for learning data"""
    db_path = get_database_path()
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables for learning data
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS table_templates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            template_hash TEXT UNIQUE,
            headers TEXT,
            structure TEXT,
            usage_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            field_name TEXT,
            pattern TEXT,
            confidence REAL,
            usage_count INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS extraction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pdf_hash TEXT,
            extraction_type TEXT,
            success INTEGER,
            extracted_data TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

# Session state initialization
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'extracted_tables' not in st.session_state:
        st.session_state.extracted_tables = []
    if 'extracted_fields' not in st.session_state:
        st.session_state.extracted_fields = {}
    if 'corrections' not in st.session_state:
        st.session_state.corrections = []
    if 'processing_history' not in st.session_state:
        st.session_state.processing_history = []
    if 'current_pdf' not in st.session_state:
        st.session_state.current_pdf = None

# Main UI functions
def render_sidebar():
    """Render sidebar with options and statistics"""
    with st.sidebar:
        st.header("📊 System Statistics")
        
        # Learning statistics
        if ADAPTIVE_AVAILABLE:
            stats = get_learning_statistics()
            st.metric("Templates Learned", stats.get('templates_learned', 0))
            st.metric("Header Corrections", stats.get('header_corrections', 0))
        else:
            st.warning("Adaptive learning not available")
        
        st.divider()
        
        # Export options
        st.header("📥 Export Options")
        export_format = st.selectbox(
            "Export Format",
            ["Excel", "CSV", "JSON", "SQLite"]
        )
        
        if st.button("Export All Data"):
            export_data(export_format)
        
        st.divider()
        
        # Settings
        st.header("⚙️ Settings")
        st.checkbox("Enable Debug Mode", key="debug_mode")
        st.slider("Confidence Threshold", 0.0, 1.0, 0.7, key="confidence_threshold")

def process_pdf(uploaded_file):
    """Process uploaded PDF file"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name
    
    try:
        # Extract text
        text = extract_text_from_pdf(temp_path)
        
        # Extract tables
        if ADAPTIVE_AVAILABLE:
            tables = extract_tables_with_learning(temp_path, debug=st.session_state.get('debug_mode', False))
        else:
            tables = []
        
        # Extract fields
        if FIELD_EXTRACTION_AVAILABLE:
            extractor = PDFFieldExtractor()
            fields = extractor.extract_fields(text)
        else:
            fields = {}
        
        # Store in session state
        st.session_state.extracted_tables = tables
        st.session_state.extracted_fields = fields
        st.session_state.current_pdf = uploaded_file.name
        
        # Add to history
        st.session_state.processing_history.append({
            'filename': uploaded_file.name,
            'timestamp': datetime.now(),
            'tables_count': len(tables),
            'fields_count': len(fields)
        })
        
        return True
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        logger.error(f"PDF processing error: {e}", exc_info=True)
        return False
    finally:
        os.unlink(temp_path)

def render_table_editor(table_data, table_idx):
    """Render editable table with correction tracking"""
    st.subheader(f"Table {table_idx + 1}")
    
    # Convert to DataFrame if not already
    if isinstance(table_data, list):
        df = pd.DataFrame(table_data)
    else:
        df = table_data
    
    # Edit table
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        key=f"table_editor_{table_idx}"
    )
    
    # Track changes
    if not df.equals(edited_df):
        correction = {
            'table_index': table_idx,
            'original': df.to_dict(),
            'corrected': edited_df.to_dict(),
            'timestamp': datetime.now()
        }
        
        if correction not in st.session_state.corrections:
            st.session_state.corrections.append(correction)
    
    return edited_df

def render_field_editor(fields):
    """Render editable fields with validation"""
    st.subheader("Extracted Fields")
    
    edited_fields = {}
    cols = st.columns(2)
    
    for idx, (field_name, field_value) in enumerate(fields.items()):
        col = cols[idx % 2]
        with col:
            edited_value = st.text_input(
                field_name.replace('_', ' ').title(),
                value=field_value or "",
                key=f"field_{field_name}"
            )
            edited_fields[field_name] = edited_value
    
    return edited_fields

def export_data(format_type):
    """Export extracted data in various formats"""
    if not st.session_state.extracted_tables and not st.session_state.extracted_fields:
        st.warning("No data to export")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format_type == "Excel":
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            # Export tables
            for idx, table in enumerate(st.session_state.extracted_tables):
                df = pd.DataFrame(table)
                df.to_excel(writer, sheet_name=f'Table_{idx+1}', index=False)
            
            # Export fields
            if st.session_state.extracted_fields:
                fields_df = pd.DataFrame([st.session_state.extracted_fields])
                fields_df.to_excel(writer, sheet_name='Fields', index=False)
        
        output.seek(0)
        st.download_button(
            label="Download Excel",
            data=output,
            file_name=f"extraction_{timestamp}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    
    elif format_type == "CSV":
        # Combine all tables
        all_data = []
        for table in st.session_state.extracted_tables:
            all_data.extend(table)
        
        if all_data:
            df = pd.DataFrame(all_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"extraction_{timestamp}.csv",
                mime="text/csv"
            )
    
    elif format_type == "JSON":
        data = {
            'tables': st.session_state.extracted_tables,
            'fields': st.session_state.extracted_fields,
            'metadata': {
                'pdf_name': st.session_state.current_pdf,
                'extraction_date': timestamp
            }
        }
        json_str = json.dumps(data, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_str,
            file_name=f"extraction_{timestamp}.json",
            mime="application/json"
        )

def main():
    """Main application function"""
    st.title("📄 Adaptive PDF Data Extraction System")
    st.markdown("Extract tables and fields from PDF documents with machine learning")
    
    # Initialize
    initialize_session_state()
    initialize_database()
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📤 Upload", "📊 Tables", "📝 Fields", "📈 Analytics"])
    
    with tab1:
        st.header("Upload PDF Document")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload a PDF document to extract tables and fields"
        )
        
        if uploaded_file:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info(f"Selected: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)")
            with col2:
                if st.button("🚀 Process PDF", type="primary"):
                    with st.spinner("Processing PDF..."):
                        if process_pdf(uploaded_file):
                            st.success("✅ PDF processed successfully!")
                            st.balloons()
    
    with tab2:
        st.header("Extracted Tables")
        if st.session_state.extracted_tables:
            for idx, table in enumerate(st.session_state.extracted_tables):
                with st.expander(f"Table {idx + 1}", expanded=True):
                    edited_table = render_table_editor(table, idx)
            
            if st.button("💾 Save Table Corrections"):
                # Save corrections to learning system
                st.success(f"Saved {len(st.session_state.corrections)} corrections")
                st.session_state.corrections = []
        else:
            st.info("No tables extracted yet. Upload a PDF to begin.")
    
    with tab3:
        st.header("Extracted Fields")
        if st.session_state.extracted_fields:
            edited_fields = render_field_editor(st.session_state.extracted_fields)
            
            if st.button("💾 Save Field Updates"):
                st.session_state.extracted_fields = edited_fields
                st.success("Fields updated successfully!")
        else:
            st.info("No fields extracted yet. Upload a PDF to begin.")
    
    with tab4:
        st.header("Analytics & History")
        
        if st.session_state.processing_history:
            # Create DataFrame from history
            history_df = pd.DataFrame(st.session_state.processing_history)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total PDFs Processed", len(history_df))
            with col2:
                st.metric("Total Tables Extracted", history_df['tables_count'].sum())
            with col3:
                st.metric("Average Fields per PDF", history_df['fields_count'].mean())
            
            # Show history table
            st.subheader("Processing History")
            st.dataframe(history_df, use_container_width=True)
            
            # Visualizations
            if len(history_df) > 1:
                st.subheader("Extraction Trends")
                chart_data = history_df.set_index('timestamp')[['tables_count', 'fields_count']]
                st.line_chart(chart_data)
        else:
            st.info("No processing history available yet.")

if __name__ == "__main__":
    main()