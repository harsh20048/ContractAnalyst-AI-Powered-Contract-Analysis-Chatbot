import streamlit as st
import os
import tempfile
import traceback
from dotenv import load_dotenv

# Import from the new package structure
from extraction.field_extraction import FieldExtractor
from learning.database import LearningDatabase
from validation.field_validation import FieldValidator
from output.excel_export import ExcelExporter

# Legacy imports for existing functionality - these need to be copied to the workspace
import sys
sys.path.append('../../')
try:
    from pdf_processor import PDFProcessor
    from chatbot import Chatbot
except ImportError:
    # Fallback for development
    PDFProcessor = None
    Chatbot = None

# Load environment variables
load_dotenv()

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for API key
if not os.getenv("GOOGLE_API_KEY"):
    st.error("No Google API key found. Please set the GOOGLE_API_KEY environment variable.")
    st.stop()

# Set page config
st.set_page_config(
    page_title="Secure PDF Extractor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #f0f2f6;
    }
    .bot-message {
        background-color: #e8f0fe;
    }
    .field-card {
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .confidence-high {
        border-left: 4px solid #28a745;
    }
    .confidence-medium {
        border-left: 4px solid #ffc107;
    }
    .confidence-low {
        border-left: 4px solid #dc3545;
    }
    .correction-status {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🔒 Secure PDF Extractor")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False
if 'extracted_fields' not in st.session_state:
    st.session_state.extracted_fields = {}
if 'document_hash' not in st.session_state:
    st.session_state.document_hash = ""
if 'pdf_name' not in st.session_state:
    st.session_state.pdf_name = ""
if 'field_extractor' not in st.session_state:
    st.session_state.field_extractor = FieldExtractor()
if 'validator' not in st.session_state:
    st.session_state.validator = FieldValidator()
if 'exporter' not in st.session_state:
    st.session_state.exporter = ExcelExporter()

# Create tabs for different functionality
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Field Extraction", 
    "📊 Validation", 
    "📤 Export", 
    "💬 Document Chat", 
    "📈 Analytics"
])

with tab1:
    st.header("🔍 Advanced Field Extraction")
    
    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📁 Document Upload")
        pdf_file = st.file_uploader("Upload your PDF document", type=['pdf'])
        
        if pdf_file:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.getvalue())
                temp_file_path = tmp_file.name
                
            try:
                # Initialize PDF processor for chatbot functionality
                pdf_processor = PDFProcessor(api_key=os.getenv("GOOGLE_API_KEY"))
                
                # Process PDF for both extraction and chat
                with st.spinner("Processing document..."):
                    try:
                        # Extract fields first
                        document_hash, extracted_fields = st.session_state.field_extractor.extract_fields_from_pdf(temp_file_path)
                        st.session_state.extracted_fields = extracted_fields
                        st.session_state.document_hash = document_hash
                        st.session_state.pdf_name = pdf_file.name
                        
                        # Process for chatbot
                        st.session_state.vector_store, suggested_questions = pdf_processor.process_pdf(temp_file_path)
                        
                        # Initialize chatbot
                        st.session_state.chatbot = Chatbot(
                            st.session_state.vector_store,
                            api_key=os.getenv("GOOGLE_API_KEY")
                        )
                        
                        st.success("✅ Document processed successfully!")
                        
                        # Show document info
                        db = LearningDatabase()
                        doc_info = db.get_document_info(document_hash)
                        if doc_info:
                            st.info(f"📄 **{doc_info['pdf_name']}**\n"
                                   f"🔍 Processing count: {doc_info['processing_count']}\n"
                                   f"✏️ Corrections saved: {doc_info['correction_count']}")
                        else:
                            st.info("📄 New document - no previous corrections found")
                        
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
                        logger.error(traceback.format_exc())
                    
            except Exception as e:
                st.error(f"Error handling file: {str(e)}")
                logger.error(traceback.format_exc())
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.error(f"Error removing temporary file: {str(e)}")
        
        # Statistics and management
        st.divider()
        if st.button("📊 Refresh Stats"):
            st.rerun()
        
        if st.button("🗑️ Clear All Data"):
            if st.checkbox("I understand this will delete all learning data"):
                st.session_state.field_extractor.db.clear_all_corrections()
                st.success("All data cleared!")
    
    # Main field extraction interface
    if st.session_state.extracted_fields:
        st.subheader(f"📋 Extracted Fields from: {st.session_state.pdf_name}")
        st.write(f"🔐 Document Hash: `{st.session_state.document_hash[:16]}...`")
        
        # Create form for field corrections
        with st.form("field_corrections"):
            st.write("### ✏️ Review and correct extracted fields:")
            
            corrections = {}
            col1, col2 = st.columns([2, 1])
            
            for field_name, field_data in st.session_state.extracted_fields.items():
                with col1:
                    # Determine confidence styling
                    if field_data.confidence >= 0.8:
                        confidence_class = "confidence-high"
                        confidence_color = "🟢"
                    elif field_data.confidence >= 0.6:
                        confidence_class = "confidence-medium"
                        confidence_color = "🟡"
                    else:
                        confidence_class = "confidence-low"
                        confidence_color = "🔴"
                    
                    # Display field information
                    st.markdown(f"""
                        <div class="field-card {confidence_class}">
                            <h4>{field_name.replace('_', ' ').title()}</h4>
                            <p><strong>Extracted Value:</strong> {field_data.value}</p>
                            <p><strong>Confidence:</strong> {confidence_color} {field_data.confidence:.2f}</p>
                            <p><strong>Source:</strong> {field_data.source}</p>
                            <p><strong>Page:</strong> {field_data.page_number}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Input field for corrections
                    corrected_value = st.text_input(
                        f"Correct {field_name.replace('_', ' ').title()}:",
                        value=field_data.value,
                        key=f"correction_{field_name}",
                        help=f"Original: {field_data.value}\nContext: {field_data.context[:100]}..."
                    )
                    corrections[field_name] = corrected_value
                    
                with col2:
                    if field_data.context:
                        st.text_area(
                            "Context:",
                            value=field_data.context,
                            height=100,
                            key=f"context_{field_name}",
                            disabled=True
                        )
            
            # Submit corrections
            submitted = st.form_submit_button("💾 Save Corrections")
            
            if submitted:
                # Save corrections
                success = st.session_state.field_extractor.save_field_corrections(
                    document_hash=st.session_state.document_hash,
                    pdf_name=st.session_state.pdf_name,
                    corrections=corrections,
                    original_fields=st.session_state.extracted_fields
                )
                
                if success:
                    st.markdown("""
                        <div class="correction-status">
                            ✅ <strong>Corrections saved successfully!</strong><br>
                            The system has learned from your corrections and will improve future extractions.
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Update session state with corrected values
                    for field_name, corrected_value in corrections.items():
                        if corrected_value:
                            st.session_state.extracted_fields[field_name].value = corrected_value
                            st.session_state.extracted_fields[field_name].source = 'user_correction'
                            st.session_state.extracted_fields[field_name].confidence = 0.95
                    
                    st.rerun()
                else:
                    st.error("❌ Failed to save corrections. Please try again.")
        
        # Display field summary
        st.subheader("📈 Field Summary")
        summary_cols = st.columns(3)
        
        total_fields = len(st.session_state.extracted_fields)
        high_confidence = sum(1 for f in st.session_state.extracted_fields.values() if f.confidence >= 0.8)
        corrections_count = sum(1 for f in st.session_state.extracted_fields.values() if f.source == 'user_correction')
        
        with summary_cols[0]:
            st.metric("Total Fields", total_fields)
        with summary_cols[1]:
            st.metric("High Confidence", high_confidence, delta=f"{high_confidence/total_fields*100:.1f}%")
        with summary_cols[2]:
            st.metric("User Corrections", corrections_count)
    
    else:
        st.info("👈 Please upload a PDF document to extract fields.")
        
        st.markdown("""
        ### 🔍 What fields can be extracted?
        
        This system can automatically extract and learn from:
        - **Angebot/Quote Numbers** - Document reference numbers
        - **Dates** - Contract dates, delivery dates, deadlines
        - **Company Names** - Client and vendor information  
        - **Amounts** - Pricing, fees, totals
        - **Contact Information** - People responsible for the contract
        - **Custom Fields** - The system learns new patterns from your corrections
        
        ### 🧠 How Learning Works
        
        1. **Upload** a PDF document
        2. **Review** the automatically extracted fields
        3. **Correct** any incorrect values
        4. **Save** corrections to teach the system
        5. **Re-upload** the same document to see persistent corrections
        
        The system will progressively improve its accuracy based on your feedback!
        """)

with tab2:
    st.header("✅ Field Validation")
    
    if st.session_state.extracted_fields:
        # Validate all fields
        validation_results = st.session_state.validator.validate_fields(st.session_state.extracted_fields)
        
        # Display validation summary
        summary = st.session_state.validator.get_validation_summary(validation_results)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Fields", summary['total_fields'])
        with col2:
            st.metric("Valid Fields", summary['valid_fields'])
        with col3:
            st.metric("Validation Rate", f"{summary['validation_rate']:.1%}")
        with col4:
            st.metric("Avg Confidence", f"{summary['average_confidence']:.2f}")
        
        # Display individual validation results
        st.subheader("📝 Field Validation Results")
        
        for field_name, result in validation_results.items():
            with st.expander(f"{field_name.replace('_', ' ').title()} - {'✅ Valid' if result.is_valid else '❌ Invalid'}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Value:** {st.session_state.extracted_fields[field_name].value}")
                    st.write(f"**Confidence:** {result.confidence:.2f}")
                    st.write(f"**Status:** {'Valid' if result.is_valid else 'Invalid'}")
                
                with col2:
                    if result.errors:
                        st.error("**Errors:**")
                        for error in result.errors:
                            st.write(f"• {error}")
                    
                    if result.warnings:
                        st.warning("**Warnings:**")
                        for warning in result.warnings:
                            st.write(f"• {warning}")
                    
                    if result.suggestions:
                        st.info("**Suggestions:**")
                        for suggestion in result.suggestions:
                            st.write(f"• {suggestion}")
    
    else:
        st.info("👈 Please extract fields first to validate them.")

with tab3:
    st.header("📤 Export Options")
    
    if st.session_state.extracted_fields:
        export_format = st.selectbox("Choose export format:", ["Excel", "CSV", "JSON"])
        
        if export_format == "Excel":
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Extracted Fields"):
                    try:
                        # Prepare data for export
                        export_data = {st.session_state.pdf_name: st.session_state.extracted_fields}
                        
                        # Generate Excel file
                        excel_data = st.session_state.exporter.export_extracted_fields(export_data)
                        
                        # Create download button
                        st.download_button(
                            label="💾 Download Excel File",
                            data=excel_data,
                            file_name=f"extracted_fields_{st.session_state.pdf_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("Excel export ready for download!")
                        
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
            
            with col2:
                if st.button("📋 Export Validation Results"):
                    try:
                        # Validate fields
                        validation_results = st.session_state.validator.validate_fields(st.session_state.extracted_fields)
                        validation_data = {st.session_state.pdf_name: validation_results}
                        
                        # Generate Excel file
                        excel_data = st.session_state.exporter.export_validation_results(validation_data)
                        
                        # Create download button
                        st.download_button(
                            label="💾 Download Validation Report",
                            data=excel_data,
                            file_name=f"validation_results_{st.session_state.pdf_name}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                        
                        st.success("Validation report ready for download!")
                        
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
        
        # Preview export data
        st.subheader("📋 Export Preview")
        if st.checkbox("Show export preview"):
            preview_data = []
            for field_name, field_data in st.session_state.extracted_fields.items():
                preview_data.append({
                    'Field': field_name,
                    'Value': field_data.value,
                    'Confidence': field_data.confidence,
                    'Source': field_data.source
                })
            
            st.table(preview_data)
    
    else:
        st.info("👈 Please extract fields first to export them.")

with tab4:
    st.header("💬 Chat with your document")
    
    # Only show chat if we have a processed document
    if st.session_state.vector_store is not None and st.session_state.chatbot is not None:
        # Display chat history
        for sender, message in st.session_state.chat_history:
            with st.container():
                if sender == "You":
                    st.markdown(f"""
                        <div class="chat-message user-message">
                            <b>👤 You:</b><br>{message}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class="chat-message bot-message">
                            <b>🤖 Assistant:</b><br>{message}
                        </div>
                    """, unsafe_allow_html=True)
        
        # User input
        with st.form(key='question_form'):
            user_question = st.text_input(
                "Ask a question about your document:",
                key="user_input",
                placeholder="e.g., What are the main terms of this contract?"
            )
            submit_button = st.form_submit_button("Send")
        
        # Process the question if submitted
        if submit_button and user_question:
            try:
                with st.spinner("Analyzing document..."):
                    response = st.session_state.chatbot.ask_question(user_question)
                
                # Update chat history
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response))
                
                # Rerun to update display
                st.rerun()
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Clear conversation button
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_conversation()
            st.success("Conversation cleared!")

    else:
        st.info("👈 Please upload a PDF document first to enable chat functionality.")

with tab5:
    st.header("📈 System Analytics")
    
    if st.button("🔄 Refresh Analytics"):
        pass  # Just rerun the tab
    
    try:
        # Get comprehensive statistics
        stats = st.session_state.field_extractor.get_extraction_statistics()
        db_stats = stats['database_stats']
        pattern_stats = stats['pattern_stats']
        
        # Database statistics
        st.subheader("🗃️ Database Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Corrections", db_stats['total_corrections'])
        with col2:
            st.metric("Unique Documents", db_stats['unique_documents'])
        with col3:
            st.metric("Learned Patterns", pattern_stats['total_patterns'])
        with col4:
            avg_corrections = db_stats['total_corrections'] / max(db_stats['unique_documents'], 1)
            st.metric("Avg Corrections/Doc", f"{avg_corrections:.1f}")
        
        # Top corrected fields
        if db_stats['top_corrected_fields']:
            st.subheader("🔝 Most Corrected Fields")
            for field_name, count in db_stats['top_corrected_fields']:
                st.write(f"**{field_name.replace('_', ' ').title()}**: {count} corrections")
        
        # Recent corrections
        if db_stats['recent_corrections']:
            st.subheader("🕒 Recent Corrections")
            for pdf_name, field_name, corrected_value, timestamp in db_stats['recent_corrections']:
                st.write(f"📄 **{pdf_name}** - {field_name}: `{corrected_value}` ({timestamp})")
        
        # Pattern statistics
        if pattern_stats['patterns_by_type']:
            st.subheader("🧠 Learned Patterns by Type")
            for field_type, count, avg_confidence in pattern_stats['patterns_by_type']:
                st.write(f"**{field_type.replace('_', ' ').title()}**: {count} patterns (avg confidence: {avg_confidence:.2f})")
        
        # Top performing patterns
        if pattern_stats['top_patterns']:
            st.subheader("🏆 Top Performing Patterns")
            for field_type, pattern, success_count, total_count, success_rate in pattern_stats['top_patterns']:
                st.write(f"**{field_type}**: {success_count}/{total_count} ({success_rate:.1%})")
                st.code(pattern, language="regex")
    
    except Exception as e:
        st.error(f"Error loading analytics: {str(e)}")
        logger.error(traceback.format_exc())
    
    st.subheader("🔧 System Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Learning Data"):
            st.info("Export functionality available in Export tab!")
    
    with col2:
        if st.button("🔄 Reset Learning System"):
            if st.checkbox("⚠️ I understand this will delete ALL learning data"):
                st.session_state.field_extractor.db.clear_all_corrections()
                st.success("Learning system reset!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | **Secure PDF Extractor v1.0** | Advanced PDF Processing with Machine Learning")