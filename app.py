import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os
import tempfile
import traceback
from dotenv import load_dotenv
from pdf_processor import PDFProcessor
from chatbot import Chatbot
from field_extractor import FieldExtractor
from database import LearningDatabase

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
    page_title="Contract Analysis & Field Extraction",
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

st.title("📄 Contract Analysis & Field Extraction")

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

# Create tabs for different functionality
tab1, tab2, tab3 = st.tabs(["📋 Field Extraction", "💬 Document Chat", "📊 Learning Statistics"])

with tab1:
    st.header("🔍 Document Field Extraction")
    
    # Sidebar for file upload and processing
    with st.sidebar:
        st.header("📁 Document Upload")
        pdf_file = st.file_uploader("Upload your contract/PDF", type=['pdf'])
        
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
        if st.button("📊 View Statistics"):
            st.session_state.show_stats = True
        
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
        
        # Handle form submission
        def handle_submit():
            st.session_state.question_submitted = True
        
        # User input
        with st.form(key='question_form'):
            user_question = st.text_input(
                "Ask a question about your document:",
                key="user_input",
                placeholder="e.g., What are the main terms of this contract?"
            )
            submit_button = st.form_submit_button("Send", on_click=handle_submit)
        
        # Process the question if submitted
        if st.session_state.question_submitted and user_question:
            try:
                with st.spinner("Analyzing document..."):
                    response = st.session_state.chatbot.ask_question(user_question)
                
                # Update chat history
                st.session_state.chat_history.append(("You", user_question))
                st.session_state.chat_history.append(("Bot", response))
                
                # Reset submission state
                st.session_state.question_submitted = False
                
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

with tab3:
    st.header("📊 Learning Statistics")
    
    if st.button("🔄 Refresh Statistics"):
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
        st.error(f"Error loading statistics: {str(e)}")
        logger.error(traceback.format_exc())
    
    st.subheader("🔧 System Management")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Export Learning Data"):
            st.info("Export functionality coming soon!")
    
    with col2:
        if st.button("🔄 Reset Learning System"):
            if st.checkbox("⚠️ I understand this will delete ALL learning data"):
                st.session_state.field_extractor.db.clear_all_corrections()
                st.success("Learning system reset!")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit | Advanced PDF Processing with Machine Learning")