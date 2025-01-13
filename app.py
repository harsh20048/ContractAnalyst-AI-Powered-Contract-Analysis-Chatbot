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
    page_title="Contract Analysis Chatbot",
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
    </style>
    """, unsafe_allow_html=True)

st.title("📄 Contract Analysis Chatbot")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'question_submitted' not in st.session_state:
    st.session_state.question_submitted = False

# Sidebar
with st.sidebar:
    st.header("📁 Document Upload")
    pdf_file = st.file_uploader("Upload your contract/PDF", type=['pdf'])
    
    if pdf_file:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_file_path = tmp_file.name
            
        try:
            # Initialize PDF processor
            pdf_processor = PDFProcessor(api_key=os.getenv("GOOGLE_API_KEY"))
            
            # Process PDF
            with st.spinner("Processing document..."):
                try:
                    st.session_state.vector_store, suggested_questions = pdf_processor.process_pdf(temp_file_path)
                    st.success("✅ Document processed successfully!")
                    
                    # Initialize chatbot
                    st.session_state.chatbot = Chatbot(
                        st.session_state.vector_store,
                        api_key=os.getenv("GOOGLE_API_KEY")
                    )
                    
                    # Display suggested questions
                    st.subheader("💡 Suggested Questions")
                    for question in suggested_questions[:5]:
                        if st.button(question):
                            st.session_state.current_question = question
                    
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
    
    # Clear conversation button
    if st.session_state.vector_store is not None:
        if st.button("🗑️ Clear Conversation"):
            st.session_state.chat_history = []
            if st.session_state.chatbot:
                st.session_state.chatbot.reset_conversation()
            st.success("Conversation cleared!")

# Main chat interface
if st.session_state.vector_store is not None and st.session_state.chatbot is not None:
    st.header("💬 Chat with your document")
    
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
    
    # Handle suggested question if selected
    if 'current_question' in st.session_state:
        user_question = st.session_state.current_question
        st.session_state.question_submitted = True
        del st.session_state.current_question
    
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

else:
    st.info("👈 Please upload a contract or PDF document to begin the analysis.")
    
    st.markdown("""
    ### 🔍 What can this chatbot do?
    
    This contract analysis chatbot can help you:
    - Extract key contract terms and conditions
    - Find specific clauses and provisions
    - Answer questions about legal requirements
    - Analyze payment terms and conditions
    - Identify important dates and deadlines
    
    ### 📝 Supported Document Types
    - PDF contracts
    - Legal agreements
    - Business documents
    """)