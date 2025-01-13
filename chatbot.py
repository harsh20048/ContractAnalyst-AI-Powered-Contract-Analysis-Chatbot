import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import logging
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from document_analyzer import DocumentAnalyzer, ResponseGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_context.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    """Data class for storing chat messages"""
    role: str
    content: str
    timestamp: float = None

class Chatbot:
    def __init__(self, vector_store, api_key: str):
        """
        Initialize the chatbot with vector store and API key.
        
        Args:
            vector_store: FAISS vector store containing document embeddings
            api_key (str): Google API key for Gemini
        """
        self.api_key = api_key
        self.vector_store = vector_store
        self.chat_history = []
        self.is_processing = False
        self.analyzer = DocumentAnalyzer()
        self.response_generator = None
        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                'gemini-pro',
                generation_config={
                    'temperature': 0.2,  # Lower temperature for more focused responses
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 1000,
                    'stop_sequences': ["Human:", "Assistant:"]
                }
            )
            logger.info("Gemini model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Gemini model: {str(e)}")
            raise

    def _get_relevant_context(self, question: str, max_chunks: int = 3) -> List[str]:
        """
        Retrieve relevant context chunks for the question with enhanced logging.
        
        Args:
            question (str): User's question
            max_chunks (int): Maximum number of context chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            logger.info(f"Searching for context for question: {question}")
            
            search_results = self.vector_store.similarity_search(
                question.strip(),
                k=max_chunks,
                fetch_k=5  # Fetch more candidates for better selection
            )
            
            contexts = [doc.page_content for doc in search_results]
            
            # Log retrieved contexts with metadata
            logger.info(f"Retrieved {len(contexts)} context chunks")
            for i, doc in enumerate(search_results):
                logger.info(f"\nContext Chunk {i + 1}:")
                logger.info(f"Page: {doc.metadata.get('page', 'N/A')}")
                logger.info(f"Chunk ID: {doc.metadata.get('chunk', 'N/A')}")
                logger.info(f"Priority Score: {doc.metadata.get('score', 'N/A')}")
                logger.info(f"Key Terms: {doc.metadata.get('terms', 'N/A')}")
                logger.info(f"Content Preview: {doc.page_content[:200]}...")
                logger.info("-" * 80)
            
            if not contexts:
                logger.warning("No relevant context found for the question")
            
            return contexts
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            logger.error(f"Stack trace: {traceback.format_exc()}")
            raise


    def _validate_question(self, question: str) -> tuple[bool, str]:
        """
        Validate the user's question.
        
        Args:
            question (str): User's question
            
        Returns:
            tuple[bool, str]: (is_valid, error_message)
        """
        if not question:
            return False, "Please provide a question."
        if len(question.strip()) < 3:
            return False, "Please provide a more detailed question about the document."
        return True, ""

    def _get_relevant_context(self, question: str, max_chunks: int = 3) -> List[str]:
        """
        Retrieve relevant context chunks for the question.
        
        Args:
            question (str): User's question
            max_chunks (int): Maximum number of context chunks to retrieve
            
        Returns:
            List[str]: List of relevant text chunks
        """
        try:
            search_results = self.vector_store.similarity_search(
                question.strip(),
                k=max_chunks,
                fetch_k=5  # Fetch more candidates for better selection
            )
            return [doc.page_content for doc in search_results]
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            raise

    def _build_prompt(self, question: str, context: List[str]) -> str:
        """
        Build the prompt for the Gemini model.
        
        Args:
            question (str): User's question
            context (List[str]): Relevant context chunks
            
        Returns:
            str: Formatted prompt
        """
        context_text = "\n\n".join(context)
        return f"""You are a helpful assistant analyzing a legal document. Based on the following context, provide a clear and direct answer.
        If the specific information isn't found in the context, say "I cannot find that specific information in the document."
        
        Context sections:
        {context_text}
        
        Question: {question}
        
        Please provide a clear, direct answer based only on the above context:"""

    def _format_response(self, text: str) -> str:
        """
        Clean and format the model's response.
        
        Args:
            text (str): Raw response text
            
        Returns:
            str: Formatted response
        """
        if not text:
            return "No response generated."
            
        text = text.strip()
        
        # Remove common prefixes
        prefixes = [
            r'^based on',
            r'^according to',
            r'^from the context',
            r'^the document states?',
            r'^as per the document'
        ]
        pattern = '|'.join(prefixes)
        text = re.sub(pattern, '', text, flags=re.IGNORECASE).strip()
        
        # Clean up newlines and spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Remove excessive newlines
        
        return text

    def ask_question(self, question: str) -> str:
        """
        Process a question and return an answer based on the document context.
        
        Args:
            question (str): User's question
            
        Returns:
            str: Response to the question
        """
        if self.is_processing:
            return "Still processing previous question. Please wait."
            
        try:
            self.is_processing = True
            
            # Validate question
            is_valid, error_message = self._validate_question(question)
            if not is_valid:
                return error_message
            
            # Get relevant context
            try:
                context_chunks = self._get_relevant_context(question)
                if not context_chunks:
                    return "I couldn't find relevant information in the document. Please try asking about specific terms or sections."
            except Exception as e:
                logger.error(f"Error getting context: {str(e)}")
                return "I'm having trouble searching the document. Please try again."
            
            # Build prompt
            prompt = self._build_prompt(question, context_chunks)
            
            # Generate response with retry logic
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = self.model.generate_content(prompt)
                    if response and response.text:
                        formatted_response = self._format_response(response.text)
                        # Add to chat history
                        self.chat_history.append(ChatMessage(role="user", content=question))
                        self.chat_history.append(ChatMessage(role="assistant", content=formatted_response))
                        return formatted_response
                except Exception as e:
                    logger.error(f"Error generating response (attempt {attempt + 1}): {str(e)}")
                    if attempt == max_retries - 1:
                        raise e
                    continue
            
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "I apologize, but I'm having trouble processing your question. Please try again or rephrase your question."
        finally:
            self.is_processing = False

    def reset_conversation(self) -> None:
        """Reset the conversation history"""
        self.chat_history = []
        logger.info("Conversation history reset")

    def get_chat_history(self) -> List[ChatMessage]:
        """
        Get the conversation history.
        
        Returns:
            List[ChatMessage]: List of chat messages
        """
        return self.chat_history

    def get_message_count(self) -> int:
        """
        Get the total number of messages in the conversation.
        
        Returns:
            int: Number of messages
        """
        return len(self.chat_history)