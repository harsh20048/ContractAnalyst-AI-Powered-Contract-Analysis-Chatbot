import os
import re
import logging
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from collections import Counter


# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Data class for storing processed document chunks"""
    content: str
    page_num: int
    chunk_num: int
    priority_score: float
    key_terms: List[str]
    suggested_questions: List[str]

class PDFProcessor:
    def __init__(self, api_key: str):
        """Initialize the PDF processor with necessary configurations"""
        self.api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Initialize text splitter with optimized parameters
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=100,
            length_function=len,
            is_separator_regex=False
        )
        
        # Initialize document analysis patterns
        self.section_pattern = re.compile(r'^(?P<number>[\d.]+)\s+(?P<title>[A-Z][^.]+)')
        self.definition_pattern = re.compile(r'"(?P<term>[^"]+)"\s+means\s+(?P<definition>[^.]+)')
        
        # Initialize priority terms with weights and categories
        self.priority_terms: Dict[str, Dict[str, float]] = {
            'legal': {
                'termination': 3.0,
                'confidentiality': 3.0,
                'intellectual property': 3.0,
                'liability': 3.0,
                'indemnification': 3.0,
                'governing law': 2.5,
                'jurisdiction': 2.5,
                'force majeure': 2.5,
            },
            'financial': {
                'payment terms': 3.0,
                'fees': 2.5,
                'expenses': 2.0,
                'pricing': 2.5,
                'invoicing': 2.0,
                'budget': 2.0,
            },
            'operational': {
                'delivery': 2.0,
                'timeline': 2.0,
                'milestones': 2.0,
                'acceptance criteria': 2.0,
                'service levels': 2.5,
                'performance': 2.0,
            }
        }
        
        # Question templates for different term categories
        self.question_templates = {
            'legal': [
                "What are the {} conditions?",
                "How is {} handled in the contract?",
                "What are the {} requirements?",
            ],
            'financial': [
                "What are the {} specifications?",
                "How are {} structured?",
                "What is the {} schedule?",
            ],
            'operational': [
                "What are the {} expectations?",
                "How is {} managed?",
                "What is the {} process?",
            ]
        }

    def clean_text(self, text: str) -> str:
        """Clean text by removing unwanted characters and normalizing whitespace"""
        # Remove special characters and normalize whitespace
        text = text.replace("\f", " ").replace("\r", " ").replace("\u0000", " ")
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def _extract_text_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """Extract text and page numbers from PDF"""
        try:
            pdf = PdfReader(pdf_path)
            pages = []
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                text = self.clean_text(text)
                if text.strip():
                    pages.append((text, i + 1))
            return pages
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise

    def _extract_document_structure(self, text: str) -> Dict[str, List[str]]:
        """Extract document structure including sections and definitions"""
        structure = {
            'sections': [],
            'definitions': {},
            'references': []
        }
        
        # Extract sections
        for match in self.section_pattern.finditer(text):
            section = {
                'number': match.group('number'),
                'title': match.group('title'),
                'content': text[match.end():].split('\n\n')[0]
            }
            structure['sections'].append(section)
            
        # Extract definitions
        for match in self.definition_pattern.finditer(text):
            structure['definitions'][match.group('term')] = match.group('definition')
            
        return structure

    def _score_chunk(self, chunk: str) -> Tuple[float, List[str], List[str]]:
        """Score chunk and identify key terms and potential questions"""
        score = 0.0
        found_terms = []
        relevant_questions = []
        chunk_lower = chunk.lower()
        
        # Extract document structure
        structure = self._extract_document_structure(chunk)
        if structure['sections'] or structure['definitions']:
            score += 2.0  # Boost score for structural content
        
        # Score based on term categories and weights
        for category, terms in self.priority_terms.items():
            category_score = 0
            category_terms = []
            
            for term, weight in terms.items():
                term_lower = term.lower()
                if term_lower in chunk_lower:
                    category_terms.append(term)
                    term_score = weight
                    
                    # Boost score for exact matches
                    if re.search(r'\b' + re.escape(term_lower) + r'\b', chunk_lower):
                        term_score *= 1.5
                    
                    # Boost score for terms in headers/beginnings
                    if re.search(r'^[^.!?]*\b' + re.escape(term_lower) + r'\b', 
                               chunk_lower, re.MULTILINE):
                        term_score *= 1.3
                    
                    category_score += term_score
            
            if category_terms:
                found_terms.extend(category_terms)
                score += category_score
                
                # Generate relevant questions
                templates = self.question_templates.get(category, [])
                for term in category_terms:
                    for template in templates:
                        question = template.format(term)
                        if question not in relevant_questions:
                            relevant_questions.append(question)
        
        # Apply multiplier for chunks with multiple terms
        if len(found_terms) > 2:
            score *= 1.3
        
        return score, found_terms, relevant_questions

    def _extract_potential_questions(self, chunks: List[DocumentChunk]) -> List[str]:
        """Generate potential questions from document content"""
        questions = set()
        term_frequency = Counter()
        
        # Analyze term frequency across chunks
        for chunk in chunks:
            term_frequency.update(chunk.key_terms)
        
        # Generate questions for most frequent terms
        for term, freq in term_frequency.most_common(10):
            for category, terms in self.priority_terms.items():
                if term.lower() in [t.lower() for t in terms]:
                    templates = self.question_templates[category]
                    for template in templates:
                        questions.add(template.format(term))
        
        return list(questions)

    def process_pdf(self, pdf_path: str) -> Tuple[FAISS, List[str]]:
        """Process PDF and return vector store and suggested questions"""
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # File validation checks
            if not os.path.exists(pdf_path):
                raise ValueError("PDF file not found")
            
            file_size = os.path.getsize(pdf_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError(f"File too large ({file_size/1024/1024:.1f}MB). Please upload a file smaller than 10MB.")
            
            # Extract text from PDF
            pages = self._extract_text_from_pdf(pdf_path)
            
            # Process chunks
            processed_chunks: List[DocumentChunk] = []
            chunk_id = 0
            
            all_texts = []
            all_metadatas = []
            
            # Process each page
            for page_text, page_num in pages:
                # Split text into chunks
                page_chunks = self.text_splitter.split_text(page_text)
                
                # Process each chunk
                for chunk_num, chunk in enumerate(page_chunks):
                    # Score and analyze chunk
                    score, terms, questions = self._score_chunk(chunk)
                    
                    # Create document chunk
                    processed_chunk = DocumentChunk(
                        content=chunk,
                        page_num=page_num,
                        chunk_num=chunk_id,
                        priority_score=score,
                        key_terms=terms,
                        suggested_questions=questions
                    )
                    
                    processed_chunks.append(processed_chunk)
                    
                    # Extract document structure
                    structure = self._extract_document_structure(chunk)
                    
                    # Add to texts and metadatas for vector store
                    all_texts.append(chunk)
                    all_metadatas.append({
                        'page': page_num,
                        'chunk': chunk_id,
                        'score': score,
                        'terms': ','.join(terms),
                        'sections': [s['number'] for s in structure['sections']],
                        'definitions': list(structure['definitions'].keys())
                    })
                    
                    chunk_id += 1
            
            if not all_texts:
                raise ValueError("No text extracted from PDF")
            
            # Create vector store
            vectordb = FAISS.from_texts(
                texts=all_texts,
                embedding=self.embeddings,
                metadatas=all_metadatas
            )
            
            # Generate suggested questions
            suggested_questions = self._extract_potential_questions(processed_chunks)
            
            logger.info(f"Successfully processed PDF with {len(processed_chunks)} chunks")
            return vectordb, suggested_questions
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise

    def get_priority_terms(self) -> Dict[str, List[str]]:
        """Return organized priority terms by category"""
        return {
            category: list(terms.keys())
            for category, terms in self.priority_terms.items()
        }