"""
Extraction Engine that combines PDF processing with Hugging Face model inference.
Orchestrates the complete pipeline from PDF input to structured output.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import json

from .pdf_processor import PDFProcessor, ProcessingResult
from .models import ModelManager, InferenceResult
from .config import settings, EXTRACTION_TASKS

logger = logging.getLogger(__name__)


@dataclass
class ExtractionRequest:
    """Request for document extraction."""
    file_path: str
    tasks: List[str]
    model_preferences: Optional[Dict[str, str]] = None
    custom_questions: Optional[List[str]] = None
    extract_tables: bool = True
    extract_metadata: bool = True
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@dataclass
class ExtractionOutput:
    """Complete extraction output."""
    success: bool
    file_path: str
    tasks_completed: List[str]
    
    # Original processing results
    processing_result: Optional[ProcessingResult] = None
    
    # Task-specific results
    summary: Optional[str] = None
    key_information: Optional[Dict[str, Any]] = None
    questions_answers: Optional[List[Dict[str, Any]]] = None
    sentiment_analysis: Optional[Dict[str, Any]] = None
    named_entities: Optional[Dict[str, List[Dict[str, Any]]]] = None
    text_classification: Optional[List[Dict[str, Any]]] = None
    topic_analysis: Optional[Dict[str, Any]] = None
    
    # Additional information
    processing_time: float = 0.0
    total_words: int = 0
    total_sentences: int = 0
    language: Optional[str] = None
    key_phrases: Optional[List[str]] = None
    
    # Error information
    errors: List[str] = None
    warnings: List[str] = None


class ExtractionEngine:
    """
    Main extraction engine that orchestrates PDF processing and model inference.
    
    Provides high-level interface for document analysis using multiple models
    and extraction techniques.
    """
    
    def __init__(self):
        """Initialize the extraction engine."""
        self.pdf_processor = PDFProcessor()
        self.model_manager = ModelManager()
        
    def extract_from_pdf(self, request: ExtractionRequest) -> ExtractionOutput:
        """
        Extract information from PDF using specified tasks.
        
        Args:
            request: Extraction request with file path and tasks
            
        Returns:
            ExtractionOutput with all results
        """
        start_time = time.time()
        
        # Initialize output
        output = ExtractionOutput(
            success=False,
            file_path=request.file_path,
            tasks_completed=[],
            errors=[],
            warnings=[]
        )
        
        try:
            # Step 1: Process PDF
            logger.info(f"Processing PDF: {request.file_path}")
            processing_result = self.pdf_processor.process_pdf(request.file_path)
            
            if not processing_result.success:
                output.errors.append(f"PDF processing failed: {processing_result.error_message}")
                return output
            
            output.processing_result = processing_result
            output.total_words = processing_result.word_count
            output.total_sentences = processing_result.sentence_count
            output.language = processing_result.language
            
            # Combine all text for analysis
            all_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
            
            if not all_text.strip():
                output.errors.append("No text extracted from PDF")
                return output
            
            # Step 2: Execute requested tasks
            for task in request.tasks:
                if task not in EXTRACTION_TASKS:
                    output.warnings.append(f"Unknown task: {task}")
                    continue
                
                try:
                    logger.info(f"Executing task: {task}")
                    
                    if task == "summarization":
                        result = self._extract_summary(all_text, request.model_preferences)
                        if result:
                            output.summary = result
                            output.tasks_completed.append(task)
                    
                    elif task == "question_answering":
                        result = self._extract_qa(all_text, request.custom_questions, request.model_preferences)
                        if result:
                            output.questions_answers = result
                            output.tasks_completed.append(task)
                    
                    elif task == "key_information_extraction":
                        result = self._extract_key_information(processing_result, request.model_preferences)
                        if result:
                            output.key_information = result
                            output.tasks_completed.append(task)
                    
                    elif task == "sentiment_analysis":
                        result = self._analyze_sentiment(all_text, request.model_preferences)
                        if result:
                            output.sentiment_analysis = result
                            output.tasks_completed.append(task)
                    
                    elif task == "named_entity_recognition":
                        result = self._extract_entities(all_text, request.model_preferences)
                        if result:
                            output.named_entities = result
                            output.tasks_completed.append(task)
                    
                    elif task == "text_classification":
                        result = self._classify_text(all_text, request.model_preferences)
                        if result:
                            output.text_classification = result
                            output.tasks_completed.append(task)
                    
                    elif task == "topic_modeling":
                        result = self._analyze_topics(all_text)
                        if result:
                            output.topic_analysis = result
                            output.tasks_completed.append(task)
                    
                except Exception as e:
                    error_msg = f"Error in task {task}: {str(e)}"
                    logger.error(error_msg)
                    output.errors.append(error_msg)
            
            # Step 3: Extract additional information
            try:
                output.key_phrases = self.pdf_processor.extract_key_phrases(all_text)
            except Exception as e:
                output.warnings.append(f"Failed to extract key phrases: {str(e)}")
            
            # Mark as successful if we completed at least one task
            output.success = len(output.tasks_completed) > 0
            output.processing_time = time.time() - start_time
            
            logger.info(f"Extraction completed in {output.processing_time:.2f}s. Tasks completed: {output.tasks_completed}")
            return output
            
        except Exception as e:
            logger.error(f"Critical error in extraction: {str(e)}")
            output.errors.append(f"Critical error: {str(e)}")
            output.processing_time = time.time() - start_time
            return output
    
    def _extract_summary(self, text: str, model_preferences: Optional[Dict[str, str]]) -> Optional[str]:
        """Extract summary from text."""
        model_name = None
        if model_preferences:
            model_name = model_preferences.get("summarization")
        
        # Split text into chunks if too long
        chunks = self.pdf_processor.chunk_text(text, chunk_size=800, chunk_overlap=100)
        
        if len(chunks) == 1:
            # Single chunk - direct summarization
            result = self.model_manager.summarize_text(chunks[0], model_name=model_name)
            if result.result:
                return result.result
        else:
            # Multiple chunks - summarize each then combine
            chunk_summaries = []
            for chunk in chunks:
                result = self.model_manager.summarize_text(chunk, model_name=model_name)
                if result.result:
                    chunk_summaries.append(result.result)
            
            if chunk_summaries:
                # Summarize the summaries
                combined_summary = " ".join(chunk_summaries)
                final_result = self.model_manager.summarize_text(combined_summary, model_name=model_name)
                if final_result.result:
                    return final_result.result
        
        return None
    
    def _extract_qa(self, text: str, custom_questions: Optional[List[str]], 
                   model_preferences: Optional[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
        """Extract answers to questions."""
        model_name = None
        if model_preferences:
            model_name = model_preferences.get("question_answering")
        
        # Default questions if none provided
        if not custom_questions:
            custom_questions = [
                "What is the main topic of this document?",
                "What are the key findings or conclusions?",
                "Who are the main people or organizations mentioned?",
                "What dates or time periods are mentioned?",
                "What actions or decisions are described?"
            ]
        
        results = []
        for question in custom_questions:
            try:
                result = self.model_manager.answer_question(question, text, model_name=model_name)
                if result.result:
                    results.append({
                        "question": question,
                        "answer": result.result["answer"],
                        "confidence": result.result["confidence"],
                        "processing_time": result.processing_time
                    })
            except Exception as e:
                logger.warning(f"Failed to answer question '{question}': {str(e)}")
        
        return results if results else None
    
    def _extract_key_information(self, processing_result: ProcessingResult, 
                                model_preferences: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Extract key information using multiple techniques."""
        all_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
        
        key_info = {
            "metadata": asdict(processing_result.metadata),
            "statistics": {
                "word_count": processing_result.word_count,
                "sentence_count": processing_result.sentence_count,
                "page_count": processing_result.metadata.page_count,
                "language": processing_result.language
            }
        }
        
        # Extract key phrases
        try:
            key_info["key_phrases"] = self.pdf_processor.extract_key_phrases(all_text, top_k=15)
        except Exception as e:
            logger.warning(f"Failed to extract key phrases: {str(e)}")
        
        # Extract named entities using NLTK
        try:
            key_info["nltk_entities"] = self.pdf_processor.extract_named_entities(all_text)
        except Exception as e:
            logger.warning(f"Failed to extract NLTK entities: {str(e)}")
        
        # Extract tables info
        if processing_result.tables:
            key_info["tables"] = []
            for table in processing_result.tables:
                table_info = {
                    "page_number": table.page_number,
                    "rows": len(table.data),
                    "columns": len(table.data[0]) if table.data else 0,
                    "headers": table.headers
                }
                key_info["tables"].append(table_info)
        
        return key_info
    
    def _analyze_sentiment(self, text: str, model_preferences: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Analyze sentiment of the text."""
        model_name = None
        if model_preferences:
            model_name = model_preferences.get("text_classification")
        
        # Analyze overall sentiment
        result = self.model_manager.classify_text(text, model_name=model_name)
        
        if result.result:
            # Also analyze sentiment of individual chunks for variation
            chunks = self.pdf_processor.chunk_text(text, chunk_size=500, chunk_overlap=50)
            chunk_sentiments = []
            
            for i, chunk in enumerate(chunks[:10]):  # Limit to first 10 chunks
                chunk_result = self.model_manager.classify_text(chunk, model_name=model_name)
                if chunk_result.result:
                    chunk_sentiments.append({
                        "chunk_index": i,
                        "sentiment": chunk_result.result[0]["label"],
                        "confidence": chunk_result.result[0]["score"]
                    })
            
            return {
                "overall_sentiment": result.result[0]["label"],
                "overall_confidence": result.result[0]["score"],
                "processing_time": result.processing_time,
                "chunk_sentiments": chunk_sentiments
            }
        
        return None
    
    def _extract_entities(self, text: str, model_preferences: Optional[Dict[str, str]]) -> Optional[Dict[str, List[Dict[str, Any]]]]:
        """Extract named entities using Hugging Face models."""
        model_name = None
        if model_preferences:
            model_name = model_preferences.get("named_entity_recognition")
        
        result = self.model_manager.extract_entities(text, model_name=model_name)
        
        if result.result:
            return result.result
        
        return None
    
    def _classify_text(self, text: str, model_preferences: Optional[Dict[str, str]]) -> Optional[List[Dict[str, Any]]]:
        """Classify text into categories."""
        model_name = None
        if model_preferences:
            model_name = model_preferences.get("text_classification")
        
        result = self.model_manager.classify_text(text, model_name=model_name)
        
        if result.result:
            return result.result
        
        return None
    
    def _analyze_topics(self, text: str) -> Optional[Dict[str, Any]]:
        """Analyze topics in the text using simple keyword extraction."""
        try:
            # Extract key phrases as topics
            key_phrases = self.pdf_processor.extract_key_phrases(text, top_k=20)
            
            # Simple topic clustering based on phrase similarity
            topics = {}
            for phrase in key_phrases:
                words = phrase.split()
                for word in words:
                    if len(word) > 3:  # Skip short words
                        if word not in topics:
                            topics[word] = []
                        topics[word].append(phrase)
            
            # Sort topics by frequency
            topic_scores = {topic: len(phrases) for topic, phrases in topics.items()}
            sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "top_topics": [{"topic": topic, "frequency": freq} for topic, freq in sorted_topics[:10]],
                "key_phrases": key_phrases
            }
            
        except Exception as e:
            logger.warning(f"Failed to analyze topics: {str(e)}")
            return None
    
    def extract_from_text(self, text: str, tasks: List[str], 
                         model_preferences: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Extract information from plain text (not PDF).
        
        Args:
            text: Input text to process
            tasks: List of extraction tasks
            model_preferences: Preferred models for each task
            
        Returns:
            Dictionary with extraction results
        """
        results = {}
        
        for task in tasks:
            if task not in EXTRACTION_TASKS:
                continue
                
            try:
                if task == "summarization":
                    result = self._extract_summary(text, model_preferences)
                    if result:
                        results[task] = result
                
                elif task == "sentiment_analysis":
                    result = self._analyze_sentiment(text, model_preferences)
                    if result:
                        results[task] = result
                
                elif task == "named_entity_recognition":
                    result = self._extract_entities(text, model_preferences)
                    if result:
                        results[task] = result
                
                elif task == "text_classification":
                    result = self._classify_text(text, model_preferences)
                    if result:
                        results[task] = result
                
                # Add other tasks as needed
                
            except Exception as e:
                logger.error(f"Error in task {task}: {str(e)}")
                results[f"{task}_error"] = str(e)
        
        return results
    
    def get_supported_tasks(self) -> List[str]:
        """Get list of supported extraction tasks."""
        return EXTRACTION_TASKS.copy()
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all loaded models."""
        return {
            "model_info": self.model_manager.get_model_info(),
            "memory_usage": self.model_manager.get_memory_usage()
        }
    
    def warm_up(self, tasks: List[str] = None):
        """Pre-load models for faster inference."""
        self.model_manager.warm_up_models(tasks)


# Global extraction engine instance
extraction_engine = ExtractionEngine()