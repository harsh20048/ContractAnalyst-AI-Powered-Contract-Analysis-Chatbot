"""
Model management for Hugging Face models.
Supports various NLP tasks including summarization, Q&A, classification, and more.
"""

import logging
import torch
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
import time
from pathlib import Path
import json

from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification, pipeline, Pipeline
)
from sentence_transformers import SentenceTransformer
import numpy as np

from .config import settings, MODEL_CONFIGS

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    task: str
    model_size: str
    is_loaded: bool = False
    load_time: float = 0.0
    memory_usage: Optional[int] = None
    parameters: Optional[int] = None


@dataclass
class InferenceResult:
    """Result from model inference."""
    task: str
    model_name: str
    result: Any
    processing_time: float
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelManager:
    """
    Manages loading and inference with multiple Hugging Face models.
    Supports caching, dynamic loading, and various NLP tasks.
    """
    
    def __init__(self):
        """Initialize the model manager."""
        self.loaded_models: Dict[str, Pipeline] = {}
        self.embedding_models: Dict[str, SentenceTransformer] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize cache directory
        self.cache_dir = Path("./model_cache")
        self.cache_dir.mkdir(exist_ok=True)
        
    def load_model(self, model_name: str, task: str, force_reload: bool = False) -> bool:
        """
        Load a Hugging Face model for a specific task.
        
        Args:
            model_name: Name of the model to load
            task: Task type (summarization, question-answering, etc.)
            force_reload: Whether to reload if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        model_key = f"{model_name}_{task}"
        
        if not force_reload and model_key in self.loaded_models:
            logger.info(f"Model {model_name} for {task} already loaded")
            return True
            
        try:
            start_time = time.time()
            logger.info(f"Loading model {model_name} for task {task}...")
            
            # Create pipeline based on task
            if task == "summarization":
                pipe = pipeline(
                    "summarization",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)}
                )
            elif task == "question_answering":
                pipe = pipeline(
                    "question-answering",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)}
                )
            elif task == "text_classification" or task == "sentiment_analysis":
                pipe = pipeline(
                    "text-classification",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)}
                )
            elif task == "named_entity_recognition":
                pipe = pipeline(
                    "ner",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)},
                    aggregation_strategy="simple"
                )
            elif task == "text_generation":
                pipe = pipeline(
                    "text-generation",
                    model=model_name,
                    tokenizer=model_name,
                    device=0 if self.device == "cuda" else -1,
                    model_kwargs={"cache_dir": str(self.cache_dir)}
                )
            else:
                logger.error(f"Unsupported task: {task}")
                return False
            
            load_time = time.time() - start_time
            
            # Store model and info
            self.loaded_models[model_key] = pipe
            self.model_info[model_key] = ModelInfo(
                name=model_name,
                task=task,
                model_size="unknown",  # Could be enhanced to detect model size
                is_loaded=True,
                load_time=load_time
            )
            
            logger.info(f"Successfully loaded {model_name} for {task} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name} for {task}: {str(e)}")
            return False
    
    def load_embedding_model(self, model_name: str, force_reload: bool = False) -> bool:
        """
        Load a sentence transformer model for embeddings.
        
        Args:
            model_name: Name of the embedding model
            force_reload: Whether to reload if already loaded
            
        Returns:
            True if successful, False otherwise
        """
        if not force_reload and model_name in self.embedding_models:
            logger.info(f"Embedding model {model_name} already loaded")
            return True
            
        try:
            start_time = time.time()
            logger.info(f"Loading embedding model {model_name}...")
            
            model = SentenceTransformer(
                model_name,
                device=self.device,
                cache_folder=str(self.cache_dir)
            )
            
            load_time = time.time() - start_time
            
            self.embedding_models[model_name] = model
            self.model_info[f"{model_name}_embeddings"] = ModelInfo(
                name=model_name,
                task="embeddings",
                model_size="unknown",
                is_loaded=True,
                load_time=load_time
            )
            
            logger.info(f"Successfully loaded embedding model {model_name} in {load_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            return False
    
    def summarize_text(self, text: str, model_name: str = None, 
                      max_length: int = 150, min_length: int = 30) -> InferenceResult:
        """
        Summarize text using a summarization model.
        
        Args:
            text: Text to summarize
            model_name: Model to use (default from config)
            max_length: Maximum summary length
            min_length: Minimum summary length
            
        Returns:
            InferenceResult with summary
        """
        model_name = model_name or MODEL_CONFIGS["summarization"]["default"]
        model_key = f"{model_name}_summarization"
        
        # Load model if not loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_name, "summarization"):
                return InferenceResult(
                    task="summarization",
                    model_name=model_name,
                    result=None,
                    processing_time=0.0,
                    metadata={"error": "Failed to load model"}
                )
        
        try:
            start_time = time.time()
            
            # Truncate text if too long
            max_input_length = 1024  # Most models have limits
            if len(text) > max_input_length:
                text = text[:max_input_length]
            
            result = self.loaded_models[model_key](
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )
            
            processing_time = time.time() - start_time
            
            summary = result[0]["summary_text"] if result else ""
            
            return InferenceResult(
                task="summarization",
                model_name=model_name,
                result=summary,
                processing_time=processing_time,
                confidence=result[0].get("score") if result else None,
                metadata={
                    "input_length": len(text),
                    "output_length": len(summary),
                    "max_length": max_length,
                    "min_length": min_length
                }
            )
            
        except Exception as e:
            logger.error(f"Error in summarization: {str(e)}")
            return InferenceResult(
                task="summarization",
                model_name=model_name,
                result=None,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def answer_question(self, question: str, context: str, 
                       model_name: str = None) -> InferenceResult:
        """
        Answer a question based on provided context.
        
        Args:
            question: Question to answer
            context: Context to search for answer
            model_name: Model to use (default from config)
            
        Returns:
            InferenceResult with answer
        """
        model_name = model_name or MODEL_CONFIGS["question_answering"]["default"]
        model_key = f"{model_name}_question_answering"
        
        # Load model if not loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_name, "question_answering"):
                return InferenceResult(
                    task="question_answering",
                    model_name=model_name,
                    result=None,
                    processing_time=0.0,
                    metadata={"error": "Failed to load model"}
                )
        
        try:
            start_time = time.time()
            
            result = self.loaded_models[model_key](
                question=question,
                context=context
            )
            
            processing_time = time.time() - start_time
            
            return InferenceResult(
                task="question_answering",
                model_name=model_name,
                result={
                    "answer": result["answer"],
                    "confidence": result["score"],
                    "start": result["start"],
                    "end": result["end"]
                },
                processing_time=processing_time,
                confidence=result["score"],
                metadata={
                    "question_length": len(question),
                    "context_length": len(context)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in question answering: {str(e)}")
            return InferenceResult(
                task="question_answering",
                model_name=model_name,
                result=None,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def classify_text(self, text: str, model_name: str = None) -> InferenceResult:
        """
        Classify text using a classification model.
        
        Args:
            text: Text to classify
            model_name: Model to use (default from config)
            
        Returns:
            InferenceResult with classification
        """
        model_name = model_name or MODEL_CONFIGS["text_classification"]["default"]
        model_key = f"{model_name}_text_classification"
        
        # Load model if not loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_name, "text_classification"):
                return InferenceResult(
                    task="text_classification",
                    model_name=model_name,
                    result=None,
                    processing_time=0.0,
                    metadata={"error": "Failed to load model"}
                )
        
        try:
            start_time = time.time()
            
            result = self.loaded_models[model_key](text)
            
            processing_time = time.time() - start_time
            
            return InferenceResult(
                task="text_classification",
                model_name=model_name,
                result=result,
                processing_time=processing_time,
                confidence=result[0]["score"] if result else None,
                metadata={
                    "input_length": len(text),
                    "num_classes": len(result) if result else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error in text classification: {str(e)}")
            return InferenceResult(
                task="text_classification",
                model_name=model_name,
                result=None,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def extract_entities(self, text: str, model_name: str = None) -> InferenceResult:
        """
        Extract named entities from text.
        
        Args:
            text: Text to process
            model_name: Model to use (default from config)
            
        Returns:
            InferenceResult with entities
        """
        model_name = model_name or MODEL_CONFIGS["named_entity_recognition"]["default"]
        model_key = f"{model_name}_named_entity_recognition"
        
        # Load model if not loaded
        if model_key not in self.loaded_models:
            if not self.load_model(model_name, "named_entity_recognition"):
                return InferenceResult(
                    task="named_entity_recognition",
                    model_name=model_name,
                    result=None,
                    processing_time=0.0,
                    metadata={"error": "Failed to load model"}
                )
        
        try:
            start_time = time.time()
            
            result = self.loaded_models[model_key](text)
            
            processing_time = time.time() - start_time
            
            # Group entities by type
            entities_by_type = {}
            for entity in result:
                entity_type = entity["entity_group"]
                if entity_type not in entities_by_type:
                    entities_by_type[entity_type] = []
                entities_by_type[entity_type].append({
                    "text": entity["word"],
                    "confidence": entity["score"],
                    "start": entity["start"],
                    "end": entity["end"]
                })
            
            return InferenceResult(
                task="named_entity_recognition",
                model_name=model_name,
                result=entities_by_type,
                processing_time=processing_time,
                confidence=np.mean([e["score"] for e in result]) if result else None,
                metadata={
                    "input_length": len(text),
                    "num_entities": len(result)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in entity extraction: {str(e)}")
            return InferenceResult(
                task="named_entity_recognition",
                model_name=model_name,
                result=None,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_embeddings(self, texts: Union[str, List[str]], 
                      model_name: str = None) -> InferenceResult:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Text or list of texts to embed
            model_name: Model to use (default from config)
            
        Returns:
            InferenceResult with embeddings
        """
        model_name = model_name or MODEL_CONFIGS["embeddings"]["default"]
        
        # Load model if not loaded
        if model_name not in self.embedding_models:
            if not self.load_embedding_model(model_name):
                return InferenceResult(
                    task="embeddings",
                    model_name=model_name,
                    result=None,
                    processing_time=0.0,
                    metadata={"error": "Failed to load model"}
                )
        
        try:
            start_time = time.time()
            
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            embeddings = self.embedding_models[model_name].encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            )
            
            processing_time = time.time() - start_time
            
            return InferenceResult(
                task="embeddings",
                model_name=model_name,
                result=embeddings.tolist(),
                processing_time=processing_time,
                metadata={
                    "num_texts": len(texts),
                    "embedding_dim": embeddings.shape[1],
                    "input_lengths": [len(text) for text in texts]
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return InferenceResult(
                task="embeddings",
                model_name=model_name,
                result=None,
                processing_time=0.0,
                metadata={"error": str(e)}
            )
    
    def get_model_info(self) -> Dict[str, ModelInfo]:
        """Get information about all loaded models."""
        return self.model_info.copy()
    
    def unload_model(self, model_name: str, task: str) -> bool:
        """
        Unload a specific model to free memory.
        
        Args:
            model_name: Name of the model
            task: Task type
            
        Returns:
            True if successful
        """
        model_key = f"{model_name}_{task}"
        
        if model_key in self.loaded_models:
            del self.loaded_models[model_key]
            if model_key in self.model_info:
                self.model_info[model_key].is_loaded = False
            logger.info(f"Unloaded model {model_name} for {task}")
            return True
        
        if task == "embeddings" and model_name in self.embedding_models:
            del self.embedding_models[model_name]
            embedding_key = f"{model_name}_embeddings"
            if embedding_key in self.model_info:
                self.model_info[embedding_key].is_loaded = False
            logger.info(f"Unloaded embedding model {model_name}")
            return True
        
        return False
    
    def unload_all_models(self):
        """Unload all models to free memory."""
        self.loaded_models.clear()
        self.embedding_models.clear()
        for info in self.model_info.values():
            info.is_loaded = False
        logger.info("Unloaded all models")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        memory_info = {
            "device": self.device,
            "loaded_models": len(self.loaded_models),
            "loaded_embedding_models": len(self.embedding_models)
        }
        
        if torch.cuda.is_available():
            memory_info.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved(),
                "gpu_memory_free": torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            })
        
        return memory_info
    
    def warm_up_models(self, tasks: List[str] = None):
        """
        Pre-load models for specified tasks.
        
        Args:
            tasks: List of tasks to warm up (default: all supported tasks)
        """
        tasks = tasks or list(MODEL_CONFIGS.keys())
        
        for task in tasks:
            if task in MODEL_CONFIGS:
                model_name = MODEL_CONFIGS[task]["default"]
                logger.info(f"Warming up {task} model: {model_name}")
                
                if task == "embeddings":
                    self.load_embedding_model(model_name)
                else:
                    self.load_model(model_name, task)


# Global model manager instance
model_manager = ModelManager()