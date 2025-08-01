"""
Mistral 7B Pipeline for PDF Parameter Extraction

This module provides optimized integration for Mistral 7B Instruct v0.1
with GPU support, memory optimization, and specialized prompting.
"""

import torch
import logging
import json
import gc
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import time
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
import psutil
import GPUtil

from .pdf_processor import PDFProcessor
from .config import settings

logger = logging.getLogger(__name__)


@dataclass
class MistralConfig:
    """Configuration for Mistral 7B model."""
    model_path: str = "./mistral-7b-instruct-v0.1"
    device_map: str = "auto"
    torch_dtype: torch.dtype = torch.float16
    trust_remote_code: bool = True
    use_flash_attention: bool = True
    max_memory: Optional[Dict[str, str]] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = True  # Enable 4-bit quantization for memory efficiency
    bnb_4bit_compute_dtype: torch.dtype = torch.float16
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = True
    top_p: float = 0.9
    repetition_penalty: float = 1.1


@dataclass 
class ExtractionResult:
    """Result from Mistral 7B extraction."""
    extracted_parameters: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    token_usage: Dict[str, int]
    model_response: str
    success: bool
    error_message: Optional[str] = None


class MistralPipeline:
    """
    Optimized pipeline for Mistral 7B Instruct v0.1 integration.
    """
    
    def __init__(self, config: MistralConfig = None):
        """Initialize Mistral pipeline with memory optimization."""
        self.config = config or MistralConfig()
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.pdf_processor = PDFProcessor()
        
        # Memory and performance tracking
        self.memory_usage = {}
        self.inference_times = []
        
        # Check system resources
        self._check_system_requirements()
        
    def _check_system_requirements(self):
        """Check if system can handle Mistral 7B."""
        
        # Check RAM
        ram_gb = psutil.virtual_memory().total / (1024**3)
        logger.info(f"Available RAM: {ram_gb:.1f} GB")
        
        if ram_gb < 16:
            logger.warning("Mistral 7B requires at least 16GB RAM. Consider using quantization.")
        
        # Check GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                for gpu in gpus:
                    logger.info(f"GPU {gpu.id}: {gpu.name}, Memory: {gpu.memoryTotal}MB")
                    if gpu.memoryTotal < 8000:  # 8GB
                        logger.warning(f"GPU {gpu.id} has limited memory. Consider 4-bit quantization.")
            else:
                logger.info("No GPU detected. Will use CPU (slower).")
        except:
            logger.info("Could not detect GPU information.")
        
        # Check disk space for model
        model_path = Path(self.config.model_path)
        if model_path.exists():
            model_size = sum(f.stat().st_size for f in model_path.rglob('*') if f.is_file())
            logger.info(f"Model size: {model_size / (1024**3):.1f} GB")
        
    def load_model(self) -> bool:
        """Load Mistral 7B with optimizations."""
        
        try:
            logger.info("🚀 Loading Mistral 7B Instruct v0.1...")
            start_time = time.time()
            
            # Setup quantization config for memory efficiency
            if self.config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.config.bnb_4bit_compute_dtype,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                )
            elif self.config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            else:
                quantization_config = None
            
            # Load tokenizer
            logger.info("   📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_path,
                trust_remote_code=self.config.trust_remote_code
            )
            
            # Ensure pad token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model with optimizations
            logger.info("   🧠 Loading model with optimizations...")
            
            model_kwargs = {
                "torch_dtype": self.config.torch_dtype,
                "trust_remote_code": self.config.trust_remote_code,
                "device_map": self.config.device_map,
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            if self.config.max_memory:
                model_kwargs["max_memory"] = self.config.max_memory
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            
            # Create pipeline for easier inference
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=self.config.torch_dtype,
                device_map=self.config.device_map,
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Mistral 7B loaded successfully in {load_time:.1f}s")
            
            # Log memory usage
            self._log_memory_usage("after_model_load")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load Mistral 7B: {str(e)}")
            return False
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage."""
        
        # RAM usage
        ram_usage = psutil.virtual_memory()
        self.memory_usage[f"{stage}_ram_used_gb"] = ram_usage.used / (1024**3)
        self.memory_usage[f"{stage}_ram_percent"] = ram_usage.percent
        
        # GPU memory if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.memory_allocated(i) / (1024**3)
                self.memory_usage[f"{stage}_gpu_{i}_used_gb"] = gpu_memory
        
        logger.info(f"Memory usage at {stage}: "
                   f"RAM {self.memory_usage[f'{stage}_ram_used_gb']:.1f}GB "
                   f"({self.memory_usage[f'{stage}_ram_percent']:.1f}%)")
    
    def create_extraction_prompt(self, text: str, parameters: List[str] = None) -> str:
        """Create optimized prompt for Mistral 7B parameter extraction."""
        
        if parameters is None:
            parameters = ["date", "company_name", "company_address", "angebot", "tables"]
        
        prompt = f"""<s>[INST] You are an expert document analyzer. Extract the following information from this PDF text with high accuracy.

**TASK**: Extract these parameters from the document:
- date: Document date (format: DD.MM.YYYY or similar)
- company_name: Company name (including GmbH, AG, Inc, Ltd, etc.)
- company_address: Full company address (street, city, postal code)
- angebot: Quote/proposal number or ID
- tables: Number of tables present in the document

**DOCUMENT TEXT**:
{text[:4000]}  {# Limit text to fit context window #}

**INSTRUCTIONS**:
1. Extract ONLY the requested information
2. If information is not found, return null
3. Be precise and avoid hallucination
4. Return response in valid JSON format

**REQUIRED JSON OUTPUT FORMAT**:
{{
    "date": "extracted_date_or_null",
    "company_name": "extracted_company_or_null", 
    "company_address": "extracted_address_or_null",
    "angebot": "extracted_quote_id_or_null",
    "tables": number_of_tables_or_0,
    "confidence": {{
        "date": confidence_score_0_to_1,
        "company_name": confidence_score_0_to_1,
        "company_address": confidence_score_0_to_1,
        "angebot": confidence_score_0_to_1,
        "tables": confidence_score_0_to_1
    }}
}}

Extract the information now: [/INST]"""
        
        return prompt
    
    def extract_parameters(self, text: str, parameters: List[str] = None) -> ExtractionResult:
        """Extract parameters using Mistral 7B."""
        
        if not self.pipeline:
            return ExtractionResult(
                extracted_parameters={},
                confidence_scores={},
                processing_time=0.0,
                token_usage={},
                model_response="",
                success=False,
                error_message="Model not loaded"
            )
        
        try:
            start_time = time.time()
            
            # Create prompt
            prompt = self.create_extraction_prompt(text, parameters)
            
            # Log memory before inference
            self._log_memory_usage("before_inference")
            
            # Generate response
            logger.info("🧠 Running Mistral 7B inference...")
            
            outputs = self.pipeline(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.do_sample,
                top_p=self.config.top_p,
                repetition_penalty=self.config.repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                return_full_text=False  # Only return new tokens
            )
            
            processing_time = time.time() - start_time
            self.inference_times.append(processing_time)
            
            # Extract response text
            response_text = outputs[0]['generated_text'].strip()
            
            # Parse JSON response
            extracted_params, confidence_scores = self._parse_model_response(response_text)
            
            # Calculate token usage
            input_tokens = len(self.tokenizer.encode(prompt))
            output_tokens = len(self.tokenizer.encode(response_text))
            
            # Log memory after inference
            self._log_memory_usage("after_inference")
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            logger.info(f"✅ Extraction completed in {processing_time:.2f}s")
            
            return ExtractionResult(
                extracted_parameters=extracted_params,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                token_usage={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                model_response=response_text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ Extraction failed: {str(e)}")
            return ExtractionResult(
                extracted_parameters={},
                confidence_scores={},
                processing_time=0.0,
                token_usage={},
                model_response="",
                success=False,
                error_message=str(e)
            )
    
    def _parse_model_response(self, response: str) -> tuple[Dict[str, Any], Dict[str, float]]:
        """Parse JSON response from Mistral 7B."""
        
        try:
            # Try to find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Extract parameters
                extracted_params = {
                    "date": parsed.get("date"),
                    "company_name": parsed.get("company_name"),
                    "company_address": parsed.get("company_address"),
                    "angebot": parsed.get("angebot"),
                    "tables": parsed.get("tables", 0)
                }
                
                # Extract confidence scores
                confidence_scores = parsed.get("confidence", {})
                
                # Ensure all parameters have confidence scores
                for param in extracted_params.keys():
                    if param not in confidence_scores:
                        confidence_scores[param] = 0.5  # Default confidence
                
                return extracted_params, confidence_scores
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {str(e)}")
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
        
        # Fallback: empty results
        return {
            "date": None,
            "company_name": None, 
            "company_address": None,
            "angebot": None,
            "tables": 0
        }, {
            "date": 0.0,
            "company_name": 0.0,
            "company_address": 0.0,
            "angebot": 0.0,
            "tables": 0.0
        }
    
    def extract_from_pdf(self, pdf_path: str) -> ExtractionResult:
        """Complete pipeline: PDF -> Text -> Mistral 7B -> Parameters."""
        
        try:
            # Extract text from PDF
            logger.info(f"📄 Processing PDF: {Path(pdf_path).name}")
            processing_result = self.pdf_processor.process_pdf(pdf_path)
            
            if not processing_result.success:
                return ExtractionResult(
                    extracted_parameters={},
                    confidence_scores={},
                    processing_time=0.0,
                    token_usage={},
                    model_response="",
                    success=False,
                    error_message=f"PDF processing failed: {processing_result.error_message}"
                )
            
            # Combine text chunks
            full_text = " ".join([chunk.content for chunk in processing_result.text_chunks])
            
            # Extract parameters with Mistral 7B
            extraction_result = self.extract_parameters(full_text)
            
            # Add table count from PDF processor
            if processing_result.tables:
                extraction_result.extracted_parameters["tables"] = len(processing_result.tables)
                extraction_result.confidence_scores["tables"] = 0.9  # High confidence for direct count
            
            return extraction_result
            
        except Exception as e:
            logger.error(f"❌ PDF extraction pipeline failed: {str(e)}")
            return ExtractionResult(
                extracted_parameters={},
                confidence_scores={},
                processing_time=0.0,
                token_usage={},
                model_response="",
                success=False,
                error_message=str(e)
            )
    
    def batch_extract(self, pdf_paths: List[str]) -> List[ExtractionResult]:
        """Batch extraction with memory management."""
        
        results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"📦 Processing batch {i+1}/{len(pdf_paths)}: {Path(pdf_path).name}")
            
            result = self.extract_from_pdf(pdf_path)
            results.append(result)
            
            # Memory cleanup every 5 documents
            if (i + 1) % 5 == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                logger.info(f"🧹 Memory cleanup after {i+1} documents")
        
        return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        
        if not self.inference_times:
            return {"message": "No inference performed yet"}
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        
        return {
            "total_inferences": len(self.inference_times),
            "average_inference_time": avg_time,
            "fastest_inference": min(self.inference_times),
            "slowest_inference": max(self.inference_times),
            "memory_usage": self.memory_usage,
            "model_config": {
                "load_in_4bit": self.config.load_in_4bit,
                "torch_dtype": str(self.config.torch_dtype),
                "device_map": self.config.device_map
            }
        }
    
    def unload_model(self):
        """Unload model to free memory."""
        
        logger.info("🧹 Unloading Mistral 7B model...")
        
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ Model unloaded and memory freed")
    
    def __del__(self):
        """Cleanup on destruction."""
        self.unload_model()


def main():
    """Example usage of Mistral pipeline."""
    
    # Configure for your system
    config = MistralConfig(
        model_path="./mistral-7b-instruct-v0.1",
        load_in_4bit=True,  # Enable for lower memory usage
        max_new_tokens=512
    )
    
    # Initialize pipeline
    pipeline = MistralPipeline(config)
    
    # Load model
    if pipeline.load_model():
        
        # Test extraction
        pdf_path = "test_document.pdf"
        result = pipeline.extract_from_pdf(pdf_path)
        
        if result.success:
            print("✅ Extraction successful!")
            print(f"Parameters: {result.extracted_parameters}")
            print(f"Confidence: {result.confidence_scores}")
            print(f"Processing time: {result.processing_time:.2f}s")
        else:
            print(f"❌ Extraction failed: {result.error_message}")
        
        # Performance stats
        stats = pipeline.get_performance_stats()
        print(f"Performance: {stats}")
        
        # Unload model
        pipeline.unload_model()


if __name__ == "__main__":
    main()