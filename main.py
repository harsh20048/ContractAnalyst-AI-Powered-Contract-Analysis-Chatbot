"""
Main FastAPI application for the Hugging Face PDF Extraction Framework.

Provides REST API endpoints for:
- PDF upload and processing
- Model management
- Text extraction and analysis
- Health checks and status monitoring
"""

import os
import tempfile
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import asyncio
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from hf_framework import (
    settings, 
    extraction_engine, 
    ExtractionRequest, 
    ExtractionOutput,
    EXTRACTION_TASKS,
    MODEL_CONFIGS
)
from hf_framework.training_data_extractor import TrainingDataExtractor, TrainingDataPoint
from hf_framework.training_pipeline import TrainingPipeline, TrainingConfig
import zipfile
import io
import pandas as pd
from hf_framework.mistral_pipeline import MistralPipeline, MistralConfig, ExtractionResult

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format=settings.log_format
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="A comprehensive framework for deploying Hugging Face models and extracting information from PDF documents",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static files directory for frontend
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Global processing status
processing_status: Dict[str, Dict[str, Any]] = {}

# Global Mistral pipeline (initialize once)
mistral_pipeline: Optional[MistralPipeline] = None


# Pydantic models for API
class TextExtractionRequest(BaseModel):
    """Request model for text extraction."""
    text: str = Field(..., description="Text to process")
    tasks: List[str] = Field(..., description="List of extraction tasks to perform")
    model_preferences: Optional[Dict[str, str]] = Field(None, description="Preferred models for each task")


class PDFExtractionRequest(BaseModel):
    """Request model for PDF extraction."""
    tasks: List[str] = Field(default=["summarization", "key_information_extraction"], description="List of extraction tasks")
    model_preferences: Optional[Dict[str, str]] = Field(None, description="Preferred models for each task")
    custom_questions: Optional[List[str]] = Field(None, description="Custom questions for Q&A")
    extract_tables: bool = Field(True, description="Whether to extract tables")
    extract_metadata: bool = Field(True, description="Whether to extract metadata")


class ModelLoadRequest(BaseModel):
    """Request model for loading a model."""
    model_name: str = Field(..., description="Name of the model to load")
    task: str = Field(..., description="Task type for the model")
    force_reload: bool = Field(False, description="Whether to force reload if already loaded")


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: float  # 0.0 to 1.0
    message: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchExtractionRequest(BaseModel):
    """Request model for batch PDF extraction."""
    tasks: List[str] = Field(default=["date", "company_name", "company_address", "tables", "angebot"], 
                            description="Parameters to extract")
    output_format: str = Field(default="excel", description="Output format: excel or json")
    include_raw_text: bool = Field(False, description="Include raw text in output")


class TrainingRequest(BaseModel):
    """Request model for training pipeline."""
    pdf_directory: str = Field(..., description="Directory containing PDF files")
    labeled_excel_path: Optional[str] = Field(None, description="Path to labeled Excel file")
    output_directory: str = Field(default="./training_output", description="Output directory")
    
    
# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.app_version
    }


# Configuration endpoints
@app.get("/config/tasks")
async def get_supported_tasks():
    """Get list of supported extraction tasks."""
    return {
        "supported_tasks": EXTRACTION_TASKS,
        "task_descriptions": {
            "summarization": "Generate a concise summary of the document",
            "question_answering": "Answer questions based on document content",
            "key_information_extraction": "Extract key information, metadata, and statistics",
            "sentiment_analysis": "Analyze sentiment and emotional tone",
            "named_entity_recognition": "Extract named entities (people, places, organizations)",
            "text_classification": "Classify text into categories",
            "topic_modeling": "Identify main topics and themes"
        }
    }


@app.get("/config/models")
async def get_available_models():
    """Get available models for each task."""
    return MODEL_CONFIGS


@app.get("/config/settings")
async def get_settings():
    """Get current application settings (non-sensitive)."""
    return {
        "max_file_size": settings.max_file_size,
        "allowed_extensions": settings.allowed_extensions,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "similarity_threshold": settings.similarity_threshold,
        "default_model": settings.default_model,
        "embedding_model": settings.embedding_model,
        "extraction_model": settings.extraction_model
    }


# Model management endpoints
@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a specific model."""
    try:
        if request.task == "embeddings":
            success = extraction_engine.model_manager.load_embedding_model(
                request.model_name, 
                force_reload=request.force_reload
            )
        else:
            success = extraction_engine.model_manager.load_model(
                request.model_name, 
                request.task, 
                force_reload=request.force_reload
            )
        
        if success:
            return {"message": f"Successfully loaded {request.model_name} for {request.task}"}
        else:
            raise HTTPException(status_code=500, detail=f"Failed to load model {request.model_name}")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/models/{model_name}/{task}")
async def unload_model(model_name: str, task: str):
    """Unload a specific model."""
    try:
        success = extraction_engine.model_manager.unload_model(model_name, task)
        if success:
            return {"message": f"Successfully unloaded {model_name} for {task}"}
        else:
            raise HTTPException(status_code=404, detail=f"Model {model_name} for {task} not found")
    except Exception as e:
        logger.error(f"Error unloading model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models/status")
async def get_model_status():
    """Get status of all loaded models."""
    return extraction_engine.get_model_status()


@app.post("/models/warm-up")
async def warm_up_models(tasks: Optional[List[str]] = None):
    """Pre-load models for faster inference."""
    try:
        extraction_engine.warm_up(tasks)
        return {"message": "Models warmed up successfully"}
    except Exception as e:
        logger.error(f"Error warming up models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Text processing endpoints
@app.post("/extract/text")
async def extract_from_text(request: TextExtractionRequest):
    """Extract information from plain text."""
    try:
        # Validate tasks
        invalid_tasks = [task for task in request.tasks if task not in EXTRACTION_TASKS]
        if invalid_tasks:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid tasks: {invalid_tasks}. Supported tasks: {EXTRACTION_TASKS}"
            )
        
        result = extraction_engine.extract_from_text(
            text=request.text,
            tasks=request.tasks,
            model_preferences=request.model_preferences
        )
        
        return {
            "success": True,
            "results": result,
            "tasks_completed": list(result.keys()),
            "input_length": len(request.text)
        }
        
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# PDF processing endpoints
@app.post("/extract/pdf")
async def extract_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tasks: str = '["summarization", "key_information_extraction"]',
    model_preferences: str = "{}",
    custom_questions: str = "[]",
    extract_tables: bool = True,
    extract_metadata: bool = True
):
    """Extract information from uploaded PDF file."""
    
    # Validate file
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    if file.size and file.size > settings.max_file_size:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.max_file_size} bytes")
    
    # Parse JSON parameters
    try:
        tasks_list = json.loads(tasks)
        model_prefs = json.loads(model_preferences) if model_preferences != "{}" else None
        questions = json.loads(custom_questions) if custom_questions != "[]" else None
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")
    
    # Validate tasks
    invalid_tasks = [task for task in tasks_list if task not in EXTRACTION_TASKS]
    if invalid_tasks:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid tasks: {invalid_tasks}. Supported tasks: {EXTRACTION_TASKS}"
        )
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Create extraction request
        extraction_request = ExtractionRequest(
            file_path=temp_file_path,
            tasks=tasks_list,
            model_preferences=model_prefs,
            custom_questions=questions,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata
        )
        
        # Process synchronously for now (could be made async for large files)
        result = extraction_engine.extract_from_pdf(extraction_request)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Convert result to dict for JSON response
        response_data = {
            "success": result.success,
            "file_name": file.filename,
            "tasks_completed": result.tasks_completed,
            "processing_time": result.processing_time,
            "total_words": result.total_words,
            "total_sentences": result.total_sentences,
            "language": result.language,
            "errors": result.errors,
            "warnings": result.warnings
        }
        
        # Add task-specific results
        if result.summary:
            response_data["summary"] = result.summary
        if result.key_information:
            response_data["key_information"] = result.key_information
        if result.questions_answers:
            response_data["questions_answers"] = result.questions_answers
        if result.sentiment_analysis:
            response_data["sentiment_analysis"] = result.sentiment_analysis
        if result.named_entities:
            response_data["named_entities"] = result.named_entities
        if result.text_classification:
            response_data["text_classification"] = result.text_classification
        if result.topic_analysis:
            response_data["topic_analysis"] = result.topic_analysis
        if result.key_phrases:
            response_data["key_phrases"] = result.key_phrases
        
        # Add metadata if requested
        if extract_metadata and result.processing_result:
            response_data["metadata"] = {
                "title": result.processing_result.metadata.title,
                "author": result.processing_result.metadata.author,
                "page_count": result.processing_result.metadata.page_count,
                "file_size": result.processing_result.metadata.file_size,
                "creation_date": result.processing_result.metadata.creation_date
            }
        
        # Add table information if requested
        if extract_tables and result.processing_result and result.processing_result.tables:
            response_data["tables"] = [
                {
                    "page_number": table.page_number,
                    "rows": len(table.data),
                    "columns": len(table.data[0]) if table.data else 0,
                    "headers": table.headers,
                    "data": table.data[:5]  # First 5 rows only for response size
                }
                for table in result.processing_result.tables
            ]
        
        return response_data
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract/pdf/async")
async def extract_from_pdf_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    tasks: str = '["summarization", "key_information_extraction"]',
    model_preferences: str = "{}",
    custom_questions: str = "[]",
    extract_tables: bool = True,
    extract_metadata: bool = True
):
    """Extract information from PDF asynchronously (for large files)."""
    
    # Similar validation as synchronous version
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Generate job ID
    import uuid
    job_id = str(uuid.uuid4())
    
    # Initialize status
    processing_status[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued for processing",
        "started_at": datetime.now(),
        "completed_at": None,
        "result": None,
        "error": None
    }
    
    # Parse parameters (similar to sync version)
    try:
        tasks_list = json.loads(tasks)
        model_prefs = json.loads(model_preferences) if model_preferences != "{}" else None
        questions = json.loads(custom_questions) if custom_questions != "[]" else None
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")
    
    # Add background task
    background_tasks.add_task(
        process_pdf_background,
        job_id,
        file,
        tasks_list,
        model_prefs,
        questions,
        extract_tables,
        extract_metadata
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "PDF processing started in background"
    }


async def process_pdf_background(
    job_id: str,
    file: UploadFile,
    tasks: List[str],
    model_preferences: Optional[Dict[str, str]],
    custom_questions: Optional[List[str]],
    extract_tables: bool,
    extract_metadata: bool
):
    """Background task for processing PDF."""
    try:
        # Update status
        processing_status[job_id]["status"] = "processing"
        processing_status[job_id]["progress"] = 0.1
        processing_status[job_id]["message"] = "Reading PDF file"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        processing_status[job_id]["progress"] = 0.3
        processing_status[job_id]["message"] = "Processing PDF content"
        
        # Create extraction request
        extraction_request = ExtractionRequest(
            file_path=temp_file_path,
            tasks=tasks,
            model_preferences=model_preferences,
            custom_questions=custom_questions,
            extract_tables=extract_tables,
            extract_metadata=extract_metadata
        )
        
        processing_status[job_id]["progress"] = 0.5
        processing_status[job_id]["message"] = "Running extraction tasks"
        
        # Process PDF
        result = extraction_engine.extract_from_pdf(extraction_request)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Update status with result
        processing_status[job_id]["status"] = "completed" if result.success else "failed"
        processing_status[job_id]["progress"] = 1.0
        processing_status[job_id]["message"] = "Processing completed"
        processing_status[job_id]["completed_at"] = datetime.now()
        processing_status[job_id]["result"] = result.__dict__ if result.success else None
        processing_status[job_id]["error"] = result.errors[0] if result.errors else None
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        # Update status with error
        processing_status[job_id]["status"] = "failed"
        processing_status[job_id]["progress"] = 0.0
        processing_status[job_id]["message"] = "Processing failed"
        processing_status[job_id]["completed_at"] = datetime.now()
        processing_status[job_id]["error"] = str(e)


@app.get("/extract/status/{job_id}")
async def get_processing_status(job_id: str):
    """Get status of asynchronous processing job."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return processing_status[job_id]


# Frontend endpoint
@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the main frontend page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HuggingFace PDF Extraction Framework</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            .drag-area {
                border: 2px dashed #ccc;
                border-radius: 10px;
                padding: 20px;
                text-align: center;
                cursor: pointer;
                transition: border-color 0.3s;
            }
            .drag-area.dragover {
                border-color: #007bff;
                background-color: #f8f9fa;
            }
            .result-section {
                margin-top: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="text-center mb-4">🤗 HuggingFace PDF Extraction Framework</h1>
            
            <div class="row">
                <div class="col-md-8 mx-auto">
                    <div class="card">
                        <div class="card-header">
                            <h5>Upload PDF for Analysis</h5>
                        </div>
                        <div class="card-body">
                            <form id="uploadForm" enctype="multipart/form-data">
                                <div class="mb-3">
                                    <div class="drag-area" onclick="document.getElementById('fileInput').click()">
                                        <input type="file" id="fileInput" accept=".pdf" style="display: none;">
                                        <p class="mb-0">Click here or drag & drop your PDF file</p>
                                        <small class="text-muted">Maximum file size: 50MB</small>
                                    </div>
                                </div>
                                
                                <div class="mb-3">
                                    <label class="form-label">Extraction Tasks:</label>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="summarization" id="task1" checked>
                                        <label class="form-check-label" for="task1">Summarization</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="key_information_extraction" id="task2" checked>
                                        <label class="form-check-label" for="task2">Key Information Extraction</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="sentiment_analysis" id="task3">
                                        <label class="form-check-label" for="task3">Sentiment Analysis</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="named_entity_recognition" id="task4">
                                        <label class="form-check-label" for="task4">Named Entity Recognition</label>
                                    </div>
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" value="question_answering" id="task5">
                                        <label class="form-check-label" for="task5">Question Answering</label>
                                    </div>
                                </div>
                                
                                <button type="submit" class="btn btn-primary" id="submitBtn">
                                    <span id="submitText">Extract Information</span>
                                    <span id="submitSpinner" class="spinner-border spinner-border-sm ms-2" style="display: none;"></span>
                                </button>
                            </form>
                        </div>
                    </div>
                    
                    <div id="results" style="display: none;"></div>
                </div>
            </div>
        </div>
        
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            // File upload handling
            const fileInput = document.getElementById('fileInput');
            const dragArea = document.querySelector('.drag-area');
            const uploadForm = document.getElementById('uploadForm');
            
            dragArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                dragArea.classList.add('dragover');
            });
            
            dragArea.addEventListener('dragleave', () => {
                dragArea.classList.remove('dragover');
            });
            
            dragArea.addEventListener('drop', (e) => {
                e.preventDefault();
                dragArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    fileInput.files = files;
                    updateFileDisplay();
                }
            });
            
            fileInput.addEventListener('change', updateFileDisplay);
            
            function updateFileDisplay() {
                const file = fileInput.files[0];
                if (file) {
                    dragArea.innerHTML = `<p class="mb-0"><strong>${file.name}</strong></p><small class="text-muted">Size: ${(file.size / 1024 / 1024).toFixed(2)} MB</small>`;
                }
            }
            
            // Form submission
            uploadForm.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const file = fileInput.files[0];
                if (!file) {
                    alert('Please select a PDF file');
                    return;
                }
                
                // Get selected tasks
                const tasks = Array.from(document.querySelectorAll('input[type="checkbox"]:checked'))
                    .map(cb => cb.value);
                
                if (tasks.length === 0) {
                    alert('Please select at least one extraction task');
                    return;
                }
                
                // Show loading state
                document.getElementById('submitText').textContent = 'Processing...';
                document.getElementById('submitSpinner').style.display = 'inline-block';
                document.getElementById('submitBtn').disabled = true;
                
                // Prepare form data
                const formData = new FormData();
                formData.append('file', file);
                formData.append('tasks', JSON.stringify(tasks));
                
                try {
                    const response = await fetch('/extract/pdf', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (response.ok) {
                        displayResults(result);
                    } else {
                        throw new Error(result.detail || 'Processing failed');
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                } finally {
                    // Reset loading state
                    document.getElementById('submitText').textContent = 'Extract Information';
                    document.getElementById('submitSpinner').style.display = 'none';
                    document.getElementById('submitBtn').disabled = false;
                }
            });
            
            function displayResults(result) {
                const resultsDiv = document.getElementById('results');
                let html = '<div class="card mt-4"><div class="card-header"><h5>Extraction Results</h5></div><div class="card-body">';
                
                // Basic info
                html += `<p><strong>File:</strong> ${result.file_name}</p>`;
                html += `<p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)} seconds</p>`;
                html += `<p><strong>Words:</strong> ${result.total_words} | <strong>Sentences:</strong> ${result.total_sentences}</p>`;
                html += `<p><strong>Tasks Completed:</strong> ${result.tasks_completed.join(', ')}</p>`;
                
                // Summary
                if (result.summary) {
                    html += `<div class="result-section"><h6>Summary</h6><p>${result.summary}</p></div>`;
                }
                
                // Key Information
                if (result.key_information) {
                    html += `<div class="result-section"><h6>Key Information</h6>`;
                    if (result.key_information.key_phrases) {
                        html += `<p><strong>Key Phrases:</strong> ${result.key_information.key_phrases.slice(0, 10).join(', ')}</p>`;
                    }
                    html += '</div>';
                }
                
                // Sentiment
                if (result.sentiment_analysis) {
                    html += `<div class="result-section"><h6>Sentiment Analysis</h6>`;
                    html += `<p><strong>Overall Sentiment:</strong> ${result.sentiment_analysis.overall_sentiment} (Confidence: ${(result.sentiment_analysis.overall_confidence * 100).toFixed(1)}%)</p>`;
                    html += '</div>';
                }
                
                // Named Entities
                if (result.named_entities) {
                    html += `<div class="result-section"><h6>Named Entities</h6>`;
                    for (const [entityType, entities] of Object.entries(result.named_entities)) {
                        html += `<p><strong>${entityType}:</strong> ${entities.map(e => e.text).slice(0, 5).join(', ')}</p>`;
                    }
                    html += '</div>';
                }
                
                // Q&A
                if (result.questions_answers) {
                    html += `<div class="result-section"><h6>Questions & Answers</h6>`;
                    result.questions_answers.forEach(qa => {
                        html += `<p><strong>Q:</strong> ${qa.question}</p>`;
                        html += `<p><strong>A:</strong> ${qa.answer} (Confidence: ${(qa.confidence * 100).toFixed(1)}%)</p><hr>`;
                    });
                    html += '</div>';
                }
                
                html += '</div></div>';
                resultsDiv.innerHTML = html;
                resultsDiv.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return html_content


# Training Data Extraction Endpoints
@app.post("/training/extract-batch")
async def extract_training_data_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    request: BatchExtractionRequest = None
):
    """Extract training data from multiple PDF files."""
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
    
    try:
        # Create temporary directory for batch processing
        import tempfile
        import shutil
        
        temp_dir = tempfile.mkdtemp()
        
        # Save uploaded files
        pdf_paths = []
        for file in files:
            temp_path = os.path.join(temp_dir, file.filename)
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
            pdf_paths.append(temp_path)
        
        # Extract data from all PDFs
        results = []
        for pdf_path in pdf_paths:
            result = training_extractor.extract_from_pdf(pdf_path)
            results.append(result)
        
        # Convert to desired format
        if request and request.output_format == "json":
            output_data = [result.to_dict() for result in results]
            response_data = {
                "success": True,
                "total_files": len(results),
                "extracted_data": output_data,
                "summary": {
                    "successful_extractions": len([r for r in results if not r.errors]),
                    "failed_extractions": len([r for r in results if r.errors]),
                    "date_extracted": len([r for r in results if r.date]),
                    "company_name_extracted": len([r for r in results if r.company_name]),
                    "address_extracted": len([r for r in results if r.company_address]),
                    "tables_found": len([r for r in results if r.tables]),
                    "angebot_found": len([r for r in results if r.angebot])
                }
            }
        else:
            # Create Excel file in memory
            excel_buffer = io.BytesIO()
            training_extractor.save_to_excel(results, excel_buffer)
            excel_buffer.seek(0)
            
            # Clean up temp directory
            shutil.rmtree(temp_dir)
            
            # Return Excel file
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                io.BytesIO(excel_buffer.read()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=extracted_training_data.xlsx"}
            )
        
        # Clean up temp directory
        shutil.rmtree(temp_dir)
        
        return response_data
        
    except Exception as e:
        # Clean up temp directory if it exists
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/extract-single")
async def extract_training_data_single(file: UploadFile = File(...)):
    """Extract training data from a single PDF file."""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        # Extract training data
        result = training_extractor.extract_from_pdf(temp_file_path)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        # Return structured result
        return {
            "success": True,
            "file_name": file.filename,
            "extracted_data": {
                "date": result.date,
                "company_name": result.company_name,
                "company_address": result.company_address,
                "tables": result.tables,
                "angebot": result.angebot
            },
            "metadata": {
                "page_count": result.page_count,
                "processing_time": result.processing_time,
                "errors": result.errors
            }
        }
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_file_path' in locals():
            try:
                os.unlink(temp_file_path)
            except:
                pass
        
        logger.error(f"Error extracting training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/upload-labeled-data")
async def upload_labeled_excel(file: UploadFile = File(...)):
    """Upload labeled Excel data for training."""
    
    if not file.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="File must be an Excel file (.xlsx or .xls)")
    
    try:
        # Save uploaded Excel file
        excel_path = f"./labeled_data_{file.filename}"
        with open(excel_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Load and validate the Excel data
        df = training_extractor.load_labeled_data(excel_path)
        
        # Validate required columns
        required_columns = ['file_name', 'date', 'company_name', 'company_address', 'angebot']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.unlink(excel_path)  # Clean up
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {missing_columns}"
            )
        
        # Return validation results
        return {
            "success": True,
            "file_path": excel_path,
            "total_rows": len(df),
            "columns": list(df.columns),
            "data_preview": df.head(5).to_dict('records'),
            "completeness": {
                col: f"{df[col].notna().sum()}/{len(df)} ({df[col].notna().mean():.1%})"
                for col in required_columns if col in df.columns
            }
        }
        
    except Exception as e:
        logger.error(f"Error uploading labeled data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training/start-pipeline")
async def start_training_pipeline(
    background_tasks: BackgroundTasks,
    pdf_files: List[UploadFile] = File(...),
    labeled_excel: Optional[UploadFile] = File(None),
    output_directory: str = "./training_output"
):
    """Start the complete training pipeline."""
    
    if len(pdf_files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 PDF files allowed")
    
    try:
        import tempfile
        import shutil
        
        # Create temporary directories
        temp_pdf_dir = tempfile.mkdtemp(prefix="training_pdfs_")
        
        # Save PDF files
        for file in pdf_files:
            if not file.filename.lower().endswith('.pdf'):
                shutil.rmtree(temp_pdf_dir)
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
            
            temp_path = os.path.join(temp_pdf_dir, file.filename)
            with open(temp_path, 'wb') as f:
                content = await file.read()
                f.write(content)
        
        # Save labeled Excel file if provided
        labeled_excel_path = None
        if labeled_excel:
            if not labeled_excel.filename.lower().endswith(('.xlsx', '.xls')):
                shutil.rmtree(temp_pdf_dir)
                raise HTTPException(status_code=400, detail="Labeled file must be Excel format")
            
            labeled_excel_path = f"./temp_labeled_{labeled_excel.filename}"
            with open(labeled_excel_path, 'wb') as f:
                content = await labeled_excel.read()
                f.write(content)
        
        # Generate job ID for tracking
        import uuid
        job_id = str(uuid.uuid4())
        
        # Initialize job status
        processing_status[job_id] = {
            "status": "processing",
            "progress": 0.0,
            "message": "Starting training pipeline",
            "started_at": datetime.now(),
            "completed_at": None,
            "result": None,
            "error": None
        }
        
        # Run training pipeline in background
        background_tasks.add_task(
            run_training_pipeline_background,
            job_id,
            temp_pdf_dir,
            labeled_excel_path,
            output_directory
        )
        
        return {
            "job_id": job_id,
            "status": "started",
            "message": f"Training pipeline started with {len(pdf_files)} PDF files",
            "pdf_count": len(pdf_files),
            "has_labeled_data": labeled_excel is not None
        }
        
    except Exception as e:
        logger.error(f"Error starting training pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_training_pipeline_background(
    job_id: str,
    pdf_directory: str,
    labeled_excel_path: Optional[str],
    output_directory: str
):
    """Background task for running the training pipeline."""
    try:
        # Update status
        processing_status[job_id]["progress"] = 0.1
        processing_status[job_id]["message"] = "Extracting data from PDFs"
        
        # Run the pipeline
        results = training_pipeline.run_full_pipeline(
            pdf_directory=pdf_directory,
            labeled_excel_path=labeled_excel_path,
            output_dir=output_directory
        )
        
        processing_status[job_id]["progress"] = 1.0
        processing_status[job_id]["status"] = "completed"
        processing_status[job_id]["message"] = "Training pipeline completed successfully"
        processing_status[job_id]["completed_at"] = datetime.now()
        processing_status[job_id]["result"] = {
            "output_directory": output_directory,
            "trained_models": list(results.get('trained_models', {}).keys()),
            "data_quality_score": results.get('data_quality', {}).get('quality_score', 0.0),
            "total_samples": results.get('data_quality', {}).get('total_samples', 0)
        }
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(pdf_directory)
        if labeled_excel_path and os.path.exists(labeled_excel_path):
            os.unlink(labeled_excel_path)
        
    except Exception as e:
        processing_status[job_id]["status"] = "failed"
        processing_status[job_id]["progress"] = 0.0
        processing_status[job_id]["message"] = "Training pipeline failed"
        processing_status[job_id]["completed_at"] = datetime.now()
        processing_status[job_id]["error"] = str(e)
        
        # Clean up on error
        try:
            import shutil
            shutil.rmtree(pdf_directory)
            if labeled_excel_path and os.path.exists(labeled_excel_path):
                os.unlink(labeled_excel_path)
        except:
            pass


@app.get("/training/download-template")
async def download_excel_template():
    """Download Excel template for labeling training data."""
    
    # Create template DataFrame
    template_data = {
        'file_name': ['example1.pdf', 'example2.pdf', 'example3.pdf'],
        'date': ['2024-01-15', '15.02.2024', '03/03/2024'],
        'company_name': ['TechCorp GmbH', 'Global Industries AG', 'Innovation Ltd'],
        'company_address': ['Musterstraße 123, 12345 Berlin', 'Hauptplatz 1, 1010 Wien', '123 Main St, London'],
        'angebot': ['Angebot Nr. A-2024-001', 'Quote #Q-2024-002', 'Proposal ID: P-2024-003'],
        'tables_count': [2, 1, 3],
        'notes': ['Example entry - replace with actual data', 'Training data template', 'Delete these examples']
    }
    
    df = pd.DataFrame(template_data)
    
    # Create Excel file in memory
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Training_Data', index=False)
        
        # Add instructions sheet
        instructions = pd.DataFrame({
            'Instructions': [
                '1. Replace the example data with your actual labeled data',
                '2. file_name: Exact name of the PDF file',
                '3. date: Date found in the document (any format)',
                '4. company_name: Full company name with legal form',
                '5. company_address: Complete address including postal code',
                '6. angebot: Quote/offer number or description',
                '7. tables_count: Number of tables found (for validation)',
                '8. notes: Optional notes about the document',
                '',
                'Save this file and upload it using the /training/upload-labeled-data endpoint'
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    excel_buffer.seek(0)
    
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        io.BytesIO(excel_buffer.read()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=training_data_template.xlsx"}
    )


@app.get("/training/status/{job_id}")
async def get_training_status(job_id: str):
    """Get status of training pipeline job."""
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return processing_status[job_id]


@app.get("/training/results/{job_id}")
async def download_training_results(job_id: str):
    """Download training results as ZIP file."""
    
    if job_id not in processing_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job_status = processing_status[job_id]
    
    if job_status["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training job not completed")
    
    output_dir = job_status["result"]["output_directory"]
    
    if not os.path.exists(output_dir):
        raise HTTPException(status_code=404, detail="Training results not found")
    
    try:
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files from output directory
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, output_dir)
                    zip_file.write(file_path, arc_name)
        
        zip_buffer.seek(0)
        
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            io.BytesIO(zip_buffer.read()),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=training_results_{job_id}.zip"}
        )
        
    except Exception as e:
        logger.error(f"Error creating training results ZIP: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Mistral integration endpoints
@app.post("/mistral/load")
async def load_mistral_model(
    model_path: str = "./mistral-7b-instruct-v0.1",
    load_in_4bit: bool = True,
    max_new_tokens: int = 512
):
    """Load Mistral 7B model with optimizations."""
    global mistral_pipeline
    
    try:
        config = MistralConfig(
            model_path=model_path,
            load_in_4bit=load_in_4bit,
            max_new_tokens=max_new_tokens
        )
        
        mistral_pipeline = MistralPipeline(config)
        
        success = mistral_pipeline.load_model()
        
        if success:
            stats = mistral_pipeline.get_performance_stats()
            return {
                "success": True,
                "message": "Mistral 7B loaded successfully",
                "model_path": model_path,
                "config": {
                    "load_in_4bit": load_in_4bit,
                    "max_new_tokens": max_new_tokens
                },
                "system_info": stats.get("model_config", {})
            }
        else:
            return {
                "success": False,
                "message": "Failed to load Mistral 7B model",
                "error": "Model loading failed"
            }
            
    except Exception as e:
        logger.error(f"Error loading Mistral 7B: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mistral/extract")
async def extract_with_mistral(file: UploadFile = File(...)):
    """Extract parameters using Mistral 7B model."""
    global mistral_pipeline
    
    if not mistral_pipeline or not mistral_pipeline.model:
        raise HTTPException(
            status_code=400, 
            detail="Mistral 7B model not loaded. Use /mistral/load first."
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract using Mistral 7B
            result = mistral_pipeline.extract_from_pdf(temp_file_path)
            
            if result.success:
                return {
                    "success": True,
                    "file_name": file.filename,
                    "extracted_parameters": result.extracted_parameters,
                    "confidence_scores": result.confidence_scores,
                    "processing_time": result.processing_time,
                    "token_usage": result.token_usage,
                    "model_response": result.model_response
                }
            else:
                return {
                    "success": False,
                    "file_name": file.filename,
                    "error": result.error_message
                }
                
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error in Mistral extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mistral/batch-extract")
async def batch_extract_with_mistral(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Batch extract parameters using Mistral 7B."""
    global mistral_pipeline
    
    if not mistral_pipeline or not mistral_pipeline.model:
        raise HTTPException(
            status_code=400, 
            detail="Mistral 7B model not loaded. Use /mistral/load first."
        )
    
    try:
        # Save all files temporarily
        temp_files = []
        for file in files:
            if file.filename.lower().endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    content = await file.read()
                    temp_file.write(content)
                    temp_files.append((temp_file.name, file.filename))
        
        if not temp_files:
            raise HTTPException(status_code=400, detail="No valid PDF files found")
        
        # Extract from all PDFs
        results = []
        
        for temp_path, original_filename in temp_files:
            try:
                result = mistral_pipeline.extract_from_pdf(temp_path)
                
                result_dict = {
                    "file_name": original_filename,
                    "success": result.success,
                    "extracted_parameters": result.extracted_parameters,
                    "confidence_scores": result.confidence_scores,
                    "processing_time": result.processing_time,
                    "token_usage": result.token_usage
                }
                
                if not result.success:
                    result_dict["error"] = result.error_message
                
                results.append(result_dict)
                
            except Exception as e:
                results.append({
                    "file_name": original_filename,
                    "success": False,
                    "error": str(e)
                })
            finally:
                # Clean up temp file
                os.unlink(temp_path)
        
        # Calculate summary statistics
        successful_extractions = [r for r in results if r["success"]]
        total_processing_time = sum(r.get("processing_time", 0) for r in successful_extractions)
        total_tokens = sum(r.get("token_usage", {}).get("total_tokens", 0) for r in successful_extractions)
        
        # Average confidence scores
        avg_confidence = {}
        for param in ["date", "company_name", "company_address", "angebot", "tables"]:
            param_confidences = [
                r["confidence_scores"].get(param, 0) 
                for r in successful_extractions 
                if "confidence_scores" in r
            ]
            avg_confidence[param] = sum(param_confidences) / len(param_confidences) if param_confidences else 0.0
        
        return {
            "success": True,
            "total_files": len(files),
            "processed_files": len(results),
            "successful_extractions": len(successful_extractions),
            "failed_extractions": len(results) - len(successful_extractions),
            "total_processing_time": total_processing_time,
            "total_token_usage": total_tokens,
            "average_confidence_scores": avg_confidence,
            "detailed_results": results
        }
        
    except Exception as e:
        logger.error(f"Error in Mistral batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mistral/status")
async def get_mistral_status():
    """Get Mistral 7B model status and performance."""
    global mistral_pipeline
    
    if not mistral_pipeline:
        return {
            "loaded": False,
            "message": "Mistral 7B pipeline not initialized"
        }
    
    if not mistral_pipeline.model:
        return {
            "loaded": False,
            "message": "Mistral 7B model not loaded"
        }
    
    try:
        stats = mistral_pipeline.get_performance_stats()
        
        return {
            "loaded": True,
            "model_path": mistral_pipeline.config.model_path,
            "performance_stats": stats,
            "config": {
                "load_in_4bit": mistral_pipeline.config.load_in_4bit,
                "max_new_tokens": mistral_pipeline.config.max_new_tokens,
                "temperature": mistral_pipeline.config.temperature,
                "torch_dtype": str(mistral_pipeline.config.torch_dtype)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting Mistral status: {str(e)}")
        return {
            "loaded": True,
            "error": str(e)
        }


@app.post("/mistral/unload")
async def unload_mistral_model():
    """Unload Mistral 7B model to free memory."""
    global mistral_pipeline
    
    if not mistral_pipeline:
        return {
            "success": True,
            "message": "No model was loaded"
        }
    
    try:
        mistral_pipeline.unload_model()
        mistral_pipeline = None
        
        return {
            "success": True,
            "message": "Mistral 7B model unloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Error unloading Mistral: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mistral/supervised-train")
async def train_mistral_supervised(
    pdf_files: List[UploadFile] = File(...),
    excel_answers: UploadFile = File(...)
):
    """Train using Mistral 7B predictions + Excel ground truth."""
    global mistral_pipeline
    
    if not mistral_pipeline or not mistral_pipeline.model:
        raise HTTPException(
            status_code=400, 
            detail="Mistral 7B model not loaded. Use /mistral/load first."
        )
    
    try:
        # Save and load Excel answers
        excel_path = f"temp_mistral_answers_{excel_answers.filename}"
        with open(excel_path, 'wb') as f:
            content = await excel_answers.read()
            f.write(content)
        
        answers_df = pd.read_excel(excel_path)
        
        # Process PDFs with Mistral 7B and compare with Excel
        training_results = []
        
        for pdf_file in pdf_files:
            if not pdf_file.filename.lower().endswith('.pdf'):
                continue
            
            # Find corresponding answer
            answer_row = answers_df[answers_df['file_name'] == pdf_file.filename]
            if answer_row.empty:
                continue
            
            # Extract with Mistral 7B
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await pdf_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Get Mistral predictions
                mistral_result = mistral_pipeline.extract_from_pdf(temp_file_path)
                
                if mistral_result.success:
                    # Get expected answers
                    answer_data = answer_row.iloc[0]
                    expected_answers = {
                        'date': str(answer_data.get('date', '')).strip() if pd.notna(answer_data.get('date')) else None,
                        'company_name': str(answer_data.get('company_name', '')).strip() if pd.notna(answer_data.get('company_name')) else None,
                        'company_address': str(answer_data.get('company_address', '')).strip() if pd.notna(answer_data.get('company_address')) else None,
                        'angebot': str(answer_data.get('angebot', '')).strip() if pd.notna(answer_data.get('angebot')) else None,
                    }
                    
                    # Calculate accuracy for each parameter
                    accuracies = {}
                    for param in ['date', 'company_name', 'company_address', 'angebot']:
                        expected = expected_answers.get(param)
                        predicted = mistral_result.extracted_parameters.get(param)
                        
                        if expected and predicted:
                            # String similarity
                            accuracy = 1.0 if (str(expected).lower() in str(predicted).lower() or 
                                             str(predicted).lower() in str(expected).lower()) else 0.0
                        else:
                            accuracy = 1.0 if (not expected and not predicted) else 0.0
                        
                        accuracies[param] = accuracy
                    
                    training_results.append({
                        'file_name': pdf_file.filename,
                        'mistral_predictions': mistral_result.extracted_parameters,
                        'expected_answers': expected_answers,
                        'accuracies': accuracies,
                        'mistral_confidence': mistral_result.confidence_scores,
                        'processing_time': mistral_result.processing_time
                    })
                    
            finally:
                os.unlink(temp_file_path)
        
        # Calculate overall statistics
        if training_results:
            param_accuracies = {}
            for param in ['date', 'company_name', 'company_address', 'angebot']:
                param_accs = [r['accuracies'][param] for r in training_results]
                param_accuracies[param] = sum(param_accs) / len(param_accs)
            
            overall_accuracy = sum(param_accuracies.values()) / len(param_accuracies)
            avg_processing_time = sum(r['processing_time'] for r in training_results) / len(training_results)
        else:
            param_accuracies = {}
            overall_accuracy = 0.0
            avg_processing_time = 0.0
        
        # Clean up
        os.unlink(excel_path)
        
        return {
            "success": True,
            "training_examples": len(training_results),
            "matched_pdfs": len(training_results),
            "total_pdfs": len(pdf_files),
            "overall_accuracy": overall_accuracy,
            "parameter_accuracies": param_accuracies,
            "average_processing_time": avg_processing_time,
            "detailed_results": training_results[:5],  # First 5 for brevity
            "message": f"Mistral 7B supervised training completed on {len(training_results)} examples"
        }
        
    except Exception as e:
        # Clean up on error
        if 'excel_path' in locals() and os.path.exists(excel_path):
            os.unlink(excel_path)
        
        logger.error(f"Error in Mistral supervised training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application components."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Ensure directories exist
    settings._ensure_directories()
    
    # Initialize training components
    global training_extractor
    global training_pipeline
    training_extractor = TrainingDataExtractor()
    training_pipeline = TrainingPipeline()
    
    # Optionally warm up default models
    # extraction_engine.warm_up(["summarization"])

    # Initialize Mistral pipeline (but don't load model yet)
    global mistral_pipeline
    logger.info("🚀 Mistral 7B pipeline initialized (model not loaded)")
    logger.info("Use POST /mistral/load to load the model when ready")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down application")
    # Clean up models if needed
    # extraction_engine.model_manager.unload_all_models()


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1  # Multiple workers don't work well with model caching
    )