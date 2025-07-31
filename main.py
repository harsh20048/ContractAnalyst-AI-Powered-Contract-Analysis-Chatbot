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


# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Ensure directories exist
    settings._ensure_directories()
    
    # Optionally warm up default models
    # extraction_engine.warm_up(["summarization"])


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