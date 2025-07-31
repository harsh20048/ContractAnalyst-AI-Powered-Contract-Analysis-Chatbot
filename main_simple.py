"""
Simplified version of the HuggingFace PDF Extraction Framework for testing.
This version works without the heavy ML dependencies.
"""

import os
import logging
import tempfile
import zipfile
import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import PyPDF2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="HuggingFace PDF Extraction Framework (Simple)",
    description="Simplified version for testing PDF parameter extraction",
    version="1.0.0-simple"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple processing status storage
processing_status: Dict[str, Any] = {}

# Pydantic models
class ExtractionRequest(BaseModel):
    text: str = Field(..., description="Text to process")
    tasks: List[str] = Field(default=["summarization"], description="Tasks to perform")

class BatchExtractionRequest(BaseModel):
    tasks: List[str] = Field(default=["date", "company_name", "company_address", "tables", "angebot"], 
                            description="Parameters to extract")
    output_format: str = Field(default="excel", description="Output format: excel or json")
    include_raw_text: bool = Field(False, description="Include raw text in output")

# Simple PDF text extraction
def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
    return text

# Simple parameter extraction
def extract_parameters(text: str) -> Dict[str, Any]:
    """Extract the 5 parameters using simple pattern matching."""
    import re
    
    result = {
        "date": None,
        "company_name": None,
        "company_address": None,
        "tables": None,
        "angebot": None
    }
    
    # Date patterns
    date_patterns = [
        r'\b\d{1,2}[./]\d{1,2}[./]\d{4}\b',
        r'\b\d{4}[./]\d{1,2}[./]\d{1,2}\b',
        r'\b\d{1,2}\.\s*\w+\s*\d{4}\b',
    ]
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result["date"] = matches[0]
            break
    
    # Company patterns
    company_patterns = [
        r'\b[\w\s]+\s+(GmbH|AG|KG|OHG|UG|e\.V\.)\b',
        r'\b[\w\s]+\s+(Inc|LLC|Ltd|Corp|Corporation|Company)\b',
    ]
    
    for pattern in company_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result["company_name"] = matches[0]
            break
    
    # Address patterns (simple)
    address_patterns = [
        r'\b\d{5}\s+[A-Za-zäöüÄÖÜß\s]+\b',
        r'\b[A-Za-zäöüÄÖÜß\s]+str\.\s*\d+[a-z]?\b',
    ]
    
    for pattern in address_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            result["company_address"] = matches[0]
            break
    
    # Angebot patterns
    angebot_patterns = [
        r'(Angebot|Quote|Quotation)\s*[Nr\.#:]*\s*([A-Za-z0-9\-_]+)',
        r'([A-Z]-\d{4}-\d{3})',
    ]
    
    for pattern in angebot_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                result["angebot"] = matches[0][1] if len(matches[0]) > 1 else matches[0][0]
            else:
                result["angebot"] = matches[0]
            break
    
    # Simple table detection
    if "|" in text or "Position" in text or "Item" in text:
        result["tables"] = [{"table_id": 1, "rows": 1, "detected": True}]
    
    return result

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "HuggingFace PDF Extraction Framework (Simple) is running",
        "version": "1.0.0-simple",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config/tasks")
async def get_supported_tasks():
    """Get supported extraction tasks."""
    return {
        "supported_tasks": ["date", "company_name", "company_address", "tables", "angebot"],
        "description": "Simplified extraction for testing"
    }

@app.get("/config/models")
async def get_model_configurations():
    """Get available model configurations."""
    return {
        "available_models": {
            "simple_extractor": {
                "name": "Simple Pattern Extractor",
                "type": "rule_based",
                "supported_tasks": ["date", "company_name", "company_address", "tables", "angebot"]
            }
        }
    }

@app.get("/config/settings")
async def get_framework_settings():
    """Get framework settings."""
    return {
        "framework": "HuggingFace PDF Extraction (Simple)",
        "version": "1.0.0-simple",
        "max_file_size": "50MB",
        "supported_formats": ["pdf"],
        "batch_limit": 50
    }

@app.post("/extract/text")
async def extract_from_text(request: ExtractionRequest):
    """Extract information from text."""
    try:
        # Simple text processing
        parameters = extract_parameters(request.text)
        
        return {
            "success": True,
            "input_length": len(request.text),
            "tasks_completed": request.tasks,
            "results": {
                "parameter_extraction": parameters,
                "text_stats": {
                    "length": len(request.text),
                    "words": len(request.text.split()),
                    "lines": len(request.text.split('\n'))
                }
            }
        }
    except Exception as e:
        logger.error(f"Error in text extraction: {str(e)}")
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
        
        # Extract text from PDF
        text = extract_pdf_text(temp_file_path)
        
        # Extract parameters
        parameters = extract_parameters(text)
        
        # Clean up temp file
        os.unlink(temp_file_path)
        
        return {
            "success": True,
            "file_name": file.filename,
            "extracted_data": parameters,
            "metadata": {
                "text_length": len(text),
                "processing_time": 0.1,  # Simulated
                "errors": [] if text else ["No text extracted"]
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

@app.post("/training/extract-batch")
async def extract_training_data_batch(
    files: List[UploadFile] = File(...),
    output_format: str = "excel"
):
    """Extract training data from multiple PDF files."""
    
    if len(files) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    # Validate files
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
    
    try:
        results = []
        
        for file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Extract text and parameters
                text = extract_pdf_text(temp_file_path)
                parameters = extract_parameters(text)
                
                # Create result record
                result = {
                    "file_name": file.filename,
                    "date": parameters.get("date"),
                    "company_name": parameters.get("company_name"),
                    "company_address": parameters.get("company_address"),
                    "tables_count": len(parameters.get("tables", [])),
                    "tables_data": json.dumps(parameters.get("tables")) if parameters.get("tables") else "",
                    "angebot": parameters.get("angebot"),
                    "page_count": 1,  # Simplified
                    "processing_time": 0.1,
                    "errors": "" if text else "No text extracted"
                }
                
                results.append(result)
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
        
        # Return based on format
        if output_format == "json":
            return {
                "success": True,
                "total_files": len(results),
                "extracted_data": results,
                "summary": {
                    "successful_extractions": len([r for r in results if not r["errors"]]),
                    "failed_extractions": len([r for r in results if r["errors"]]),
                    "date_extracted": len([r for r in results if r["date"]]),
                    "company_name_extracted": len([r for r in results if r["company_name"]]),
                    "address_extracted": len([r for r in results if r["company_address"]]),
                    "tables_found": len([r for r in results if r["tables_count"] > 0]),
                    "angebot_found": len([r for r in results if r["angebot"]])
                }
            }
        else:
            # Create Excel file
            df = pd.DataFrame(results)
            excel_buffer = io.BytesIO()
            
            with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Extracted_Data', index=False)
                
                # Summary sheet
                summary_data = {
                    'Total Files': len(results),
                    'Successful Extractions': len([r for r in results if not r["errors"]]),
                    'Failed Extractions': len([r for r in results if r["errors"]]),
                    'Date Extracted': len([r for r in results if r["date"]]),
                    'Company Name Extracted': len([r for r in results if r["company_name"]]),
                    'Address Extracted': len([r for r in results if r["company_address"]]),
                    'Tables Found': len([r for r in results if r["tables_count"] > 0]),
                    'Angebot Found': len([r for r in results if r["angebot"]])
                }
                
                summary_df = pd.DataFrame(list(summary_data.items()), columns=['Metric', 'Count'])
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            excel_buffer.seek(0)
            
            return StreamingResponse(
                io.BytesIO(excel_buffer.read()),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={"Content-Disposition": "attachment; filename=extracted_training_data.xlsx"}
            )
        
    except Exception as e:
        logger.error(f"Error in batch extraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

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
                'Save this file and upload it for training'
            ]
        })
        instructions.to_excel(writer, sheet_name='Instructions', index=False)
    
    excel_buffer.seek(0)
    
    return StreamingResponse(
        io.BytesIO(excel_buffer.read()),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=training_data_template.xlsx"}
    )

@app.get("/models/status")
async def get_model_status():
    """Get model status."""
    return {
        "model_info": {
            "simple_extractor": {
                "name": "Simple Pattern Extractor",
                "task": "parameter_extraction",
                "is_loaded": True,
                "memory_usage": "minimal"
            }
        },
        "memory_usage": {
            "device": "cpu",
            "total_models": 1
        }
    }

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Simple web interface."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HuggingFace PDF Extraction Framework (Simple)</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 20px; text-align: center; margin: 20px 0; }
            button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .result { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤗 HuggingFace PDF Extraction Framework (Simple)</h1>
            <p>Simplified version for testing PDF parameter extraction</p>
            
            <h2>📄 Single PDF Test</h2>
            <div class="upload-area">
                <input type="file" id="singleFile" accept=".pdf">
                <br><br>
                <button onclick="testSinglePDF()">Extract Parameters</button>
            </div>
            
            <h2>📦 Batch PDF Test</h2>
            <div class="upload-area">
                <input type="file" id="batchFiles" accept=".pdf" multiple>
                <br><br>
                <button onclick="testBatchPDF()">Extract Batch</button>
                <button onclick="downloadTemplate()">Download Template</button>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            async function testSinglePDF() {
                const fileInput = document.getElementById('singleFile');
                if (!fileInput.files[0]) {
                    alert('Please select a PDF file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    const response = await fetch('/training/extract-single', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    document.getElementById('results').innerHTML = '<div class="result"><h3>Single PDF Results</h3><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('results').innerHTML = '<div class="result"><h3>Error</h3><p>' + error.message + '</p></div>';
                }
            }
            
            async function testBatchPDF() {
                const fileInput = document.getElementById('batchFiles');
                if (!fileInput.files.length) {
                    alert('Please select PDF files');
                    return;
                }
                
                const formData = new FormData();
                for (let file of fileInput.files) {
                    formData.append('files', file);
                }
                formData.append('output_format', 'json');
                
                try {
                    const response = await fetch('/training/extract-batch', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    document.getElementById('results').innerHTML = '<div class="result"><h3>Batch PDF Results</h3><pre>' + JSON.stringify(result, null, 2) + '</pre></div>';
                } catch (error) {
                    document.getElementById('results').innerHTML = '<div class="result"><h3>Error</h3><p>' + error.message + '</p></div>';
                }
            }
            
            async function downloadTemplate() {
                window.open('/training/download-template', '_blank');
            }
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)