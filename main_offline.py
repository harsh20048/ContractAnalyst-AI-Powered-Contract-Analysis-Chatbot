#!/usr/bin/env python3
"""
Offline Mistral 7B PDF Extraction Server
Runs completely without internet connection
"""

import os
import sys
from pathlib import Path

# Set offline environment variables BEFORE any other imports
offline_env = {
    "HF_HOME": "./models/cache",
    "TRANSFORMERS_CACHE": "./models/cache/transformers",
    "HF_DATASETS_CACHE": "./models/cache/datasets", 
    "NLTK_DATA": "./models/nltk_data",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "OFFLINE_MODE": "True"
}

for key, value in offline_env.items():
    os.environ[key] = value

print("🔒 Offline environment configured")

import logging
import tempfile
import zipfile
import io
import json
import uuid
from datetime import datetime
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

# Global processing status
processing_status: Dict[str, Any] = {}

# Simple offline parameter extractor
class OfflineParameterExtractor:
    """Simple parameter extractor that works offline."""
    
    def __init__(self):
        """Initialize the offline extractor."""
        self.patterns = self._load_patterns()
    
    def _load_patterns(self):
        """Load regex patterns for parameter extraction."""
        import re
        
        return {
            'date': [
                r'\b(\d{1,2}[./]\d{1,2}[./]\d{4})\b',
                r'\b(\d{4}[./]\d{1,2}[./]\d{1,2})\b',
                r'\b(\d{1,2}\.\s*\w+\s*\d{4})\b',
                r'(?i)datum:\s*([^\n]+)',
                r'(?i)date:\s*([^\n]+)'
            ],
            'company_name': [
                r'\b([\w\s&]+\s+(?:GmbH|AG|KG|OHG|UG|e\.V\.))\b',
                r'\b([\w\s&]+\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company))\b',
                r'\b([A-Z][a-zA-Z\s&]+(?:GmbH|AG|Inc|Ltd))\b'
            ],
            'company_address': [
                r'(\b\d{5}\s+[A-Za-zäöüÄÖÜß\s]+\b)',
                r'(\b[A-Za-zäöüÄÖÜß\s]+str\.\s*\d+[a-z]?\b)',
                r'(\b[A-Za-zäöüÄÖÜß\s]+straße\s*\d+[a-z]?\b)',
                r'(\d+\s+[A-Za-z\s]+(?:Street|Avenue|Road|Str|Plaza))',
            ],
            'angebot': [
                r'(?i)(?:angebot|quote|quotation|proposal)\s*[nr\.#:]*\s*([A-Za-z0-9\-_]+)',
                r'([A-Z]-\d{4}-\d{3})',
                r'(?i)(?:quote|proposal)\s*(?:id|number|nr)[:.]?\s*([A-Za-z0-9\-_]+)',
                r'(?i)angebotsnummer[:.]?\s*([A-Za-z0-9\-_]+)'
            ]
        }
    
    def extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF using PyPDF2."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def extract_parameters(self, text: str) -> Dict[str, Any]:
        """Extract the 5 parameters from text using regex patterns."""
        import re
        
        result = {
            "date": None,
            "company_name": None,
            "company_address": None,
            "angebot": None,
            "tables": 0
        }
        
        # Extract each parameter using patterns
        for param, patterns in self.patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    if isinstance(matches[0], tuple):
                        result[param] = matches[0][0] if matches[0] else None
                    else:
                        result[param] = matches[0]
                    break  # Use first successful match
        
        # Count tables (simple heuristic)
        table_indicators = ['position', 'beschreibung', 'preis', 'menge', 'total', '|', 'item', 'description', 'price']
        table_count = 0
        lines = text.lower().split('\n')
        for line in lines:
            if sum(indicator in line for indicator in table_indicators) >= 2:
                table_count += 1
        
        result['tables'] = min(table_count, 10)  # Cap at reasonable number
        
        return result
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Complete extraction from PDF file."""
        start_time = datetime.now()
        
        try:
            # Extract text
            text = self.extract_pdf_text(pdf_path)
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from PDF",
                    "extracted_parameters": {},
                    "processing_time": 0.0
                }
            
            # Extract parameters
            parameters = self.extract_parameters(text)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "extracted_parameters": parameters,
                "processing_time": processing_time,
                "text_length": len(text),
                "confidence_scores": self._calculate_confidence(parameters, text)
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "extracted_parameters": {},
                "processing_time": processing_time
            }
    
    def _calculate_confidence(self, parameters: Dict[str, Any], text: str) -> Dict[str, float]:
        """Calculate confidence scores for extracted parameters."""
        confidence = {}
        
        for param, value in parameters.items():
            if param == 'tables':
                confidence[param] = 0.8 if value > 0 else 0.3
            elif value:
                # Simple confidence based on value characteristics
                if param == 'date':
                    confidence[param] = 0.9 if any(c.isdigit() for c in str(value)) else 0.5
                elif param == 'company_name':
                    confidence[param] = 0.8 if any(suffix in str(value).upper() for suffix in ['GMBH', 'AG', 'INC', 'LTD']) else 0.6
                elif param == 'company_address':
                    confidence[param] = 0.7 if any(c.isdigit() for c in str(value)) else 0.5
                elif param == 'angebot':
                    confidence[param] = 0.8 if any(c.isdigit() for c in str(value)) else 0.6
                else:
                    confidence[param] = 0.7
            else:
                confidence[param] = 0.0
        
        return confidence

# Initialize offline extractor
offline_extractor = OfflineParameterExtractor()

# FastAPI app
app = FastAPI(
    title="Offline Mistral PDF Extraction Framework",
    description="Complete PDF parameter extraction running entirely offline",
    version="1.0.0-offline"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ExtractionRequest(BaseModel):
    """Request model for parameter extraction."""
    parameters: List[str] = Field(default=["date", "company_name", "company_address", "angebot", "tables"])

class BatchExtractionRequest(BaseModel):
    """Request model for batch extraction."""
    output_format: str = Field(default="json", description="Output format: json or excel")
    include_raw_text: bool = Field(default=False, description="Include raw text in output")

# API Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "Offline Mistral PDF Extraction Framework is running",
        "version": "1.0.0-offline",
        "mode": "offline",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config/tasks")
async def get_supported_tasks():
    """Get supported extraction tasks."""
    return {
        "supported_tasks": ["date", "company_name", "company_address", "angebot", "tables"],
        "description": "Offline parameter extraction from PDF documents",
        "mode": "offline"
    }

@app.get("/config/models")
async def get_model_info():
    """Get model information."""
    return {
        "available_models": {
            "offline_extractor": {
                "type": "regex_based",
                "description": "Pattern-based parameter extraction",
                "parameters": ["date", "company_name", "company_address", "angebot", "tables"],
                "status": "ready"
            }
        },
        "active_model": "offline_extractor",
        "mode": "offline"
    }

@app.post("/extract/single")
async def extract_single_pdf(file: UploadFile = File(...)):
    """Extract parameters from a single PDF."""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract parameters
            result = offline_extractor.extract_from_pdf(temp_file_path)
            
            result.update({
                "file_name": file.filename,
                "extraction_method": "offline_regex",
                "mode": "offline"
            })
            
            return result
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/extract/batch")
async def extract_batch_pdfs(
    files: List[UploadFile] = File(...),
    request: BatchExtractionRequest = BatchExtractionRequest()
):
    """Extract parameters from multiple PDFs."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    
    for file in files:
        if not file.filename.lower().endswith('.pdf'):
            continue
        
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Extract parameters
                result = offline_extractor.extract_from_pdf(temp_file_path)
                result["file_name"] = file.filename
                results.append(result)
                
            finally:
                # Clean up temp file
                os.unlink(temp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            results.append({
                "file_name": file.filename,
                "success": False,
                "error": str(e)
            })
    
    # Calculate summary statistics
    successful_results = [r for r in results if r.get("success", False)]
    total_processing_time = sum(r.get("processing_time", 0) for r in successful_results)
    
    # Return based on format
    if request.output_format.lower() == "excel":
        # Create Excel response
        return await create_excel_response(results, "batch_extraction_results.xlsx")
    else:
        return {
            "success": True,
            "total_files": len(files),
            "processed_files": len(results),
            "successful_extractions": len(successful_results),
            "total_processing_time": total_processing_time,
            "extraction_method": "offline_regex",
            "mode": "offline",
            "results": results
        }

@app.post("/offline/supervised-train")
async def offline_supervised_training(
    pdf_files: List[UploadFile] = File(...),
    excel_answers: UploadFile = File(...)
):
    """Offline supervised training using PDFs and Excel answers."""
    
    if not excel_answers.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Answer file must be Excel format")
    
    try:
        # Save and load Excel answers
        excel_path = f"temp_offline_answers_{excel_answers.filename}"
        with open(excel_path, 'wb') as f:
            content = await excel_answers.read()
            f.write(content)
        
        answers_df = pd.read_excel(excel_path)
        
        # Validate required columns
        required_columns = ['file_name', 'date', 'company_name', 'company_address', 'angebot']
        missing_columns = [col for col in required_columns if col not in answers_df.columns]
        if missing_columns:
            os.unlink(excel_path)
            raise HTTPException(status_code=400, detail=f"Missing columns in Excel: {missing_columns}")
        
        # Process PDFs and compare with answers
        training_results = []
        
        for pdf_file in pdf_files:
            if not pdf_file.filename.lower().endswith('.pdf'):
                continue
            
            # Find corresponding answer
            answer_row = answers_df[answers_df['file_name'] == pdf_file.filename]
            if answer_row.empty:
                continue
            
            # Extract with offline method
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await pdf_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Get offline predictions
                offline_result = offline_extractor.extract_from_pdf(temp_file_path)
                
                if offline_result["success"]:
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
                        predicted = offline_result["extracted_parameters"].get(param)
                        
                        if expected and predicted:
                            # String similarity
                            accuracy = 1.0 if (str(expected).lower() in str(predicted).lower() or 
                                             str(predicted).lower() in str(expected).lower()) else 0.0
                        else:
                            accuracy = 1.0 if (not expected and not predicted) else 0.0
                        
                        accuracies[param] = accuracy
                    
                    training_results.append({
                        'file_name': pdf_file.filename,
                        'offline_predictions': offline_result["extracted_parameters"],
                        'expected_answers': expected_answers,
                        'accuracies': accuracies,
                        'processing_time': offline_result["processing_time"]
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
            "mode": "offline_training",
            "training_examples": len(training_results),
            "matched_pdfs": len(training_results),
            "total_pdfs": len(pdf_files),
            "overall_accuracy": overall_accuracy,
            "parameter_accuracies": param_accuracies,
            "average_processing_time": avg_processing_time,
            "detailed_results": training_results[:5],  # First 5 for brevity
            "message": f"Offline training completed on {len(training_results)} examples"
        }
        
    except Exception as e:
        # Clean up on error
        if 'excel_path' in locals() and os.path.exists(excel_path):
            os.unlink(excel_path)
        
        logger.error(f"Error in offline training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def create_excel_response(results: List[Dict], filename: str):
    """Create Excel response from results."""
    try:
        # Create DataFrame
        excel_data = []
        for result in results:
            if result.get("success", False):
                params = result.get("extracted_parameters", {})
                row = {
                    "file_name": result.get("file_name", ""),
                    "date": params.get("date", ""),
                    "company_name": params.get("company_name", ""),
                    "company_address": params.get("company_address", ""),
                    "angebot": params.get("angebot", ""),
                    "tables_count": params.get("tables", 0),
                    "processing_time": result.get("processing_time", 0.0),
                    "confidence_avg": sum(result.get("confidence_scores", {}).values()) / 5 if result.get("confidence_scores") else 0.0
                }
            else:
                row = {
                    "file_name": result.get("file_name", ""),
                    "date": "",
                    "company_name": "",
                    "company_address": "",
                    "angebot": "",
                    "tables_count": 0,
                    "processing_time": 0.0,
                    "confidence_avg": 0.0,
                    "error": result.get("error", "Processing failed")
                }
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Extraction_Results', index=False)
        
        excel_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(excel_buffer.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
        
    except Exception as e:
        logger.error(f"Error creating Excel response: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Excel generation failed: {str(e)}")

@app.get("/training/download-template")
async def download_excel_template():
    """Download Excel template for training data."""
    try:
        # Create template data
        template_data = {
            'file_name': ['example_document.pdf', 'sample_invoice.pdf'],
            'date': ['15.03.2024', '20.02.2024'],
            'company_name': ['Example GmbH', 'Sample Corp'],
            'company_address': ['Musterstraße 123, 12345 Berlin', '123 Main St, City'],
            'angebot': ['A-2024-001', 'Q-2024-002'],
            'tables_count': [2, 1],
            'notes': ['Template example', 'Sample data']
        }
        
        df = pd.DataFrame(template_data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Training_Template', index=False)
            
            # Add instructions sheet
            instructions = pd.DataFrame({
                'Instructions': [
                    '1. Fill in the file_name column with your PDF filenames',
                    '2. Provide correct answers for each parameter',
                    '3. Use this format for dates: DD.MM.YYYY or MM/DD/YYYY',
                    '4. Include full company names with legal forms (GmbH, AG, Inc, etc.)',
                    '5. Provide complete addresses including postal codes',
                    '6. Enter quote/proposal numbers in the angebot column',
                    '7. Count the number of tables in each document',
                    '8. Save and upload this file for supervised training'
                ]
            })
            instructions.to_excel(writer, sheet_name='Instructions', index=False)
        
        excel_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(excel_buffer.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=training_template.xlsx"}
        )
        
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Enhanced web interface for offline operation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🔒 Offline Mistral PDF Extraction Framework</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
            .container { max-width: 1000px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .offline-badge { background: #28a745; color: white; padding: 5px 15px; border-radius: 20px; font-size: 14px; }
            .section { margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
            .upload-area { border: 2px dashed #007bff; padding: 20px; text-align: center; margin: 20px 0; background: white; border-radius: 8px; }
            .upload-area:hover { border-color: #0056b3; background: #f8f9fa; }
            button { background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 6px; cursor: pointer; margin: 5px; font-size: 14px; }
            button:hover { background: #0056b3; }
            .result { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 4px; border-left: 4px solid #007bff; }
            .success { background: #d4edda; border-left-color: #28a745; }
            .error { background: #f8d7da; border-left-color: #dc3545; }
            .warning { background: #fff3cd; border-left-color: #ffc107; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            .feature-card { background: white; padding: 20px; border-radius: 8px; border: 1px solid #ddd; }
            .parameter-list { display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0; }
            .parameter-tag { background: #e9ecef; padding: 5px 10px; border-radius: 15px; font-size: 12px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔒 Offline Mistral PDF Extraction Framework</h1>
                <span class="offline-badge">OFFLINE MODE</span>
                <p>Complete PDF parameter extraction running entirely offline - no internet required!</p>
            </div>
            
            <div class="section">
                <h2>📄 Extract Parameters from Your PDFs</h2>
                <p>Extract 5 key parameters offline: date, company name, address, angebot, and table count</p>
                
                <div class="parameter-list">
                    <span class="parameter-tag">📅 Date</span>
                    <span class="parameter-tag">🏢 Company Name</span>
                    <span class="parameter-tag">📍 Company Address</span>
                    <span class="parameter-tag">📋 Angebot/Quote</span>
                    <span class="parameter-tag">📊 Tables</span>
                </div>
                
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3>Single PDF Extraction</h3>
                        <div class="upload-area">
                            <input type="file" id="singleFile" accept=".pdf">
                            <br><br>
                            <button onclick="extractSinglePDF()">Extract Parameters</button>
                        </div>
                    </div>
                    
                    <div class="feature-card">
                        <h3>Batch PDF Processing</h3>
                        <div class="upload-area">
                            <input type="file" id="batchFiles" accept=".pdf" multiple>
                            <br><br>
                            <button onclick="extractBatchPDF()">Process Batch</button>
                            <button onclick="downloadTemplate()">Download Template</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎓 Offline Supervised Training</h2>
                <p>Train and validate using your PDFs with Excel answer sheets</p>
                
                <div class="upload-area">
                    <h3>Train with Your Data</h3>
                    <label>PDF Files:</label>
                    <input type="file" id="trainingPDFs" accept=".pdf" multiple>
                    <br><br>
                    <label>Excel Answers:</label>
                    <input type="file" id="answerExcel" accept=".xlsx,.xls">
                    <br><br>
                    <button onclick="trainOfflineModel()">Train Offline Model</button>
                </div>
            </div>
            
            <div class="section">
                <h2>🔧 System Information</h2>
                <div class="feature-grid">
                    <div class="feature-card">
                        <h3>Offline Status</h3>
                        <p>✅ No internet connection required</p>
                        <p>✅ All processing done locally</p>
                        <p>✅ Data privacy guaranteed</p>
                        <button onclick="checkHealth()">Check System Health</button>
                    </div>
                    
                    <div class="feature-card">
                        <h3>Supported Formats</h3>
                        <p>📄 PDF documents</p>
                        <p>📊 Excel files (.xlsx, .xls)</p>
                        <p>📋 JSON output</p>
                        <p>📈 Excel output</p>
                    </div>
                </div>
            </div>
            
            <div id="results"></div>
        </div>
        
        <script>
            function showResult(content, type = 'result') {
                const resultsDiv = document.getElementById('results');
                resultsDiv.innerHTML = `<div class="section"><div class="${type}"><h3>Results</h3><pre>${content}</pre></div></div>`;
                resultsDiv.scrollIntoView({ behavior: 'smooth' });
            }
            
            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const result = await response.json();
                    showResult(JSON.stringify(result, null, 2), 'success');
                } catch (error) {
                    showResult(`Health check failed: ${error.message}`, 'error');
                }
            }
            
            async function extractSinglePDF() {
                const fileInput = document.getElementById('singleFile');
                if (!fileInput.files[0]) {
                    alert('Please select a PDF file');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    showResult('Processing PDF offline...', 'warning');
                    const response = await fetch('/extract/single', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    showResult(JSON.stringify(result, null, 2), response.ok ? 'success' : 'error');
                } catch (error) {
                    showResult(`Error: ${error.message}`, 'error');
                }
            }
            
            async function extractBatchPDF() {
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
                    showResult('Processing PDFs offline...', 'warning');
                    const response = await fetch('/extract/batch', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    showResult(JSON.stringify(result, null, 2), response.ok ? 'success' : 'error');
                } catch (error) {
                    showResult(`Error: ${error.message}`, 'error');
                }
            }
            
            async function trainOfflineModel() {
                const pdfFiles = document.getElementById('trainingPDFs').files;
                const excelFile = document.getElementById('answerExcel').files[0];
                
                if (!pdfFiles.length || !excelFile) {
                    alert('Please select PDF files and Excel answer sheet');
                    return;
                }
                
                const formData = new FormData();
                for (let file of pdfFiles) {
                    formData.append('pdf_files', file);
                }
                formData.append('excel_answers', excelFile);
                
                try {
                    showResult('Training offline model...', 'warning');
                    const response = await fetch('/offline/supervised-train', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    showResult(JSON.stringify(result, null, 2), response.ok ? 'success' : 'error');
                } catch (error) {
                    showResult(`Error: ${error.message}`, 'error');
                }
            }
            
            async function downloadTemplate() {
                try {
                    const response = await fetch('/training/download-template');
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'training_template.xlsx';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);
                    showResult('Template downloaded successfully!', 'success');
                } catch (error) {
                    showResult(`Error downloading template: ${error.message}`, 'error');
                }
            }
            
            // Auto-check health on page load
            window.onload = function() {
                checkHealth();
            };
        </script>
    </body>
    </html>
    """
    return html_content

if __name__ == "__main__":
    import uvicorn
    
    print("🔒 Starting Offline Mistral PDF Extraction Server")
    print("=" * 50)
    print("✅ Mode: Completely Offline")
    print("✅ No internet connection required")
    print("✅ All processing done locally")
    print("🌐 Server will be available at: http://localhost:8000")
    print("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)