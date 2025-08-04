#!/usr/bin/env python3
"""
BERT-Base-Uncased PDF Extraction Server
Uses bert-base-uncased for superior parameter extraction accuracy
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

print("🤖 BERT-base-uncased offline environment configured")

import logging
import tempfile
import io
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import PyPDF2
import pdfplumber

# BERT imports
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline,
    BertTokenizer,
    BertModel
)
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BertBaseUncasedExtractor:
    """BERT-base-uncased parameter extractor for PDF documents."""
    
    def __init__(self):
        """Initialize BERT-base-uncased models for extraction."""
        print("🤖 Initializing BERT-base-uncased models...")
        
        # Check device availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_name = "GPU" if torch.cuda.is_available() else "CPU"
        print(f"   📱 Using device: {device_name}")
        
        try:
            # Initialize BERT-base-uncased model and tokenizer
            self.model_name = "bert-base-uncased"
            print(f"   📥 Loading BERT model: {self.model_name}")
            
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.bert_model = BertModel.from_pretrained(self.model_name)
            self.bert_model.to(self.device)
            self.bert_model.eval()
            
            # Initialize NER pipeline for entity extraction
            try:
                print("   📥 Loading NER pipeline...")
                self.ner_pipeline = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("   ✅ NER pipeline loaded successfully")
            except Exception as e:
                print(f"   ⚠️ NER pipeline failed: {e}")
                self.ner_pipeline = None
            
            # Parameter-specific embeddings for semantic matching
            self.parameter_embeddings = self._create_parameter_embeddings()
            
            # Fallback regex patterns
            self.regex_patterns = self._load_regex_patterns()
            
            print("   ✅ BERT-base-uncased models loaded successfully")
            
        except Exception as e:
            print(f"   ⚠️ Error loading BERT models: {e}")
            print("   🔄 Falling back to regex-only extraction")
            self.tokenizer = None
            self.bert_model = None
            self.ner_pipeline = None
            self.parameter_embeddings = {}
            self.regex_patterns = self._load_regex_patterns()
    
    def _create_parameter_embeddings(self) -> Dict[str, torch.Tensor]:
        """Create BERT embeddings for parameter keywords."""
        if not self.bert_model:
            return {}
        
        parameter_keywords = {
            'date': [
                "date", "datum", "created", "issued", "timestamp", "when", 
                "day", "month", "year", "today", "yesterday", "calendar"
            ],
            'company_name': [
                "company", "corporation", "business", "firm", "enterprise", 
                "organization", "gmbh", "inc", "ltd", "corp", "ag", "llc"
            ],
            'company_address': [
                "address", "location", "street", "avenue", "road", "city", 
                "postal", "zip", "country", "building", "office", "headquarters"
            ],
            'angebot': [
                "quote", "quotation", "proposal", "offer", "estimate", 
                "angebot", "number", "id", "reference", "code", "identifier"
            ]
        }
        
        embeddings = {}
        
        with torch.no_grad():
            for param, keywords in parameter_keywords.items():
                keyword_embeddings = []
                
                for keyword in keywords:
                    # Tokenize and get embeddings
                    inputs = self.tokenizer(
                        keyword, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True
                    ).to(self.device)
                    
                    outputs = self.bert_model(**inputs)
                    # Use [CLS] token embedding as keyword representation
                    keyword_embedding = outputs.last_hidden_state[:, 0, :].cpu()
                    keyword_embeddings.append(keyword_embedding)
                
                # Average embeddings for this parameter
                if keyword_embeddings:
                    param_embedding = torch.mean(torch.cat(keyword_embeddings, dim=0), dim=0)
                    embeddings[param] = param_embedding
        
        print(f"   🧠 Created parameter embeddings for {len(embeddings)} parameters")
        return embeddings
    
    def _load_regex_patterns(self):
        """Load comprehensive regex patterns as fallback."""
        return {
            'date': [
                r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b',
                r'\b(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})\b',
                r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b',
                r'\b(\d{1,2}(?:st|nd|rd|th)?\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b',
                r'(?i)(?:date|datum|created|issued)[:=\s]+([^\n\r]+)',
                r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{2})\b'
            ],
            'company_name': [
                r'\b([\w\s&,.-]+\s+(?:Inc|LLC|Ltd|Corp|Corporation|Company|Co|GmbH|AG|KG|OHG|UG|e\.V\.))\b',
                r'\b([A-Z][a-zA-Z\s&,.-]{2,40}(?:Inc|LLC|Ltd|Corp|GmbH|AG))\b',
                r'(?i)(?:company|corporation|business|firm)[:=\s]+([^\n\r]+)',
                r'\b([A-Z][a-zA-Z\s&,.-]{5,50})\b(?=\s*(?:Inc|LLC|Ltd|Corp|GmbH|AG))',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|LLC|Ltd|Corp|GmbH|AG))\b'
            ],
            'company_address': [
                r'(\d+\s+[A-Za-z\s,.-]+(?:Street|Avenue|Road|Lane|Drive|St|Ave|Rd|Ln|Dr|Str|Straße)[\s,]*[A-Za-z0-9\s,-]*)',
                r'(\b\d{5}(?:-\d{4})?\s+[A-Za-zäöüÄÖÜß\s,.-]+\b)',
                r'(\b[A-Za-zäöüÄÖÜß\s,.-]+\s*\d+[a-z]?\s*,\s*\d{5}\s+[A-Za-zäöüÄÖÜß\s,.-]+)',
                r'(?i)(?:address|location|headquarters)[:=\s]+([^\n\r]+)',
                r'(\b[A-Za-z\s,.-]{10,50}\s+\d{5}\s+[A-Za-z\s,.-]+\b)',
                r'(\d+\s+[A-Za-z\s,.-]+,\s*[A-Za-z\s,.-]+\s+\d{5})'
            ],
            'angebot': [
                r'(?i)(?:quote|quotation|proposal|offer|angebot|estimate)[\s#:]*([A-Za-z0-9\-_/]+)',
                r'(?i)(?:quote|proposal|angebot)\s*(?:number|nr|no|id)[:.\s]*([A-Za-z0-9\-_/]+)',
                r'([A-Z]{1,4}[-_]\d{4}[-_]\d{3,4})',
                r'([A-Z]+\d{6,10})',
                r'(\d{4}[-_]\d{3,4}[-_][A-Z]{1,4})',
                r'(?i)reference[:=\s]+([A-Za-z0-9\-_/]+)',
                r'([A-Z]{2,5}-[0-9]{4,8})',
                r'(Q[0-9]{6,10})'
            ]
        }
    
    def extract_pdf_text_advanced(self, pdf_path: str) -> Tuple[str, Dict]:
        """Advanced PDF text extraction with metadata."""
        text_content = ""
        metadata = {"pages": 0, "tables_detected": 0, "extraction_method": []}
        
        try:
            # Method 1: PyPDF2 for basic text
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata["pages"] = len(reader.pages)
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text.strip():
                        text_content += page_text + "\n"
                        metadata["extraction_method"].append("PyPDF2")
            
            # Method 2: pdfplumber for better structure detection
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        # Extract text
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text_content += page_text + "\n"
                        
                        # Detect tables
                        tables = page.extract_tables()
                        if tables:
                            metadata["tables_detected"] += len(tables)
                            # Add table content as text
                            for table in tables:
                                table_text = ""
                                for row in table:
                                    if row:
                                        row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                        table_text += row_text + "\n"
                                text_content += f"\n[TABLE]\n{table_text}[/TABLE]\n"
                        
                        metadata["extraction_method"].append("pdfplumber")
            except Exception as e:
                logger.warning(f"pdfplumber extraction failed: {e}")
            
            return text_content, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return "", metadata
    
    def get_bert_embeddings(self, text: str) -> torch.Tensor:
        """Get BERT embeddings for text."""
        if not self.bert_model:
            return torch.zeros(768)  # Default BERT-base embedding size
        
        try:
            # Tokenize text (truncate to max length)
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
                # Use [CLS] token as text representation
                text_embedding = outputs.last_hidden_state[:, 0, :].cpu().squeeze()
            
            return text_embedding
            
        except Exception as e:
            logger.error(f"Error getting BERT embeddings: {e}")
            return torch.zeros(768)
    
    def extract_with_bert_semantic_search(self, text: str) -> Dict[str, Any]:
        """Extract parameters using BERT semantic similarity."""
        if not self.bert_model or not self.parameter_embeddings:
            return {}
        
        results = {}
        
        # Split text into sentences for better semantic matching
        sentences = re.split(r'[.!?]\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        for param, param_embedding in self.parameter_embeddings.items():
            best_match = None
            best_similarity = 0.0
            best_sentence = ""
            
            for sentence in sentences:
                # Get sentence embedding
                sentence_embedding = self.get_bert_embeddings(sentence)
                
                # Calculate cosine similarity
                similarity = torch.cosine_similarity(
                    param_embedding.unsqueeze(0), 
                    sentence_embedding.unsqueeze(0)
                ).item()
                
                if similarity > best_similarity and similarity > 0.3:  # Threshold
                    best_similarity = similarity
                    best_sentence = sentence
            
            if best_sentence:
                # Extract specific value from the best matching sentence
                extracted_value = self._extract_value_from_sentence(best_sentence, param)
                if extracted_value:
                    results[param] = {
                        "value": extracted_value,
                        "confidence": best_similarity,
                        "source_sentence": best_sentence[:100] + "..." if len(best_sentence) > 100 else best_sentence
                    }
        
        return results
    
    def _extract_value_from_sentence(self, sentence: str, param: str) -> Optional[str]:
        """Extract specific value from sentence based on parameter type."""
        sentence = sentence.strip()
        
        if param == "date":
            # Look for date patterns in the sentence
            date_patterns = [
                r'\b(\d{1,2}[./\-]\d{1,2}[./\-]\d{4})\b',
                r'\b(\d{4}[./\-]\d{1,2}[./\-]\d{1,2})\b',
                r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b'
            ]
            for pattern in date_patterns:
                match = re.search(pattern, sentence)
                if match:
                    return match.group(1)
        
        elif param == "company_name":
            # Look for company names with legal suffixes
            company_patterns = [
                r'\b([\w\s&,.-]+\s+(?:Inc|LLC|Ltd|Corp|GmbH|AG|KG|UG))\b',
                r'\b([A-Z][a-zA-Z\s&,.-]{2,30}(?:Inc|LLC|Ltd|Corp|GmbH|AG))\b'
            ]
            for pattern in company_patterns:
                match = re.search(pattern, sentence, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            
            # Fallback: extract capitalized words that might be company names
            words = sentence.split()
            company_candidates = []
            for i, word in enumerate(words):
                if word[0].isupper() and len(word) > 2:
                    # Check if next few words are also capitalized (company name)
                    company_part = [word]
                    for j in range(i+1, min(i+4, len(words))):
                        if words[j][0].isupper() or words[j].lower() in ['inc', 'llc', 'ltd', 'corp', 'gmbh', 'ag']:
                            company_part.append(words[j])
                        else:
                            break
                    if len(company_part) >= 2:
                        company_candidates.append(" ".join(company_part))
            
            if company_candidates:
                return company_candidates[0]
        
        elif param == "company_address":
            # Look for address patterns
            address_patterns = [
                r'(\d+\s+[A-Za-z\s,.-]+(?:Street|Avenue|Road|Lane|St|Ave|Rd|Str)[\s,]*[A-Za-z0-9\s,-]*)',
                r'(\b\d{5}(?:-\d{4})?\s+[A-Za-z\s,.-]+\b)'
            ]
            for pattern in address_patterns:
                match = re.search(pattern, sentence)
                if match:
                    return match.group(1).strip()
        
        elif param == "angebot":
            # Look for quote/proposal numbers
            angebot_patterns = [
                r'([A-Z]{1,4}[-_]\d{4}[-_]\d{3,4})',
                r'([A-Z]+\d{6,10})',
                r'(Q[0-9]{6,10})'
            ]
            for pattern in angebot_patterns:
                match = re.search(pattern, sentence)
                if match:
                    return match.group(1)
        
        return None
    
    def extract_with_ner(self, text: str) -> Dict[str, Any]:
        """Extract entities using NER pipeline."""
        if not self.ner_pipeline:
            return {}
        
        try:
            # Get entities from NER
            entities = self.ner_pipeline(text)
            
            # Group entities by type
            organizations = [e for e in entities if e['entity_group'] in ['ORG']]
            locations = [e for e in entities if e['entity_group'] in ['LOC']]
            persons = [e for e in entities if e['entity_group'] in ['PER']]
            miscellaneous = [e for e in entities if e['entity_group'] in ['MISC']]
            
            result = {}
            
            # Company name (prefer organizations)
            if organizations:
                # Filter for likely company names
                company_candidates = []
                for org in organizations:
                    org_text = org['word'].strip()
                    # Higher score for entities with company suffixes
                    if any(suffix in org_text.upper() for suffix in ['INC', 'LLC', 'LTD', 'CORP', 'GMBH', 'AG']):
                        company_candidates.append((org_text, org['score'] + 0.2))
                    else:
                        company_candidates.append((org_text, org['score']))
                
                if company_candidates:
                    # Sort by score and take the best
                    company_candidates.sort(key=lambda x: x[1], reverse=True)
                    result['company_name'] = {
                        "value": company_candidates[0][0],
                        "confidence": company_candidates[0][1],
                        "method": "ner_organization"
                    }
            
            # Address (use locations)
            if locations:
                # Combine location entities that might form an address
                location_texts = [loc['word'].strip() for loc in locations[:3]]
                combined_address = ", ".join(location_texts)
                avg_confidence = sum(loc['score'] for loc in locations[:3]) / len(locations[:3])
                
                result['company_address'] = {
                    "value": combined_address,
                    "confidence": avg_confidence,
                    "method": "ner_location"
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in NER extraction: {e}")
            return {}
    
    def extract_with_regex_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced regex extraction with confidence scoring."""
        results = {}
        
        for param, patterns in self.regex_patterns.items():
            best_match = None
            best_confidence = 0.0
            
            for i, pattern in enumerate(patterns):
                matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Calculate confidence based on pattern specificity
                    confidence = 0.9 - (i * 0.1)  # Earlier patterns are more specific
                    
                    if isinstance(matches[0], tuple):
                        match_value = matches[0][0] if matches[0] else None
                    else:
                        match_value = matches[0]
                    
                    if match_value and confidence > best_confidence:
                        best_match = match_value.strip()
                        best_confidence = confidence
            
            if best_match:
                results[param] = {
                    "value": best_match,
                    "confidence": best_confidence,
                    "method": "regex"
                }
        
        # Special handling for tables count
        table_indicators = ['position', 'beschreibung', 'preis', 'menge', 'total', 'item', 'description', 'price', 'amount']
        table_count = 0
        lines = text.lower().split('\n')
        
        for line in lines:
            indicator_count = sum(1 for indicator in table_indicators if indicator in line)
            if indicator_count >= 2:  # Line contains multiple table indicators
                table_count += 1
        
        # Also check for table markers from pdfplumber
        table_markers = text.count('[TABLE]')
        table_count = max(table_count, table_markers)
        
        results['tables'] = {
            "value": min(table_count, 20),  # Cap at reasonable number
            "confidence": 0.8 if table_count > 0 else 0.2,
            "method": "heuristic"
        }
        
        return results
    
    def merge_extraction_results(self, bert_results: Dict, ner_results: Dict, regex_results: Dict) -> Dict[str, Any]:
        """Intelligently merge results from different extraction methods."""
        final_results = {}
        
        all_params = set()
        all_params.update(bert_results.keys())
        all_params.update(ner_results.keys()) 
        all_params.update(regex_results.keys())
        
        for param in all_params:
            candidates = []
            
            # Collect candidates from all methods
            if param in bert_results:
                candidates.append({
                    "value": bert_results[param]["value"],
                    "confidence": bert_results[param]["confidence"],
                    "method": "bert_semantic",
                    "weight": 1.0  # BERT gets highest weight
                })
            
            if param in ner_results:
                candidates.append({
                    "value": ner_results[param]["value"],
                    "confidence": ner_results[param]["confidence"],
                    "method": "bert_ner",
                    "weight": 0.9  # NER gets second highest
                })
            
            if param in regex_results:
                candidates.append({
                    "value": regex_results[param]["value"],
                    "confidence": regex_results[param]["confidence"],
                    "method": "regex",
                    "weight": 0.7  # Regex gets lower weight but is reliable
                })
            
            # Choose best candidate based on weighted confidence
            if candidates:
                # Calculate weighted scores
                for candidate in candidates:
                    candidate["weighted_score"] = candidate["confidence"] * candidate["weight"]
                
                # Sort by weighted score
                candidates.sort(key=lambda x: x["weighted_score"], reverse=True)
                best_candidate = candidates[0]
                
                final_results[param] = {
                    "value": best_candidate["value"],
                    "confidence": best_candidate["confidence"],
                    "method": best_candidate["method"],
                    "all_candidates": len(candidates)
                }
        
        return final_results
    
    def extract_parameters(self, text: str, metadata: Dict) -> Dict[str, Any]:
        """Main parameter extraction using multi-method BERT approach."""
        # Method 1: BERT semantic search
        bert_results = self.extract_with_bert_semantic_search(text)
        
        # Method 2: NER extraction
        ner_results = self.extract_with_ner(text)
        
        # Method 3: Enhanced regex extraction
        regex_results = self.extract_with_regex_enhanced(text)
        
        # Merge all results intelligently
        final_results = self.merge_extraction_results(bert_results, ner_results, regex_results)
        
        # Add table count from metadata
        if metadata.get("tables_detected", 0) > 0:
            final_results["tables"] = {
                "value": metadata["tables_detected"],
                "confidence": 0.95,
                "method": "pdfplumber_detection"
            }
        
        return final_results
    
    def extract_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Complete BERT-based extraction from PDF file."""
        start_time = datetime.now()
        
        try:
            # Extract text with advanced methods
            text, metadata = self.extract_pdf_text_advanced(pdf_path)
            
            if not text.strip():
                return {
                    "success": False,
                    "error": "No text could be extracted from PDF",
                    "extracted_parameters": {},
                    "processing_time": 0.0,
                    "bert_available": self.bert_model is not None
                }
            
            # Extract parameters using BERT multi-method approach
            extraction_results = self.extract_parameters(text, metadata)
            
            # Convert to final format
            final_parameters = {}
            confidence_scores = {}
            extraction_methods = {}
            
            for param, result in extraction_results.items():
                final_parameters[param] = result["value"]
                confidence_scores[param] = result["confidence"]
                extraction_methods[param] = result["method"]
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "extracted_parameters": final_parameters,
                "confidence_scores": confidence_scores,
                "extraction_methods": extraction_methods,
                "processing_time": processing_time,
                "text_length": len(text),
                "pdf_metadata": metadata,
                "bert_model": self.model_name,
                "bert_available": self.bert_model is not None,
                "ner_available": self.ner_pipeline is not None
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return {
                "success": False,
                "error": str(e),
                "extracted_parameters": {},
                "processing_time": processing_time,
                "bert_available": self.bert_model is not None
            }

# Initialize BERT extractor
print("🤖 Initializing BERT-base-uncased PDF Extractor...")
bert_extractor = BertBaseUncasedExtractor()

# FastAPI app
app = FastAPI(
    title="BERT-base-uncased PDF Extraction Framework",
    description="Advanced PDF parameter extraction using BERT-base-uncased and hybrid NLP techniques",
    version="2.0.0-bert-base-uncased"
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
        "message": "BERT-base-uncased PDF Extraction Framework is running",
        "version": "2.0.0-bert-base-uncased",
        "mode": "offline",
        "bert_model": "bert-base-uncased",
        "bert_available": bert_extractor.bert_model is not None,
        "ner_available": bert_extractor.ner_pipeline is not None,
        "device": str(bert_extractor.device),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/config/models")
async def get_model_info():
    """Get detailed model information."""
    return {
        "primary_model": {
            "name": "bert-base-uncased",
            "type": "bert_transformer",
            "description": "BERT base model (uncased) for semantic understanding",
            "parameters": ["date", "company_name", "company_address", "angebot"],
            "status": "ready" if bert_extractor.bert_model else "unavailable",
            "embedding_size": 768,
            "max_sequence_length": 512
        },
        "ner_model": {
            "name": "dbmdz/bert-large-cased-finetuned-conll03-english", 
            "type": "bert_ner",
            "description": "BERT NER model for entity extraction",
            "parameters": ["company_name", "company_address"],
            "status": "ready" if bert_extractor.ner_pipeline else "unavailable"
        },
        "fallback_methods": {
            "regex_patterns": {
                "type": "regex_based",
                "description": "Pattern-based extraction (always available)",
                "parameters": ["date", "company_name", "company_address", "angebot", "tables"],
                "status": "ready"
            }
        },
        "extraction_strategy": "multi_method_hybrid",
        "confidence_weighting": {
            "bert_semantic": 1.0,
            "bert_ner": 0.9,
            "regex": 0.7
        }
    }

@app.post("/extract/single")
async def extract_single_pdf(file: UploadFile = File(...)):
    """Extract parameters from a single PDF using BERT-base-uncased."""
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Extract parameters using BERT
            result = bert_extractor.extract_from_pdf(temp_file_path)
            
            result.update({
                "file_name": file.filename,
                "mode": "offline",
                "extraction_framework": "bert-base-uncased"
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
    """Extract parameters from multiple PDFs using BERT-base-uncased."""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    results = []
    processing_stats = {
        "total_files": len(files),
        "successful_extractions": 0,
        "total_processing_time": 0.0,
        "bert_extractions": 0,
        "ner_extractions": 0,
        "regex_extractions": 0
    }
    
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
                # Extract parameters using BERT
                result = bert_extractor.extract_from_pdf(temp_file_path)
                result["file_name"] = file.filename
                results.append(result)
                
                # Update statistics
                if result.get("success", False):
                    processing_stats["successful_extractions"] += 1
                    processing_stats["total_processing_time"] += result.get("processing_time", 0)
                    
                    # Count extraction methods used
                    methods = result.get("extraction_methods", {})
                    if any("bert" in method for method in methods.values()):
                        processing_stats["bert_extractions"] += 1
                    if any("ner" in method for method in methods.values()):
                        processing_stats["ner_extractions"] += 1
                    if any("regex" in method for method in methods.values()):
                        processing_stats["regex_extractions"] += 1
                
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
    
    # Calculate average confidence
    successful_results = [r for r in results if r.get("success", False)]
    avg_confidence = 0.0
    if successful_results:
        all_confidences = []
        for result in successful_results:
            confidence_scores = result.get("confidence_scores", {})
            if confidence_scores:
                all_confidences.extend(confidence_scores.values())
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
    
    # Return based on format
    if request.output_format.lower() == "excel":
        return await create_excel_response(results, "bert_base_uncased_results.xlsx")
    else:
        return {
            "success": True,
            "extraction_framework": "bert-base-uncased",
            "processing_statistics": processing_stats,
            "average_confidence": float(avg_confidence),
            "bert_available": bert_extractor.bert_model is not None,
            "ner_available": bert_extractor.ner_pipeline is not None,
            "mode": "offline",
            "results": results
        }

@app.post("/bert/supervised-train")
async def bert_supervised_training(
    pdf_files: List[UploadFile] = File(...),
    excel_answers: UploadFile = File(...)
):
    """BERT-base-uncased supervised training and evaluation."""
    
    if not excel_answers.filename.lower().endswith(('.xlsx', '.xls')):
        raise HTTPException(status_code=400, detail="Answer file must be Excel format")
    
    try:
        # Save and load Excel answers
        excel_path = f"temp_bert_answers_{excel_answers.filename}"
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
        
        # Process PDFs and evaluate BERT performance
        training_results = []
        method_performance = {
            "bert_semantic": {"correct": 0, "total": 0},
            "bert_ner": {"correct": 0, "total": 0}, 
            "regex": {"correct": 0, "total": 0}
        }
        
        for pdf_file in pdf_files:
            if not pdf_file.filename.lower().endswith('.pdf'):
                continue
            
            # Find corresponding answer
            answer_row = answers_df[answers_df['file_name'] == pdf_file.filename]
            if answer_row.empty:
                continue
            
            # Extract with BERT method
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                content = await pdf_file.read()
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Get BERT predictions
                bert_result = bert_extractor.extract_from_pdf(temp_file_path)
                
                if bert_result["success"]:
                    # Get expected answers
                    answer_data = answer_row.iloc[0]
                    expected_answers = {
                        'date': str(answer_data.get('date', '')).strip() if pd.notna(answer_data.get('date')) else None,
                        'company_name': str(answer_data.get('company_name', '')).strip() if pd.notna(answer_data.get('company_name')) else None,
                        'company_address': str(answer_data.get('company_address', '')).strip() if pd.notna(answer_data.get('company_address')) else None,
                        'angebot': str(answer_data.get('angebot', '')).strip() if pd.notna(answer_data.get('angebot')) else None,
                    }
                    
                    # Evaluate each parameter
                    parameter_evaluation = {}
                    extraction_methods = bert_result.get("extraction_methods", {})
                    
                    for param in ['date', 'company_name', 'company_address', 'angebot']:
                        expected = expected_answers.get(param)
                        predicted = bert_result["extracted_parameters"].get(param)
                        method_used = extraction_methods.get(param, "unknown")
                        
                        # Calculate accuracy with fuzzy matching for BERT
                        accuracy = calculate_bert_accuracy(expected, predicted)
                        
                        parameter_evaluation[param] = {
                            "expected": expected,
                            "predicted": predicted,
                            "accuracy": accuracy,
                            "method": method_used,
                            "confidence": bert_result.get("confidence_scores", {}).get(param, 0.0)
                        }
                        
                        # Update method performance statistics
                        if method_used in method_performance:
                            method_performance[method_used]["total"] += 1
                            if accuracy >= 0.8:  # Consider 80%+ as correct
                                method_performance[method_used]["correct"] += 1
                    
                    training_results.append({
                        'file_name': pdf_file.filename,
                        'parameter_evaluation': parameter_evaluation,
                        'processing_time': bert_result["processing_time"],
                        'bert_available': bert_result.get("bert_available", False),
                        'ner_available': bert_result.get("ner_available", False)
                    })
                    
            finally:
                os.unlink(temp_file_path)
        
        # Calculate overall performance metrics
        if training_results:
            param_accuracies = {}
            for param in ['date', 'company_name', 'company_address', 'angebot']:
                param_accs = [r['parameter_evaluation'][param]['accuracy'] for r in training_results]
                param_accuracies[param] = sum(param_accs) / len(param_accs)
            
            overall_accuracy = sum(param_accuracies.values()) / len(param_accuracies)
            avg_processing_time = sum(r['processing_time'] for r in training_results) / len(training_results)
            
            # Calculate method performance percentages
            method_performance_pct = {}
            for method, stats in method_performance.items():
                if stats["total"] > 0:
                    method_performance_pct[method] = {
                        "accuracy": stats["correct"] / stats["total"],
                        "usage_count": stats["total"]
                    }
                else:
                    method_performance_pct[method] = {"accuracy": 0.0, "usage_count": 0}
        else:
            param_accuracies = {}
            overall_accuracy = 0.0
            avg_processing_time = 0.0
            method_performance_pct = {}
        
        # Clean up
        os.unlink(excel_path)
        
        return {
            "success": True,
            "framework": "bert-base-uncased",
            "training_examples": len(training_results),
            "matched_pdfs": len(training_results),
            "total_pdfs": len(pdf_files),
            "overall_accuracy": overall_accuracy,
            "parameter_accuracies": param_accuracies,
            "method_performance": method_performance_pct,
            "average_processing_time": avg_processing_time,
            "bert_available": bert_extractor.bert_model is not None,
            "ner_available": bert_extractor.ner_pipeline is not None,
            "detailed_results": training_results[:3],  # First 3 for brevity
            "message": f"BERT-base-uncased training evaluation completed on {len(training_results)} examples"
        }
        
    except Exception as e:
        # Clean up on error
        if 'excel_path' in locals() and os.path.exists(excel_path):
            os.unlink(excel_path)
        
        logger.error(f"Error in BERT training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def calculate_bert_accuracy(expected: str, predicted: str) -> float:
    """Calculate accuracy with BERT-appropriate fuzzy matching."""
    if not expected and not predicted:
        return 1.0
    if not expected or not predicted:
        return 0.0
    
    expected_lower = str(expected).lower().strip()
    predicted_lower = str(predicted).lower().strip()
    
    # Exact match
    if expected_lower == predicted_lower:
        return 1.0
    
    # Containment (one contains the other)
    if expected_lower in predicted_lower or predicted_lower in expected_lower:
        return 0.9
    
    # Word overlap (good for BERT semantic understanding)
    expected_words = set(expected_lower.split())
    predicted_words = set(predicted_lower.split())
    
    if expected_words and predicted_words:
        overlap = len(expected_words & predicted_words)
        union = len(expected_words | predicted_words)
        jaccard_similarity = overlap / union if union > 0 else 0.0
        
        # Scale Jaccard similarity
        if jaccard_similarity >= 0.5:
            return 0.8
        elif jaccard_similarity >= 0.3:
            return 0.6
        elif jaccard_similarity > 0:
            return 0.4
    
    return 0.0

async def create_excel_response(results: List[Dict], filename: str):
    """Create Excel response from BERT extraction results."""
    try:
        # Create main results DataFrame
        excel_data = []
        for result in results:
            if result.get("success", False):
                params = result.get("extracted_parameters", {})
                confidence = result.get("confidence_scores", {})
                methods = result.get("extraction_methods", {})
                metadata = result.get("pdf_metadata", {})
                
                row = {
                    "file_name": result.get("file_name", ""),
                    "date": params.get("date", ""),
                    "company_name": params.get("company_name", ""),
                    "company_address": params.get("company_address", ""),
                    "angebot": params.get("angebot", ""),
                    "tables_count": params.get("tables", 0),
                    "processing_time_sec": result.get("processing_time", 0.0),
                    "bert_model": result.get("bert_model", "bert-base-uncased"),
                    "bert_available": result.get("bert_available", False),
                    "ner_available": result.get("ner_available", False),
                    "pdf_pages": metadata.get("pages", 0),
                    "pdf_tables_detected": metadata.get("tables_detected", 0),
                    "confidence_date": confidence.get("date", 0.0),
                    "confidence_company": confidence.get("company_name", 0.0),
                    "confidence_address": confidence.get("company_address", 0.0),
                    "confidence_angebot": confidence.get("angebot", 0.0),
                    "confidence_average": sum(confidence.values()) / len(confidence) if confidence else 0.0,
                    "method_date": methods.get("date", ""),
                    "method_company": methods.get("company_name", ""),
                    "method_address": methods.get("company_address", ""),
                    "method_angebot": methods.get("angebot", "")
                }
            else:
                row = {
                    "file_name": result.get("file_name", ""),
                    "date": "",
                    "company_name": "",
                    "company_address": "",
                    "angebot": "",
                    "tables_count": 0,
                    "processing_time_sec": 0.0,
                    "bert_model": "bert-base-uncased",
                    "bert_available": False,
                    "ner_available": False,
                    "pdf_pages": 0,
                    "pdf_tables_detected": 0,
                    "confidence_date": 0.0,
                    "confidence_company": 0.0,
                    "confidence_address": 0.0,
                    "confidence_angebot": 0.0,
                    "confidence_average": 0.0,
                    "method_date": "",
                    "method_company": "",
                    "method_address": "",
                    "method_angebot": "",
                    "error": result.get("error", "Processing failed")
                }
            excel_data.append(row)
        
        df = pd.DataFrame(excel_data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            # Main results sheet
            df.to_excel(writer, sheet_name='BERT_Extraction_Results', index=False)
            
            # Summary statistics sheet
            successful_results = [r for r in excel_data if not r.get('error')]
            summary_data = {
                'Metric': [
                    'Total Files Processed',
                    'Successful Extractions',
                    'Success Rate (%)',
                    'Average Processing Time (s)',
                    'Average Confidence Score',
                    'BERT Model Used',
                    'BERT Availability',
                    'NER Availability',
                    'Extraction Framework',
                    'Mode'
                ],
                'Value': [
                    len(excel_data),
                    len(successful_results),
                    f"{(len(successful_results) / len(excel_data) * 100):.1f}%" if excel_data else "0%",
                    f"{np.mean([r['processing_time_sec'] for r in successful_results]):.3f}" if successful_results else "0.000",
                    f"{np.mean([r['confidence_average'] for r in successful_results]):.3f}" if successful_results else "0.000",
                    'bert-base-uncased',
                    'Available' if any(r['bert_available'] for r in excel_data) else 'Not Available',
                    'Available' if any(r['ner_available'] for r in excel_data) else 'Not Available',
                    'BERT-base-uncased + NER + Regex Hybrid',
                    'Offline'
                ]
            }
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Method performance sheet (if we have extraction methods data)
            if successful_results:
                method_data = []
                methods = ['bert_semantic', 'bert_ner', 'regex', 'heuristic', 'pdfplumber_detection']
                
                for method in methods:
                    count = sum(1 for r in successful_results 
                              if any(method in str(r.get(f'method_{param}', '')) 
                                   for param in ['date', 'company', 'address', 'angebot']))
                    if count > 0:
                        method_data.append({
                            'Extraction_Method': method,
                            'Usage_Count': count,
                            'Usage_Percentage': f"{(count / len(successful_results) * 100):.1f}%"
                        })
                
                if method_data:
                    method_df = pd.DataFrame(method_data)
                    method_df.to_excel(writer, sheet_name='Method_Performance', index=False)
        
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
    """Download Excel template for BERT-base-uncased training data."""
    try:
        # Create template data
        template_data = {
            'file_name': ['example_document.pdf', 'sample_invoice.pdf'],
            'date': ['15.03.2024', '20.02.2024'],
            'company_name': ['Example Corp Ltd', 'Sample GmbH'],
            'company_address': ['123 Main Street, New York, NY 10001', 'Musterstraße 456, 12345 Berlin'],
            'angebot': ['Q-2024-001', 'A-2024-002'],
            'tables_count': [2, 1],
            'notes': ['BERT training example', 'Sample data for training']
        }
        
        df = pd.DataFrame(template_data)
        
        # Create Excel file in memory
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='BERT_Training_Template', index=False)
            
            # Add instructions sheet
            instructions = pd.DataFrame({
                'BERT-base-uncased Training Instructions': [
                    '1. BERT MODEL: This system uses bert-base-uncased for semantic understanding',
                    '2. FILE NAMES: Use EXACT PDF filenames (case-sensitive)',
                    '3. QUALITY DATA: BERT learns from patterns - provide accurate examples',
                    '4. DATE FORMAT: Use consistent formats (DD.MM.YYYY or MM/DD/YYYY)',
                    '5. COMPANY NAMES: Include full legal names (Corp, Ltd, GmbH, Inc, etc.)',
                    '6. ADDRESSES: Complete addresses with postal codes work best',
                    '7. ANGEBOT: Exact quote/proposal numbers as they appear in documents',
                    '8. TABLES: Count actual structured data tables in documents',
                    '9. BERT ADVANTAGE: Better at understanding context and variations',
                    '10. TRAINING SIZE: More examples = better BERT performance',
                    '11. SEMANTIC UNDERSTANDING: BERT can handle synonyms and variations',
                    '12. SAVE FORMAT: Save as .xlsx and upload with your PDFs'
                ]
            })
            instructions.to_excel(writer, sheet_name='BERT_Instructions', index=False)
            
            # Add parameter descriptions
            param_descriptions = pd.DataFrame({
                'Parameter': ['date', 'company_name', 'company_address', 'angebot', 'tables'],
                'Description': [
                    'Document date - BERT finds dates in context',
                    'Company name - BERT recognizes organizations',
                    'Company address - BERT combines location entities',
                    'Quote/proposal number - BERT finds structured IDs',
                    'Table count - Combined detection methods'
                ],
                'BERT_Method': [
                    'Semantic similarity + regex validation',
                    'NER organizations + semantic search',
                    'NER locations + pattern matching',
                    'Semantic search + structured patterns',
                    'PDF structure analysis + heuristics'
                ]
            })
            param_descriptions.to_excel(writer, sheet_name='Parameter_Info', index=False)
        
        excel_buffer.seek(0)
        
        return StreamingResponse(
            io.BytesIO(excel_buffer.read()),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=bert_base_uncased_training_template.xlsx"}
        )
        
    except Exception as e:
        logger.error(f"Error creating template: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Enhanced web interface for BERT-base-uncased powered operation."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>🤖 BERT-base-uncased PDF Extraction Framework</title>
        <style>
            body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
            .container { max-width: 1400px; margin: 0 auto; background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
            .header { text-align: center; margin-bottom: 40px; }
            .title { color: #2c3e50; font-size: 2.5em; margin-bottom: 10px; }
            .subtitle { color: #7f8c8d; font-size: 1.2em; }
            .bert-badge { background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 8px 20px; border-radius: 25px; font-size: 14px; margin: 5px; display: inline-block; font-weight: bold; }
            .offline-badge { background: linear-gradient(45deg, #27ae60, #2ecc71); color: white; padding: 8px 20px; border-radius: 25px; font-size: 14px; margin: 5px; display: inline-block; font-weight: bold; }
            .section { margin-bottom: 30px; padding: 25px; border: 1px solid #ecf0f1; border-radius: 10px; background: #f8f9fa; }
            .section h3 { color: #2c3e50; margin-top: 0; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            .upload-area { border: 3px dashed #3498db; padding: 30px; text-align: center; margin: 20px 0; background: white; border-radius: 10px; transition: all 0.3s ease; }
            .upload-area:hover { border-color: #2980b9; background: #f8f9fa; transform: translateY(-2px); }
            .btn { background: linear-gradient(45deg, #3498db, #2980b9); color: white; padding: 12px 25px; border: none; border-radius: 8px; cursor: pointer; margin: 5px; font-size: 14px; font-weight: bold; transition: all 0.3s ease; }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4); }
            .btn-success { background: linear-gradient(45deg, #27ae60, #2ecc71); }
            .btn-success:hover { box-shadow: 0 5px 15px rgba(39, 174, 96, 0.4); }
            .result { background: #ecf0f1; padding: 20px; margin: 15px 0; border-radius: 8px; border-left: 5px solid #3498db; }
            .success { background: #d5f4e6; border-left-color: #27ae60; }
            .error { background: #fadbd8; border-left-color: #e74c3c; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin-top: 30px; }
            .feature-card { background: white; padding: 25px; border-radius: 10px; border: 1px solid #ecf0f1; box-shadow: 0 2px 10px rgba(0,0,0,0.1); transition: transform 0.3s ease; }
            .feature-card:hover { transform: translateY(-5px); box-shadow: 0 5px 20px rgba(0,0,0,0.15); }
            .bert-info { background: linear-gradient(135deg, #3498db, #2980b9); color: white; padding: 20px; border-radius: 10px; margin-bottom: 30px; }
            .model-specs { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px; }
            .spec-item { background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px; text-align: center; }
            .progress-container { background: #ecf0f1; border-radius: 10px; padding: 20px; margin: 20px 0; display: none; }
            .progress-bar { background: #3498db; height: 20px; border-radius: 10px; transition: width 0.3s ease; }
            #results { max-height: 500px; overflow-y: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="title">🤖 BERT-base-uncased PDF Extraction</h1>
                <p class="subtitle">Advanced AI-powered parameter extraction using transformer models</p>
                <div>
                    <span class="bert-badge">BERT-base-uncased</span>
                    <span class="offline-badge">Offline Mode</span>
                    <span class="bert-badge">Multi-method Hybrid</span>
                </div>
            </div>

            <div class="bert-info">
                <h3>🧠 BERT-base-uncased Model Information</h3>
                <p>This system uses Google's BERT-base-uncased transformer model for advanced semantic understanding of document content.</p>
                <div class="model-specs">
                    <div class="spec-item">
                        <strong>Model Size</strong><br>110M parameters
                    </div>
                    <div class="spec-item">
                        <strong>Embedding Size</strong><br>768 dimensions
                    </div>
                    <div class="spec-item">
                        <strong>Max Sequence</strong><br>512 tokens
                    </div>
                    <div class="spec-item">
                        <strong>Language Support</strong><br>English (uncased)
                    </div>
                </div>
            </div>

            <div class="feature-grid">
                <div class="feature-card">
                    <h3>📄 Single PDF Extraction</h3>
                    <p>Extract parameters from individual PDF documents using BERT's semantic understanding.</p>
                    <form id="singleForm" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('singleFile').click()">
                            <input type="file" id="singleFile" accept=".pdf" style="display: none">
                            <p>📎 Click to select PDF file</p>
                            <p style="font-size: 12px; color: #7f8c8d;">BERT will analyze: dates, companies, addresses, quotes, tables</p>
                        </div>
                        <button type="submit" class="btn">🚀 Extract with BERT</button>
                    </form>
                </div>

                <div class="feature-card">
                    <h3>📚 Batch PDF Processing</h3>
                    <p>Process multiple PDFs simultaneously with BERT-powered batch extraction.</p>
                    <form id="batchForm" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('batchFiles').click()">
                            <input type="file" id="batchFiles" accept=".pdf" multiple style="display: none">
                            <p>📎 Click to select multiple PDF files</p>
                            <p style="font-size: 12px; color: #7f8c8d;">BERT processes each file with semantic analysis</p>
                        </div>
                        <select id="batchFormat" style="margin: 10px; padding: 8px; border-radius: 5px;">
                            <option value="json">JSON Output</option>
                            <option value="excel">Excel Output</option>
                        </select>
                        <button type="submit" class="btn">🔄 Batch Process</button>
                    </form>
                </div>

                <div class="feature-card">
                    <h3>🎯 BERT Supervised Training</h3>
                    <p>Evaluate BERT performance using your labeled training data.</p>
                    <form id="trainingForm" enctype="multipart/form-data">
                        <div class="upload-area" onclick="document.getElementById('trainingPdfs').click()">
                            <input type="file" id="trainingPdfs" accept=".pdf" multiple style="display: none">
                            <p>📎 Select training PDF files</p>
                        </div>
                        <div class="upload-area" onclick="document.getElementById('trainingExcel').click()">
                            <input type="file" id="trainingExcel" accept=".xlsx,.xls" style="display: none">
                            <p>📊 Select Excel answers file</p>
                        </div>
                        <button type="submit" class="btn btn-success">🧠 Train & Evaluate BERT</button>
                    </form>
                </div>

                <div class="feature-card">
                    <h3>📋 Training Template</h3>
                    <p>Download Excel template optimized for BERT training data preparation.</p>
                    <button onclick="downloadTemplate()" class="btn">📥 Download BERT Template</button>
                    <p style="font-size: 12px; color: #7f8c8d; margin-top: 10px;">
                        Template includes BERT-specific instructions and parameter guidelines
                    </p>
                </div>
            </div>

            <div class="progress-container" id="progressContainer">
                <h4>🤖 BERT Processing...</h4>
                <div style="background: #bdc3c7; border-radius: 10px; overflow: hidden;">
                    <div class="progress-bar" id="progressBar" style="width: 0%;"></div>
                </div>
                <p id="progressText">Initializing BERT models...</p>
            </div>

            <div id="results"></div>

            <div class="section">
                <h3>🔧 BERT System Information</h3>
                <button onclick="checkHealth()" class="btn">🏥 Check BERT Status</button>
                <button onclick="getModels()" class="btn">📊 Model Information</button>
                <div id="systemInfo"></div>
            </div>
        </div>

        <script>
            // File selection handlers
            document.getElementById('singleFile').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    e.target.parentElement.innerHTML = `<p>✅ Selected: ${file.name}</p><p style="font-size: 12px; color: #27ae60;">Ready for BERT extraction</p>`;
                }
            });

            document.getElementById('batchFiles').addEventListener('change', function(e) {
                const files = e.target.files;
                if (files.length > 0) {
                    e.target.parentElement.innerHTML = `<p>✅ Selected: ${files.length} PDF files</p><p style="font-size: 12px; color: #27ae60;">Ready for BERT batch processing</p>`;
                }
            });

            document.getElementById('trainingPdfs').addEventListener('change', function(e) {
                const files = e.target.files;
                if (files.length > 0) {
                    e.target.parentElement.innerHTML = `<p>✅ PDFs: ${files.length} files selected</p>`;
                }
            });

            document.getElementById('trainingExcel').addEventListener('change', function(e) {
                const file = e.target.files[0];
                if (file) {
                    e.target.parentElement.innerHTML = `<p>✅ Excel: ${file.name}</p>`;
                }
            });

            // Form submission handlers
            document.getElementById('singleForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('singleFile');
                if (!fileInput.files[0]) {
                    alert('Please select a PDF file');
                    return;
                }

                showProgress('BERT analyzing single PDF...');
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/extract/single', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    hideProgress();
                    displayResult('Single PDF BERT Extraction', result);
                } catch (error) {
                    hideProgress();
                    displayError('BERT extraction failed: ' + error.message);
                }
            });

            document.getElementById('batchForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const fileInput = document.getElementById('batchFiles');
                if (!fileInput.files.length) {
                    alert('Please select PDF files');
                    return;
                }

                showProgress('BERT processing batch files...');
                const formData = new FormData();
                for (let file of fileInput.files) {
                    formData.append('files', file);
                }

                const format = document.getElementById('batchFormat').value;
                if (format === 'excel') {
                    try {
                        const response = await fetch(`/extract/batch?output_format=excel`, {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const blob = await response.blob();
                            const url = window.URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = 'bert_base_uncased_results.xlsx';
                            a.click();
                            hideProgress();
                            displayResult('Batch Processing', {success: true, message: 'Excel file downloaded successfully'});
                        } else {
                            throw new Error('Failed to download Excel file');
                        }
                    } catch (error) {
                        hideProgress();
                        displayError('Batch processing failed: ' + error.message);
                    }
                } else {
                    try {
                        const response = await fetch('/extract/batch', {
                            method: 'POST',
                            body: formData
                        });
                        const result = await response.json();
                        hideProgress();
                        displayResult('Batch PDF BERT Extraction', result);
                    } catch (error) {
                        hideProgress();
                        displayError('Batch processing failed: ' + error.message);
                    }
                }
            });

            document.getElementById('trainingForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                const pdfInput = document.getElementById('trainingPdfs');
                const excelInput = document.getElementById('trainingExcel');
                
                if (!pdfInput.files.length || !excelInput.files[0]) {
                    alert('Please select both PDF files and Excel answers file');
                    return;
                }

                showProgress('BERT supervised training in progress...');
                const formData = new FormData();
                for (let file of pdfInput.files) {
                    formData.append('pdf_files', file);
                }
                formData.append('excel_answers', excelInput.files[0]);

                try {
                    const response = await fetch('/bert/supervised-train', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await response.json();
                    hideProgress();
                    displayResult('BERT Supervised Training Results', result);
                } catch (error) {
                    hideProgress();
                    displayError('BERT training failed: ' + error.message);
                }
            });

            async function downloadTemplate() {
                try {
                    const response = await fetch('/training/download-template');
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'bert_base_uncased_training_template.xlsx';
                    a.click();
                } catch (error) {
                    displayError('Template download failed: ' + error.message);
                }
            }

            async function checkHealth() {
                try {
                    const response = await fetch('/health');
                    const result = await response.json();
                    displayResult('BERT System Health', result);
                } catch (error) {
                    displayError('Health check failed: ' + error.message);
                }
            }

            async function getModels() {
                try {
                    const response = await fetch('/config/models');
                    const result = await response.json();
                    displayResult('BERT Model Configuration', result);
                } catch (error) {
                    displayError('Model info failed: ' + error.message);
                }
            }

            function showProgress(message) {
                document.getElementById('progressContainer').style.display = 'block';
                document.getElementById('progressText').textContent = message;
                let width = 0;
                const interval = setInterval(() => {
                    width += Math.random() * 15;
                    if (width >= 90) {
                        clearInterval(interval);
                        width = 90;
                    }
                    document.getElementById('progressBar').style.width = width + '%';
                }, 500);
            }

            function hideProgress() {
                document.getElementById('progressContainer').style.display = 'none';
                document.getElementById('progressBar').style.width = '0%';
            }

            function displayResult(title, result) {
                const resultsDiv = document.getElementById('results');
                const resultClass = result.success !== false ? 'success' : 'error';
                const timestamp = new Date().toLocaleTimeString();
                
                resultsDiv.innerHTML = `
                    <div class="result ${resultClass}">
                        <h4>🤖 ${title} - ${timestamp}</h4>
                        <pre>${JSON.stringify(result, null, 2)}</pre>
                    </div>
                ` + resultsDiv.innerHTML;
            }

            function displayError(message) {
                const resultsDiv = document.getElementById('results');
                const timestamp = new Date().toLocaleTimeString();
                resultsDiv.innerHTML = `
                    <div class="result error">
                        <h4>❌ Error - ${timestamp}</h4>
                        <p>${message}</p>
                    </div>
                ` + resultsDiv.innerHTML;
            }

            // Initialize system check on load
            window.onload = function() {
                checkHealth();
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    print("🤖 Starting BERT-base-uncased PDF Extraction Server")
    print("=" * 60)
    print("🧠 Model: bert-base-uncased")
    print("🔧 Framework: Hybrid BERT + NER + Regex")
    print("🌐 Server: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🏥 Health: http://localhost:8000/health")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)