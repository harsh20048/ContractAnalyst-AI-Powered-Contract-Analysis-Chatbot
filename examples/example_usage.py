#!/usr/bin/env python3
"""
Example usage of the HuggingFace PDF Extraction Framework.

This script demonstrates how to use the framework programmatically
for various PDF processing and text extraction tasks.
"""

import requests
import json
import time
from pathlib import Path


class HFPDFExtractor:
    """Client for the HuggingFace PDF Extraction API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the client."""
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check if the service is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_supported_tasks(self):
        """Get list of supported extraction tasks."""
        response = requests.get(f"{self.base_url}/config/tasks")
        return response.json()
    
    def get_available_models(self):
        """Get available models for each task."""
        response = requests.get(f"{self.base_url}/config/models")
        return response.json()
    
    def load_model(self, model_name: str, task: str, force_reload: bool = False):
        """Load a specific model."""
        payload = {
            "model_name": model_name,
            "task": task,
            "force_reload": force_reload
        }
        response = requests.post(f"{self.base_url}/models/load", json=payload)
        return response.json()
    
    def get_model_status(self):
        """Get status of all loaded models."""
        response = requests.get(f"{self.base_url}/models/status")
        return response.json()
    
    def extract_from_pdf(self, pdf_path: str, tasks: list, 
                        model_preferences: dict = None,
                        custom_questions: list = None,
                        extract_tables: bool = True,
                        extract_metadata: bool = True):
        """Extract information from a PDF file."""
        
        # Prepare the file
        files = {'file': open(pdf_path, 'rb')}
        
        # Prepare form data
        data = {
            'tasks': json.dumps(tasks),
            'model_preferences': json.dumps(model_preferences or {}),
            'custom_questions': json.dumps(custom_questions or []),
            'extract_tables': extract_tables,
            'extract_metadata': extract_metadata
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/extract/pdf",
                files=files,
                data=data
            )
            return response.json()
        finally:
            files['file'].close()
    
    def extract_from_pdf_async(self, pdf_path: str, tasks: list,
                              model_preferences: dict = None,
                              custom_questions: list = None):
        """Extract information from PDF asynchronously."""
        
        files = {'file': open(pdf_path, 'rb')}
        data = {
            'tasks': json.dumps(tasks),
            'model_preferences': json.dumps(model_preferences or {}),
            'custom_questions': json.dumps(custom_questions or [])
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/extract/pdf/async",
                files=files,
                data=data
            )
            return response.json()
        finally:
            files['file'].close()
    
    def get_processing_status(self, job_id: str):
        """Get status of an async processing job."""
        response = requests.get(f"{self.base_url}/extract/status/{job_id}")
        return response.json()
    
    def extract_from_text(self, text: str, tasks: list, 
                         model_preferences: dict = None):
        """Extract information from plain text."""
        payload = {
            "text": text,
            "tasks": tasks,
            "model_preferences": model_preferences
        }
        response = requests.post(f"{self.base_url}/extract/text", json=payload)
        return response.json()


def example_basic_usage():
    """Basic usage example."""
    print("=== Basic Usage Example ===")
    
    # Initialize client
    client = HFPDFExtractor()
    
    # Check if service is running
    if not client.health_check():
        print("❌ Service is not running. Please start the application first.")
        return
    
    print("✅ Service is running")
    
    # Get supported tasks
    tasks = client.get_supported_tasks()
    print(f"📋 Supported tasks: {tasks['supported_tasks']}")
    
    # Example text processing
    sample_text = """
    Artificial Intelligence (AI) is transforming industries worldwide. 
    Companies are investing heavily in machine learning technologies 
    to improve efficiency and customer experience. The future looks 
    promising for AI applications in healthcare, finance, and education.
    """
    
    print("\n🔍 Processing sample text...")
    result = client.extract_from_text(
        text=sample_text,
        tasks=["summarization", "sentiment_analysis", "named_entity_recognition"]
    )
    
    print(f"✅ Text processing completed")
    print(f"📊 Results: {json.dumps(result, indent=2)}")


def example_pdf_processing():
    """PDF processing example."""
    print("\n=== PDF Processing Example ===")
    
    client = HFPDFExtractor()
    
    # For this example, you would need a PDF file
    pdf_path = "sample_document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file '{pdf_path}' not found. Skipping PDF example.")
        return
    
    print(f"📄 Processing PDF: {pdf_path}")
    
    # Extract information from PDF
    result = client.extract_from_pdf(
        pdf_path=pdf_path,
        tasks=[
            "summarization",
            "key_information_extraction",
            "question_answering",
            "named_entity_recognition"
        ],
        custom_questions=[
            "What is the main topic of this document?",
            "What are the key findings?",
            "Who are the main stakeholders mentioned?"
        ]
    )
    
    if result.get("success"):
        print("✅ PDF processing completed successfully")
        print(f"📊 Processing time: {result['processing_time']:.2f} seconds")
        print(f"📝 Words: {result['total_words']}, Sentences: {result['total_sentences']}")
        
        # Display summary if available
        if result.get("summary"):
            print(f"\n📋 Summary:\n{result['summary']}")
        
        # Display Q&A results if available
        if result.get("questions_answers"):
            print("\n❓ Question & Answers:")
            for qa in result["questions_answers"]:
                print(f"Q: {qa['question']}")
                print(f"A: {qa['answer']} (Confidence: {qa['confidence']:.2f})")
                print()
        
        # Display named entities if available
        if result.get("named_entities"):
            print("🏷️  Named Entities:")
            for entity_type, entities in result["named_entities"].items():
                entity_texts = [e['text'] for e in entities[:5]]  # First 5
                print(f"  {entity_type}: {', '.join(entity_texts)}")
    
    else:
        print("❌ PDF processing failed")
        if result.get("errors"):
            print(f"Errors: {result['errors']}")


def example_async_processing():
    """Asynchronous processing example."""
    print("\n=== Async Processing Example ===")
    
    client = HFPDFExtractor()
    
    pdf_path = "sample_document.pdf"
    
    if not Path(pdf_path).exists():
        print(f"⚠️  PDF file '{pdf_path}' not found. Skipping async example.")
        return
    
    print(f"🚀 Starting async processing of: {pdf_path}")
    
    # Start async processing
    result = client.extract_from_pdf_async(
        pdf_path=pdf_path,
        tasks=["summarization", "key_information_extraction"]
    )
    
    if "job_id" in result:
        job_id = result["job_id"]
        print(f"📋 Job ID: {job_id}")
        
        # Poll for completion
        while True:
            status = client.get_processing_status(job_id)
            print(f"📊 Status: {status['status']} ({status['progress']*100:.1f}%)")
            
            if status["status"] in ["completed", "failed"]:
                break
            
            time.sleep(2)
        
        if status["status"] == "completed":
            print("✅ Async processing completed")
            print(f"📋 Results available in status response")
        else:
            print("❌ Async processing failed")
            if status.get("error"):
                print(f"Error: {status['error']}")
    
    else:
        print("❌ Failed to start async processing")


def example_model_management():
    """Model management example."""
    print("\n=== Model Management Example ===")
    
    client = HFPDFExtractor()
    
    # Get current model status
    status = client.get_model_status()
    print(f"📊 Current model status:")
    print(json.dumps(status, indent=2))
    
    # Load a specific model
    print("\n🔄 Loading summarization model...")
    result = client.load_model(
        model_name="facebook/bart-large-cnn",
        task="summarization"
    )
    print(f"✅ Model loading result: {result}")
    
    # Check updated status
    status = client.get_model_status()
    print(f"\n📊 Updated model status:")
    loaded_models = status.get("model_info", {})
    for model_key, info in loaded_models.items():
        if info.get("is_loaded"):
            print(f"  ✅ {info['name']} ({info['task']}) - Load time: {info['load_time']:.2f}s")


def example_custom_workflow():
    """Custom workflow example combining multiple features."""
    print("\n=== Custom Workflow Example ===")
    
    client = HFPDFExtractor()
    
    # Step 1: Pre-load models for better performance
    print("🔄 Pre-loading models...")
    models_to_load = [
        ("facebook/bart-large-cnn", "summarization"),
        ("distilbert-base-cased-distilled-squad", "question_answering"),
        ("dslim/bert-base-NER", "named_entity_recognition")
    ]
    
    for model_name, task in models_to_load:
        result = client.load_model(model_name, task)
        print(f"  {'✅' if 'Successfully' in result.get('message', '') else '❌'} {model_name}")
    
    # Step 2: Process multiple documents with custom preferences
    sample_texts = [
        "The quarterly report shows a 15% increase in revenue, driven by strong performance in the technology sector.",
        "John Smith, CEO of TechCorp, announced a new partnership with Global Industries to expand into emerging markets.",
        "The research study indicates that remote work has improved employee satisfaction by 23% while reducing operational costs."
    ]
    
    print("\n📊 Processing multiple texts with custom model preferences...")
    
    model_preferences = {
        "summarization": "facebook/bart-large-cnn",
        "question_answering": "distilbert-base-cased-distilled-squad",
        "named_entity_recognition": "dslim/bert-base-NER"
    }
    
    for i, text in enumerate(sample_texts, 1):
        print(f"\n📄 Processing text {i}...")
        result = client.extract_from_text(
            text=text,
            tasks=["summarization", "named_entity_recognition"],
            model_preferences=model_preferences
        )
        
        if result.get("success"):
            print(f"  ✅ Completed in {result.get('processing_time', 0):.2f}s")
            if "summarization" in result.get("results", {}):
                summary = result["results"]["summarization"]
                print(f"  📋 Summary: {summary}")
        else:
            print(f"  ❌ Failed to process text {i}")


def main():
    """Run all examples."""
    print("🤗 HuggingFace PDF Extraction Framework - Usage Examples")
    print("=" * 60)
    
    try:
        example_basic_usage()
        example_model_management()
        example_custom_workflow()
        
        # These require actual PDF files
        example_pdf_processing()
        example_async_processing()
        
    except KeyboardInterrupt:
        print("\n⚠️  Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
    
    print("\n✅ Examples completed!")


if __name__ == "__main__":
    main()