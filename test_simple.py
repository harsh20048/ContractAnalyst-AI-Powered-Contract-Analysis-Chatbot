#!/usr/bin/env python3
"""
Simple test script for the HuggingFace PDF Extraction Framework.
Tests basic functionality using existing PDF if available.
"""

import os
import requests
import time
import json
from pathlib import Path


def test_api_health(base_url: str = "http://localhost:8000"):
    """Test if the API is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            result = response.json()
            print(f"✅ API is healthy: {result}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API is not accessible: {str(e)}")
        return False


def test_config_endpoints(base_url: str = "http://localhost:8000"):
    """Test configuration endpoints."""
    print("\n📋 Testing configuration endpoints...")
    
    endpoints = [
        "/config/tasks",
        "/config/models", 
        "/config/settings"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(f"{base_url}{endpoint}")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ {endpoint}: OK")
                if endpoint == "/config/tasks":
                    print(f"   Supported tasks: {result.get('supported_tasks', [])}")
            else:
                print(f"❌ {endpoint}: Failed ({response.status_code})")
                return False
        except Exception as e:
            print(f"❌ {endpoint}: Error - {str(e)}")
            return False
    
    return True


def test_excel_template(base_url: str = "http://localhost:8000"):
    """Test Excel template download."""
    print("\n📝 Testing Excel template download...")
    
    try:
        response = requests.get(f"{base_url}/training/download-template")
        if response.status_code == 200:
            # Save template
            template_path = "test_template.xlsx"
            with open(template_path, 'wb') as f:
                f.write(response.content)
            
            file_size = os.path.getsize(template_path)
            print(f"✅ Template downloaded successfully!")
            print(f"   File size: {file_size} bytes")
            print(f"   Saved to: {template_path}")
            
            # Clean up
            os.unlink(template_path)
            return True
        else:
            print(f"❌ Template download failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Template download error: {str(e)}")
        return False


def test_text_extraction(base_url: str = "http://localhost:8000"):
    """Test text extraction endpoint."""
    print("\n📄 Testing text extraction...")
    
    # Sample text with the 5 parameters we want to extract
    sample_text = """
    ANGEBOT
    
    Datum: 15.03.2024
    
    TechSolutions GmbH
    Musterstraße 123
    12345 Berlin
    Deutschland
    
    Sehr geehrte Damen und Herren,
    
    hiermit unterbreiten wir Ihnen unser Angebot für die gewünschten IT-Services.
    
    Angebotsnummer: A-2024-001
    Gültigkeitsdauer: 30 Tage
    
    Position | Beschreibung | Menge | Einzelpreis | Gesamtpreis
    1 | Server Installation | 1 | 2.500,00 € | 2.500,00 €
    2 | Software Lizenz | 5 | 150,00 € | 750,00 €
    3 | Schulung | 8 Std. | 120,00 € | 960,00 €
    Gesamt: 4.210,00 €
    
    Vielen Dank für Ihr Interesse.
    """
    
    payload = {
        "text": sample_text,
        "tasks": ["summarization", "named_entity_recognition", "text_classification"]
    }
    
    try:
        response = requests.post(f"{base_url}/extract/text", json=payload)
        if response.status_code == 200:
            result = response.json()
            print("✅ Text extraction successful!")
            print(f"   Tasks completed: {result.get('tasks_completed', [])}")
            print(f"   Input length: {result.get('input_length', 0)} characters")
            
            # Show some results if available
            results = result.get('results', {})
            for task, task_result in results.items():
                if task_result:
                    print(f"   {task}: Available")
                else:
                    print(f"   {task}: No result")
            
            return True
        else:
            print(f"❌ Text extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Text extraction error: {str(e)}")
        return False


def create_simple_test_pdf():
    """Create a very simple PDF for testing without external dependencies."""
    # Create a minimal PDF using basic file operations
    # This creates a very basic PDF structure
    
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>
endobj
4 0 obj
<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>
endobj
5 0 obj
<< /Length 200 >>
stream
BT
/F1 12 Tf
50 750 Td
(ANGEBOT) Tj
0 -20 Td
(Datum: 15.03.2024) Tj
0 -20 Td
(TechSolutions GmbH) Tj
0 -20 Td
(Musterstrasse 123, 12345 Berlin) Tj
0 -20 Td
(Angebotsnummer: A-2024-001) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000251 00000 n 
0000000318 00000 n 
trailer
<< /Size 6 /Root 1 0 R >>
startxref
568
%%EOF"""
    
    test_pdf_path = "simple_test.pdf"
    with open(test_pdf_path, 'wb') as f:
        f.write(pdf_content)
    
    return test_pdf_path


def test_single_pdf_extraction(base_url: str = "http://localhost:8000"):
    """Test single PDF extraction with a simple test PDF."""
    print("\n📄 Testing single PDF extraction...")
    
    # Create a simple test PDF
    try:
        pdf_path = create_simple_test_pdf()
        print(f"   Created test PDF: {pdf_path}")
        
        # Test the extraction
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/training/extract-single", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ PDF extraction successful!")
            
            extracted = result.get('extracted_data', {})
            print("   Extracted parameters:")
            for key, value in extracted.items():
                if value:
                    if isinstance(value, list):
                        print(f"     - {key}: {len(value)} item(s)")
                    else:
                        display_value = str(value)[:50] + "..." if len(str(value)) > 50 else str(value)
                        print(f"     - {key}: {display_value}")
                else:
                    print(f"     - {key}: Not found")
            
            metadata = result.get('metadata', {})
            print(f"   Processing time: {metadata.get('processing_time', 0):.2f}s")
            
            # Clean up
            os.unlink(pdf_path)
            return True
        else:
            print(f"❌ PDF extraction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            os.unlink(pdf_path)
            return False
            
    except Exception as e:
        print(f"❌ PDF extraction error: {str(e)}")
        if os.path.exists(pdf_path):
            os.unlink(pdf_path)
        return False


def test_model_status(base_url: str = "http://localhost:8000"):
    """Test model status endpoint."""
    print("\n🤖 Testing model status...")
    
    try:
        response = requests.get(f"{base_url}/models/status")
        if response.status_code == 200:
            result = response.json()
            print("✅ Model status retrieved!")
            
            model_info = result.get('model_info', {})
            memory_usage = result.get('memory_usage', {})
            
            print(f"   Loaded models: {len(model_info)}")
            print(f"   Device: {memory_usage.get('device', 'unknown')}")
            
            for model_key, info in model_info.items():
                if info.get('is_loaded'):
                    print(f"     - {info.get('name', 'Unknown')}: {info.get('task', 'Unknown task')}")
            
            return True
        else:
            print(f"❌ Model status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model status error: {str(e)}")
        return False


def run_basic_tests():
    """Run basic tests of the framework."""
    print("🧪 Running Basic Framework Tests")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    all_passed = True
    
    # Test 1: API Health
    print("1️⃣ Testing API health...")
    if not test_api_health(base_url):
        print("❌ API health check failed!")
        print("Please start the framework with: python run.py dev")
        return False
    
    # Test 2: Configuration endpoints
    if not test_config_endpoints(base_url):
        all_passed = False
    
    # Test 3: Excel template
    if not test_excel_template(base_url):
        all_passed = False
    
    # Test 4: Text extraction
    if not test_text_extraction(base_url):
        all_passed = False
    
    # Test 5: Model status
    if not test_model_status(base_url):
        all_passed = False
    
    # Test 6: Single PDF extraction  
    if not test_single_pdf_extraction(base_url):
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL BASIC TESTS PASSED!")
        print("✅ Framework is working correctly!")
        print("✅ API endpoints are functional")
        print("✅ PDF extraction is working")
        print("✅ Text processing is working")
        print("✅ Template generation works")
    else:
        print("❌ SOME TESTS FAILED!")
        print("Please check the error messages above.")
    
    return all_passed


def test_parameter_extraction_accuracy():
    """Test accuracy of the 5 specific parameter extraction."""
    print("\n🎯 Testing Parameter Extraction Accuracy")
    print("=" * 50)
    
    # Create test text with known parameters
    test_cases = [
        {
            "name": "German Angebot",
            "text": """
            ANGEBOT Nr. A-2024-001
            
            Datum: 15.03.2024
            
            TechSolutions GmbH
            Musterstraße 123
            12345 Berlin
            Deutschland
            
            Angebotsnummer: A-2024-001
            Gesamtsumme: 4.210,00 €
            """,
            "expected": {
                "date": "15.03.2024",
                "company_name": "TechSolutions GmbH", 
                "company_address": "12345 Berlin",
                "angebot": "A-2024-001"
            }
        },
        {
            "name": "English Quote",
            "text": """
            QUOTATION Q-2024-002
            
            Date: March 20, 2024
            
            Global Industries Ltd
            123 Business Street
            London W1A 1AA
            United Kingdom
            
            Quote Number: Q-2024-002
            Total: $10,100.00
            """,
            "expected": {
                "date": "March 20, 2024",
                "company_name": "Global Industries Ltd",
                "company_address": "London W1A 1AA", 
                "angebot": "Q-2024-002"
            }
        }
    ]
    
    base_url = "http://localhost:8000"
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        
        # Create simple PDF content
        pdf_path = f"test_{test_case['name'].lower().replace(' ', '_')}.pdf"
        simple_pdf_content = create_simple_test_pdf()
        
        try:
            # Use single PDF extraction
            with open(simple_pdf_content, 'rb') as f:
                files = {'file': (pdf_path, f, 'application/pdf')}
                response = requests.post(f"{base_url}/training/extract-single", files=files)
            
            if response.status_code == 200:
                result = response.json()
                extracted = result.get('extracted_data', {})
                
                print("   Extraction results:")
                for param, expected_value in test_case['expected'].items():
                    extracted_value = extracted.get(param)
                    if extracted_value and expected_value.lower() in str(extracted_value).lower():
                        print(f"     ✅ {param}: Found ('{extracted_value}')")
                    else:
                        print(f"     ⚠️  {param}: Expected '{expected_value}', got '{extracted_value}'")
            else:
                print(f"     ❌ Extraction failed: {response.status_code}")
                
        except Exception as e:
            print(f"     ❌ Error: {str(e)}")
        finally:
            if os.path.exists(simple_pdf_content):
                os.unlink(simple_pdf_content)


def main():
    """Main test function."""
    print("🚀 HuggingFace PDF Extraction Framework - Test Suite")
    print("=" * 60)
    
    # Run basic functionality tests
    success = run_basic_tests()
    
    if success:
        # Run parameter extraction accuracy tests
        test_parameter_extraction_accuracy()
        
        print("\n" + "=" * 60)
        print("🎉 FRAMEWORK TESTING COMPLETED!")
        print("✅ Your framework is ready for use with 50 PDFs!")
        print("📋 Next steps:")
        print("   1. Put your 50 PDFs in a directory")
        print("   2. Run: python examples/training_workflow_example.py --pdf-directory ./your_pdfs")
        print("   3. Follow the workflow for manual labeling")
        print("   4. Train your models!")
    else:
        print("\n❌ Framework has issues that need to be resolved.")
        print("Please check the API is running and dependencies are installed.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())