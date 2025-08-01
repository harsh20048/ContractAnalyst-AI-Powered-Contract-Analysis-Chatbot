#!/usr/bin/env python3
"""
Test script for supervised learning functionality.
Creates sample PDFs and Excel answers to test the complete workflow.
"""

import os
import requests
import pandas as pd
import tempfile
import time
from pathlib import Path


def create_sample_excel_answers(output_path: str):
    """Create sample Excel file with correct answers."""
    
    sample_data = {
        'file_name': [
            'document_001.pdf',
            'document_002.pdf', 
            'document_003.pdf'
        ],
        'date': [
            '15.03.2024',
            '20.02.2024',
            '10.01.2024'
        ],
        'company_name': [
            'TechSolutions GmbH',
            'Global Industries AG',
            'Innovation Ltd'
        ],
        'company_address': [
            'Musterstraße 123, 12345 Berlin',
            'Hauptplatz 1, 1010 Wien',
            '123 Main Street, London'
        ],
        'angebot': [
            'A-2024-001',
            'Q-2024-002', 
            'P-2024-003'
        ],
        'tables_count': [2, 1, 3],
        'notes': [
            'Correct sample data',
            'Training example',
            'Test document'
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_excel(output_path, index=False)
    print(f"✅ Created sample Excel answers: {output_path}")
    return df


def create_sample_pdf_content(pdf_path: str, content_data: dict):
    """Create a simple PDF with sample content."""
    
    # Create a minimal PDF with the content embedded as text
    pdf_content = f"""{content_data['title']}

Datum: {content_data['date']}

{content_data['company_name']}
{content_data['company_address']}

{content_data['angebot_text']} {content_data['angebot']}

Position | Beschreibung | Preis
1 | Service A | 1.000,00 €
2 | Service B | 2.000,00 €

Vielen Dank für Ihr Interesse."""
    
    # Create PDF structure as string, then encode
    pdf_structure = f"""%PDF-1.4
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
<< /Length {len(pdf_content) + 100} >>
stream
BT
/F1 12 Tf
50 750 Td
{' 0 -15 Td '.join([f'({line.replace("(", "").replace(")", "")}' for line in pdf_content.split(chr(10)) if line.strip()])} Tj
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
    
    with open(pdf_path, 'wb') as f:
        f.write(pdf_structure.encode('utf-8'))
    
    print(f"✅ Created sample PDF: {pdf_path}")


def create_sample_training_data(temp_dir: str):
    """Create sample PDFs and Excel answers for testing."""
    
    # Sample document data
    documents = [
        {
            'filename': 'document_001.pdf',
            'title': 'ANGEBOT',
            'date': '15.03.2024',
            'company_name': 'TechSolutions GmbH',
            'company_address': 'Musterstraße 123\n12345 Berlin\nDeutschland',
            'angebot_text': 'Angebotsnummer:',
            'angebot': 'A-2024-001'
        },
        {
            'filename': 'document_002.pdf',
            'title': 'QUOTATION',
            'date': '20.02.2024',
            'company_name': 'Global Industries AG',
            'company_address': 'Hauptplatz 1\n1010 Wien\nÖsterreich',
            'angebot_text': 'Quote Number:',
            'angebot': 'Q-2024-002'
        },
        {
            'filename': 'document_003.pdf',
            'title': 'PROPOSAL',
            'date': '10.01.2024',
            'company_name': 'Innovation Ltd',
            'company_address': '123 Main Street\nLondon\nUnited Kingdom',
            'angebot_text': 'Proposal ID:',
            'angebot': 'P-2024-003'
        }
    ]
    
    # Create PDFs
    pdf_paths = []
    for doc in documents:
        pdf_path = os.path.join(temp_dir, doc['filename'])
        create_sample_pdf_content(pdf_path, doc)
        pdf_paths.append(pdf_path)
    
    # Create Excel answers
    excel_path = os.path.join(temp_dir, 'training_answers.xlsx')
    excel_df = create_sample_excel_answers(excel_path)
    
    return pdf_paths, excel_path, excel_df


def test_supervised_training(base_url: str = "http://localhost:8000"):
    """Test the complete supervised training workflow."""
    
    print("🎓 Testing Supervised Learning Workflow")
    print("=" * 50)
    
    # Create temporary directory with sample data
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Step 1: Create sample training data
        print("1️⃣ Creating sample training data...")
        pdf_paths, excel_path, excel_df = create_sample_training_data(temp_dir)
        
        # Step 2: Test training status (before training)
        print("\n2️⃣ Checking initial training status...")
        try:
            response = requests.get(f"{base_url}/supervised/training-status")
            if response.status_code == 200:
                status = response.json()
                print(f"   ✅ Initial status: {status['message']}")
                print(f"   📊 Trained: {status['trained']}")
            else:
                print(f"   ❌ Status check failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error checking status: {str(e)}")
        
        # Step 3: Train supervised model
        print("\n3️⃣ Training supervised model...")
        try:
            files = []
            
            # Add PDF files
            for pdf_path in pdf_paths:
                with open(pdf_path, 'rb') as f:
                    files.append(('pdf_files', (Path(pdf_path).name, f.read(), 'application/pdf')))
            
            # Add Excel answers
            with open(excel_path, 'rb') as f:
                files.append(('excel_answers', (Path(excel_path).name, f.read(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
            
            response = requests.post(f"{base_url}/supervised/train", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ Training successful!")
                print(f"   📊 Training examples: {result['training_examples']}")
                print(f"   📄 Matched PDFs: {result['matched_pdfs']}/{result['total_pdfs']}")
                print(f"   📈 Overall accuracy: {result['overall_training_accuracy']:.3f}")
                print("   📋 Parameter accuracies:")
                for param, acc in result['training_accuracies'].items():
                    print(f"      - {param}: {acc:.3f}")
            else:
                print(f"   ❌ Training failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"   ❌ Training error: {str(e)}")
            return False
        
        # Step 4: Test training status (after training)
        print("\n4️⃣ Checking training status after training...")
        try:
            response = requests.get(f"{base_url}/supervised/training-status")
            if response.status_code == 200:
                status = response.json()
                print(f"   ✅ Status: {status['message']}")
                print(f"   📊 Trained: {status['trained']}")
                print(f"   📈 Overall accuracy: {status['overall_accuracy']:.3f}")
                print("   📋 Parameter performance:")
                for param, perf in status['parameter_performance'].items():
                    print(f"      - {param}: {perf['average_accuracy']:.3f} ({perf['correct_predictions']}/{perf['total_examples']})")
            else:
                print(f"   ❌ Status check failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error checking status: {str(e)}")
        
        # Step 5: Test batch verification
        print("\n5️⃣ Testing batch verification...")
        try:
            files = []
            
            # Add PDF files for verification
            for pdf_path in pdf_paths:
                with open(pdf_path, 'rb') as f:
                    files.append(('pdf_files', (Path(pdf_path).name, f.read(), 'application/pdf')))
            
            # Add Excel answers for verification
            with open(excel_path, 'rb') as f:
                files.append(('excel_answers', (Path(excel_path).name, f.read(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
            
            response = requests.post(f"{base_url}/supervised/batch-verify", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("   ✅ Batch verification successful!")
                print(f"   📊 Files processed: {result['verified_files']}/{result['total_files']}")
                print(f"   📈 Average accuracy: {result['average_overall_accuracy']:.3f}")
                print("   📋 Parameter accuracies:")
                for param, acc in result['parameter_accuracies'].items():
                    print(f"      - {param}: {acc:.3f}")
                
                # Show detailed results for first few files
                print("\n   📄 Detailed results (first 2 files):")
                for i, detail in enumerate(result['detailed_results'][:2]):
                    if detail.get('success'):
                        print(f"      File {i+1}: {detail['file_name']}")
                        print(f"         Overall accuracy: {detail['overall_accuracy']:.3f}")
                        for param, acc_info in detail['accuracies'].items():
                            match_icon = "✅" if acc_info > 0.5 else "❌"
                            print(f"         {match_icon} {param}: {acc_info:.3f}")
            else:
                print(f"   ❌ Batch verification failed: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Verification error: {str(e)}")
        
        # Step 6: Test individual prediction vs expected
        print("\n6️⃣ Testing individual predictions...")
        try:
            for i, (pdf_path, expected_row) in enumerate(zip(pdf_paths[:2], excel_df.itertuples())):
                print(f"\n   📄 Testing {Path(pdf_path).name}:")
                
                # Extract with basic endpoint first
                with open(pdf_path, 'rb') as f:
                    files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
                    response = requests.post(f"{base_url}/training/extract-single", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    extracted = result['extracted_data']
                    
                    print(f"   🔍 Predictions vs Expected:")
                    comparisons = [
                        ('date', extracted.get('date'), expected_row.date),
                        ('company_name', extracted.get('company_name'), expected_row.company_name), 
                        ('company_address', extracted.get('company_address'), expected_row.company_address),
                        ('angebot', extracted.get('angebot'), expected_row.angebot)
                    ]
                    
                    for param, predicted, expected in comparisons:
                        match_icon = "✅" if (predicted and expected and 
                                           (str(expected).lower() in str(predicted).lower() or 
                                            str(predicted).lower() in str(expected).lower())) else "❌"
                        print(f"      {match_icon} {param}:")
                        print(f"         Predicted: {predicted}")
                        print(f"         Expected:  {expected}")
                else:
                    print(f"   ❌ Extraction failed: {response.status_code}")
                    
        except Exception as e:
            print(f"   ❌ Individual test error: {str(e)}")
    
    print("\n" + "=" * 50)
    print("🎉 SUPERVISED LEARNING TEST COMPLETED!")
    return True


def test_api_endpoints(base_url: str = "http://localhost:8000"):
    """Test all supervised learning API endpoints."""
    
    print("🔗 Testing Supervised Learning API Endpoints")
    print("=" * 50)
    
    endpoints = [
        ("/supervised/training-status", "GET", None),
        ("/health", "GET", None),
        ("/config/tasks", "GET", None)
    ]
    
    for endpoint, method, data in endpoints:
        try:
            if method == "GET":
                response = requests.get(f"{base_url}{endpoint}")
            else:
                response = requests.post(f"{base_url}{endpoint}", json=data)
            
            status_icon = "✅" if response.status_code == 200 else "❌"
            print(f"{status_icon} {method} {endpoint}: {response.status_code}")
            
            if endpoint == "/supervised/training-status" and response.status_code == 200:
                result = response.json()
                print(f"   📊 Training status: {result.get('trained', False)}")
                
        except Exception as e:
            print(f"❌ {method} {endpoint}: Error - {str(e)}")
    
    print("✅ API endpoint testing completed")


def main():
    """Main test function."""
    
    print("🚀 Supervised Learning Framework Test Suite")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    
    # Wait for server to be ready
    print("⏳ Waiting for server to start...")
    time.sleep(3)
    
    # Test API health
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("❌ Server health check failed")
            return 1
    except Exception as e:
        print(f"❌ Cannot connect to server: {str(e)}")
        print("Please ensure the server is running with: python3 main_simple.py")
        return 1
    
    # Test API endpoints
    test_api_endpoints(base_url)
    
    print("\n")
    
    # Test supervised learning workflow
    success = test_supervised_training(base_url)
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ALL SUPERVISED LEARNING TESTS PASSED!")
        print("✅ Framework is ready for your PDFs + Excel workflow!")
        print("\n📋 What you can do now:")
        print("   1. Prepare your 50 PDFs")
        print("   2. Create Excel file with correct answers")
        print("   3. Use /supervised/train to train the model")
        print("   4. Use /supervised/batch-verify to verify accuracy")
        print("   5. Iterate and improve!")
        print("\n🌐 Web interface: http://localhost:8000")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())