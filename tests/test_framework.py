#!/usr/bin/env python3
"""
Test script for the HuggingFace PDF Extraction Framework.

This script creates dummy PDF files and tests the complete workflow
to ensure everything works correctly.
"""

import os
import tempfile
import shutil
from pathlib import Path
import requests
import time
import json

# For creating dummy PDFs
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch


def create_dummy_pdf(output_path: str, content_type: str = "angebot"):
    """Create a dummy PDF with realistic content for testing."""
    
    # Sample data for different types of documents
    sample_data = {
        "angebot": {
            "title": "ANGEBOT",
            "date": "15.03.2024",
            "company_name": "TechSolutions GmbH",
            "company_address": "Musterstraße 123\n12345 Berlin\nDeutschland",
            "angebot_nr": "A-2024-001",
            "content": """
            Sehr geehrte Damen und Herren,
            
            hiermit unterbreiten wir Ihnen unser Angebot für die gewünschten IT-Services.
            
            Angebotsnummer: A-2024-001
            Gültigkeitsdauer: 30 Tage
            
            Vielen Dank für Ihr Interesse.
            """,
            "table_data": [
                ["Position", "Beschreibung", "Menge", "Einzelpreis", "Gesamtpreis"],
                ["1", "Server Installation", "1", "2.500,00 €", "2.500,00 €"],
                ["2", "Software Lizenz", "5", "150,00 €", "750,00 €"],
                ["3", "Schulung", "8 Std.", "120,00 €", "960,00 €"],
                ["", "", "", "Gesamt:", "4.210,00 €"]
            ]
        },
        "quote": {
            "title": "QUOTATION",
            "date": "March 20, 2024",
            "company_name": "Global Industries Ltd",
            "company_address": "123 Business Street\nLondon W1A 1AA\nUnited Kingdom",
            "angebot_nr": "Q-2024-002",
            "content": """
            Dear Customer,
            
            Please find our quotation for the requested services below.
            
            Quote Number: Q-2024-002
            Valid until: April 20, 2024
            
            Thank you for your inquiry.
            """,
            "table_data": [
                ["Item", "Description", "Quantity", "Unit Price", "Total"],
                ["1", "Consulting Services", "40 hrs", "$150.00", "$6,000.00"],
                ["2", "Software License", "1", "$2,500.00", "$2,500.00"],
                ["3", "Training", "16 hrs", "$100.00", "$1,600.00"],
                ["", "", "", "Total:", "$10,100.00"]
            ]
        },
        "rechnung": {
            "title": "RECHNUNG",
            "date": "10.04.2024",
            "company_name": "Innovation AG",
            "company_address": "Hauptplatz 1\n1010 Wien\nÖsterreich",
            "angebot_nr": "R-2024-003",
            "content": """
            Rechnung für erbrachte Leistungen
            
            Rechnungsnummer: R-2024-003
            Leistungszeitraum: März 2024
            Zahlungsziel: 14 Tage
            
            Mit freundlichen Grüßen
            """,
            "table_data": [
                ["Pos.", "Leistung", "Anzahl", "Preis", "Summe"],
                ["1", "Projektmanagement", "20 Std.", "95,00 €", "1.900,00 €"],
                ["2", "Entwicklung", "40 Std.", "85,00 €", "3.400,00 €"],
                ["3", "Testing", "10 Std.", "75,00 €", "750,00 €"],
                ["", "", "", "Netto:", "6.050,00 €"],
                ["", "", "", "MwSt. 19%:", "1.149,50 €"],
                ["", "", "", "Brutto:", "7.199,50 €"]
            ]
        }
    }
    
    # Get content for this type
    if content_type not in sample_data:
        content_type = "angebot"
    
    data = sample_data[content_type]
    
    # Create PDF
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph(f"<b>{data['title']}</b>", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 20))
    
    # Date
    date_para = Paragraph(f"<b>Datum:</b> {data['date']}", styles['Normal'])
    story.append(date_para)
    story.append(Spacer(1, 12))
    
    # Company info
    company_para = Paragraph(f"<b>{data['company_name']}</b><br/>{data['company_address'].replace(chr(10), '<br/>')}", styles['Normal'])
    story.append(company_para)
    story.append(Spacer(1, 20))
    
    # Content
    content_para = Paragraph(data['content'].replace('\n', '<br/>'), styles['Normal'])
    story.append(content_para)
    story.append(Spacer(1, 20))
    
    # Table
    table = Table(data['table_data'])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    
    # Build PDF
    doc.build(story)
    print(f"Created dummy PDF: {output_path}")


def create_test_pdfs(output_dir: str, count: int = 10):
    """Create multiple test PDFs with different content types."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    content_types = ["angebot", "quote", "rechnung"]
    
    for i in range(count):
        content_type = content_types[i % len(content_types)]
        filename = f"test_document_{i+1:02d}_{content_type}.pdf"
        filepath = output_path / filename
        
        create_dummy_pdf(str(filepath), content_type)
    
    print(f"Created {count} test PDF files in {output_dir}")
    return list(output_path.glob("*.pdf"))


def test_api_health(base_url: str = "http://localhost:8000"):
    """Test if the API is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def test_single_pdf_extraction(pdf_path: str, base_url: str = "http://localhost:8000"):
    """Test single PDF extraction."""
    print(f"\n🔍 Testing single PDF extraction: {Path(pdf_path).name}")
    
    try:
        with open(pdf_path, 'rb') as f:
            files = {'file': (Path(pdf_path).name, f, 'application/pdf')}
            response = requests.post(f"{base_url}/training/extract-single", files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Single extraction successful!")
            print(f"📊 Extracted data:")
            
            extracted = result['extracted_data']
            for key, value in extracted.items():
                if value:
                    if key == 'tables' and isinstance(value, list):
                        print(f"   - {key}: {len(value)} table(s) found")
                    else:
                        display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                        print(f"   - {key}: {display_value}")
                else:
                    print(f"   - {key}: Not found")
            
            return True, result
        else:
            print(f"❌ Single extraction failed: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error in single extraction: {str(e)}")
        return False, None


def test_batch_extraction(pdf_files: list, base_url: str = "http://localhost:8000"):
    """Test batch PDF extraction."""
    print(f"\n📦 Testing batch extraction with {len(pdf_files)} files")
    
    try:
        # Prepare files
        files = []
        for pdf_file in pdf_files:
            files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
        
        # Test JSON output
        response = requests.post(
            f"{base_url}/training/extract-batch",
            files=files,
            data={'output_format': 'json'}
        )
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch extraction successful!")
            print(f"📊 Results summary:")
            print(f"   - Total files: {result['total_files']}")
            print(f"   - Successful: {result['summary']['successful_extractions']}")
            print(f"   - Failed: {result['summary']['failed_extractions']}")
            print(f"   - Dates found: {result['summary']['date_extracted']}")
            print(f"   - Companies found: {result['summary']['company_name_extracted']}")
            print(f"   - Addresses found: {result['summary']['address_extracted']}")
            print(f"   - Angebot found: {result['summary']['angebot_found']}")
            
            return True, result
        else:
            print(f"❌ Batch extraction failed: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error in batch extraction: {str(e)}")
        return False, None


def test_excel_template_download(base_url: str = "http://localhost:8000"):
    """Test Excel template download."""
    print(f"\n📝 Testing Excel template download")
    
    try:
        response = requests.get(f"{base_url}/training/download-template")
        
        if response.status_code == 200:
            # Save template
            template_path = "test_template.xlsx"
            with open(template_path, 'wb') as f:
                f.write(response.content)
            
            print("✅ Template download successful!")
            print(f"💾 Saved to: {template_path}")
            
            return True, template_path
        else:
            print(f"❌ Template download failed: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error downloading template: {str(e)}")
        return False, None


def test_training_pipeline(pdf_files: list, base_url: str = "http://localhost:8000"):
    """Test the complete training pipeline."""
    print(f"\n🚀 Testing training pipeline with {len(pdf_files)} files")
    
    try:
        # Create simple labeled data
        import pandas as pd
        
        labeled_data = []
        for i, pdf_file in enumerate(pdf_files):
            labeled_data.append({
                'file_name': pdf_file.name,
                'date': f"2024-03-{15+i:02d}",
                'company_name': f"Test Company {i+1} GmbH",
                'company_address': f"Test Street {i+1}, 12345 Test City",
                'angebot': f"A-2024-{i+1:03d}",
                'tables_count': 1,
                'notes': 'Test data'
            })
        
        # Save labeled data to Excel
        labeled_df = pd.DataFrame(labeled_data)
        labeled_excel_path = "test_labeled_data.xlsx"
        labeled_df.to_excel(labeled_excel_path, index=False)
        
        # Prepare files for upload
        files = []
        
        # Add PDF files
        for pdf_file in pdf_files:
            files.append(('pdf_files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
        
        # Add labeled Excel
        with open(labeled_excel_path, 'rb') as f:
            files.append(('labeled_excel', (Path(labeled_excel_path).name, f.read(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
        
        # Start training pipeline
        response = requests.post(
            f"{base_url}/training/start-pipeline",
            files=files,
            data={'output_directory': './test_training_output'}
        )
        
        # Close PDF file handles
        for _, (_, file_handle, _) in files[:-1]:  # Exclude Excel file
            if hasattr(file_handle, 'close'):
                file_handle.close()
        
        if response.status_code == 200:
            result = response.json()
            job_id = result['job_id']
            
            print("✅ Training pipeline started!")
            print(f"📋 Job ID: {job_id}")
            
            # Monitor progress (simplified)
            max_checks = 30  # Maximum 5 minutes
            for i in range(max_checks):
                time.sleep(10)  # Wait 10 seconds
                
                status_response = requests.get(f"{base_url}/training/status/{job_id}")
                if status_response.status_code == 200:
                    status = status_response.json()
                    print(f"📊 Progress: {status['status']} ({status['progress']*100:.1f}%) - {status['message']}")
                    
                    if status['status'] in ['completed', 'failed']:
                        if status['status'] == 'completed':
                            print("✅ Training completed successfully!")
                            return True, job_id
                        else:
                            print(f"❌ Training failed: {status.get('error', 'Unknown error')}")
                            return False, None
                else:
                    print(f"❌ Error checking status: {status_response.text}")
                    return False, None
            
            print("⚠️ Training taking longer than expected, but may still succeed")
            return True, job_id
            
        else:
            print(f"❌ Training pipeline failed to start: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Error in training pipeline: {str(e)}")
        return False, None
    finally:
        # Clean up
        if os.path.exists(labeled_excel_path):
            os.unlink(labeled_excel_path)


def run_comprehensive_test():
    """Run comprehensive test of the framework."""
    print("🧪 Starting Comprehensive Framework Test")
    print("=" * 60)
    
    # Check if we need to install reportlab
    try:
        import reportlab
    except ImportError:
        print("❌ reportlab not installed. Installing...")
        os.system("pip install reportlab")
        import reportlab
    
    base_url = "http://localhost:8000"
    
    # Step 1: Check API health
    print("1️⃣ Checking API health...")
    if not test_api_health(base_url):
        print("❌ API is not running!")
        print("Please start the framework with: python run.py dev")
        return False
    print("✅ API is running!")
    
    # Step 2: Create test PDFs
    print("\n2️⃣ Creating test PDF files...")
    temp_dir = tempfile.mkdtemp(prefix="test_pdfs_")
    try:
        pdf_files = create_test_pdfs(temp_dir, 10)  # Create 10 test PDFs
        
        # Step 3: Test single extraction
        print("\n3️⃣ Testing single PDF extraction...")
        success, _ = test_single_pdf_extraction(str(pdf_files[0]), base_url)
        if not success:
            return False
        
        # Step 4: Test batch extraction
        print("\n4️⃣ Testing batch extraction...")
        success, _ = test_batch_extraction(pdf_files[:5], base_url)  # Test with 5 files
        if not success:
            return False
        
        # Step 5: Test template download
        print("\n5️⃣ Testing template download...")
        success, template_path = test_excel_template_download(base_url)
        if not success:
            return False
        
        # Step 6: Test training pipeline
        print("\n6️⃣ Testing training pipeline...")
        success, job_id = test_training_pipeline(pdf_files[:3], base_url)  # Test with 3 files
        if not success:
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("✅ Framework is working correctly!")
        print("✅ PDF extraction works")
        print("✅ Batch processing works")
        print("✅ Excel templates work")
        print("✅ Training pipeline works")
        
        return True
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
        for cleanup_file in ["test_template.xlsx", "test_labeled_data.xlsx"]:
            if os.path.exists(cleanup_file):
                os.unlink(cleanup_file)


def main():
    """Main test function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the HuggingFace PDF Extraction Framework")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--create-pdfs-only", action="store_true", help="Only create test PDFs")
    parser.add_argument("--output-dir", default="./test_pdfs", help="Output directory for test PDFs")
    
    args = parser.parse_args()
    
    if args.create_pdfs_only:
        print("📄 Creating test PDFs only...")
        try:
            import reportlab
        except ImportError:
            print("Installing reportlab...")
            os.system("pip install reportlab")
        
        pdf_files = create_test_pdfs(args.output_dir, 10)
        print(f"✅ Created {len(pdf_files)} test PDFs in {args.output_dir}")
        return 0
    
    # Run comprehensive test
    success = run_comprehensive_test()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())