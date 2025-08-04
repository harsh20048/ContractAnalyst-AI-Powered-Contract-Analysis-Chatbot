#!/usr/bin/env python3
"""
Create Your Excel Answer Template
Perfect for your data preparation: 1 PDF + 1 Excel answer sheet
"""

import pandas as pd
import sys
import os

def create_excel_template(pdf_filename: str, output_excel: str = None):
    """Create an Excel template for a specific PDF file."""
    
    if output_excel is None:
        # Generate Excel filename based on PDF
        base_name = os.path.splitext(pdf_filename)[0]
        output_excel = f"{base_name}_answers.xlsx"
    
    print(f"📄 Creating Excel template for: {pdf_filename}")
    print(f"📊 Output Excel file: {output_excel}")
    
    # Create template data
    template_data = {
        'file_name': [pdf_filename],  # Exact PDF filename
        'date': ['DD.MM.YYYY or MM/DD/YYYY'],  # Date format example
        'company_name': ['Company Name Here'],  # Company name
        'company_address': ['Full Address Here'],  # Complete address
        'angebot': ['Quote/Proposal Number'],  # Angebot/quote number
        'tables_count': [0],  # Number of tables (optional)
        'notes': ['Add any notes here']  # Optional notes
    }
    
    # Create DataFrame
    df = pd.DataFrame(template_data)
    
    # Save to Excel
    try:
        df.to_excel(output_excel, index=False)
        print(f"✅ Excel template created: {output_excel}")
        print()
        print("📋 Next Steps:")
        print(f"1. Open {output_excel} in Excel or LibreOffice")
        print("2. Replace the example values with the CORRECT answers from your PDF")
        print("3. Save the file")
        print(f"4. Run: python3 single_pdf_accuracy_test.py {pdf_filename} {output_excel}")
        print()
        print("🎯 IMPORTANT:")
        print("- file_name must match your PDF filename EXACTLY")
        print("- Provide accurate answers for BERT to measure accuracy")
        print("- Empty fields are okay if the PDF doesn't contain that information")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create Excel template: {e}")
        return False

def main():
    """Main function to create Excel template."""
    
    if len(sys.argv) < 2:
        print("📊 Excel Template Creator for BERT Accuracy Testing")
        print("=" * 60)
        print("Usage: python3 create_your_excel_template.py <your_pdf_file.pdf> [output_excel.xlsx]")
        print()
        print("Examples:")
        print("  python3 create_your_excel_template.py invoice_001.pdf")
        print("  python3 create_your_excel_template.py contract.pdf contract_answers.xlsx")
        print()
        print("This will create an Excel template with the correct structure for accuracy testing.")
        sys.exit(1)
    
    pdf_filename = sys.argv[1]
    output_excel = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if PDF exists (optional warning)
    if not os.path.exists(pdf_filename):
        print(f"⚠️  Warning: PDF file '{pdf_filename}' not found in current directory")
        print("   (You can still create the template and move the PDF later)")
        print()
    
    # Create template
    success = create_excel_template(pdf_filename, output_excel)
    
    if success:
        print("\n🎉 Template created successfully!")
        print("📝 Fill in the correct answers and test with BERT!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()