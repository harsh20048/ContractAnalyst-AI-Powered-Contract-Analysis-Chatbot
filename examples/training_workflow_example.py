#!/usr/bin/env python3
"""
Complete Training Workflow Example for PDF Parameter Extraction

This script demonstrates the complete workflow for your specific use case:
1. Extract 5 parameters from 50 PDFs (date, company name, address, tables, angebot)
2. Generate Excel file with extracted data
3. Manual labeling process
4. Training pipeline for model improvement

Usage:
    python training_workflow_example.py --pdf-directory ./your_pdfs --output-directory ./training_output
"""

import argparse
import requests
import json
import time
import os
from pathlib import Path
import pandas as pd


class TrainingWorkflowManager:
    """Manages the complete training workflow for PDF parameter extraction."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the workflow manager."""
        self.base_url = base_url.rstrip('/')
        
    def health_check(self):
        """Check if the service is running."""
        try:
            response = requests.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def step1_extract_batch_data(self, pdf_directory: str, output_excel: str):
        """
        Step 1: Extract training data from batch of PDFs.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_excel: Path to save extracted data Excel file
        """
        print("🔍 Step 1: Extracting data from PDF batch...")
        
        pdf_dir = Path(pdf_directory)
        if not pdf_dir.exists():
            raise FileNotFoundError(f"Directory {pdf_directory} not found")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"Found {len(pdf_files)} PDF files")
        
        if len(pdf_files) > 50:
            print(f"⚠️  More than 50 files found. Processing first 50 files.")
            pdf_files = pdf_files[:50]
        
        # Prepare files for upload
        files = []
        for pdf_file in pdf_files:
            files.append(('files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
        
        try:
            # Send batch extraction request
            response = requests.post(
                f"{self.base_url}/training/extract-batch",
                files=files,
                data={
                    'output_format': 'excel',
                    'include_raw_text': 'false'
                }
            )
            
            if response.status_code == 200:
                # Save Excel file
                with open(output_excel, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Extracted data saved to {output_excel}")
                return True
            else:
                print(f"❌ Extraction failed: {response.text}")
                return False
                
        finally:
            # Close all file handles
            for _, (_, file_handle, _) in files:
                file_handle.close()
    
    def step2_download_labeling_template(self, template_path: str):
        """
        Step 2: Download Excel template for manual labeling.
        
        Args:
            template_path: Path to save the template file
        """
        print("📝 Step 2: Downloading labeling template...")
        
        response = requests.get(f"{self.base_url}/training/download-template")
        
        if response.status_code == 200:
            with open(template_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ Template saved to {template_path}")
            return True
        else:
            print(f"❌ Failed to download template: {response.text}")
            return False
    
    def step3_merge_and_prepare_labels(self, extracted_excel: str, template_excel: str, 
                                      output_excel: str):
        """
        Step 3: Merge extracted data with template for manual labeling.
        
        Args:
            extracted_excel: Path to extracted data Excel
            template_excel: Path to template Excel
            output_excel: Path to save the merged file for labeling
        """
        print("🔄 Step 3: Preparing data for manual labeling...")
        
        try:
            # Read extracted data
            extracted_df = pd.read_excel(extracted_excel, sheet_name='Extracted_Data')
            
            # Read template to understand structure
            template_df = pd.read_excel(template_excel, sheet_name='Training_Data')
            
            # Create labeling DataFrame with extracted data as starting point
            labeling_df = pd.DataFrame()
            
            # Map the columns appropriately
            labeling_df['file_name'] = extracted_df['file_name']
            labeling_df['date'] = extracted_df['date']  # Pre-filled with extracted data
            labeling_df['company_name'] = extracted_df['company_name']
            labeling_df['company_address'] = extracted_df['company_address']
            labeling_df['angebot'] = extracted_df['angebot']
            labeling_df['tables_count'] = extracted_df['tables_count']
            labeling_df['notes'] = 'Please review and correct the extracted data'
            
            # Save to Excel with multiple sheets
            with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
                # Main labeling sheet
                labeling_df.to_excel(writer, sheet_name='Data_to_Label', index=False)
                
                # Instructions sheet
                instructions = pd.DataFrame({
                    'Instructions for Manual Labeling': [
                        '1. Review each row for accuracy',
                        '2. Correct any wrongly extracted data',
                        '3. Fill in missing data where possible',
                        '4. Date format: Use any readable format (DD.MM.YYYY, DD/MM/YYYY, etc.)',
                        '5. Company name: Include full legal name with GmbH, AG, etc.',
                        '6. Address: Complete address with postal code and city',
                        '7. Angebot: Quote number, offer ID, or description',
                        '8. Tables count: Number of tables found (for validation)',
                        '9. Notes: Add any observations about the document',
                        '',
                        '10. Save this file when complete and upload using training pipeline',
                        '',
                        'Quality Tips:',
                        '- Focus on accuracy over completeness',
                        '- Mark uncertain data in notes',
                        '- Use consistent formatting',
                        '- Double-check company names and addresses'
                    ]
                })
                instructions.to_excel(writer, sheet_name='Instructions', index=False)
                
                # Summary sheet
                summary = pd.DataFrame({
                    'Extraction Summary': [
                        f'Total Files: {len(labeling_df)}',
                        f'Date Extracted: {labeling_df["date"].notna().sum()}',
                        f'Company Name Extracted: {labeling_df["company_name"].notna().sum()}',
                        f'Address Extracted: {labeling_df["company_address"].notna().sum()}',
                        f'Angebot Extracted: {labeling_df["angebot"].notna().sum()}',
                        f'Tables Found: {labeling_df["tables_count"].sum()}',
                        '',
                        'Next Steps:',
                        '1. Review and correct data in "Data_to_Label" sheet',
                        '2. Save the file',
                        '3. Use the training pipeline to train models'
                    ]
                })
                summary.to_excel(writer, sheet_name='Summary', index=False)
            
            print(f"✅ Labeling file prepared: {output_excel}")
            print(f"📊 Ready for labeling: {len(labeling_df)} documents")
            print(f"📋 Pre-filled data:")
            print(f"   - Dates: {labeling_df['date'].notna().sum()}/{len(labeling_df)}")
            print(f"   - Company names: {labeling_df['company_name'].notna().sum()}/{len(labeling_df)}")
            print(f"   - Addresses: {labeling_df['company_address'].notna().sum()}/{len(labeling_df)}")
            print(f"   - Angebot: {labeling_df['angebot'].notna().sum()}/{len(labeling_df)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error preparing labeling file: {str(e)}")
            return False
    
    def step4_validate_labeled_data(self, labeled_excel: str):
        """
        Step 4: Validate manually labeled data.
        
        Args:
            labeled_excel: Path to manually labeled Excel file
        """
        print("✅ Step 4: Validating labeled data...")
        
        try:
            with open(labeled_excel, 'rb') as f:
                files = {'file': (Path(labeled_excel).name, f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
                
                response = requests.post(
                    f"{self.base_url}/training/upload-labeled-data",
                    files=files
                )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Labeled data validation successful")
                print(f"📊 Total rows: {result['total_rows']}")
                print(f"📋 Completeness:")
                for field, completeness in result['completeness'].items():
                    print(f"   - {field}: {completeness}")
                
                return True, result
            else:
                print(f"❌ Validation failed: {response.text}")
                return False, None
                
        except Exception as e:
            print(f"❌ Error validating labeled data: {str(e)}")
            return False, None
    
    def step5_start_training_pipeline(self, pdf_directory: str, labeled_excel: str, 
                                    output_directory: str):
        """
        Step 5: Start the complete training pipeline.
        
        Args:
            pdf_directory: Directory containing PDF files
            labeled_excel: Path to labeled Excel file
            output_directory: Directory to save training results
        """
        print("🚀 Step 5: Starting training pipeline...")
        
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))[:50]  # Limit to 50 files
        
        # Prepare PDF files
        files = []
        for pdf_file in pdf_files:
            files.append(('pdf_files', (pdf_file.name, open(pdf_file, 'rb'), 'application/pdf')))
        
        # Add labeled Excel file
        with open(labeled_excel, 'rb') as f:
            files.append(('labeled_excel', (Path(labeled_excel).name, f.read(), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')))
        
        try:
            response = requests.post(
                f"{self.base_url}/training/start-pipeline",
                files=files,
                data={'output_directory': output_directory}
            )
            
            if response.status_code == 200:
                result = response.json()
                job_id = result['job_id']
                
                print(f"✅ Training pipeline started")
                print(f"📋 Job ID: {job_id}")
                print(f"📄 Processing {result['pdf_count']} PDF files")
                
                return job_id
            else:
                print(f"❌ Failed to start training pipeline: {response.text}")
                return None
                
        finally:
            # Close file handles
            for _, (_, file_handle, _) in files[:-1]:  # PDF files
                if hasattr(file_handle, 'close'):
                    file_handle.close()
    
    def step6_monitor_training(self, job_id: str):
        """
        Step 6: Monitor training pipeline progress.
        
        Args:
            job_id: Training job ID
        """
        print(f"⏳ Step 6: Monitoring training progress for job {job_id}...")
        
        while True:
            try:
                response = requests.get(f"{self.base_url}/training/status/{job_id}")
                
                if response.status_code == 200:
                    status = response.json()
                    
                    print(f"📊 Status: {status['status']} ({status['progress']*100:.1f}%)")
                    print(f"💬 Message: {status['message']}")
                    
                    if status['status'] in ['completed', 'failed']:
                        if status['status'] == 'completed':
                            print("✅ Training completed successfully!")
                            result = status.get('result', {})
                            print(f"📈 Data quality score: {result.get('data_quality_score', 0):.2f}")
                            print(f"📊 Total samples: {result.get('total_samples', 0)}")
                            print(f"🤖 Models trained: {', '.join(result.get('trained_models', []))}")
                        else:
                            print(f"❌ Training failed: {status.get('error', 'Unknown error')}")
                        
                        return status['status'] == 'completed', status
                    
                    time.sleep(10)  # Wait 10 seconds before next check
                else:
                    print(f"❌ Error checking status: {response.text}")
                    return False, None
                    
            except Exception as e:
                print(f"❌ Error monitoring training: {str(e)}")
                return False, None
    
    def step7_download_results(self, job_id: str, results_path: str):
        """
        Step 7: Download training results.
        
        Args:
            job_id: Training job ID
            results_path: Path to save results ZIP file
        """
        print(f"📥 Step 7: Downloading training results...")
        
        try:
            response = requests.get(f"{self.base_url}/training/results/{job_id}")
            
            if response.status_code == 200:
                with open(results_path, 'wb') as f:
                    f.write(response.content)
                
                print(f"✅ Results downloaded to {results_path}")
                print(f"📦 Extract the ZIP file to view:")
                print(f"   - Trained models")
                print(f"   - Data quality report")
                print(f"   - Training metrics")
                print(f"   - Prepared datasets")
                
                return True
            else:
                print(f"❌ Failed to download results: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error downloading results: {str(e)}")
            return False
    
    def run_complete_workflow(self, pdf_directory: str, output_directory: str):
        """
        Run the complete training workflow.
        
        Args:
            pdf_directory: Directory containing PDF files
            output_directory: Directory to save all outputs
        """
        print("🚀 Starting Complete Training Workflow")
        print("=" * 50)
        
        # Create output directory
        output_dir = Path(output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        extracted_excel = output_dir / "01_extracted_data.xlsx"
        template_excel = output_dir / "02_labeling_template.xlsx"
        labeling_excel = output_dir / "03_data_for_labeling.xlsx"
        results_zip = output_dir / "04_training_results.zip"
        
        try:
            # Step 1: Extract batch data
            if not self.step1_extract_batch_data(pdf_directory, str(extracted_excel)):
                return False
            
            # Step 2: Download template
            if not self.step2_download_labeling_template(str(template_excel)):
                return False
            
            # Step 3: Prepare labeling file
            if not self.step3_merge_and_prepare_labels(
                str(extracted_excel), str(template_excel), str(labeling_excel)
            ):
                return False
            
            print("\n" + "="*50)
            print("⚠️  MANUAL STEP REQUIRED")
            print("="*50)
            print(f"📝 Please manually review and correct the data in:")
            print(f"   {labeling_excel}")
            print(f"")
            print(f"📋 Instructions:")
            print(f"   1. Open the Excel file")
            print(f"   2. Review the 'Data_to_Label' sheet")
            print(f"   3. Correct any extraction errors")
            print(f"   4. Fill in missing information")
            print(f"   5. Save the file")
            print(f"   6. Press Enter to continue...")
            
            input("Press Enter when manual labeling is complete...")
            
            # Step 4: Validate labeled data
            success, validation_result = self.step4_validate_labeled_data(str(labeling_excel))
            if not success:
                return False
            
            # Step 5: Start training
            job_id = self.step5_start_training_pipeline(
                pdf_directory, str(labeling_excel), str(output_dir / "training_output")
            )
            if not job_id:
                return False
            
            # Step 6: Monitor training
            success, training_status = self.step6_monitor_training(job_id)
            if not success:
                return False
            
            # Step 7: Download results
            if not self.step7_download_results(job_id, str(results_zip)):
                return False
            
            print("\n" + "="*50)
            print("🎉 WORKFLOW COMPLETED SUCCESSFULLY!")
            print("="*50)
            print(f"📁 All outputs saved to: {output_directory}")
            print(f"📊 Files created:")
            print(f"   - {extracted_excel.name}: Raw extracted data")
            print(f"   - {template_excel.name}: Labeling template")
            print(f"   - {labeling_excel.name}: Manually labeled data")
            print(f"   - {results_zip.name}: Complete training results")
            print(f"")
            print(f"🤖 Your models are now trained and ready to use!")
            
            return True
            
        except Exception as e:
            print(f"❌ Workflow failed: {str(e)}")
            return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Complete Training Workflow for PDF Parameter Extraction"
    )
    
    parser.add_argument(
        "--pdf-directory",
        required=True,
        help="Directory containing PDF files (max 50)"
    )
    
    parser.add_argument(
        "--output-directory", 
        default="./training_workflow_output",
        help="Directory to save all outputs"
    )
    
    parser.add_argument(
        "--api-url",
        default="http://localhost:8000",
        help="Base URL of the API server"
    )
    
    args = parser.parse_args()
    
    # Initialize workflow manager
    workflow = TrainingWorkflowManager(args.api_url)
    
    # Check if service is running
    if not workflow.health_check():
        print("❌ API service is not running!")
        print("Please start the service with: python run.py dev")
        return 1
    
    print("✅ API service is running")
    
    # Run complete workflow
    success = workflow.run_complete_workflow(
        args.pdf_directory,
        args.output_directory
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())