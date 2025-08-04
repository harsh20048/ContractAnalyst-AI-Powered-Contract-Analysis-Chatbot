#!/usr/bin/env python3
"""
Single PDF + Excel Answer Accuracy Test
Perfect for your data preparation approach: 1 PDF + 1 Excel sheet for accuracy rate
"""

import requests
import pandas as pd
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any

class SinglePDFAccuracyTester:
    """Test accuracy with 1 PDF + 1 Excel answer sheet."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {}
    
    def calculate_parameter_accuracy(self, expected: str, predicted: str) -> float:
        """Calculate accuracy between expected and predicted values."""
        if not expected and not predicted:
            return 1.0
        if not expected or not predicted:
            return 0.0
        
        expected_clean = str(expected).lower().strip()
        predicted_clean = str(predicted).lower().strip()
        
        # Exact match
        if expected_clean == predicted_clean:
            return 1.0
        
        # Containment (one contains the other)
        if expected_clean in predicted_clean or predicted_clean in expected_clean:
            return 0.8
        
        # Word overlap (good for BERT semantic understanding)
        expected_words = set(expected_clean.split())
        predicted_words = set(predicted_clean.split())
        
        if expected_words and predicted_words:
            overlap = len(expected_words & predicted_words)
            union = len(expected_words | predicted_words)
            jaccard_similarity = overlap / union if union > 0 else 0.0
            
            if jaccard_similarity >= 0.5:
                return 0.7
            elif jaccard_similarity >= 0.3:
                return 0.5
            elif jaccard_similarity > 0:
                return 0.3
        
        return 0.0
    
    def test_single_pdf_accuracy(self, pdf_path: str, excel_path: str) -> Dict[str, Any]:
        """Test accuracy with 1 PDF + 1 Excel answer sheet."""
        
        print(f"🤖 BERT-base-uncased Accuracy Test")
        print(f"📄 PDF: {pdf_path}")
        print(f"📊 Excel: {excel_path}")
        print("=" * 60)
        
        # Validate files exist
        if not os.path.exists(pdf_path):
            return {"error": f"PDF file not found: {pdf_path}"}
        
        if not os.path.exists(excel_path):
            return {"error": f"Excel file not found: {excel_path}"}
        
        try:
            # Load Excel answers
            print("📥 Loading Excel answers...")
            answers_df = pd.read_excel(excel_path)
            
            # Validate Excel structure
            required_columns = ['file_name', 'date', 'company_name', 'company_address', 'angebot']
            missing_columns = [col for col in required_columns if col not in answers_df.columns]
            if missing_columns:
                return {"error": f"Missing columns in Excel: {missing_columns}"}
            
            # Get PDF filename
            pdf_filename = os.path.basename(pdf_path)
            
            # Find matching row in Excel
            matching_rows = answers_df[answers_df['file_name'] == pdf_filename]
            if matching_rows.empty:
                return {"error": f"No matching row found for {pdf_filename} in Excel"}
            
            expected_answers = matching_rows.iloc[0]
            print(f"✅ Found expected answers for: {pdf_filename}")
            
            # Extract parameters using BERT
            print("🧠 Extracting with BERT-base-uncased...")
            start_time = time.time()
            
            with open(pdf_path, 'rb') as f:
                files = {'file': (pdf_filename, f, 'application/pdf')}
                response = requests.post(
                    f"{self.base_url}/extract/single",
                    files=files,
                    timeout=60
                )
            
            if response.status_code != 200:
                return {"error": f"BERT extraction failed: HTTP {response.status_code}"}
            
            bert_result = response.json()
            extraction_time = time.time() - start_time
            
            if not bert_result.get("success", False):
                return {"error": f"BERT extraction failed: {bert_result.get('error', 'Unknown error')}"}
            
            print(f"✅ BERT extraction completed in {extraction_time:.2f}s")
            
            # Get extracted parameters
            extracted_params = bert_result.get("extracted_parameters", {})
            confidence_scores = bert_result.get("confidence_scores", {})
            extraction_methods = bert_result.get("extraction_methods", {})
            
            # Calculate accuracy for each parameter
            parameter_results = {}
            overall_scores = []
            
            for param in ['date', 'company_name', 'company_address', 'angebot']:
                expected_value = str(expected_answers.get(param, "")).strip() if pd.notna(expected_answers.get(param)) else ""
                predicted_value = str(extracted_params.get(param, "")).strip()
                
                accuracy = self.calculate_parameter_accuracy(expected_value, predicted_value)
                confidence = confidence_scores.get(param, 0.0)
                method = extraction_methods.get(param, "unknown")
                
                parameter_results[param] = {
                    "expected": expected_value,
                    "predicted": predicted_value,
                    "accuracy": accuracy,
                    "confidence": confidence,
                    "extraction_method": method,
                    "match_status": self.get_match_status(accuracy)
                }
                
                overall_scores.append(accuracy)
                
                # Print parameter result
                status_emoji = "✅" if accuracy >= 0.8 else "⚠️" if accuracy >= 0.5 else "❌"
                print(f"{status_emoji} {param.upper()}:")
                print(f"   Expected: {expected_value}")
                print(f"   Predicted: {predicted_value}")
                print(f"   Accuracy: {accuracy:.1%}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Method: {method}")
                print()
            
            # Calculate overall metrics
            overall_accuracy = sum(overall_scores) / len(overall_scores)
            
            # Count BERT methods used
            bert_methods_count = sum(1 for method in extraction_methods.values() if 'bert' in method.lower())
            
            # Compile results
            results = {
                "pdf_file": pdf_filename,
                "excel_file": os.path.basename(excel_path),
                "overall_accuracy": overall_accuracy,
                "parameter_results": parameter_results,
                "extraction_time": extraction_time,
                "bert_model": bert_result.get("bert_model", "bert-base-uncased"),
                "bert_available": bert_result.get("bert_available", False),
                "ner_available": bert_result.get("ner_available", False),
                "bert_methods_used": bert_methods_count,
                "total_parameters": len(parameter_results),
                "excellent_matches": sum(1 for r in parameter_results.values() if r["accuracy"] >= 0.8),
                "good_matches": sum(1 for r in parameter_results.values() if 0.5 <= r["accuracy"] < 0.8),
                "poor_matches": sum(1 for r in parameter_results.values() if r["accuracy"] < 0.5),
                "average_confidence": sum(r["confidence"] for r in parameter_results.values()) / len(parameter_results),
                "performance_grade": self.get_performance_grade(overall_accuracy)
            }
            
            # Print summary
            print("=" * 60)
            print("🎯 ACCURACY SUMMARY")
            print("=" * 60)
            print(f"📊 Overall Accuracy: {overall_accuracy:.1%}")
            print(f"🎯 Performance Grade: {results['performance_grade']}")
            print(f"🧠 BERT Methods Used: {bert_methods_count}/4 parameters")
            print(f"⚡ Processing Time: {extraction_time:.2f} seconds")
            print(f"✅ Excellent Matches: {results['excellent_matches']}/4")
            print(f"⚠️  Good Matches: {results['good_matches']}/4")
            print(f"❌ Poor Matches: {results['poor_matches']}/4")
            print(f"🔮 Average Confidence: {results['average_confidence']:.3f}")
            print("=" * 60)
            
            return results
            
        except Exception as e:
            return {"error": f"Test failed: {str(e)}"}
    
    def get_match_status(self, accuracy: float) -> str:
        """Get match status based on accuracy."""
        if accuracy >= 0.8:
            return "excellent"
        elif accuracy >= 0.5:
            return "good"
        else:
            return "poor"
    
    def get_performance_grade(self, accuracy: float) -> str:
        """Get performance grade based on overall accuracy."""
        if accuracy >= 0.9:
            return "A+ (Excellent)"
        elif accuracy >= 0.8:
            return "A (Very Good)"
        elif accuracy >= 0.7:
            return "B (Good)"
        elif accuracy >= 0.6:
            return "C (Fair)"
        elif accuracy >= 0.5:
            return "D (Poor)"
        else:
            return "F (Failed)"
    
    def save_results(self, results: Dict[str, Any], output_file: str = "accuracy_results.json"):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"📁 Results saved to: {output_file}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")
    
    def create_detailed_report(self, results: Dict[str, Any], report_file: str = "accuracy_report.md"):
        """Create detailed markdown report."""
        try:
            with open(report_file, 'w') as f:
                f.write(f"# BERT-base-uncased PDF Extraction Accuracy Report\n\n")
                f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"## Test Files\n")
                f.write(f"- **PDF:** {results['pdf_file']}\n")
                f.write(f"- **Excel:** {results['excel_file']}\n\n")
                
                f.write(f"## Overall Performance\n")
                f.write(f"- **Overall Accuracy:** {results['overall_accuracy']:.1%}\n")
                f.write(f"- **Performance Grade:** {results['performance_grade']}\n")
                f.write(f"- **Processing Time:** {results['extraction_time']:.2f} seconds\n")
                f.write(f"- **BERT Model:** {results['bert_model']}\n")
                f.write(f"- **BERT Methods Used:** {results['bert_methods_used']}/4 parameters\n\n")
                
                f.write(f"## Parameter Details\n\n")
                for param, details in results['parameter_results'].items():
                    f.write(f"### {param.upper()}\n")
                    f.write(f"- **Expected:** {details['expected']}\n")
                    f.write(f"- **Predicted:** {details['predicted']}\n")
                    f.write(f"- **Accuracy:** {details['accuracy']:.1%}\n")
                    f.write(f"- **Confidence:** {details['confidence']:.3f}\n")
                    f.write(f"- **Method:** {details['extraction_method']}\n")
                    f.write(f"- **Status:** {details['match_status']}\n\n")
                
                f.write(f"## Summary Statistics\n")
                f.write(f"- **Excellent Matches (≥80%):** {results['excellent_matches']}/4\n")
                f.write(f"- **Good Matches (50-79%):** {results['good_matches']}/4\n")
                f.write(f"- **Poor Matches (<50%):** {results['poor_matches']}/4\n")
                f.write(f"- **Average Confidence:** {results['average_confidence']:.3f}\n")
            
            print(f"📋 Detailed report saved to: {report_file}")
        except Exception as e:
            print(f"❌ Failed to create report: {e}")

def main():
    """Main function for single PDF accuracy testing."""
    
    if len(sys.argv) < 3:
        print("🤖 BERT-base-uncased Single PDF Accuracy Tester")
        print("=" * 60)
        print("Usage: python3 single_pdf_accuracy_test.py <pdf_file> <excel_file>")
        print()
        print("Examples:")
        print("  python3 single_pdf_accuracy_test.py my_document.pdf my_answers.xlsx")
        print("  python3 single_pdf_accuracy_test.py test_offline.pdf test_answers.xlsx")
        print()
        print("Excel Requirements:")
        print("  - Must have columns: file_name, date, company_name, company_address, angebot")
        print("  - file_name must match the PDF filename exactly")
        print()
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    excel_path = sys.argv[2]
    
    # Check if BERT server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ BERT server is not responding correctly")
            print("💡 Please start the server with: ./start_bert_system.sh")
            sys.exit(1)
    except requests.exceptions.RequestException:
        print("❌ BERT server is not running")
        print("💡 Please start the server with: ./start_bert_system.sh")
        sys.exit(1)
    
    # Run accuracy test
    tester = SinglePDFAccuracyTester()
    results = tester.test_single_pdf_accuracy(pdf_path, excel_path)
    
    # Check for errors
    if "error" in results:
        print(f"❌ Test failed: {results['error']}")
        sys.exit(1)
    
    # Save results
    tester.save_results(results)
    tester.create_detailed_report(results)
    
    print("\n🎉 Accuracy test completed successfully!")
    print(f"📊 Final Score: {results['overall_accuracy']:.1%} - {results['performance_grade']}")

if __name__ == "__main__":
    main()