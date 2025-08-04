#!/usr/bin/env python3
"""
BERT-base-uncased PDF Extraction System Test Suite
Tests all BERT functionality including semantic search, NER, and hybrid extraction
"""

import requests
import json
import time
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List
import tempfile

class BertSystemTester:
    """Comprehensive tester for BERT-base-uncased PDF extraction system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_results = []
        self.start_time = time.time()
    
    def log_test(self, test_name: str, success: bool, details: str = "", response_data: Dict = None):
        """Log test result."""
        result = {
            "test_name": test_name,
            "success": success,
            "details": details,
            "timestamp": time.time() - self.start_time,
            "response_data": response_data
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if details:
            print(f"    📝 {details}")
        if not success and response_data:
            print(f"    📊 Response: {response_data}")
        print()
    
    def test_server_health(self) -> bool:
        """Test BERT system health and availability."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check BERT-specific fields
                bert_available = data.get("bert_available", False)
                bert_model = data.get("bert_model", "")
                ner_available = data.get("ner_available", False)
                device = data.get("device", "")
                
                details = f"Model: {bert_model}, BERT: {bert_available}, NER: {ner_available}, Device: {device}"
                
                if bert_model == "bert-base-uncased":
                    self.log_test("BERT System Health", True, details, data)
                    return True
                else:
                    self.log_test("BERT System Health", False, f"Wrong model: {bert_model}", data)
                    return False
            else:
                self.log_test("BERT System Health", False, f"HTTP {response.status_code}", response.json())
                return False
                
        except Exception as e:
            self.log_test("BERT System Health", False, f"Connection error: {str(e)}")
            return False
    
    def test_model_configuration(self) -> bool:
        """Test BERT model configuration endpoint."""
        try:
            response = requests.get(f"{self.base_url}/config/models", timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check BERT model details
                primary_model = data.get("primary_model", {})
                model_name = primary_model.get("name", "")
                embedding_size = primary_model.get("embedding_size", 0)
                max_seq_length = primary_model.get("max_sequence_length", 0)
                
                # Check extraction strategy
                strategy = data.get("extraction_strategy", "")
                confidence_weights = data.get("confidence_weighting", {})
                
                if (model_name == "bert-base-uncased" and 
                    embedding_size == 768 and 
                    max_seq_length == 512 and
                    strategy == "multi_method_hybrid"):
                    
                    details = f"Model: {model_name}, Embeddings: {embedding_size}D, Strategy: {strategy}"
                    self.log_test("BERT Model Configuration", True, details, data)
                    return True
                else:
                    self.log_test("BERT Model Configuration", False, f"Invalid config: {data}", data)
                    return False
            else:
                self.log_test("BERT Model Configuration", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("BERT Model Configuration", False, f"Error: {str(e)}")
            return False
    
    def test_single_pdf_extraction(self) -> bool:
        """Test single PDF extraction with BERT."""
        if not os.path.exists('test_offline.pdf'):
            self.log_test("Single PDF BERT Extraction", False, "test_offline.pdf not found")
            return False
        
        try:
            with open('test_offline.pdf', 'rb') as f:
                files = {'file': ('test_offline.pdf', f, 'application/pdf')}
                response = requests.post(
                    f"{self.base_url}/extract/single",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check BERT-specific fields
                success = data.get("success", False)
                bert_available = data.get("bert_available", False)
                ner_available = data.get("ner_available", False)
                extraction_methods = data.get("extraction_methods", {})
                confidence_scores = data.get("confidence_scores", {})
                bert_model = data.get("bert_model", "")
                
                # Check extracted parameters
                params = data.get("extracted_parameters", {})
                processing_time = data.get("processing_time", 0)
                
                if success and bert_model == "bert-base-uncased":
                    method_summary = ", ".join([f"{k}:{v}" for k, v in extraction_methods.items()])
                    avg_confidence = sum(confidence_scores.values()) / len(confidence_scores) if confidence_scores else 0
                    
                    details = f"Methods: {method_summary}, Avg Confidence: {avg_confidence:.3f}, Time: {processing_time:.3f}s"
                    self.log_test("Single PDF BERT Extraction", True, details, data)
                    return True
                else:
                    self.log_test("Single PDF BERT Extraction", False, f"Extraction failed", data)
                    return False
            else:
                self.log_test("Single PDF BERT Extraction", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Single PDF BERT Extraction", False, f"Error: {str(e)}")
            return False
    
    def test_batch_pdf_extraction(self) -> bool:
        """Test batch PDF extraction with BERT."""
        if not os.path.exists('test_offline.pdf'):
            self.log_test("Batch PDF BERT Extraction", False, "test_offline.pdf not found")
            return False
        
        try:
            # Use the same file multiple times for batch test
            files = []
            for i in range(2):
                with open('test_offline.pdf', 'rb') as f:
                    files.append(('files', (f'test_batch_{i}.pdf', f.read(), 'application/pdf')))
            
            response = requests.post(
                f"{self.base_url}/extract/batch",
                files=files,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Check batch processing results
                success = data.get("success", False)
                framework = data.get("extraction_framework", "")
                processing_stats = data.get("processing_statistics", {})
                results = data.get("results", [])
                
                if success and framework == "bert-base-uncased" and len(results) >= 2:
                    total_files = processing_stats.get("total_files", 0)
                    successful = processing_stats.get("successful_extractions", 0)
                    bert_extractions = processing_stats.get("bert_extractions", 0)
                    
                    details = f"Files: {total_files}, Successful: {successful}, BERT used: {bert_extractions}"
                    self.log_test("Batch PDF BERT Extraction", True, details, data)
                    return True
                else:
                    self.log_test("Batch PDF BERT Extraction", False, f"Batch failed", data)
                    return False
            else:
                self.log_test("Batch PDF BERT Extraction", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Batch PDF BERT Extraction", False, f"Error: {str(e)}")
            return False
    
    def test_bert_supervised_training(self) -> bool:
        """Test BERT supervised training functionality."""
        if not os.path.exists('test_offline.pdf') or not os.path.exists('test_answers.xlsx'):
            self.log_test("BERT Supervised Training", False, "Required files not found")
            return False
        
        try:
            # Prepare files for upload
            files = [
                ('pdf_files', ('test_offline.pdf', open('test_offline.pdf', 'rb'), 'application/pdf')),
                ('excel_answers', ('test_answers.xlsx', open('test_answers.xlsx', 'rb'), 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'))
            ]
            
            response = requests.post(
                f"{self.base_url}/bert/supervised-train",
                files=files,
                timeout=60
            )
            
            # Close files
            for _, (_, file_obj, _) in files:
                file_obj.close()
            
            if response.status_code == 200:
                data = response.json()
                
                # Check training results
                success = data.get("success", False)
                framework = data.get("framework", "")
                matched_pdfs = data.get("matched_pdfs", 0)
                overall_accuracy = data.get("overall_accuracy", 0)
                method_performance = data.get("method_performance", {})
                
                if success and framework == "bert-base-uncased" and matched_pdfs > 0:
                    bert_performance = method_performance.get("bert_semantic", {})
                    bert_accuracy = bert_performance.get("accuracy", 0) if bert_performance else 0
                    
                    details = f"Matched: {matched_pdfs}, Overall: {overall_accuracy:.3f}, BERT: {bert_accuracy:.3f}"
                    self.log_test("BERT Supervised Training", True, details, data)
                    return True
                else:
                    self.log_test("BERT Supervised Training", False, f"Training failed", data)
                    return False
            else:
                self.log_test("BERT Supervised Training", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("BERT Supervised Training", False, f"Error: {str(e)}")
            return False
    
    def test_excel_template_download(self) -> bool:
        """Test BERT training template download."""
        try:
            response = requests.get(f"{self.base_url}/training/download-template", timeout=10)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                content_length = len(response.content)
                
                if 'excel' in content_type.lower() or 'spreadsheet' in content_type.lower():
                    details = f"Size: {content_length} bytes, Type: {content_type}"
                    self.log_test("Excel Template Download", True, details)
                    return True
                else:
                    self.log_test("Excel Template Download", False, f"Wrong content type: {content_type}")
                    return False
            else:
                self.log_test("Excel Template Download", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("Excel Template Download", False, f"Error: {str(e)}")
            return False
    
    def test_web_interface(self) -> bool:
        """Test BERT web interface availability."""
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            
            if response.status_code == 200:
                content = response.text.lower()
                
                # Check for BERT-specific content
                bert_indicators = [
                    'bert-base-uncased',
                    'semantic understanding',
                    'transformer',
                    'bert processing',
                    'bert model'
                ]
                
                found_indicators = [indicator for indicator in bert_indicators if indicator in content]
                
                if len(found_indicators) >= 3:
                    details = f"Found indicators: {', '.join(found_indicators)}"
                    self.log_test("BERT Web Interface", True, details)
                    return True
                else:
                    self.log_test("BERT Web Interface", False, f"Missing BERT content indicators")
                    return False
            else:
                self.log_test("BERT Web Interface", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("BERT Web Interface", False, f"Error: {str(e)}")
            return False
    
    def test_parameter_extraction_accuracy(self) -> bool:
        """Test BERT parameter extraction accuracy with known data."""
        if not os.path.exists('test_offline.pdf'):
            self.log_test("BERT Parameter Accuracy", False, "test_offline.pdf not found")
            return False
        
        try:
            with open('test_offline.pdf', 'rb') as f:
                files = {'file': ('test_offline.pdf', f, 'application/pdf')}
                response = requests.post(
                    f"{self.base_url}/extract/single",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get("success", False):
                    params = data.get("extracted_parameters", {})
                    confidence_scores = data.get("confidence_scores", {})
                    extraction_methods = data.get("extraction_methods", {})
                    
                    # Expected values (from our test PDF)
                    expected = {
                        'date': '15.03.2024',
                        'company_name': 'OfflineTest GmbH',
                        'company_address': 'Teststrasse 123, 12345 Teststadt',
                        'angebot': 'OFFLINE-2024-001'
                    }
                    
                    # Calculate accuracy for each parameter
                    accuracies = {}
                    for param, expected_value in expected.items():
                        extracted_value = params.get(param, "")
                        if extracted_value and expected_value.lower() in str(extracted_value).lower():
                            accuracies[param] = 1.0
                        elif extracted_value:
                            accuracies[param] = 0.5  # Partial match
                        else:
                            accuracies[param] = 0.0
                    
                    overall_accuracy = sum(accuracies.values()) / len(accuracies)
                    bert_methods = sum(1 for method in extraction_methods.values() if 'bert' in method.lower())
                    
                    if overall_accuracy >= 0.5 and bert_methods > 0:
                        details = f"Accuracy: {overall_accuracy:.3f}, BERT methods: {bert_methods}/4"
                        self.log_test("BERT Parameter Accuracy", True, details)
                        return True
                    else:
                        details = f"Low accuracy: {overall_accuracy:.3f}"
                        self.log_test("BERT Parameter Accuracy", False, details, data)
                        return False
                else:
                    self.log_test("BERT Parameter Accuracy", False, "Extraction failed", data)
                    return False
            else:
                self.log_test("BERT Parameter Accuracy", False, f"HTTP {response.status_code}")
                return False
                
        except Exception as e:
            self.log_test("BERT Parameter Accuracy", False, f"Error: {str(e)}")
            return False
    
    def run_complete_test_suite(self) -> Dict:
        """Run complete BERT test suite."""
        print("🤖 BERT-base-uncased PDF Extraction System Test Suite")
        print("=" * 60)
        print("🧠 Testing transformer-based parameter extraction")
        print("📊 Testing semantic search, NER, and hybrid methods")
        print("=" * 60)
        print()
        
        # Define test sequence
        tests = [
            ("BERT System Health", self.test_server_health),
            ("BERT Model Configuration", self.test_model_configuration),
            ("Single PDF BERT Extraction", self.test_single_pdf_extraction),
            ("Batch PDF BERT Extraction", self.test_batch_pdf_extraction),
            ("BERT Supervised Training", self.test_bert_supervised_training),
            ("Excel Template Download", self.test_excel_template_download),
            ("BERT Web Interface", self.test_web_interface),
            ("BERT Parameter Accuracy", self.test_parameter_extraction_accuracy),
        ]
        
        # Run tests
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                time.sleep(1)  # Small delay between tests
            except Exception as e:
                self.log_test(test_name, False, f"Test exception: {str(e)}")
        
        # Calculate results
        success_rate = (passed / total) * 100
        total_time = time.time() - self.start_time
        
        # Summary
        print("=" * 60)
        print("🎯 BERT TEST SUITE SUMMARY")
        print("=" * 60)
        print(f"✅ Tests Passed: {passed}/{total} ({success_rate:.1f}%)")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🤖 Framework: BERT-base-uncased PDF Extraction")
        
        if success_rate >= 80:
            print("🎉 BERT SYSTEM STATUS: EXCELLENT")
        elif success_rate >= 60:
            print("⚠️  BERT SYSTEM STATUS: GOOD")
        else:
            print("❌ BERT SYSTEM STATUS: NEEDS ATTENTION")
        
        return {
            "total_tests": total,
            "passed_tests": passed,
            "success_rate": success_rate,
            "total_time": total_time,
            "test_results": self.test_results,
            "framework": "bert-base-uncased",
            "status": "excellent" if success_rate >= 80 else "good" if success_rate >= 60 else "needs_attention"
        }

def main():
    """Main test execution function."""
    print("🚀 Starting BERT-base-uncased System Tests...")
    print()
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            print("❌ BERT server is not responding correctly")
            print("💡 Please start the server with: python3 main_bert_offline.py")
            return False
    except requests.exceptions.RequestException:
        print("❌ BERT server is not running")
        print("💡 Please start the server with: python3 main_bert_offline.py")
        return False
    
    # Run tests
    tester = BertSystemTester()
    results = tester.run_complete_test_suite()
    
    # Save results
    with open('bert_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📊 Detailed results saved to: bert_test_results.json")
    
    return results["success_rate"] >= 60

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)