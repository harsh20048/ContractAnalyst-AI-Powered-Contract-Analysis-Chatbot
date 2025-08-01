#!/usr/bin/env python3
"""
Verify that all components work offline
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_offline_env():
    """Set up offline environment variables."""
    offline_vars = {
        "HF_HOME": "./models/cache",
        "TRANSFORMERS_CACHE": "./models/cache/transformers", 
        "HF_DATASETS_CACHE": "./models/cache/datasets",
        "NLTK_DATA": "./models/nltk_data",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    
    for key, value in offline_vars.items():
        os.environ[key] = value

def test_imports():
    """Test that all required imports work offline."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import nltk
        print(f"✅ NLTK {nltk.__version__}")
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False
        
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__}")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    return True

def test_model_loading():
    """Test that Mistral model can be loaded offline."""
    print("\n🤖 Testing Mistral model loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_path = "./models/mistral-7b-instruct-v0.1"
        
        if not Path(model_path).exists():
            print(f"❌ Model not found at {model_path}")
            return False
        
        print("   📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        print("   ✅ Tokenizer loaded successfully")
        
        print("   🧠 Loading model (this may take a moment)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        print("   ✅ Model loaded successfully")
        
        # Test tokenization
        test_text = "Hello, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"   🔤 Tokenization test: '{test_text}' -> {len(tokens)} tokens -> '{decoded}'")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_pdf_processing():
    """Test PDF processing capabilities."""
    print("\n📄 Testing PDF processing...")
    
    try:
        import PyPDF2
        import pdfplumber
        
        # Test with sample PDF
        sample_pdf = "./data/sample_pdfs/offline_test_1.pdf"
        
        if not Path(sample_pdf).exists():
            print(f"❌ Sample PDF not found: {sample_pdf}")
            return False
        
        # Test PyPDF2
        with open(sample_pdf, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = reader.pages[0].extract_text()
            print(f"   ✅ PyPDF2 extracted {len(text)} characters")
        
        # Test pdfplumber
        with pdfplumber.open(sample_pdf) as pdf:
            text = pdf.pages[0].extract_text()
            print(f"   ✅ pdfplumber extracted {len(text)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        return False

def test_excel_processing():
    """Test Excel processing capabilities."""
    print("\n📊 Testing Excel processing...")
    
    try:
        import pandas as pd
        
        # Test reading sample Excel
        excel_file = "./data/excel/answers/offline_test_answers.xlsx"
        
        if not Path(excel_file).exists():
            print(f"❌ Sample Excel not found: {excel_file}")
            return False
        
        df = pd.read_excel(excel_file)
        print(f"   ✅ Read Excel with {len(df)} rows and {len(df.columns)} columns")
        
        # Test writing Excel
        test_data = {'test': [1, 2, 3], 'data': ['a', 'b', 'c']}
        test_df = pd.DataFrame(test_data)
        
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=True) as tmp:
            test_df.to_excel(tmp.name, index=False)
            verification_df = pd.read_excel(tmp.name)
            print(f"   ✅ Excel write/read test successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Excel processing failed: {e}")
        return False

def test_nltk_offline():
    """Test NLTK offline capabilities."""
    print("\n📚 Testing NLTK offline...")
    
    try:
        import nltk
        
        # Test tokenization
        from nltk.tokenize import word_tokenize, sent_tokenize
        
        test_text = "Hello world. This is a test sentence for NLTK."
        
        words = word_tokenize(test_text)
        sentences = sent_tokenize(test_text)
        
        print(f"   ✅ Tokenization: {len(words)} words, {len(sentences)} sentences")
        
        # Test stopwords
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        print(f"   ✅ Stopwords: {len(stop_words)} English stopwords loaded")
        
        return True
        
    except Exception as e:
        print(f"❌ NLTK offline test failed: {e}")
        return False

def main():
    """Run all offline verification tests."""
    print("🔒 Offline Setup Verification")
    print("=" * 50)
    
    # Setup offline environment
    setup_offline_env()
    
    tests = [
        ("Import Tests", test_imports),
        ("Model Loading", test_model_loading),
        ("PDF Processing", test_pdf_processing), 
        ("Excel Processing", test_excel_processing),
        ("NLTK Offline", test_nltk_offline),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 ALL TESTS PASSED! Offline setup is complete and working!")
        print("🔒 You can now run the pipeline completely offline")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    exit(main())
