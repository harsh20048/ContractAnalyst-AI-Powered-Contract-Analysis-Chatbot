#!/bin/bash
# Complete Offline Setup for Mistral 7B PDF Pipeline
# This script downloads everything needed to run completely offline

echo "🔒 Setting up Mistral 7B PDF Pipeline for OFFLINE operation"
echo "=========================================================="

# Configuration
MISTRAL_MODEL_DIR="./models/mistral-7b-instruct-v0.1"
CACHE_DIR="./models/cache"
NLTK_DATA_DIR="./models/nltk_data"

# Create directories
mkdir -p "$MISTRAL_MODEL_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$NLTK_DATA_DIR"
mkdir -p "./models/tokenizers"
mkdir -p "./data/sample_pdfs"

echo "📁 Created offline directories"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install huggingface_hub if not present
echo "📦 Installing offline tools..."
pip install --break-system-packages huggingface_hub datasets nltk

echo ""
echo "🤖 Step 1: Downloading Mistral 7B Instruct v0.1..."
echo "=================================================="

# Check if model already exists
if [ -f "$MISTRAL_MODEL_DIR/config.json" ]; then
    echo "✅ Mistral 7B model already exists at $MISTRAL_MODEL_DIR"
else
    echo "📥 Downloading Mistral 7B Instruct v0.1 (this will take 15-30 minutes)..."
    
    # Download with huggingface-cli for better reliability
    if command_exists huggingface-cli; then
        huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 \
            --local-dir "$MISTRAL_MODEL_DIR" \
            --local-dir-use-symlinks False
    else
        # Fallback to Python script
        python3 << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "mistralai/Mistral-7B-Instruct-v0.1"
local_dir = "$MISTRAL_MODEL_DIR"

print("📥 Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(local_dir)

print("📥 Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    low_cpu_mem_usage=True
)
model.save_pretrained(local_dir)

print("✅ Mistral 7B downloaded successfully!")
EOF
    fi
fi

echo ""
echo "📚 Step 2: Downloading NLTK Data..."
echo "==================================="

# Download NLTK data for offline use
export NLTK_DATA="$NLTK_DATA_DIR"
python3 << EOF
import nltk
import os

# Set NLTK data path for offline use
nltk_data_dir = "$NLTK_DATA_DIR"
nltk.data.path.append(nltk_data_dir)

# Download essential NLTK data
datasets = [
    'punkt',
    'punkt_tab', 
    'stopwords',
    'averaged_perceptron_tagger',
    'averaged_perceptron_tagger_eng',
    'wordnet',
    'omw-1.4',
    'vader_lexicon',
    'brown',
    'names'
]

for dataset in datasets:
    try:
        print(f"📥 Downloading {dataset}...")
        nltk.download(dataset, download_dir=nltk_data_dir, quiet=True)
        print(f"✅ {dataset} downloaded")
    except Exception as e:
        print(f"⚠️  Failed to download {dataset}: {e}")

print("✅ NLTK data setup complete")
EOF

echo ""
echo "🔧 Step 3: Setting up Offline Configuration..."
echo "=============================================="

# Create offline configuration
cat > "./config/offline_config.py" << 'EOF'
"""
Offline configuration for Mistral 7B PDF Pipeline
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODEL_DIR / "cache"
NLTK_DATA_DIR = MODEL_DIR / "nltk_data"

# Offline model paths
MISTRAL_MODEL_PATH = MODEL_DIR / "mistral-7b-instruct-v0.1"

# Environment variables for offline operation
OFFLINE_CONFIG = {
    "HF_HOME": str(CACHE_DIR),
    "TRANSFORMERS_CACHE": str(CACHE_DIR / "transformers"),
    "HF_DATASETS_CACHE": str(CACHE_DIR / "datasets"),
    "NLTK_DATA": str(NLTK_DATA_DIR),
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

def setup_offline_environment():
    """Set up environment variables for offline operation."""
    for key, value in OFFLINE_CONFIG.items():
        os.environ[key] = value
    
    # Create cache directories
    for path in OFFLINE_CONFIG.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print("🔒 Offline environment configured")

def verify_offline_setup():
    """Verify that all required files are available offline."""
    checks = []
    
    # Check Mistral model files
    model_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    for file in model_files:
        file_path = MISTRAL_MODEL_PATH / file
        checks.append(("Mistral " + file, file_path.exists()))
    
    # Check NLTK data
    nltk_datasets = ["punkt", "stopwords", "averaged_perceptron_tagger", "wordnet"]
    for dataset in nltk_datasets:
        dataset_path = NLTK_DATA_DIR / "tokenizers" / dataset
        checks.append((f"NLTK {dataset}", dataset_path.exists() or (NLTK_DATA_DIR / "corpora" / dataset).exists()))
    
    # Print results
    print("\n🔍 Offline Setup Verification:")
    print("=" * 40)
    all_good = True
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 All offline components are ready!")
    else:
        print("\n⚠️  Some components missing - check downloads")
    
    return all_good

if __name__ == "__main__":
    setup_offline_environment()
    verify_offline_setup()
EOF

echo ""
echo "📄 Step 4: Creating Sample Test PDFs..."
echo "======================================="

# Create sample PDFs for testing
python3 << 'EOF'
import os
from pathlib import Path

def create_test_pdf(filename, content_data):
    """Create a simple test PDF with content."""
    
    pdf_content = f"""
{content_data['title']}

Datum: {content_data['date']}

{content_data['company_name']}
{content_data['company_address']}

{content_data['angebot_text']} {content_data['angebot']}

Position | Beschreibung | Preis
1 | Service A | 1.000,00 €
2 | Service B | 2.000,00 €

Vielen Dank für Ihr Interesse.
"""
    
    # Create minimal PDF structure
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
{' 0 -15 Td '.join([f'({line.replace("(", "").replace(")", "")}' for line in pdf_content.split('\n') if line.strip()])} Tj
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
    
    with open(filename, 'wb') as f:
        f.write(pdf_structure.encode('utf-8'))
    
    print(f"✅ Created {filename}")

# Create sample PDFs
sample_docs = [
    {
        'filename': './data/sample_pdfs/offline_test_1.pdf',
        'title': 'ANGEBOT OFFLINE TEST',
        'date': '15.03.2024',
        'company_name': 'OfflineTest GmbH',
        'company_address': 'Teststraße 123\n12345 Teststadt\nDeutschland',
        'angebot_text': 'Angebotsnummer:',
        'angebot': 'OFFLINE-2024-001'
    },
    {
        'filename': './data/sample_pdfs/offline_test_2.pdf',
        'title': 'QUOTATION OFFLINE',
        'date': '20.02.2024',
        'company_name': 'Offline Industries AG',
        'company_address': 'Offline Plaza 1\n1010 Vienna\nAustria',
        'angebot_text': 'Quote Number:',
        'angebot': 'Q-OFFLINE-002'
    }
]

for doc in sample_docs:
    create_test_pdf(doc['filename'], doc)

print("✅ Sample test PDFs created")
EOF

echo ""
echo "📊 Step 5: Creating Offline Test Excel..."
echo "========================================"

# Create sample Excel for testing
python3 << 'EOF'
import pandas as pd

# Create sample training data
sample_data = {
    'file_name': [
        'offline_test_1.pdf',
        'offline_test_2.pdf'
    ],
    'date': [
        '15.03.2024',
        '20.02.2024'
    ],
    'company_name': [
        'OfflineTest GmbH',
        'Offline Industries AG'
    ],
    'company_address': [
        'Teststraße 123, 12345 Teststadt',
        'Offline Plaza 1, 1010 Vienna'
    ],
    'angebot': [
        'OFFLINE-2024-001',
        'Q-OFFLINE-002'
    ],
    'tables_count': [2, 2],
    'notes': [
        'Offline test document 1',
        'Offline test document 2'
    ]
}

df = pd.DataFrame(sample_data)
df.to_excel('./data/excel/answers/offline_test_answers.xlsx', index=False)

print("✅ Created offline test Excel with answers")
EOF

echo ""
echo "⚙️  Step 6: Configuring Offline Environment..."
echo "=============================================="

# Set up offline environment variables in .env
cat >> .env << 'EOF'

# Offline Configuration
OFFLINE_MODE=True
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
HF_DATASETS_OFFLINE=1
HF_HOME=./models/cache
TRANSFORMERS_CACHE=./models/cache/transformers
HF_DATASETS_CACHE=./models/cache/datasets
NLTK_DATA=./models/nltk_data

# Offline Model Paths
MISTRAL_MODEL_PATH=./models/mistral-7b-instruct-v0.1
EOF

echo "✅ Updated .env for offline operation"

echo ""
echo "🧪 Step 7: Setting up Offline Verification..."
echo "============================================="

# Create offline verification script
cat > "./scripts/verify_offline.py" << 'EOF'
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
EOF

chmod +x "./scripts/verify_offline.py"

echo ""
echo "🔒 Step 8: Final Setup..."
echo "========================"

# Create offline startup script  
cat > "./scripts/start_offline.sh" << 'EOF'
#!/bin/bash
# Start Mistral PDF Extraction Pipeline in OFFLINE mode

echo "🔒 Starting Mistral PDF Pipeline in OFFLINE MODE"
echo "================================================"

# Set offline environment variables
export OFFLINE_MODE=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=./models/cache
export TRANSFORMERS_CACHE=./models/cache/transformers
export HF_DATASETS_CACHE=./models/cache/datasets
export NLTK_DATA=./models/nltk_data

echo "🔒 Offline environment configured"

# Verify offline setup first
echo "🧪 Verifying offline components..."
python3 ./scripts/verify_offline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting offline server..."
    python3 main_simple.py
else
    echo "❌ Offline verification failed. Please run setup_offline.sh first."
    exit 1
fi
EOF

chmod +x "./scripts/start_offline.sh"

echo "✅ Created offline startup script"

# Verify the setup
echo ""
echo "🔍 Step 9: Initial Verification..."
echo "=================================="

python3 ./config/offline_config.py

echo ""
echo "=========================================================="
echo "🎉 OFFLINE SETUP COMPLETE!"
echo "=========================================================="
echo ""
echo "📋 What was downloaded/configured:"
echo "✅ Mistral 7B Instruct v0.1 model (~13GB)"
echo "✅ NLTK data for text processing"
echo "✅ Offline configuration files"
echo "✅ Sample test PDFs and Excel files"
echo "✅ Verification and startup scripts"
echo ""
echo "🚀 Next steps:"
echo "1. Run verification: python3 scripts/verify_offline.py"
echo "2. Start offline:    ./scripts/start_offline.sh"
echo "3. Test offline:     python3 tests/test_supervised.py"
echo ""
echo "🔒 Your pipeline is now ready for COMPLETELY OFFLINE operation!"
EOF

chmod +x scripts/setup_offline.sh