#!/bin/bash

# BERT-base-uncased PDF Extraction System Startup Script
# Comprehensive startup with dependency checking and system information

echo "🤖 BERT-base-uncased PDF Extraction System Startup"
echo "=" * 60

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        echo "✅ Python 3 found: $PYTHON_VERSION"
        return 0
    else
        echo "❌ Python 3 not found"
        return 1
    fi
}

# Function to check if port is in use
check_port() {
    if lsof -i:8000 >/dev/null 2>&1; then
        echo "⚠️  Port 8000 is already in use"
        echo "🔄 Attempting to stop existing process..."
        pkill -f "main_bert_offline.py" 2>/dev/null || true
        sleep 2
        if lsof -i:8000 >/dev/null 2>&1; then
            echo "❌ Could not free port 8000"
            return 1
        else
            echo "✅ Port 8000 is now available"
            return 0
        fi
    else
        echo "✅ Port 8000 is available"
        return 0
    fi
}

# Function to check PyTorch installation
check_pytorch() {
    echo "🔍 Checking PyTorch installation..."
    python3 -c "import torch; print(f'✅ PyTorch {torch.__version__} found')" 2>/dev/null && return 0
    echo "❌ PyTorch not found - required for BERT"
    return 1
}

# Function to check transformers library
check_transformers() {
    echo "🔍 Checking transformers library..."
    python3 -c "import transformers; print(f'✅ Transformers {transformers.__version__} found')" 2>/dev/null && return 0
    echo "❌ Transformers library not found - required for BERT"
    return 1
}

# Function to check BERT model availability
check_bert_model() {
    echo "🔍 Checking BERT-base-uncased model..."
    python3 -c "
from transformers import BertTokenizer, BertModel
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
    model = BertModel.from_pretrained('bert-base-uncased', local_files_only=True)
    print('✅ BERT-base-uncased model found locally')
except:
    print('⚠️  BERT-base-uncased model will be downloaded on first run')
" 2>/dev/null
}

# Function to check GPU availability
check_gpu() {
    echo "🔍 Checking GPU availability..."
    python3 -c "
import torch
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else 'Unknown'
    print(f'✅ GPU available: {gpu_name} ({gpu_count} device(s))')
    print(f'   CUDA version: {torch.version.cuda}')
else:
    print('ℹ️  No GPU detected - will use CPU')
    print('   📝 For better performance, consider using a GPU-enabled environment')
" 2>/dev/null
}

# Function to install dependencies
install_dependencies() {
    echo "📦 Installing BERT system dependencies..."
    if [ -f "requirements_bert.txt" ]; then
        echo "📥 Installing from requirements_bert.txt..."
        python3 -m pip install -r requirements_bert.txt --break-system-packages --quiet
        if [ $? -eq 0 ]; then
            echo "✅ Dependencies installed successfully"
            return 0
        else
            echo "❌ Failed to install dependencies"
            return 1
        fi
    else
        echo "⚠️  requirements_bert.txt not found, installing core dependencies..."
        python3 -m pip install --break-system-packages --quiet \
            torch transformers tokenizers numpy scikit-learn \
            fastapi uvicorn python-multipart PyPDF2 pdfplumber \
            pandas openpyxl aiofiles python-dotenv pydantic requests
        return $?
    fi
}

# Function to create test files if missing
create_test_files() {
    echo "🔍 Checking test files..."
    
    # Create test PDF if missing
    if [ ! -f "test_offline.pdf" ]; then
        echo "📄 Creating test PDF..."
        python3 -c "
pdf_content = '''%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj  
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 5 0 R/Resources<</ProcSet[/PDF/Text]/Font<</F1 4 0 R>>>>>>endobj
4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
5 0 obj<</Length 200>>stream
BT
/F1 12 Tf
72 720 Td
(BERT TEST DOCUMENT) Tj
0 -20 Td  
(Date: 15.03.2024) Tj
0 -20 Td
(OfflineTest GmbH) Tj
0 -20 Td
(Teststrasse 123, 12345 Teststadt) Tj
0 -20 Td
(Quote Number: OFFLINE-2024-001) Tj
0 -20 Td
(Position | Description | Price) Tj
0 -20 Td
(1 | Service A | 1000 EUR) Tj
0 -20 Td
(2 | Service B | 2000 EUR) Tj
ET
endstream
endobj
xref
0 6
0000000000 65535 f
0000000015 00000 n
0000000068 00000 n
0000000125 00000 n
0000000281 00000 n
0000000348 00000 n
trailer<</Size 6/Root 1 0 R>>
startxref
596
%%EOF'''
with open('test_offline.pdf', 'wb') as f:
    f.write(pdf_content.encode('utf-8'))
print('✅ Created test_offline.pdf')
"
    else
        echo "✅ test_offline.pdf exists"
    fi
    
    # Create test Excel if missing
    if [ ! -f "test_answers.xlsx" ]; then
        echo "📊 Creating test Excel answers..."
        python3 -c "
import pandas as pd
sample_data = {
    'file_name': ['test_offline.pdf'],
    'date': ['15.03.2024'],
    'company_name': ['OfflineTest GmbH'],
    'company_address': ['Teststrasse 123, 12345 Teststadt'],
    'angebot': ['OFFLINE-2024-001'],
    'tables_count': [2],
    'notes': ['BERT test data']
}
df = pd.DataFrame(sample_data)
df.to_excel('test_answers.xlsx', index=False)
print('✅ Created test_answers.xlsx')
"
    else
        echo "✅ test_answers.xlsx exists"
    fi
}

# Main startup sequence
echo "🔍 Running pre-flight checks..."
echo

# Check Python
if ! check_python; then
    echo "💡 Please install Python 3.8 or higher"
    exit 1
fi

# Check main script
if [ ! -f "main_bert_offline.py" ]; then
    echo "❌ main_bert_offline.py not found"
    echo "💡 Please ensure you're in the correct directory"
    exit 1
fi
echo "✅ main_bert_offline.py found"

# Check port availability
if ! check_port; then
    echo "💡 Please stop any services using port 8000"
    exit 1
fi

# Check dependencies
echo
echo "🔍 Checking BERT dependencies..."
if ! check_pytorch || ! check_transformers; then
    echo "📦 Installing missing dependencies..."
    if ! install_dependencies; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    # Re-check after installation
    check_pytorch
    check_transformers
fi

# Check BERT model
check_bert_model

# Check GPU
check_gpu

# Create test files
echo
create_test_files

# Display system information
echo
echo "=" * 60
echo "🤖 BERT-base-uncased System Information"
echo "=" * 60
echo "🧠 Model: bert-base-uncased (110M parameters)"
echo "📏 Embedding size: 768 dimensions"
echo "📝 Max sequence length: 512 tokens"
echo "🔧 Framework: Hybrid BERT + NER + Regex"
echo "🌐 Web interface: http://localhost:8000"
echo "📚 API documentation: http://localhost:8000/docs"
echo "🏥 Health check: http://localhost:8000/health"
echo "📊 Model info: http://localhost:8000/config/models"
echo "=" * 60

# Start the server
echo "🚀 Starting BERT-base-uncased PDF extraction server..."
echo
echo "📋 Quick Commands:"
echo "   Health Check:    curl http://localhost:8000/health"
echo "   Test Single PDF: python3 test_bert_system.py"
echo "   Stop Server:     Ctrl+C"
echo
echo "🔄 Server starting..."
echo

# Set environment variables for offline mode
export HF_HOME="./models/cache"
export TRANSFORMERS_CACHE="./models/cache/transformers"
export HF_DATASETS_CACHE="./models/cache/datasets"
export NLTK_DATA="./models/nltk_data"
export TRANSFORMERS_OFFLINE="1"
export HF_HUB_OFFLINE="1"
export HF_DATASETS_OFFLINE="1"
export OFFLINE_MODE="True"

# Start the server
python3 main_bert_offline.py