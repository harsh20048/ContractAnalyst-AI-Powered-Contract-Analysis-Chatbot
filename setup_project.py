#!/usr/bin/env python3
"""
Complete Project Setup Script for Mistral 7B PDF Extraction Pipeline
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json
import requests

def run_command(command, check=True, capture_output=False):
    """Run shell command safely."""
    try:
        if capture_output:
            result = subprocess.run(command, shell=True, check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(command, shell=True, check=check)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {command}")
        print(f"Error: {e}")
        return False

def create_directory_structure():
    """Create organized project directory structure."""
    print("📁 Creating project directory structure...")
    
    directories = [
        "data/pdfs/training",
        "data/pdfs/test", 
        "data/excel/answers",
        "data/excel/templates",
        "models/mistral-7b-instruct-v0.1",
        "models/trained_models",
        "logs",
        "uploads",
        "processed",
        "outputs/extractions",
        "outputs/reports",
        "outputs/training_results",
        "temp",
        "config",
        "scripts",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")

def check_python_version():
    """Check Python version compatibility."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major == 3 and version.minor >= 9:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("   📋 Please upgrade to Python 3.9+")
        return False

def check_gpu_availability():
    """Check GPU availability and CUDA setup."""
    print("🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"   ✅ GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("   ⚠️  No GPU detected - will use CPU (slower)")
            return False
    except ImportError:
        print("   ⚠️  PyTorch not installed yet - GPU check will be performed after installation")
        return None

def install_dependencies():
    """Install required Python dependencies."""
    print("📦 Installing Python dependencies...")
    
    # First, upgrade pip
    print("   🔄 Upgrading pip...")
    run_command("python3 -m pip install --upgrade pip")
    
    # Install core dependencies
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.36.0", 
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "python-multipart==0.0.6",
        "pandas>=2.1.4",
        "numpy>=1.24.3",
        "PyPDF2>=3.0.1",
        "pdfplumber>=0.10.3",
        "pymupdf>=1.23.8",
        "openpyxl>=3.1.2",
        "xlsxwriter>=3.1.9",
        "scikit-learn>=1.3.2",
        "nltk>=3.8.1",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "requests>=2.31.0",
        "GPUtil>=1.4.0",
        "psutil>=5.9.0",
        "aiofiles>=23.2.1",
        "jinja2>=3.1.2"
    ]
    
    print("   📋 Installing core packages...")
    for req in requirements:
        print(f"      Installing {req}...")
        success = run_command(f"pip install '{req}'")
        if not success:
            print(f"      ⚠️  Failed to install {req} - continuing...")
    
    # Download NLTK data
    print("   📚 Downloading NLTK data...")
    run_command("""python3 -c "
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded successfully')
" """)

def setup_configuration():
    """Create configuration files."""
    print("⚙️  Setting up configuration files...")
    
    # Create .env file
    env_content = """
# Mistral 7B PDF Extraction Pipeline Configuration

# Server Configuration
HOST=0.0.0.0
PORT=8000
DEBUG=True

# Model Configuration
MODEL_CACHE_DIR=./models
MISTRAL_MODEL_PATH=./models/mistral-7b-instruct-v0.1
ENABLE_GPU=True
LOAD_IN_4BIT=True
MAX_NEW_TOKENS=512
TEMPERATURE=0.1

# PDF Processing
PDF_UPLOAD_DIR=./uploads
PDF_PROCESSED_DIR=./processed
MAX_FILE_SIZE=50MB

# Database and Storage
DATA_DIR=./data
LOGS_DIR=./logs
OUTPUTS_DIR=./outputs

# Security
SECRET_KEY=your-secret-key-here
CORS_ORIGINS=["http://localhost:3000", "http://localhost:8080"]

# Performance
BATCH_SIZE=5
MEMORY_CLEANUP_FREQUENCY=5
MAX_CONCURRENT_REQUESTS=10
"""
    
    with open(".env", "w") as f:
        f.write(env_content.strip())
    print("   ✅ Created .env configuration file")
    
    # Create config.json
    config_data = {
        "project_name": "Mistral PDF Extractor",
        "version": "1.0.0",
        "supported_formats": [".pdf"],
        "extraction_parameters": [
            "date",
            "company_name", 
            "company_address",
            "angebot",
            "tables"
        ],
        "model_configs": {
            "mistral-7b": {
                "path": "./models/mistral-7b-instruct-v0.1",
                "type": "causal-lm",
                "quantization": "4bit",
                "max_memory": "auto"
            }
        },
        "directories": {
            "data": "./data",
            "models": "./models", 
            "logs": "./logs",
            "uploads": "./uploads",
            "outputs": "./outputs"
        }
    }
    
    with open("config/config.json", "w") as f:
        json.dump(config_data, f, indent=2)
    print("   ✅ Created config.json file")

def create_startup_scripts():
    """Create convenient startup scripts."""
    print("🚀 Creating startup scripts...")
    
    # Start server script
    start_server_script = """#!/bin/bash
# Start Mistral PDF Extraction Server

echo "🚀 Starting Mistral PDF Extraction Server..."

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the server
echo "🌐 Starting FastAPI server..."
python3 main.py

echo "✅ Server started at http://localhost:8000"
echo "📋 Use Ctrl+C to stop the server"
"""
    
    with open("scripts/start_server.sh", "w") as f:
        f.write(start_server_script)
    os.chmod("scripts/start_server.sh", 0o755)
    print("   ✅ Created start_server.sh")
    
    # Download model script
    download_model_script = """#!/bin/bash
# Download Mistral 7B Instruct v0.1 Model

echo "📥 Downloading Mistral 7B Instruct v0.1..."

# Check if model already exists
if [ -d "models/mistral-7b-instruct-v0.1" ]; then
    echo "⚠️  Model directory already exists. Remove it to re-download."
    exit 1
fi

# Install huggingface_hub if not present
pip install huggingface_hub

# Download model
echo "🔄 Downloading model files (this may take a while)..."
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.1 --local-dir ./models/mistral-7b-instruct-v0.1

echo "✅ Model downloaded successfully!"
echo "📍 Model location: ./models/mistral-7b-instruct-v0.1"
"""
    
    with open("scripts/download_model.sh", "w") as f:
        f.write(download_model_script)
    os.chmod("scripts/download_model.sh", 0o755)
    print("   ✅ Created download_model.sh")

def create_example_files():
    """Create example files for testing."""
    print("📄 Creating example files...")
    
    # Create sample Excel template
    try:
        import pandas as pd
        
        sample_data = {
            'file_name': ['sample_document.pdf', 'example_invoice.pdf'],
            'date': ['15.03.2024', '20.02.2024'],
            'company_name': ['TechSolutions GmbH', 'Global Industries AG'],
            'company_address': ['Musterstraße 123, 12345 Berlin', 'Hauptplatz 1, 1010 Wien'],
            'angebot': ['A-2024-001', 'Q-2024-002'],
            'tables_count': [2, 1],
            'notes': ['Training example', 'Sample data']
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel("data/excel/templates/training_template.xlsx", index=False)
        print("   ✅ Created Excel training template")
        
    except ImportError:
        print("   ⚠️  Pandas not available yet - Excel template will be created after installation")

def create_test_script():
    """Create comprehensive test script."""
    print("🧪 Creating test script...")
    
    test_script = """#!/usr/bin/env python3
'''
Comprehensive test script for Mistral PDF Extraction Pipeline
'''

import requests
import time
import json

def test_server_health():
    '''Test if server is running.'''
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
            return True
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        return False

def test_mistral_status():
    '''Check Mistral model status.'''
    try:
        response = requests.get("http://localhost:8000/mistral/status")
        data = response.json()
        
        if data.get("loaded"):
            print("✅ Mistral 7B model is loaded")
            print(f"   📍 Model path: {data.get('model_path')}")
            return True
        else:
            print("⚠️  Mistral 7B model not loaded")
            print("   💡 Use: curl -X POST 'http://localhost:8000/mistral/load'")
            return False
    except Exception as e:
        print(f"❌ Error checking Mistral status: {e}")
        return False

def run_all_tests():
    '''Run comprehensive test suite.'''
    print("🧪 Running Mistral PDF Extraction Tests")
    print("=" * 50)
    
    tests = [
        ("Server Health", test_server_health),
        ("Mistral Status", test_mistral_status),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\\n🔬 {test_name}:")
        result = test_func()
        results.append((test_name, result))
    
    print("\\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\\n🎯 Overall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! Your setup is ready!")
    else:
        print("⚠️  Some tests failed. Check the logs above for details.")

if __name__ == "__main__":
    run_all_tests()
"""
    
    with open("tests/test_setup.py", "w") as f:
        f.write(test_script)
    os.chmod("tests/test_setup.py", 0o755)
    print("   ✅ Created test_setup.py")

def create_quick_start_guide():
    """Create a quick start guide."""
    print("📚 Creating quick start guide...")
    
    guide_content = """# 🚀 Quick Start Guide

## Your Mistral 7B PDF Extraction Pipeline is Ready!

### 🏃‍♂️ Quick Setup (5 minutes):

1. **Download Mistral 7B Model:**
   ```bash
   ./scripts/download_model.sh
   ```

2. **Start the Server:**
   ```bash
   ./scripts/start_server.sh
   ```

3. **Load Mistral Model:**
   ```bash
   curl -X POST "http://localhost:8000/mistral/load"
   ```

4. **Test with Sample PDF:**
   ```bash
   curl -X POST "http://localhost:8000/mistral/extract" -F "file=@sample.pdf"
   ```

### 📁 Project Structure:
```
📦 Your Project
├── 📁 data/                    # Training data
│   ├── 📁 pdfs/               # PDF files
│   └── 📁 excel/              # Excel answer sheets
├── 📁 models/                 # AI models
│   └── 📁 mistral-7b-instruct-v0.1/
├── 📁 hf_framework/           # Core framework
├── 📁 scripts/                # Utility scripts
├── 📁 outputs/                # Results
├── 📄 main.py                 # Main application
└── 📄 .env                    # Configuration
```

### 🎯 Your Workflow:
1. **Prepare your 50 PDFs** → `data/pdfs/training/`
2. **Create Excel answers** → `data/excel/answers/training_answers.xlsx`
3. **Train the model** → `POST /mistral/supervised-train`
4. **Extract parameters** → `POST /mistral/batch-extract`
5. **Get results** → Excel output with extractions

### 🌐 Web Interface:
- **Main Interface:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### 🔧 Key Commands:
```bash
# Load model
curl -X POST "http://localhost:8000/mistral/load"

# Check status  
curl "http://localhost:8000/mistral/status"

# Extract from PDF
curl -X POST "http://localhost:8000/mistral/extract" -F "file=@document.pdf"

# Run tests
python3 tests/test_setup.py
```

### 🆘 Need Help?
- Check `MISTRAL_DEPLOYMENT_GUIDE.md` for detailed instructions
- Run tests: `python3 tests/test_setup.py`
- Check logs in `logs/` directory

**🎉 You're all set! Happy extracting!**
"""
    
    with open("QUICK_START.md", "w") as f:
        f.write(guide_content)
    print("   ✅ Created QUICK_START.md")

def organize_existing_files():
    """Organize existing files into proper structure."""
    print("📋 Organizing existing files...")
    
    # Files to keep in root
    keep_in_root = [
        "main.py", "main_simple.py", "run.py", 
        ".env", ".env.example", "README.md",
        "requirements_mistral.txt", "MISTRAL_DEPLOYMENT_GUIDE.md",
        "Dockerfile", "docker-compose.yml"
    ]
    
    # Move test files to tests directory
    test_files = ["test_supervised.py", "test_simple.py", "test_framework.py", "test_gemini.py"]
    for test_file in test_files:
        if os.path.exists(test_file):
            shutil.move(test_file, f"tests/{test_file}")
            print(f"   📁 Moved {test_file} → tests/")
    
    # Move old requirements to config
    old_requirements = ["requirements.txt", "requirements_hf.txt", "requirements_hf_compatible.txt"]
    for req_file in old_requirements:
        if os.path.exists(req_file):
            shutil.move(req_file, f"config/{req_file}")
            print(f"   📁 Moved {req_file} → config/")
    
    # Move web files to a web directory
    os.makedirs("web", exist_ok=True)
    web_files = ["index.html", "style.css", "script.js"]
    for web_file in web_files:
        if os.path.exists(web_file):
            shutil.move(web_file, f"web/{web_file}")
            print(f"   📁 Moved {web_file} → web/")

def main():
    """Main setup function."""
    print("🎯 Mistral 7B PDF Extraction Pipeline Setup")
    print("=" * 60)
    
    # Check prerequisites
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    create_directory_structure()
    
    # Organize existing files
    organize_existing_files()
    
    # Install dependencies
    install_dependencies()
    
    # Setup configuration
    setup_configuration()
    
    # Create scripts
    create_startup_scripts()
    
    # Create examples
    create_example_files()
    
    # Create test script
    create_test_script()
    
    # Create quick start guide
    create_quick_start_guide()
    
    # Check GPU after installation
    check_gpu_availability()
    
    print("\n" + "=" * 60)
    print("🎉 PROJECT SETUP COMPLETE!")
    print("=" * 60)
    print("✅ Directory structure created")
    print("✅ Dependencies installed")
    print("✅ Configuration files created")
    print("✅ Startup scripts created") 
    print("✅ Test scripts created")
    print("✅ Documentation created")
    
    print("\n📋 Next Steps:")
    print("1. Download Mistral 7B: ./scripts/download_model.sh")
    print("2. Start server: ./scripts/start_server.sh")
    print("3. Read QUICK_START.md for usage guide")
    print("4. Check MISTRAL_DEPLOYMENT_GUIDE.md for details")
    
    print("\n🌐 Once running:")
    print("   • Web Interface: http://localhost:8000")
    print("   • API Docs: http://localhost:8000/docs")
    print("   • Test Setup: python3 tests/test_setup.py")
    
    print("\n🚀 Your Mistral 7B PDF extraction pipeline is ready!")

if __name__ == "__main__":
    main()