# 🚀 Quick Start Guide

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
