# 🤗 HuggingFace PDF Extraction Framework

A comprehensive framework for deploying Hugging Face models and extracting information from PDF documents. This framework provides a complete solution for document analysis using state-of-the-art NLP models.

## ✨ Features

- **PDF Processing**: Advanced PDF text extraction using multiple libraries (PyMuPDF, pdfplumber, PyPDF2)
- **Multiple NLP Tasks**: 
  - Document summarization
  - Question answering
  - Named entity recognition
  - Sentiment analysis
  - Text classification
  - Key information extraction
  - Topic modeling
- **Hugging Face Integration**: Support for any Hugging Face model
- **REST API**: Complete FastAPI-based REST API
- **Web Interface**: Built-in web interface for easy document upload
- **Docker Support**: Full containerization for easy deployment
- **Async Processing**: Background processing for large documents
- **Model Management**: Dynamic model loading/unloading
- **Robust Processing**: Multiple fallback strategies for difficult PDFs

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hf-pdf-extraction-framework
   ```

2. **Create environment file**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Access the application**:
   - Web Interface: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Alternative Docs: http://localhost:8000/redoc

### Option 2: Local Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements_hf.txt
   ```

2. **Download additional models**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Run the application**:
   ```bash
   python main.py
   ```

## 📖 Usage

### Web Interface

1. Navigate to http://localhost:8000
2. Upload a PDF file
3. Select extraction tasks
4. Click "Extract Information"
5. View results

### API Usage

#### Upload and Process PDF

```bash
curl -X POST "http://localhost:8000/extract/pdf" \
  -F "file=@your_document.pdf" \
  -F "tasks=[\"summarization\", \"key_information_extraction\"]"
```

#### Process Text Directly

```bash
curl -X POST "http://localhost:8000/extract/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your text here...",
    "tasks": ["summarization", "sentiment_analysis"]
  }'
```

#### Model Management

```bash
# Load a specific model
curl -X POST "http://localhost:8000/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "facebook/bart-large-cnn",
    "task": "summarization"
  }'

# Check model status
curl "http://localhost:8000/models/status"
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `.env`:

```bash
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# File Processing
MAX_FILE_SIZE=52428800  # 50MB
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Models
DEFAULT_MODEL=microsoft/DialoGPT-medium
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EXTRACTION_MODEL=facebook/bart-large-cnn
```

### Supported Models

The framework supports any Hugging Face model compatible with the following tasks:

- **Summarization**: `facebook/bart-large-cnn`, `t5-small`, `google/pegasus-xsum`
- **Question Answering**: `distilbert-base-cased-distilled-squad`, `deepset/roberta-base-squad2`
- **Text Classification**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Named Entity Recognition**: `dslim/bert-base-NER`, `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`

## 📊 API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/health` | GET | Health check |
| `/docs` | GET | API documentation |
| `/extract/pdf` | POST | Process PDF file |
| `/extract/text` | POST | Process text directly |
| `/extract/pdf/async` | POST | Async PDF processing |
| `/extract/status/{job_id}` | GET | Check async job status |

### Configuration Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/config/tasks` | GET | Supported extraction tasks |
| `/config/models` | GET | Available models |
| `/config/settings` | GET | Current settings |

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/models/load` | POST | Load a model |
| `/models/{model}/{task}` | DELETE | Unload a model |
| `/models/status` | GET | Model status |
| `/models/warm-up` | POST | Pre-load models |

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │───▶│  PDF Processor   │───▶│ Extraction      │
│                 │    │                  │    │ Engine          │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web UI        │    │ Multiple PDF     │    │ Model Manager   │
│                 │    │ Libraries        │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                       │
                                ▼                       ▼
                    ┌──────────────────┐    ┌─────────────────┐
                    │ PyMuPDF          │    │ HuggingFace     │
                    │ pdfplumber       │    │ Transformers    │
                    │ PyPDF2           │    │                 │
                    └──────────────────┘    └─────────────────┘
```

### Components

- **FastAPI App**: Main web application and API
- **PDF Processor**: Multi-library PDF text extraction
- **Extraction Engine**: Orchestrates PDF processing and model inference
- **Model Manager**: Handles loading/unloading of HuggingFace models
- **Web UI**: Built-in web interface for document upload

## 🔧 Development

### Setting up Development Environment

```bash
# Clone repository
git clone <repository-url>
cd hf-pdf-extraction-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_hf.txt

# Install development dependencies
pip install pytest black flake8

# Run in development mode
python main.py
```

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black .
flake8 .
```

## 📋 Supported Extraction Tasks

1. **Summarization**: Generate concise summaries of documents
2. **Question Answering**: Answer questions based on document content
3. **Key Information Extraction**: Extract metadata, statistics, and key phrases
4. **Sentiment Analysis**: Analyze emotional tone and sentiment
5. **Named Entity Recognition**: Extract people, places, organizations
6. **Text Classification**: Classify documents into categories
7. **Topic Modeling**: Identify main topics and themes

## 🚀 Deployment

### Production Deployment

1. **Update environment variables** for production:
   ```bash
   DEBUG=false
   LOG_LEVEL=INFO
   SECRET_KEY=your-secure-secret-key
   ```

2. **Use production WSGI server**:
   ```bash
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

3. **Set up reverse proxy** (Nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

### Docker Production

```bash
# Build and run with docker-compose
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

## 🎯 Performance Considerations

### Memory Management

- Models are cached for reuse
- Automatic model unloading available
- GPU support for faster inference
- Configurable file size limits

### Optimization Tips

1. **Use GPU** if available for faster model inference
2. **Warm up models** on startup for common tasks
3. **Adjust chunk sizes** based on your documents
4. **Use async processing** for large files
5. **Configure memory limits** appropriately

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce model size or use CPU-only models
2. **PDF Processing Failed**: Check if PDF is text-based vs scanned
3. **Model Download Issues**: Check internet connection and HuggingFace access
4. **Slow Processing**: Consider using smaller models or GPU acceleration

### Debugging

```bash
# Enable debug mode
DEBUG=true LOG_LEVEL=DEBUG python main.py

# Check logs
docker-compose logs -f hf-pdf-extractor
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

- Create an issue for bug reports
- Check documentation at `/docs` endpoint
- Review example usage in the `examples/` directory

## 🔮 Roadmap

- [ ] Support for more document formats (DOCX, TXT, etc.)
- [ ] Integration with vector databases
- [ ] Custom model fine-tuning interface
- [ ] Batch processing capabilities
- [ ] Advanced caching mechanisms
- [ ] Multi-language support
- [ ] OCR integration for scanned PDFs
