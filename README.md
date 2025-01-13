# ContractAnalyst: AI-Powered Contract Analysis Chatbot

## 🔍 Overview
ContractAnalyst is an intelligent document analysis system that uses Google's Gemini AI to help users understand and analyze legal contracts and documents. The system provides an intuitive chat interface where users can upload PDFs and ask questions about their content.

## ✨ Key Features
- **PDF Document Analysis**: Upload and process PDF contracts and legal documents
- **Interactive Chat Interface**: Ask questions about your documents in natural language
- **Smart Context Understanding**: System maintains context for follow-up questions
- **Suggested Questions**: AI-generated relevant questions based on document content
- **Structure Recognition**: Automatically identifies document sections, definitions, and key terms
- **Priority Scoring**: Intelligent scoring system for identifying important document sections

## 🛠️ Technical Stack
- **Frontend**: Streamlit
- **AI/ML**: 
  - Google Gemini AI for natural language processing
  - FAISS for vector similarity search
  - LangChain for document processing
- **Text Processing**: 
  - NLTK for text analysis
  - spaCy for advanced NLP tasks
  - PyPDF2 for PDF processing

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Google API Key (Gemini AI)

### Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/contract-analyst.git
cd contract-analyst
```

2. Install dependencies:
```bash
pip install -r requirements.txt
python setup_nltk.py
```

3. Set up environment variables:
Create a `.env` file with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

4. Run the application:
```bash
streamlit run app.py
```

## 📖 Usage
1. Launch the application
2. Upload a PDF contract or legal document
3. Wait for the document to be processed
4. Ask questions about the document using the chat interface
5. Use suggested questions or ask your own

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details

## 🙏 Acknowledgments
- Google Gemini AI for providing the language model
- Streamlit for the web interface framework
- LangChain for document processing capabilities
