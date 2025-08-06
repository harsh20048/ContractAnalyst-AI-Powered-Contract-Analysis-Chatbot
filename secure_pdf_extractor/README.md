# 🔒 Secure PDF Extractor

Advanced PDF processing system with intelligent field extraction, machine learning, and adaptive pattern recognition.

## 📁 Project Structure

```
secure_pdf_extractor/
├── __init__.py                 # Package initialization
├── requirements.txt            # Dependencies
├── run.py                     # Main application entry point
├── README.md                  # This file
├── extraction/                # Field and content extraction
│   ├── __init__.py
│   ├── text_extraction.py     # Text extraction from PDFs
│   ├── field_extraction.py    # Advanced field extraction with ML
│   └── table_extraction.py    # Table detection and parsing
├── learning/                  # Machine learning and pattern recognition
│   ├── __init__.py
│   ├── database.py           # Database for storing corrections and patterns
│   └── pattern_learner.py    # Adaptive pattern learning system
├── validation/                # Data quality and validation
│   ├── __init__.py
│   └── field_validation.py   # Field validation and quality checks
├── output/                    # Export and output functionality
│   ├── __init__.py
│   └── excel_export.py       # Excel export with formatting
└── gui/                       # User interface
    ├── __init__.py
    └── app.py                # Streamlit web application
```

## 🚀 Features

### 🔍 **Advanced Field Extraction**
- **Smart Pattern Recognition**: Automatically detects common field patterns
- **Machine Learning**: Learns from user corrections to improve accuracy
- **Multi-Format Support**: Handles various PDF layouts and formats
- **Context-Aware**: Uses surrounding text for better field identification

### 🧠 **Adaptive Learning System**
- **Pattern Learning**: Automatically generates regex patterns from corrections
- **Performance Tracking**: Monitors pattern success rates
- **Database Persistence**: Saves corrections and patterns for reuse
- **Document Hashing**: Content-based identification for consistent processing

### ✅ **Data Validation**
- **Field Type Validation**: Custom validation rules for different field types
- **Quality Checks**: Identifies potential errors and inconsistencies
- **Confidence Scoring**: Provides reliability metrics for extracted data
- **Suggestion System**: Offers correction suggestions

### 📤 **Export Options**
- **Excel Export**: Formatted spreadsheets with multiple sheets
- **Validation Reports**: Detailed validation results and statistics
- **Learning Analytics**: System performance and pattern statistics
- **Template Generation**: Creates templates for manual data entry

### 🔒 **Security & Reliability**
- **Content-Based Hashing**: Secure document identification
- **Error Handling**: Robust error recovery and logging
- **Data Persistence**: Reliable storage of corrections and patterns
- **Performance Monitoring**: System health and usage analytics

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Google API Key (for AI features)

### Quick Start

1. **Clone or download the repository**
   ```bash
   git clone <repository-url>
   cd secure_pdf_extractor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the parent directory:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the application**
   ```bash
   python run.py
   ```

## 📖 Usage

### Basic Workflow

1. **Upload PDF**: Use the file uploader in the sidebar
2. **Review Fields**: Check automatically extracted fields
3. **Make Corrections**: Edit any incorrect values
4. **Save Corrections**: Click "Save Corrections" to teach the system
5. **Export Data**: Use the Export tab to download results

### Advanced Features

#### **Field Extraction**
- Upload PDFs and automatically extract key fields
- System detects: dates, amounts, company names, contact info, reference numbers
- Visual confidence indicators show extraction reliability

#### **Validation**
- Comprehensive field validation with error detection
- Custom validation rules for different field types
- Suggestions for improving data quality

#### **Learning System**
- System learns from every correction you make
- Improves accuracy for future documents
- Pattern performance tracking and optimization

#### **Export Options**
- Excel files with multiple formatted sheets
- Validation reports with detailed analysis
- System statistics and learning progress

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for AI-powered extraction features
- `DATABASE_PATH`: Optional, defaults to "learning.db"

### Supported Field Types
- **Dates**: Various formats (DD.MM.YYYY, MM/DD/YYYY, etc.)
- **Amounts**: Monetary values with currency symbols
- **Company Names**: Business entity names with legal suffixes
- **Contact Persons**: Names with titles and formatting
- **Reference Numbers**: Angebot numbers, quote IDs, etc.
- **Custom Fields**: System learns new field types automatically

## 📊 Analytics

The system provides comprehensive analytics:

- **Extraction Statistics**: Success rates and confidence metrics
- **Learning Progress**: Pattern improvement over time
- **Document Processing**: Usage statistics and performance
- **Validation Results**: Data quality metrics and trends

## 🛡️ Error Handling

### Common Issues and Solutions

1. **Import Errors**: Check if all dependencies are installed
2. **API Key Issues**: Verify GOOGLE_API_KEY is set correctly
3. **PDF Processing Errors**: Ensure PDF is not corrupted or password-protected
4. **Memory Issues**: For large PDFs, consider splitting into smaller files

### Logging
- Application logs are available in the console
- Error details are captured for debugging
- Performance metrics are tracked automatically

## 🔄 Updates and Maintenance

### Database Management
- Learning database grows over time with corrections
- Use "Clear All Data" feature to reset if needed
- Regular backups recommended for production use

### Pattern Management
- System automatically optimizes patterns
- Manual pattern review available in Analytics tab
- Poor-performing patterns are automatically deprecated

## 🤝 Contributing

### Development Setup
1. Follow installation steps above
2. Install development dependencies
3. Run tests before submitting changes
4. Follow existing code style and patterns

### Code Structure
- **Modular Design**: Each component has specific responsibilities
- **Type Hints**: Full type annotation for better IDE support
- **Error Handling**: Comprehensive exception management
- **Documentation**: Inline docs and docstrings throughout

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the error logs in the console
2. Review this README for common solutions
3. Ensure all dependencies are properly installed
4. Verify environment variables are set correctly

## 🎯 Future Enhancements

- **Multi-language Support**: Extend to non-English documents
- **Advanced OCR**: Handle scanned PDFs and images
- **Batch Processing**: Process multiple documents simultaneously
- **API Integration**: REST API for programmatic access
- **Custom Field Types**: User-defined field categories
- **Cloud Storage**: Integration with cloud storage services

---

**Made with ❤️ using Python, Streamlit, and Machine Learning**