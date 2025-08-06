# Adaptive PDF Data Extraction System

A Streamlit-based application for extracting tables and fields from PDF documents with machine learning capabilities.

## Features

- **PDF Upload**: Easy drag-and-drop interface for PDF files
- **Table Extraction**: Automatically detect and extract tables from PDFs
- **Field Extraction**: Extract key fields like dates, company names, and addresses
- **Adaptive Learning**: The system learns from user corrections to improve future extractions
- **Data Export**: Export extracted data in multiple formats (Excel, CSV, JSON)
- **Analytics Dashboard**: Track extraction history and performance metrics
- **Interactive Editing**: Edit extracted tables and fields directly in the interface

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run adaptive_pdf_extractor.py
```

## Usage

1. **Upload PDF**: Click on the "Upload" tab and drag your PDF file into the upload area
2. **Process**: Click the "Process PDF" button to extract tables and fields
3. **Review Tables**: Navigate to the "Tables" tab to view and edit extracted tables
4. **Review Fields**: Check the "Fields" tab for extracted field data
5. **Export Data**: Use the sidebar export options to download your data in preferred format
6. **View Analytics**: The "Analytics" tab shows processing history and statistics

## Features in Detail

### Adaptive Learning
The system includes fallback implementations for adaptive learning features. When the full extraction modules are available, it will:
- Learn from user corrections
- Improve extraction accuracy over time
- Store learning data in SQLite database

### Export Formats
- **Excel**: Multi-sheet workbook with separate sheets for each table
- **CSV**: Combined data from all tables
- **JSON**: Structured data with metadata
- **SQLite**: Direct database export (coming soon)

### Database Schema
The application creates a local SQLite database (`data/learning_data.db`) with:
- `table_templates`: Stores learned table structures
- `field_patterns`: Stores field extraction patterns
- `extraction_history`: Tracks all extraction operations

## Configuration

### Settings (in sidebar)
- **Debug Mode**: Enable detailed logging
- **Confidence Threshold**: Adjust extraction confidence requirements

### Fallback Mechanisms
The application includes comprehensive fallbacks for missing dependencies:
- Uses `pdfplumber` for text extraction if primary method unavailable
- Provides stub implementations for learning features
- Gracefully handles missing extraction modules

## Troubleshooting

### Common Issues

1. **Import Errors**: The application includes fallback implementations for all optional modules
2. **PDF Processing Errors**: Check the console for detailed error messages
3. **Export Issues**: Ensure you have write permissions in the application directory

### Dependencies

Core requirements:
- streamlit
- pandas
- numpy
- pdfplumber
- xlsxwriter

Optional (for full functionality):
- secure_pdf_extractor module
- extraction modules for adaptive learning

## Future Enhancements

- Real-time collaboration features
- Advanced ML models for better extraction
- API endpoints for programmatic access
- Cloud storage integration
- Multi-language support