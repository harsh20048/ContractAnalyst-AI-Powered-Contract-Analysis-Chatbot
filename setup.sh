#!/bin/bash

# Setup script for Adaptive PDF Table Extraction System
echo "🚀 Setting up Adaptive PDF Table Extraction System..."
echo "=================================================="

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Initialize database
echo "🗄️ Initializing database..."
python3 -c "from database_manager import initialize_database; db = initialize_database(); print('Database initialized:', db.db_path)"

if [ $? -eq 0 ]; then
    echo "✅ Database initialized successfully"
else
    echo "❌ Failed to initialize database"
    exit 1
fi

# Populate sample data
echo "📊 Populating sample data..."
python3 test_database.py

if [ $? -eq 0 ]; then
    echo "✅ Sample data populated successfully"
else
    echo "❌ Failed to populate sample data"
    exit 1
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "To run the application:"
echo "  python3 run_adaptive_app.py"
echo ""
echo "Or directly with streamlit:"
echo "  streamlit run adaptive_pdf_app.py --server.port=8501"
echo ""
echo "The application will be available at: http://localhost:8501"
echo "Database location: /workspace/learning/learning.db"