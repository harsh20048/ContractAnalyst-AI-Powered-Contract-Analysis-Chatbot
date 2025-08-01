#!/bin/bash
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
