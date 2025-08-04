#!/bin/bash
# Simple Startup Script for Offline PDF Extraction System

echo "🔒 Offline PDF Extraction System Startup"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.9+"
    exit 1
fi

# Check if required files exist
if [ ! -f "main_offline.py" ]; then
    echo "❌ main_offline.py not found in current directory"
    exit 1
fi

echo "✅ Python3 found"
echo "✅ Offline server file found"

# Check if port 8000 is available
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port 8000 is already in use"
    echo "💡 Stopping existing server..."
    pkill -f "python3 main_offline.py" 2>/dev/null
    sleep 2
fi

# Start the offline server
echo "🚀 Starting offline PDF extraction server..."
echo ""
echo "📋 Access Points:"
echo "   🌐 Web Interface: http://localhost:8000"
echo "   📖 API Documentation: http://localhost:8000/docs"
echo "   🔍 Health Check: http://localhost:8000/health"
echo ""
echo "🔧 Quick Commands:"
echo "   Test System: python3 test_offline_complete.py"
echo "   Stop Server: Ctrl+C or pkill -f 'python3 main_offline.py'"
echo ""
echo "🔒 OFFLINE MODE: No internet connection required!"
echo "========================================"
echo ""

# Start the server
python3 main_offline.py