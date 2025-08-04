#!/bin/bash
# Start Mistral PDF Extraction Pipeline in OFFLINE mode

echo "🔒 Starting Mistral PDF Pipeline in OFFLINE MODE"
echo "================================================"

# Set offline environment variables
export OFFLINE_MODE=True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HOME=./models/cache
export TRANSFORMERS_CACHE=./models/cache/transformers
export HF_DATASETS_CACHE=./models/cache/datasets
export NLTK_DATA=./models/nltk_data

echo "🔒 Offline environment configured"

# Verify offline setup first
echo "🧪 Verifying offline components..."
python3 ./scripts/verify_offline.py

if [ $? -eq 0 ]; then
    echo ""
    echo "🚀 Starting offline server..."
    python3 main_simple.py
else
    echo "❌ Offline verification failed. Please run setup_offline.sh first."
    exit 1
fi
