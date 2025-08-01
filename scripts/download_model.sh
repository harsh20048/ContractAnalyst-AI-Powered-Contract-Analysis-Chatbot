#!/bin/bash
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
