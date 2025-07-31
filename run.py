#!/usr/bin/env python3
"""
Startup script for the HuggingFace PDF Extraction Framework.

This script provides an easy way to start the framework with different configurations.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import torch
        import transformers
        import fastapi
        import uvicorn
        print("✅ Core dependencies found")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements_hf.txt")
        return False


def setup_environment():
    """Set up environment variables and directories."""
    
    # Create .env file if it doesn't exist
    if not Path(".env").exists():
        print("📝 Creating .env file from template...")
        if Path(".env.example").exists():
            subprocess.run(["cp", ".env.example", ".env"])
            print("✅ Created .env file. Please review and update the configuration.")
        else:
            print("⚠️  .env.example not found. Creating basic .env file...")
            with open(".env", "w") as f:
                f.write("DEBUG=false\n")
                f.write("LOG_LEVEL=INFO\n")
                f.write("HOST=0.0.0.0\n")
                f.write("PORT=8000\n")
    
    # Create necessary directories
    directories = ["uploads", "processed", "vector_db", "model_cache", "static", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("✅ Environment setup completed")


def download_models(models=None):
    """Download and cache specified models."""
    if not models:
        # Default models to download
        models = [
            ("facebook/bart-large-cnn", "summarization"),
            ("sentence-transformers/all-MiniLM-L6-v2", "embeddings"),
            ("distilbert-base-cased-distilled-squad", "question_answering")
        ]
    
    print("📥 Pre-downloading models...")
    
    try:
        from transformers import AutoTokenizer, AutoModel, pipeline
        from sentence_transformers import SentenceTransformer
        
        for model_name, task in models:
            try:
                print(f"📦 Downloading {model_name}...")
                
                if task == "embeddings":
                    SentenceTransformer(model_name, cache_folder="./model_cache")
                else:
                    pipeline(task, model=model_name, model_kwargs={"cache_dir": "./model_cache"})
                
                print(f"✅ Downloaded {model_name}")
                
            except Exception as e:
                print(f"⚠️  Failed to download {model_name}: {e}")
    
    except ImportError:
        print("❌ Cannot download models - transformers not installed")


def run_development():
    """Run in development mode."""
    print("🚀 Starting in development mode...")
    
    os.environ["DEBUG"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])


def run_production():
    """Run in production mode."""
    print("🚀 Starting in production mode...")
    
    os.environ["DEBUG"] = "false"
    os.environ["LOG_LEVEL"] = "INFO"
    
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--workers", "1"
    ])


def run_docker():
    """Run using Docker."""
    print("🐳 Starting with Docker...")
    
    if not Path("Dockerfile").exists():
        print("❌ Dockerfile not found")
        return
    
    # Build and run with docker-compose
    try:
        subprocess.run(["docker-compose", "up", "--build"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Docker command failed. Make sure Docker is installed and running.")
    except FileNotFoundError:
        print("❌ docker-compose not found. Please install Docker Compose.")


def run_tests():
    """Run tests."""
    print("🧪 Running tests...")
    
    # Create basic test if none exist
    test_dir = Path("tests")
    test_dir.mkdir(exist_ok=True)
    
    test_file = test_dir / "test_basic.py"
    if not test_file.exists():
        test_content = '''
import pytest
import requests
import time
import subprocess
import os
from pathlib import Path


def test_health_endpoint():
    """Test health endpoint."""
    # Start the application in background
    proc = subprocess.Popen([
        "python", "-m", "uvicorn", "main:app", 
        "--host", "127.0.0.1", "--port", "8001"
    ])
    
    # Wait for startup
    time.sleep(5)
    
    try:
        # Test health endpoint
        response = requests.get("http://127.0.0.1:8001/health")
        assert response.status_code == 200
        assert "status" in response.json()
        
    finally:
        # Clean up
        proc.terminate()
        proc.wait()


def test_config_endpoints():
    """Test configuration endpoints."""
    proc = subprocess.Popen([
        "python", "-m", "uvicorn", "main:app", 
        "--host", "127.0.0.1", "--port", "8002"
    ])
    
    time.sleep(5)
    
    try:
        # Test tasks endpoint
        response = requests.get("http://127.0.0.1:8002/config/tasks")
        assert response.status_code == 200
        data = response.json()
        assert "supported_tasks" in data
        
        # Test models endpoint
        response = requests.get("http://127.0.0.1:8002/config/models")
        assert response.status_code == 200
        
    finally:
        proc.terminate()
        proc.wait()
'''
        test_file.write_text(test_content)
    
    # Run tests
    try:
        subprocess.run(["python", "-m", "pytest", "tests/", "-v"], check=True)
    except subprocess.CalledProcessError:
        print("❌ Tests failed")
    except FileNotFoundError:
        print("❌ pytest not found. Install with: pip install pytest")


def main():
    """Main function to handle command line arguments."""
    parser = argparse.ArgumentParser(
        description="HuggingFace PDF Extraction Framework Startup Script"
    )
    
    parser.add_argument(
        "command",
        choices=["dev", "prod", "docker", "test", "setup", "download-models"],
        help="Command to run"
    )
    
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Skip dependency and environment checks"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)"
    )
    
    args = parser.parse_args()
    
    print("🤗 HuggingFace PDF Extraction Framework")
    print("=" * 50)
    
    # Run setup for most commands
    if args.command != "test" and not args.skip_checks:
        if not check_dependencies():
            sys.exit(1)
        setup_environment()
    
    # Set port
    os.environ["PORT"] = str(args.port)
    
    # Execute command
    if args.command == "dev":
        run_development()
    
    elif args.command == "prod":
        run_production()
    
    elif args.command == "docker":
        run_docker()
    
    elif args.command == "test":
        run_tests()
    
    elif args.command == "setup":
        print("✅ Setup completed")
    
    elif args.command == "download-models":
        download_models()
    
    else:
        print(f"❌ Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()