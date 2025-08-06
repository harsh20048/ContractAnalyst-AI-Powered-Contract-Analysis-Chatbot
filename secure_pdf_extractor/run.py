#!/usr/bin/env python3
"""
Secure PDF Extractor - Main Entry Point
Run this file to start the application.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

def main():
    """Main function to run the application."""
    try:
        # Import streamlit and run the app
        import streamlit.web.cli as stcli
        import streamlit as st
        
        # Get the app path
        app_path = current_dir / "gui" / "app.py"
        
        if not app_path.exists():
            print(f"❌ Error: Application file not found at {app_path}")
            print("Please ensure the file structure is correct.")
            return 1
        
        print("🚀 Starting Secure PDF Extractor...")
        print(f"📁 Application path: {app_path}")
        print("🌐 Opening web interface...")
        
        # Run streamlit app
        sys.argv = [
            "streamlit", 
            "run", 
            str(app_path),
            "--server.headless", "false",
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        stcli.main()
        return 0
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user.")
        return 0
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'streamlit',
        'pandas', 
        'PyPDF2',
        'python-dotenv',
        'langchain',
        'langchain-google-genai',
        'faiss-cpu'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nInstall them with:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_environment():
    """Setup environment variables and configurations."""
    # Check for .env file
    env_file = current_dir.parent / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found.")
        print("Create a .env file with your GOOGLE_API_KEY for full functionality.")
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("⚠️  Warning: GOOGLE_API_KEY not found in environment variables.")
        print("Some features may not work without the API key.")
    else:
        print("✅ Google API key found.")
    
    return True

if __name__ == "__main__":
    print("🔒 Secure PDF Extractor v1.0")
    print("=" * 50)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    print("✅ All dependencies found.")
    
    # Setup environment
    print("🔧 Setting up environment...")
    if not setup_environment():
        sys.exit(1)
    print("✅ Environment configured.")
    
    # Run the application
    sys.exit(main())