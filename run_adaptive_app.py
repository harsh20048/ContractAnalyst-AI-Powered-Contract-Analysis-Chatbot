#!/usr/bin/env python3
"""
Launcher script for the Adaptive PDF Data Extraction System
"""

import sys
import os
import subprocess

def main():
    """Launch the Streamlit application"""
    try:
        # Change to the workspace directory
        os.chdir('/workspace')
        
        # Run the Streamlit app
        cmd = [sys.executable, '-m', 'streamlit', 'run', 'adaptive_pdf_app.py', '--server.port=8501', '--server.address=0.0.0.0']
        
        print("🚀 Starting Adaptive PDF Data Extraction System...")
        print("🌐 Application will be available at: http://localhost:8501")
        print("📊 Database path: /workspace/learning/learning.db")
        print("=" * 60)
        
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())