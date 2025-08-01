"""
Offline configuration for Mistral 7B PDF Pipeline
"""

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
CACHE_DIR = MODEL_DIR / "cache"
NLTK_DATA_DIR = MODEL_DIR / "nltk_data"

# Offline model paths
MISTRAL_MODEL_PATH = MODEL_DIR / "mistral-7b-instruct-v0.1"

# Environment variables for offline operation
OFFLINE_CONFIG = {
    "HF_HOME": str(CACHE_DIR),
    "TRANSFORMERS_CACHE": str(CACHE_DIR / "transformers"),
    "HF_DATASETS_CACHE": str(CACHE_DIR / "datasets"),
    "NLTK_DATA": str(NLTK_DATA_DIR),
    "TRANSFORMERS_OFFLINE": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
}

def setup_offline_environment():
    """Set up environment variables for offline operation."""
    for key, value in OFFLINE_CONFIG.items():
        os.environ[key] = value
    
    # Create cache directories
    for path in OFFLINE_CONFIG.values():
        Path(path).mkdir(parents=True, exist_ok=True)
    
    print("🔒 Offline environment configured")

def verify_offline_setup():
    """Verify that all required files are available offline."""
    checks = []
    
    # Check Mistral model files
    model_files = [
        "config.json",
        "tokenizer.json", 
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    
    for file in model_files:
        file_path = MISTRAL_MODEL_PATH / file
        checks.append(("Mistral " + file, file_path.exists()))
    
    # Check NLTK data
    nltk_datasets = ["punkt", "stopwords", "averaged_perceptron_tagger", "wordnet"]
    for dataset in nltk_datasets:
        dataset_path = NLTK_DATA_DIR / "tokenizers" / dataset
        checks.append((f"NLTK {dataset}", dataset_path.exists() or (NLTK_DATA_DIR / "corpora" / dataset).exists()))
    
    # Print results
    print("\n🔍 Offline Setup Verification:")
    print("=" * 40)
    all_good = True
    for name, status in checks:
        icon = "✅" if status else "❌"
        print(f"{icon} {name}")
        if not status:
            all_good = False
    
    if all_good:
        print("\n🎉 All offline components are ready!")
    else:
        print("\n⚠️  Some components missing - check downloads")
    
    return all_good

if __name__ == "__main__":
    setup_offline_environment()
    verify_offline_setup()
