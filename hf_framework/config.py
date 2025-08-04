"""
Configuration management for the Hugging Face PDF extraction framework.
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, Field
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Configuration
    app_name: str = "HuggingFace PDF Extraction Framework"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Model Configuration
    default_model: str = Field(
        default="microsoft/DialoGPT-medium", 
        env="DEFAULT_MODEL"
    )
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    extraction_model: str = Field(
        default="facebook/bart-large-cnn",
        env="EXTRACTION_MODEL"
    )
    
    # Model specific settings
    max_length: int = Field(default=512, env="MAX_LENGTH")
    temperature: float = Field(default=0.7, env="TEMPERATURE")
    top_p: float = Field(default=0.9, env="TOP_P")
    
    # PDF Processing Configuration
    max_file_size: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    allowed_extensions: List[str] = Field(default=[".pdf"], env="ALLOWED_EXTENSIONS")
    chunk_size: int = Field(default=1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, env="CHUNK_OVERLAP")
    
    # Vector Database Configuration
    vector_db_path: str = Field(default="./vector_db", env="VECTOR_DB_PATH")
    similarity_threshold: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")
    
    # Storage Configuration
    upload_dir: str = Field(default="./uploads", env="UPLOAD_DIR")
    processed_dir: str = Field(default="./processed", env="PROCESSED_DIR")
    
    # Security
    secret_key: str = Field(default="your-secret-key-here", env="SECRET_KEY")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    
    # Cache Configuration
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")  # 1 hour
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ensure_directories()
        
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.upload_dir,
            self.processed_dir,
            self.vector_db_path
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Model configurations for different tasks
MODEL_CONFIGS = {
    "summarization": {
        "models": [
            "facebook/bart-large-cnn",
            "t5-small",
            "google/pegasus-xsum"
        ],
        "default": "facebook/bart-large-cnn"
    },
    "question_answering": {
        "models": [
            "distilbert-base-cased-distilled-squad",
            "deepset/roberta-base-squad2",
            "microsoft/DialoGPT-medium"
        ],
        "default": "distilbert-base-cased-distilled-squad"
    },
    "text_classification": {
        "models": [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "nlptown/bert-base-multilingual-uncased-sentiment",
            "distilbert-base-uncased-finetuned-sst-2-english"
        ],
        "default": "distilbert-base-uncased-finetuned-sst-2-english"
    },
    "named_entity_recognition": {
        "models": [
            "dbmdz/bert-large-cased-finetuned-conll03-english",
            "dslim/bert-base-NER",
            "xlm-roberta-large-finetuned-conll03-english"
        ],
        "default": "dslim/bert-base-NER"
    },
    "embeddings": {
        "models": [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/distilbert-base-nli-mean-tokens"
        ],
        "default": "sentence-transformers/all-MiniLM-L6-v2"
    }
}

# Supported extraction tasks
EXTRACTION_TASKS = [
    "summarization",
    "question_answering", 
    "key_information_extraction",
    "sentiment_analysis",
    "named_entity_recognition",
    "topic_modeling",
    "text_classification"
]