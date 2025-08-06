"""
Secure PDF Extractor Package
Advanced PDF processing with field extraction, learning, and validation.
"""

__version__ = "1.0.0"
__author__ = "PDF Processing Team"

from .extraction.field_extraction import FieldExtractor
from .learning.database import LearningDatabase
from .learning.pattern_learner import PatternLearner

__all__ = [
    'FieldExtractor',
    'LearningDatabase', 
    'PatternLearner'
]