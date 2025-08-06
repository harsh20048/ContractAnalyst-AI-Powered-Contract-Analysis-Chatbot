"""
Learning Package
Database management and pattern learning for adaptive extraction.
"""

from .database import LearningDatabase
from .pattern_learner import PatternLearner

__all__ = [
    'LearningDatabase',
    'PatternLearner'
]