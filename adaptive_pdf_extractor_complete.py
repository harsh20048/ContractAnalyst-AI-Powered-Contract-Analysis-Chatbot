"""
Updated Adaptive PDF Table Extraction Application
Refined version with consolidated imports, improved error handling, and cleaner structure.
"""

import sys
import copy
import os
import tempfile
import sqlite3
import json
import time
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Adaptive PDF Data Extraction System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
