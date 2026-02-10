"""
Utility functions for Fashion Magazine-alytics project
"""
import os
from pathlib import Path
import pandas as pd
from datetime import datetime

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def load_metadata(metadata_path='data/processed_features/paris_metadata.csv'):
    """Load metadata CSV"""
    return pd.read_csv(metadata_path)
