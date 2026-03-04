"""
Configuration file for CAD Extraction System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, UPLOAD_DIR]:
    dir_path.mkdir(exist_ok=True)

# Image processing settings
IMAGE_PROCESSING = {
    'canny_low': 50,
    'canny_high': 150,
    'gaussian_kernel': (5, 5),
    'min_contour_area': 100,
    'min_circle_radius': 5,
    'max_circle_radius': 500,
    'min_line_length': 50,
    'max_line_gap': 10,
}

# CAD generation settings
CAD_SETTINGS = {
    'dxf_version': 'R2010',
    'view_spacing': 200.0,
    'line_color': 'black',
    'dimension_text_height': 10,
}

# PDF export settings
PDF_SETTINGS = {
    'page_size': 'A4',  # 'A4' or 'letter'
    'margin_mm': 20.0,
    'title_font_size': 16,
}

# Dimension measurement settings
DIMENSION_SETTINGS = {
    'default_unit': 'mm',
    'calibration_reference_length': None,  # Set if you have a known reference
}

# API settings
API_SETTINGS = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': True,
    'max_upload_size': 16 * 1024 * 1024,  # 16MB
    'allowed_extensions': {'png', 'jpg', 'jpeg', 'gif', 'bmp'},
}

# Training/ML settings (for future use)
ML_SETTINGS = {
    'model_type': 'cnn',  # 'cnn', 'transformer', etc.
    'batch_size': 32,
    'learning_rate': 0.001,
    'epochs': 100,
    'checkpoint_dir': MODELS_DIR / "checkpoints",
    'tensorboard_dir': MODELS_DIR / "tensorboard",
}
