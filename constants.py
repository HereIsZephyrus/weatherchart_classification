"""
Constants for the project
"""

TRAIN_DATA_DIR = 'train_data'
GALLERY_DIR = f'{TRAIN_DATA_DIR}/gallery'
RADER_RAW_DIR = f'{TRAIN_DATA_DIR}/radar-dataset'
RADER_DIR = f'{TRAIN_DATA_DIR}/radar'
INPUT_DIR = 'income'
PPT_DIR = f'{INPUT_DIR}/slides'
IMAGE_DIR = f'{INPUT_DIR}/extracted_images'
LOGO_DIR = f'{TRAIN_DATA_DIR}/logo'
IMAGE_SIZE = (320, 240)
DATASET_DIR = 'dataset'

__all__ = [
    'TRAIN_DATA_DIR',
    'GALLERY_DIR',
    'RADER_RAW_DIR',
    'RADER_DIR',
    'INPUT_DIR',
    'PPT_DIR',
    'IMAGE_DIR',
    'LOGO_DIR',
    'IMAGE_SIZE',
    'DATASET_DIR',
]
