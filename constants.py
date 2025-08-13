"""
Constants for the project
"""

TRAIN_DATA_DIR = 'train_data'
GALLERY_DIR = f'{TRAIN_DATA_DIR}/gallery'
GALLERY_MAPPING_PATH = f'{GALLERY_DIR}/gallery_mapping.json'
GALLERY_MAPPING_BILINGUAL_PATH = f'{GALLERY_DIR}/gallery_mapping_bilingual.json'
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
    'GALLERY_MAPPING_PATH',
    'GALLERY_MAPPING_BILINGUAL_PATH',
    'RADER_RAW_DIR',
    'RADER_DIR',
    'INPUT_DIR',
    'PPT_DIR',
    'IMAGE_DIR',
    'LOGO_DIR',
    'IMAGE_SIZE',
    'DATASET_DIR',
]
