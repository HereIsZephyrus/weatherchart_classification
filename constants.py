"""
Constants for the project
"""

INPUT_DIR = f'{__package__}/income'
DATASET_DIR = f'{__package__}/dataset'
RAW_DATA_DIR = f'{DATASET_DIR}/raw'
GALLERY_DIR = f'{RAW_DATA_DIR}/gallery'

GALLERY_MAPPING_PATH = f'{GALLERY_DIR}/gallery_mapping.json'
GALLERY_MAPPING_BILINGUAL_PATH = f'{GALLERY_DIR}/gallery_mapping_bilingual.json'
RADAR_RAW_DIR = f'{RAW_DATA_DIR}/radar-dataset'
RADAR_LABEL_PATH = f'{RADAR_RAW_DIR}/labels_cleaned.csv'
RADAR_DIR = f'{RAW_DATA_DIR}/radar'
LOGO_DIR = f'{RAW_DATA_DIR}/logo'
PPT_DIR = f'{INPUT_DIR}/slides'
IMAGE_DIR = f'{INPUT_DIR}/extracted_images'
MULTI_LABEL_EXPERIMENTS_DIR = f'{__package__}/multi_label_classifier/experiments'

__all__ = [
    'INPUT_DIR',
    'DATASET_DIR',
    'RAW_DATA_DIR',
    'GALLERY_DIR',
    'GALLERY_MAPPING_PATH',
    'GALLERY_MAPPING_BILINGUAL_PATH',
    'RADAR_RAW_DIR',
    'RADAR_LABEL_PATH',
    'RADAR_DIR',
    'LOGO_DIR',
    'PPT_DIR',
    'IMAGE_DIR',
    'MULTI_LABEL_EXPERIMENTS_DIR',
]
