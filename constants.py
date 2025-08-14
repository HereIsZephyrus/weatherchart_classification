"""
Constants for the project
"""

TRAIN_DATA_DIR = f'{__package__}/train_data'
GALLERY_DIR = f'{TRAIN_DATA_DIR}/gallery'
GALLERY_MAPPING_PATH = f'{GALLERY_DIR}/gallery_mapping.json'
GALLERY_MAPPING_BILINGUAL_PATH = f'{GALLERY_DIR}/gallery_mapping_bilingual.json'
RADAR_RAW_DIR = f'{TRAIN_DATA_DIR}/radar-dataset'
RADAR_LABEL_PATH = f'{RADAR_RAW_DIR}/labels_cleaned.csv'
RADAR_DIR = f'{TRAIN_DATA_DIR}/radar'
INPUT_DIR = f'{__package__}/income'
PPT_DIR = f'{INPUT_DIR}/slides'
IMAGE_DIR = f'{INPUT_DIR}/extracted_images'
LOGO_DIR = f'{TRAIN_DATA_DIR}/logo'
IMAGE_SIZE = (480, 360)
DATASET_DIR = f'{__package__}/dataset'
BATCH_NUM = 10
SINGLE_BATCH_SIZE = 500

__all__ = [
    'TRAIN_DATA_DIR',
    'GALLERY_DIR',
    'GALLERY_MAPPING_PATH',
    'GALLERY_MAPPING_BILINGUAL_PATH',
    'RADAR_RAW_DIR',
    'RADAR_LABEL_PATH',
    'RADAR_DIR',
    'INPUT_DIR',
    'PPT_DIR',
    'IMAGE_DIR',
    'LOGO_DIR',
    'IMAGE_SIZE',
    'DATASET_DIR',
    'BATCH_NUM',
    'SINGLE_BATCH_SIZE',
]
