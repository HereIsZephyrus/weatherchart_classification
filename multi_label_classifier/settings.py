"""
settings for the multi-label classifier
"""
from ..constants import *

CURRENT_DATASET_DIR = f'{DATASET_DIR}/multi_label_classifier'
IMAGE_SIZE = (224, 224)
EPOCH_NUM = 50
SAMPLE_PER_BATCH = 128
SAVE_FREQUENCY = 5
NUM_WORKERS = 8
LOCAL_RANK = -1

__all__ = [
    'GALLERY_DIR',
    'CURRENT_DATASET_DIR',
    'IMAGE_SIZE',
    'EPOCH_NUM',
    'SAMPLE_PER_BATCH',
]
