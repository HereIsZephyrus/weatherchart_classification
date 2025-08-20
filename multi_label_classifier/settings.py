"""
settings for the multi-label classifier
"""
import os
from ..constants import DATASET_DIR, RAW_DATA_DIR
CURRENT_DATASET_DIR = os.path.join(DATASET_DIR, "multi_label_classifier")
RADAR_DIR = os.path.join(RAW_DATA_DIR, "radar")
LOGO_DIR = os.path.join(RAW_DATA_DIR, "logo")
IMAGE_SIZE = (224, 224)
EPOCH_NUM = 50
SAMPLE_PER_BATCH = 256
SAVE_FREQUENCY = 5
NUM_WORKERS = 8
LOCAL_RANK = -1

__all__ = [
    'CURRENT_DATASET_DIR',
    'RADAR_DIR',
    'LOGO_DIR',
    'IMAGE_SIZE',
    'EPOCH_NUM',
    'SAMPLE_PER_BATCH',
    'SAVE_FREQUENCY',
    'NUM_WORKERS',
    'LOCAL_RANK'
]
