"""
settings for the multi-label classifier
"""
import os
from ..constants import DATASET_DIR, RAW_DATA_DIR, GALLERY_DIR
CURRENT_DATASET_DIR = os.path.join(DATASET_DIR, "gate_classifier")
NON_WEATHERCHART_DIR = os.path.join(RAW_DATA_DIR, "non_weatherchart")
RAW_CHART_DIR = os.path.join(NON_WEATHERCHART_DIR, "chart")
RAW_WEATHER_DIR = os.path.join(NON_WEATHERCHART_DIR, "weather")
IMAGE_SIZE = (224, 224)
EPOCH_NUM = 50
SAMPLE_PER_BATCH = 256
SAVE_FREQUENCY = 5
DATASET_SIZE = 10000
NUM_WORKERS = 8
LOCAL_RANK = -1

__all__ = [
    'CURRENT_DATASET_DIR',
    'NON_WEATHERCHART_DIR',
    'RAW_CHART_DIR',
    'RAW_WEATHER_DIR',
    'IMAGE_SIZE',
    'EPOCH_NUM',
    'SAMPLE_PER_BATCH',
    'SAVE_FREQUENCY',
    'DATASET_SIZE',
    'NUM_WORKERS',
    'LOCAL_RANK',
    'GALLERY_DIR'
]
