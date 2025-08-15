"""
settings for the multi-label classifier
"""
from ..constants import *

IMAGE_SIZE = (224, 224)
EPOCH_NUM = 50
BATCH_PER_EPOCH = 5
SAMPLE_PER_BATCH = 256

__all__ = [
    'GALLERY_DIR',
    'IMAGE_SIZE',
    'EPOCH_NUM',
    'BATCH_PER_EPOCH',
    'SAMPLE_PER_BATCH',
]