"""
entry point for the dataset maker
"""

import logging
from .dataset_builder import EpochBuilder, DatasetConfig
from .constants import EPOCH_NUM, SAMPLE_PER_BATCH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_dataset():
    """
    make the dataset
    """
    logger.info("make dataset")
    dataset_manager = EpochBuilder(DatasetConfig(EPOCH_NUM=EPOCH_NUM, SAMPLE_PER_BATCH=SAMPLE_PER_BATCH))
    dataset_manager.build_dataset()

if __name__ == "__main__":
    make_dataset()
