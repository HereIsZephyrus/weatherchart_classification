"""
entry point for the dataset maker
"""

import logging
from .inspector import DatasetManager, DatasetConfig
from .constants import BATCH_NUM, SINGLE_BATCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_dataset():
    """
    make the dataset
    """
    logger.info("make dataset")
    dataset_manager = DatasetManager(DatasetConfig(batch_num=BATCH_NUM, single_batch_size=SINGLE_BATCH_SIZE))
    dataset_manager.build_dataset()

if __name__ == "__main__":
    make_dataset()
