"""
entry point for the dataset maker
"""

import logging
from .inspector import DatasetManager, DatasetConfig
from .constants import EPOCH_NUM, SINGLE_EXPOCH_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def make_dataset():
    """
    make the dataset
    """
    logger.info("make dataset")
    dataset_manager = DatasetManager(DatasetConfig(EPOCH_NUM=EPOCH_NUM, SINGLE_EXPOCH_SIZE=SINGLE_EXPOCH_SIZE))
    dataset_manager.build_dataset()

if __name__ == "__main__":
    make_dataset()
