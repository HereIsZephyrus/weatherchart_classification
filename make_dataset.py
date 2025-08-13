"""
entry point for the dataset maker
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .inspector import DatasetManager, DatasetConfig

def make_dataset():
    """
    make the dataset
    """
    logger.info("make dataset")
    dataset_manager = DatasetManager(DatasetConfig(batch_num=10, single_batch_size=1000))
    dataset_manager.build_dataset()

if __name__ == "__main__":
    make_dataset()
