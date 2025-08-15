"""
Validation and test dataset generate module
Consider validatation and test a large epoch of data
"""

import logging
from .train_maker import ProgressStatus, BatchMetedata

logger = logging.getLogger(__name__)

class ValidationDatasetBuilder:
    """
    manager for the validation dataset
    """