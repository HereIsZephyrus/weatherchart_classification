"""
entry point for the radar data parser
"""

import logging
from .multi_label_classifier.preprocess import RadarDatasetParser

logger = logging.getLogger(__name__)

def parse_radardata():
    """
    main function
    """
    logger.info("Starting radar data parser")
    parser = RadarDatasetParser()
    parser.convert_dataset()

if __name__ == "__main__":
    parse_radardata()
