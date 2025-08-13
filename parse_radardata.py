"""
entry point for the radar data parser
"""

import logging
from inspector import RadarDatasetParser
from constants import RADER_RAW_DIR

logger = logging.getLogger(__name__)

def main():
    """
    main function
    """
    logger.info("Starting radar data parser")
    parser = RadarDatasetParser()
    parser.convert_dataset(RADER_RAW_DIR)

if __name__ == "__main__":
    main()