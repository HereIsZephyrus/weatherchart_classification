"""
entry point for the radar data parser
"""

import logging
from .gate_classifier import parse_non_weatherchart

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parse_non_weatherchart()
