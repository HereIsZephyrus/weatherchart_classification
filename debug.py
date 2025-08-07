#!/usr/bin/env python3
"""
Comprehensive Example: Weather Chart Gallery Crawler

This example demonstrates the complete functionality of the weather chart
gallery crawler system, including both local HTML filtering and remote
ECMWF website crawling capabilities.

Usage:
    python example_usage.py
"""

import logging
from crawler import Crawler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Demonstrate remote ECMWF website operations."""
    crawler = Crawler()

if __name__ == "__main__":
    main()
