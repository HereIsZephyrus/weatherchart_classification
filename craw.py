#!/usr/bin/env python3
"""
Comprehensive Example: Weather Chart Gallery Crawler

This example demonstrates the complete functionality of the weather chart
gallery crawler system,
including both local HTML filtering and remote
ECMWF website crawling capabilities.

Usage:
    python example_usage.py
"""

import os
import logging
import multiprocessing
from typing import Dict, List
from crawler import download_gallery_task, Crawler, GalleryInspector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

param_list =  [ 'Wind',
                'Mean sea level pressure',
                'Temperature',
                'Geopotential',
                'Precipitation',
                'Cloud',
                'Water vapour',
                'Humidity',
                'Indices',
                'Snow',
                'Ocean waves',
                'Surface characteristics',
                'Tropical cyclones']

def craw_from_ecmwf():
    """Demonstrate remote ECMWF website operations."""
    crawler = Crawler()
    # test for wind
    gallery : Dict[str, List[str]] = {}
    for param in param_list:
        logger.info("Filtering by %s", param)
        crawler.filter([param])
        gallery[param] = crawler.extract_chart_hrefs()

    gallery = crawler.reorganize_gallery(gallery)

    os.makedirs("gallery", exist_ok=True)
    with multiprocessing.Pool() as pool:
        tasks = []
        for kind, urls in gallery.items():
            task = pool.apply_async(download_gallery_task, args=(kind, urls))
            tasks.append(task)

        for task in tasks:
            task.get()

    inspector = GalleryInspector("gallery")
    inspector.inspect()
    inspector.gallery_info()

if __name__ == "__main__":
    craw_from_ecmwf()
