"""
This file is the entry point of the gallery crawler system,
including both local HTML filtering and remote ECMWF website crawling capabilities.

Usage:
    python craw.py
"""

import os
import logging
import multiprocessing
from typing import Dict, List
from crawler import download_gallery_task, Crawler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

GALLERY_DIR = 'train_data/gallery'

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

    os.makedirs(GALLERY_DIR, exist_ok=True)
    with multiprocessing.Pool() as pool:
        tasks = []
        for kind, urls in gallery.items():
            task = pool.apply_async(download_gallery_task, args=(GALLERY_DIR, kind, urls))
            tasks.append(task)

        for task in tasks:
            task.get()

    logger.info("Downloaded gallery to %s", GALLERY_DIR)

if __name__ == "__main__":
    craw_from_ecmwf()
