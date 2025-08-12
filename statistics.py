"""
This file is the entry point of the inspector system to inspect the data set,
including the coverage, missing items, and projection distribution.

Usage:
    python inspect.py           # inspect all data sets
    python inspect.py gallery   # inspect the ECMWF gallery in GALLERY_DIR
    python inspect.py ppt       # inspect the PPT slides and extracted images in income/
"""

import logging
import os
import sys
from inspector import GalleryInspector, PPTInspector
from constants import GALLERY_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_gallery(save_html: bool = True):
    """
    Inspect the ECMWF gallery in GALLERY_DIR
    """
    gallery_inspector = GalleryInspector(base_dir=GALLERY_DIR)
    gallery_inspector.inspect()
    logger.info("Inspected the ECMWF gallery")
    gallery_inspector.info()
    if save_html:
        out = os.path.abspath(os.path.join("reports", "gallery_report.html"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        gallery_inspector.save_html(out)
        logger.info("Saved gallery HTML report to %s", out)


def inspect_ppt(save_html: bool = True):
    """
    Inspect the PPT slides and extracted images in income/
    """
    ppt_inspector = PPTInspector(income_dir="income")
    ppt_inspector.inspect_all()
    logger.info("Inspected PPT slides and extracted images under income/")
    if save_html:
        out = os.path.abspath(os.path.join("reports", "ppt_report.html"))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        ppt_inspector.save_html(out)
        logger.info("Saved PPT HTML report to %s", out)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["gallery", "ppt"]:
            logger.error("Unknown mode: %s (expected 'gallery' or 'ppt')", mode)
            sys.exit(1)
        if mode == "gallery":
            logger.info("Inspecting the ECMWF gallery in GALLERY_DIR")
            inspect_gallery()
        elif mode == "ppt":
            logger.info("Inspecting the PPT slides and extracted images in INPUT_DIR")
            inspect_ppt()
        logger.info("Inspected the data set")
    else:
        logger.info("Inspecting the ECMWF gallery in GALLERY_DIR and PPT slides and extracted images in INPUT_DIR")
        inspect_gallery()
        inspect_ppt()
