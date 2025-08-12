"""
This file is the entry point of the pptx image extractor system,
including both local PPT extraction and image classification.

Usage:
    python extract.py
"""

import os
import logging
from extractor import Extractor, SourceClassifier
from .constants import PPT_DIR, IMAGE_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def extract_and_classify_source_from_pptx():
    """
    Extract images from PPTX files and classify their sources
    The extracted images are saved in the IMAGE_DIR
    The pptx to extract are saved in the PPT_DIR
    """
    ppt_extractor : Extractor = Extractor(output_dir=IMAGE_DIR)
    source_classifier : SourceClassifier = SourceClassifier(image_dir=IMAGE_DIR)
    ppt_files = ppt_extractor.find_pptx_files(PPT_DIR)
    for ppt_file in ppt_files:
        ppt_extractor.extract_images_from_ppt(ppt_file)

    nmc_image_list = []
    for image in os.listdir(IMAGE_DIR):
        source = source_classifier.classify_source(os.path.join(IMAGE_DIR, image))
        logger.debug("Image %s is from %s", image, source)
        if source == "nmc":
            nmc_image_list.append(os.path.join(IMAGE_DIR, image))

    source_classifier.check_nmc_image(nmc_image_list)
    source_classifier.save_classified_list("classified_list.json")

if __name__ == '__main__':
    extract_and_classify_source_from_pptx()
