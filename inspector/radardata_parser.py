"""
parse hugging face dataset @deepguess--weather-analysis-dataset 
Convert to image-label format and save to local TRAIN_DATA_DIR/RADAR_DIR
"""

import os
import logging
from datasets import load_dataset
import pandas as pd
from ..constants import RADAR_DIR, RADAR_RAW_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RadarDatasetParser:
    """
    @deepguess--weather-analysis-dataset parser
    """

    def __init__(self):
        self.dataset = None
        self.output_dir = os.path.abspath(RADAR_DIR)
        self.images_dir = os.path.join(self.output_dir, "images")

    def setup_directories(self):
        """
        create necessary directory structure
        """
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        logger.info("directory structure created: %s", self.output_dir)

    def load_dataset(self, dataset_path: str):
        """
        load dataset from hugging face
        """
        try:
            logger.info("loading dataset: %s", dataset_path)
            self.dataset = load_dataset(
                "deepguess/weather-analysis-dataset",
                cache_dir=os.path.abspath(dataset_path)
            )
            self.dataset = self.dataset["train"]
            logger.info("dataset loaded successfully, %d samples", len(self.dataset))
        except Exception as e:
            logger.error("dataset loading failed: %s", e)
            raise RuntimeError("dataset loading failed") from e

    def convert_dataset(self) -> None:
        """
        execute complete dataset conversion process
        """
        logger.info("starting dataset conversion...")

        # set directory
        self.setup_directories()

        # load dataset
        try:
            self.load_dataset(RADAR_RAW_DIR)
        except Exception as e:
            raise RuntimeError("dataset load failed") from e

        # convert each sample
        logger.debug("dataset info: %s", self.dataset)
        labels = pd.DataFrame(columns=['index', 'en', 'summary', 'feature'])
        index = 0
        for feature in self.dataset:
            if feature['parameters_visible'] is None:
                continue
            feature['image'].save(os.path.join(self.images_dir, f"{index}.png"))
            labels.loc[index] = {
                'index' : index,
                'en' : feature['product_type'],
                'summary' : feature['context_summary'],
                'feature' : feature['parameters_visible'].split('|'),
            }
            index += 1

        labels.to_csv(os.path.join(self.output_dir, "labels.csv"), index=False, encoding="utf-8")
        logger.info("conversion completed! successfully processed %d/%d samples", len(self.dataset), len(self.dataset))
        logger.info("data saved to: %s", self.output_dir)
