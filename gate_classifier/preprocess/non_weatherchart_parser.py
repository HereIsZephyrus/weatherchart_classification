"""
parse hugging face dataset @Peppertuna--ChartQADatasetV2 annd @davidshableski--weatherimages
Convert to image-label format and save to local TRAIN_DATA_DIR/NON_WEATHERCHART_DIR
"""

import os
import logging
import random
from datasets import load_dataset
from ..setting import RAW_CHART_DIR, RAW_WEATHER_DIR

logger = logging.getLogger(__name__)

class NonWeatherChartDatasetParser:
    """
    Base class for dataset parsers
    """
    def __init__(self, dataset_dir: str, dataset_name: str, splited: bool = False):
        self.dataset = None
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.dataset_name = dataset_name
        self.splited = splited

    def setup_directories(self):
        """
        create necessary directory structure
        """
        os.makedirs(self.dataset_dir, exist_ok=True)
        logger.info("directory structure created: %s", self.dataset_dir)

    def load_dataset(self):
        """
        load dataset from hugging face, dataset_name is the name of the dataset
        """
        try:
            logger.info("loading dataset: %s", self.dataset_dir)
            self.dataset = load_dataset(
                self.dataset_name,
                cache_dir=os.path.abspath(self.dataset_dir)
            )
            self.dataset = self.dataset
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
            self.load_dataset()
        except Exception as e:
            raise RuntimeError("dataset load failed") from e

        # convert each sample
        logger.debug("dataset info: %s", self.dataset)

        if not self.splited:
            dataset_split1 = self.dataset["train"].train_test_split(test_size=0.2, shuffle=True, seed=42)
            dataset_split2 = dataset_split1["train"].train_test_split(test_size=0.125, shuffle=True, seed=42)
            self.dataset = {
                "train": dataset_split2["train"],
                "validation": dataset_split2["test"],
                "test": dataset_split1["test"]
            }

        if self.splited:
            for dataset_name in ["train", "validation", "test"]:
                current_dir = os.path.join(self.dataset_dir, dataset_name)
                os.makedirs(current_dir, exist_ok=True)
                for index, feature in enumerate(self.dataset[dataset_name]):
                    if feature['parameters_visible'] is None:
                        continue
                    feature['image'].save(os.path.join(current_dir, f"{index}.png"))

        logger.info("conversion completed! successfully processed %d/%d samples", len(self.dataset), len(self.dataset))
        logger.info("data saved to: %s", self.dataset_dir)

def parse_non_weatherchart():
    """
    main function
    """
    logger.info("starting non weather chart data parser")
    chart_parser = NonWeatherChartDatasetParser(RAW_CHART_DIR, "Peppertuna/ChartQADatasetV2", splited=True)
    weather_parser = NonWeatherChartDatasetParser(RAW_WEATHER_DIR, "davidshableski/weatherimages", splited=False)
    chart_parser.convert_dataset()
    weather_parser.convert_dataset()
    logger.info("non weather chart data parser completed")
