"""
Validation and test dataset generate module
Consider validatation and test a large epoch of data
"""

import logging
import random
from pathlib import Path
from typing import List, Dict, Tuple, ClassVar
from enum import Enum
import pandas as pd
from pydantic import BaseModel
from .chart import Chart
from .chart_enhancer import ChartEnhancer, EnhancerConfig, EnhancerConfigPresets
from ..settings import CURRENT_DATASET_DIR, GALLERY_DIR

logger = logging.getLogger(__name__)

class SampleStrategy(str, Enum):
    """
    strategy to sample the data
    """
    RANDOM = "random"
    BALANCED = "balanced"
    NONE = "none"

class DatasetRole(str, Enum):
    """
    role of the dataset
    """
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

class DatasetMetadata(BaseModel):
    """
    metadata for a batch of data
    """
    name: DatasetRole
    sample_strategy: SampleStrategy
    source_path: str
    save_path: str

class SplitConfig(BaseModel):
    """
    config for the split
    """
    seed: int = 42
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1

class DatasetBuilder:
    """
    builder for a dataset
    """
    def __init__(self):
        self.matadata_columns = ["image_path", "label", "en_name", "zh_name", "summary"]
        self.metadata_list: Dict[DatasetRole, pd.DataFrame] = {
            DatasetRole.TRAIN: pd.DataFrame(columns=self.matadata_columns),
            DatasetRole.VALIDATION: pd.DataFrame(columns=self.matadata_columns),
            DatasetRole.TEST: pd.DataFrame(columns=self.matadata_columns),
        }

    def get_enhancer_config(self, sample_strategy: SampleStrategy) -> EnhancerConfig:
        """
        get the enhancer config
        """
        if sample_strategy == SampleStrategy.NONE:
            return EnhancerConfigPresets["None"]
        if sample_strategy == SampleStrategy.BALANCED:
            return EnhancerConfigPresets["BalancedEnhance"]

        preset_keys = list(EnhancerConfigPresets.keys())
        preset_values = list(EnhancerConfigPresets.values())
        enhancer_config_index = random.randint(0, len(preset_keys) - 1)
        if preset_keys[enhancer_config_index] == "None" or \
            preset_keys[enhancer_config_index] == "ExtremeVariation":
            # if extreme, roll again to decrease the probability
            enhancer_config_index = random.randint(0, len(preset_keys) - 1)
        return preset_values[enhancer_config_index]

    def migrate(self, metadata: DatasetMetadata, image_files: List[str]):
        """
        main function to build a data batch
        """
        enhancer = ChartEnhancer(self.get_enhancer_config(metadata.sample_strategy))
        images_dir = Path(metadata.save_path)
        images_dir.mkdir(parents=True, exist_ok=True)

        current_metadata_list = pd.DataFrame(columns=self.matadata_columns)
        preview_length = len(self.metadata_list[metadata.name])
        for i, image_file in enumerate(image_files):
            current_index = preview_length + i
            chart = enhancer(Chart(image_file, index=current_index))
            chart.save(images_dir / f"t{chart.metadata.index:05d}.png")
            current_metadata_list = current_metadata_list.append(chart.metadata)
        self.metadata_list[metadata.name] = pd.concat([self.metadata_list[metadata.name], current_metadata_list])

class DataSpliter:
    """
    split raw dataset into training, validation and test datasets
    """
    def __init__(self, split_config: SplitConfig):
        random.seed(split_config.seed)
        self.split_config = split_config
        self.input_dir = Path(GALLERY_DIR)
        self.output_dir = Path(CURRENT_DATASET_DIR)

    def split_images(self, sample_num: int, folder_path: str = "") -> Tuple[int, int, int, List[str]]:
        """
        shuffle and split the image files into train, validation and test sets
        """
        # Get list of image files in the folder
        folder = Path(folder_path)
        image_files = [str(f) for f in folder.glob("*.webp")]
        image_files.extend([str(f) for f in folder.glob("*.png")])
        if not image_files:
            return -1, -1, -1, []

        random.shuffle(image_files)
        train_num = int(sample_num * self.split_config.train_ratio)
        validation_num = int(sample_num * self.split_config.validation_ratio)
        return 0, train_num, train_num + validation_num, image_files

    def get_types(self, source_path: str) -> List[str]:
        """
        get the types(subfolder names) of the data
        """
        folder = Path(source_path)
        return [item.name for item in folder.iterdir() if item.is_dir()]

    split_num: ClassVar[int] = 4
    def split(self):
        """
        split the data into train, validation and test sets and call Migrate to build the data batch
        """
        types = self.get_types(self.input_dir)
        builder = DatasetBuilder()
        for type_name in types:
            train_index, validation_index, test_index, image_files = self.split_images(sample_num=100, folder_path=self.input_dir / type_name)
            if train_index == -1:
                logger.warning("No image files found in %s", type_name)
                continue
            train_list = image_files[train_index:validation_index-1]
            tri_list_length = len(train_list) // self.split_num
            for index in range(self.split_num):
                if index < self.split_num - 1:
                    start_index = tri_list_length * index
                    end_index = tri_list_length * (index + 1)
                else:
                    start_index = tri_list_length * index
                    end_index = len(train_list)
                builder.migrate(DatasetMetadata(
                    name=DatasetRole.TRAIN,
                    sample_strategy=SampleStrategy.RANDOM,
                    source_path=self.input_dir / type_name,
                    save_path=self.output_dir / "images" / "train" / type_name,
                ),image_files=train_list[start_index:end_index])
            logger.info("Migrate %s train data", type_name)
            builder.migrate(DatasetMetadata(
                name=DatasetRole.VALIDATION,
                sample_strategy=SampleStrategy.BALANCED,
                source_path=self.input_dir / type_name,
                save_path=self.output_dir / "images" / "validation" / type_name,
            ),image_files=train_list[validation_index:test_index])
            logger.info("Migrate %s validation data", type_name)
            builder.migrate(DatasetMetadata(
                name=DatasetRole.TEST,
                sample_strategy=SampleStrategy.BALANCED,
                source_path=self.input_dir / type_name,
                save_path=self.output_dir / "images" / "test" / type_name,
            ),image_files=train_list[test_index:])
            logger.info("Migrate %s test data", type_name)

if __name__ == "__main__":
    spliter = DataSpliter(SplitConfig())
    spliter.split()
