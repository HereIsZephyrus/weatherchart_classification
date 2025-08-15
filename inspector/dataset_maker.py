"""
Dataset generate module
"""

import json
import os
import random
import logging
import ast
from pathlib import Path
from typing import List, Dict, Any, ClassVar
from enum import Enum
from pydantic import BaseModel
import pandas as pd
from .chart import Chart, ChartMetadata
from .chart_enhancer import ChartEnhancer, EnhancerConfig, EnhancerConfigPresets
from ..constants import DATASET_DIR, GALLERY_DIR#, RADAR_DIR

logger = logging.getLogger(__name__)

class DataBatchRole(tuple[str, float], Enum):
    """
    role of the data batch.
    the first element is the role name,
    the second element is the type percentage that contained in one data batch.
    """
    TRAIN = ("train", 0.7)
    VALIDATION = ("validation", 1.0)
    TEST = ("test", 0.9)

class DataBatchStatus(str, Enum):
    """
    status of the data batch
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class DataBatchMetedata(BaseModel):
    """
    metadata for a batch of data
    """
    batch_id: int
    name: str
    path: str
    size: int
    progress: float
    status: DataBatchStatus
    role: DataBatchRole

class DatasetConfig(BaseModel):
    """
    config for the dataset
    """
    EPOCH_NUM: int
    SINGLE_EXPOCH_SIZE: int
    train_percent : ClassVar[float] = 0.7
    validation_percent : ClassVar[float] = 0.1
    test_percent : ClassVar[float] = 0.2

class DataBatchBuilder:
    """
    builder for the data batch
    """
    def __init__(self, metadata: DataBatchMetedata, enhancer_config: EnhancerConfig = EnhancerConfigPresets["None"]):
        self.metadata = metadata
        self.enhancer = ChartEnhancer(enhancer_config)

    def boostraping(self, sample_num: int, folder_path: str = "") -> List[str]:
        """
        boostraping sample list
        """
        # Get list of image files in the folder
        folder = Path(folder_path)
        image_files = [str(f) for f in folder.glob("*.webp")]
        image_files.extend([str(f) for f in folder.glob("*.png")])
        if not image_files:
            return []

        # Bootstrapping
        sampled_images = random.choices(image_files, k=sample_num)
        logger.info("Bootstrap sampling generated %d samples from %s", len(sampled_images), folder_path)
        return sampled_images

    def select_type(self) -> List[str]:
        """
        select the type of the data batch
        """
        percentage = self.metadata.role.value[1]

        # Get list of subdirectories in GALLERY_DIR
        gallery_path = Path(GALLERY_DIR)
        type_dirs = [str(d) for d in gallery_path.iterdir() if d.is_dir()]
        if not type_dirs:
            logger.warning("No subdirectories found in %s", GALLERY_DIR)
            return []

        size = int(len(type_dirs) * percentage)
        selected_types = random.choices(type_dirs, k=size)
        return selected_types

    def generate_image_dataset(self, image_path_list: List[str], save_dir: Path) -> List[ChartMetadata]:
        """
        generate the data for a data batch
        """
        metadata_list : List[ChartMetadata] = []
        index_list = list(range(len(image_path_list)))
        random.shuffle(index_list)
        for i, image_path in enumerate(image_path_list):
            index = index_list[i]
            chart = self.enhancer(Chart(image_path, index=index))
            chart.save(save_dir / f"{index:04d}.png")
            metadata_list.append(chart.metadata)

        return metadata_list

    def sample_huggingface_dataset(self, sample_num: int, save_dir: Path, source_dir: str) -> List[ChartMetadata]:
        """
        generate the radar data for a data batch
        """
        metadata_list : List[ChartMetadata] = []

        # Get list of image files in the source directory
        source_path = Path(source_dir)
        images_path = source_path / "images"
        labels_df = pd.read_csv(
            source_path / "labels.csv",
            converters={'feature': lambda x: ast.literal_eval(x) if pd.notna(x) else []}
        )
        labels_df = labels_df.sample(n=sample_num)

        save_index = self.metadata.size - 1
        for row in labels_df.itertuples():
            image_path = images_path / f"{row.index}.png"
            chart = self.enhancer(Chart(image_path, index=save_index, info=row))
            chart.save(save_dir / f"{save_index:04d}.png")
            metadata_list.append(chart.metadata)
            save_index -= 1

        return metadata_list

    def build(self) -> Dict[str, Any]:
        """
        main function to build a data batch
        """
        batch_dir = Path(self.metadata.path)
        batch_dir.mkdir(parents=True, exist_ok=True)

        images_dir = batch_dir / "images"
        images_dir.mkdir(exist_ok=True)

        types = self.select_type()
        image_num_each_type = int(self.metadata.size / (len(types)))

        sampled_images = []
        for type_folder in types:
            type_images = self.boostraping(sample_num=image_num_each_type, folder_path=type_folder)
            if not type_images:
                logger.warning("No image files found in %s", type_folder)
                continue
            sampled_images.extend(type_images)
        remaining_images = self.metadata.size - len(sampled_images)
        ramin_type = random.choice(types)
        ramin_images = self.boostraping(sample_num=remaining_images, folder_path=ramin_type)
        sampled_images.extend(ramin_images)
        ecmwf_labels = self.generate_image_dataset(
            image_path_list=sampled_images,
            save_dir=images_dir
        )

        #radar_num = self.metadata.size - len(sampled_images)
        #radar_labels = self.sample_huggingface_dataset(
        #    sample_num=radar_num,
        #    save_dir=images_dir,
        #    source_dir=RADAR_DIR
        #)
        #total_labels = ecmwf_labels
        return ecmwf_labels

class DatasetManager:
    """
    manager for the dataset
    """
    def __init__(self, config: DatasetConfig):
        self.dataset_dir = Path(DATASET_DIR)
        self.config = config
        self.batch_metadata_list: List[DataBatchMetedata] = []
        self.standardize_percentage()

    def standardize_percentage(self) -> None:
        """
        ensure the sum of the percentage of train, validation and test is 1
        """
        total = self.config.train_percent + self.config.validation_percent + self.config.test_percent

        if abs(total - 1.0) > 1e-6:
            logger.warning("The sum of the percentage is not 1 (%.3f), automatically standardize", total)

            self.config.train_percent /= total
            self.config.validation_percent /= total
            self.config.test_percent /= total

            logger.info("After standardization, the ratio - train: %.2f, validation: %.2f, test: %.2f",
                       self.config.train_percent,
                       self.config.validation_percent,
                       self.config.test_percent)

    def construct_metadata(self, bid : int, size: int, role: DataBatchRole) -> DataBatchMetedata:
        """
        construct the metadata for a batch of data
        """
        name = f"{role.value[0]}_batch_{bid:04d}"
        return DataBatchMetedata(
            batch_id=bid,
            name=name,
            path=f"{DATASET_DIR}/{name}",
            size=size,
            progress=0,
            status=DataBatchStatus.PENDING,
            role=role
        )

    def generate_dataset_build_task(self) -> None:
        """
        generate the dataset build task
        """
        self.batch_metadata_list.clear()

        # calculate the number of batches for each dataset
        train_batches = int(self.config.EPOCH_NUM * self.config.train_percent)
        val_batches = int(self.config.EPOCH_NUM * self.config.validation_percent)
        test_batches = self.config.EPOCH_NUM - train_batches - val_batches

        logger.info("Generate dataset build task - train batches: %d, validation batches: %d, test batches: %d",
                   train_batches, val_batches, test_batches)

        # generate training batches
        for i in range(train_batches):
            metadata = self.construct_metadata(
                bid=i,
                size=self.config.SINGLE_EXPOCH_SIZE,
                role=DataBatchRole.TRAIN
            )
            self.batch_metadata_list.append(metadata)

        # generate validation batches
        for i in range(val_batches):
            metadata = self.construct_metadata(
                bid=train_batches + i,
                size=self.config.SINGLE_EXPOCH_SIZE,
                role=DataBatchRole.VALIDATION
            )
            self.batch_metadata_list.append(metadata)

        # generate test batches
        for i in range(test_batches):
            metadata = self.construct_metadata(
                bid=train_batches + val_batches + i,
                size=self.config.SINGLE_EXPOCH_SIZE,
                role=DataBatchRole.TEST
            )
            self.batch_metadata_list.append(metadata)

        logger.info("Generated %d batch tasks", len(self.batch_metadata_list))

    def build_dataset(self) -> None:
        """
        main function to build the dataset
        """
        self.generate_dataset_build_task()
        # ensure the output directory exists
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        preset_keys = list(EnhancerConfigPresets.keys())
        preset_values = list(EnhancerConfigPresets.values())
        for metadata in self.batch_metadata_list:
            enhancer_config_index = random.randint(0, len(preset_keys) - 1)
            if preset_keys[enhancer_config_index] == "None" or \
               preset_keys[enhancer_config_index] == "ExtremeVariation":
                # if extreme, roll again to decrease the probability
                enhancer_config_index = random.randint(0, len(preset_keys) - 1)
            builder = DataBatchBuilder(
                metadata=metadata,
                enhancer_config=preset_values[enhancer_config_index]
            )
            total_labels = builder.build()
            with open(os.path.join(builder.metadata.path, "labels.json"), "w", encoding="utf-8") as f:
                json.dump(total_labels, f, ensure_ascii=False, indent=2)
            with open(os.path.join(builder.metadata.path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(builder.enhancer.config.model_dump(), f, ensure_ascii=False, indent=2)

            logger.info("Built batch %s", metadata.name)

        logger.info("All batches built")
