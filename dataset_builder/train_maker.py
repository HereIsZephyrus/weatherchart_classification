"""
Training dataset generate module
"""

import json
import os
import random
import logging
import ast
from pathlib import Path
from typing import List, Dict, Any
from enum import Enum
from pydantic import BaseModel
import pandas as pd
from .chart import Chart, ChartMetadata
from .chart_enhancer import ChartEnhancer, EnhancerConfig, EnhancerConfigPresets
from ..constants import GALLERY_DIR, EPOCH_NUM, SAMPLE_PER_BATCH, BATCH_PER_EPOCH, DATASET_DIR

logger = logging.getLogger(__name__)

class ProgressStatus(str, Enum):
    """
    status of the data batch
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class BatchMetedata(BaseModel):
    """
    metadata for a batch of data
    """
    batch_id: int
    name: str
    source_path: str
    save_path: str
    size: int
    status: ProgressStatus

class EpochMetedata(BaseModel):
    """
    metadata for an epoch of data
    """
    epoch_id: int
    name: str
    save_path: str
    status: ProgressStatus

class BatchBuilder:
    """
    builder for the data batch
    """
    def __init__(self, metadata: BatchMetedata, enhancer_config: EnhancerConfig = EnhancerConfigPresets["None"]):
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
        logger.debug("Bootstrap sampling generated %d samples from %s", len(sampled_images), folder_path)
        return sampled_images

    def select_type(self) -> List[str]:
        """
        select the type of the data batch
        """
        percentage = self.metadata.role.value[1]

        gallery_path = Path(self.metadata.source_path)
        type_dirs = [str(d) for d in gallery_path.iterdir() if d.is_dir()]
        if not type_dirs:
            logger.warning("No subdirectories found in %s", self.metadata.source_path)
            return []

        size = int(len(type_dirs) * percentage)
        selected_types = random.choices(type_dirs, k=size)
        return selected_types

    def generate_image_dataset(self, image_path_list: List[str]) -> List[ChartMetadata]:
        """
        generate the data for a data batch
        """
        metadata_list : List[ChartMetadata] = []
        index_list = list(range(len(image_path_list)))
        random.shuffle(index_list)
        for i, image_path in enumerate(image_path_list):
            index = index_list[i]
            chart = self.enhancer(Chart(image_path, index=index))
            chart.save(self.metadata.save_path / "images" / f"{index:04d}.png")
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
        batch_dir = Path(self.metadata.save_path)
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
        ecmwf_labels = self.generate_image_dataset(image_path_list=sampled_images)

        #radar_num = self.metadata.size - len(sampled_images)
        #radar_labels = self.sample_huggingface_dataset(
        #    sample_num=radar_num,
        #    save_dir=images_dir,
        #    source_dir=RADAR_DIR
        #)
        #total_labels = ecmwf_labels
        return ecmwf_labels

class EpochBuilder:
    """
    manager for one epoch of data
    """
    def __init__(self, metadata: EpochMetedata):
        self.metadata = metadata
        self.batch_metadata_list: List[BatchMetedata] = []

    def generate_epoch_build_task(self) -> None:
        """
        generate the dataset build task
        """
        self.batch_metadata_list.clear()
        # generate training batches
        for i in range(BATCH_PER_EPOCH):
            name = f"batch_{i:02d}"
            self.batch_metadata_list.append(
                BatchMetedata(
                    batch_id=self.metadata.epoch_id * 100 + i,
                    name=name,
                    source_path=GALLERY_DIR,
                    save_path=f"{self.metadata.save_path}/{name}",
                    size=SAMPLE_PER_BATCH,
                    status=ProgressStatus.PENDING,
                )
            )

    def build(self) -> None:
        """
        main function to build the dataset
        """
        self.generate_epoch_build_task()
        # ensure the output directory exists
        self.metadata.save_path.mkdir(parents=True, exist_ok=True)

        preset_keys = list(EnhancerConfigPresets.keys())
        preset_values = list(EnhancerConfigPresets.values())
        for metadata in self.batch_metadata_list:
            enhancer_config_index = random.randint(0, len(preset_keys) - 1)
            if preset_keys[enhancer_config_index] == "None" or \
               preset_keys[enhancer_config_index] == "ExtremeVariation":
                # if extreme, roll again to decrease the probability
                enhancer_config_index = random.randint(0, len(preset_keys) - 1)
            batch_builder = BatchBuilder(
                metadata=metadata,
                enhancer_config=preset_values[enhancer_config_index]
            )
            total_labels = batch_builder.build()
            with open(os.path.join(batch_builder.metadata.save_path, "labels.json"), "w", encoding="utf-8") as f:
                json.dump(total_labels, f, ensure_ascii=False, indent=2)
            with open(os.path.join(batch_builder.metadata.save_path, "config.json"), "w", encoding="utf-8") as f:
                json.dump(batch_builder.enhancer.config.model_dump(), f, ensure_ascii=False, indent=2)

            logger.info("Built batch %s", metadata.name)

        logger.info("All batches built for epoch %s", self.metadata.name)

class TrainingDatasetBuilder:
    """
    manager for the dataset
    """
    def __init__(self):
        self.output_dir = Path(DATASET_DIR)
        self.input_dir = Path(GALLERY_DIR)
        self.epoch_metadata_list: List[EpochMetedata] = []

    def generate_dataset_build_task(self) -> None:
        """
        generate the dataset build task
        """
        self.epoch_metadata_list.clear()
        for i in range(EPOCH_NUM):
            name = f"epoch_{i:03d}"
            self.epoch_metadata_list.append(
                EpochMetedata(
                    epoch_id=i,
                    name=name,
                    save_path=f"{self.output_dir}/{name}",
                    status=ProgressStatus.PENDING,
                )
            )

    def build(self) -> None:
        """
        main function to build the dataset
        """
        self.generate_dataset_build_task()
        for metadata in self.epoch_metadata_list:
            epoch_builder = EpochBuilder(metadata=metadata)
            epoch_builder.build()
            with open(os.path.join(epoch_builder.metadata.save_path, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(epoch_builder.metadata.model_dump(), f, ensure_ascii=False, indent=2)
            logger.info("Built epoch %s", metadata.name)

        logger.info("All epochs built")
