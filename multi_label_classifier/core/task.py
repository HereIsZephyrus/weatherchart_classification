"""
Training dataset generate module
"""

import os
import random
import logging
from enum import Enum
import ast
from pydantic import BaseModel
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from ..settings import EPOCH_NUM, SAMPLE_PER_BATCH, CURRENT_DATASET_DIR

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

class DatasetReader(Dataset):
    """
    Pandas-based weather chart dataset class
    使用Pandas读取和解析CSV数据的气象图表数据集
    """

    def __init__(self, csv_file: str):
        """
        Initialize Pandas dataset

        Args:
            csv_file: Path to CSV metadata file
            images_dir: Directory containing images
            transform: Image transform pipeline
        """
        self.csv_file = csv_file
        self.dataset_type = csv_file.split("/")[-1].split(".")[0] # train, validation, test
        self.images_dir = f"{CURRENT_DATASET_DIR}/images/{self.dataset_type}"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load CSV data using pandas
        self.df = pd.read_csv(csv_file)
        logger.info("Loaded %d samples from %s using pandas", len(self.df), csv_file)

        # Preprocess labels
        self._preprocess_labels()

    def _preprocess_labels(self):
        """Preprocess label column using pandas operations"""
        def parse_labels(label_str):
            """Parse label string to list"""
            if pd.isna(label_str) or label_str == '':
                return []
            try:
                # Handle string representation of list
                if isinstance(label_str, str):
                    return ast.literal_eval(label_str)
                return label_str if isinstance(label_str, list) else []
            except (ValueError, SyntaxError):
                logger.warning("Failed to parse labels: %s", label_str)
                return []

        self.df['parsed_labels'] = self.df['label'].apply(parse_labels)
        if "radar" in self.df['parsed_labels']:
            suffix = "png"
        else:
            suffix = "webp"
        self.df['image_path'] = self.df['index'].apply(
            lambda idx: os.path.join(self.images_dir, f"{int(float(idx)):04d}.{suffix}")
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get item by index"""
        row = self.df.iloc[idx]
        image_path = row['image_path']
        try:
            image = Image.open(image_path).convert('RGB')
        except (IOError, OSError) as e:
            logger.warning("Error loading image %s: %s", image_path, str(e))
            image = torch.zeros((3, 224, 224))

        image = self.transform(image)

        return {
            'image': image,
            'labels': row['parsed_labels'],
            'en_name': row['en_name'],
            'zh_name': row['zh_name'],
            'summary': row['summary'] if pd.notna(row['summary']) else '',
            'index': int(float(row['index'])),
            'image_path': image_path
        }

    def get_sample_info(self):
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self.df),
            'unique_labels': self.df['parsed_labels'].apply(len).sum(),
            'avg_labels_per_sample': self.df['parsed_labels'].apply(len).mean(),
            'sample_info': self.df[['en_name', 'zh_name', 'index']].head(5).to_dict('records')
        }
        return stats

class DatasetIterator:
    """
    Custom iterator for training with specified samples per batch in one epoch
    """

    def __init__(self, metadata_file: str):
        """
        Initialize dataset iterator

        Args:
            metadata_file: CSV metadata file path
        """
        self.dataset = DatasetReader(metadata_file)
        self.current_batch = 0
        self.batch_size = SAMPLE_PER_BATCH
        self.batches_per_epoch = (len(self.dataset)-1) // self.batch_size + 1
        self.indices = list(range(len(self.dataset)))

    def __iter__(self):
        self.current_batch = 0
        random.shuffle(self.indices)
        return self

    def __next__(self):
        start_idx = self.current_batch * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.dataset))
        if start_idx >= len(self.dataset):
            raise StopIteration
        batch_data = [self.dataset[i] for i in self.indices[start_idx:end_idx]]

        self.current_batch += 1
        if self.current_batch >= self.batches_per_epoch:
            self.current_batch = 0

        return batch_data

class DatasetLoader(DataLoader):
    """
    Custom DataLoader that inherits from PyTorch DataLoader
    """

    def __init__(self, dataset, epochs: int = EPOCH_NUM, **kwargs):
        """
        Initialize DatasetLoader

        Args:
            dataset: Dataset to load from
            epochs: Number of epochs to iterate
            **kwargs: Additional arguments for DataLoader
        """
        super().__init__(dataset, **kwargs)
        self.epochs = epochs
        self.current_epoch = 0

    def __iter__(self):
        """Iterate through dataset for specified number of epochs"""
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            logger.debug("Starting epoch %d/%d", epoch + 1, self.epochs)

            # Get iterator for current epoch
            epoch_iterator = super().__iter__()

            for batch in epoch_iterator:
                yield batch

    def get_epoch_iterator(self):
        """Get iterator for single epoch"""
        return super().__iter__()

    def total_batches(self):
        """Get total number of batches across all epochs"""
        return len(self) * self.epochs

class DatasetFactory:
    """
    Factory class for creating weather chart datasets and data loaders
    """

    def __init__(self, dataset_base_path: str = None):
        """
        Initialize WeatherDatasetLoader

        Args:
            dataset_base_path: Base path to dataset directory containing metadata and images folders
        """
        self.dataset_base_path = dataset_base_path or CURRENT_DATASET_DIR
        logger.info("Initialized DatasetLoader with dataset path: %s", self.dataset_base_path)

    def load_dataset(self, dataset_type: str) -> DatasetReader:
        """
        Load train, validation or test datasets from CSV files.

        Returns:
            DatasetReader
        """
        csv_file = os.path.join(self.dataset_base_path, "metadata", f"{dataset_type}.csv")

        try:
            dataset = DatasetReader(csv_file)

            logger.info("Loaded datasets - %s: %d samples", dataset_type, len(dataset))
            return dataset

        except Exception as e:
            logger.error("Error loading CSV datasets: %s", str(e))
            raise

    def create_dataloader(
        self,
        dataset: DatasetReader,
        batch_size: int = SAMPLE_PER_BATCH,
        num_workers: int = 8
    ) -> DatasetLoader:
        """
        Create PyTorch DataLoaders for training, validation and test datasets.
        """
        def collate_fn(batch):
            """Custom collate function for batching samples"""
            images = torch.stack([item['image'] for item in batch])
            labels = [item['labels'] for item in batch]
            en_names = [item['en_name'] for item in batch]
            zh_names = [item['zh_name'] for item in batch]
            summaries = [item['summary'] for item in batch]
            indices = torch.tensor([item['index'] for item in batch])
            image_paths = [item['image_path'] for item in batch]

            return {
                'images': images,
                'labels': labels,
                'en_names': en_names,
                'zh_names': zh_names,
                'summaries': summaries,
                'indices': indices,
                'image_paths': image_paths
            }

        return DatasetLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            epochs=EPOCH_NUM
        )
