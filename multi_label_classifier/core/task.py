"""
Training dataset generate module
"""

import json
import os
import random
import logging
from typing import Tuple
from enum import Enum
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torchvision import transforms
from ..preprocess import Chart, ChartMetadata, ChartEnhancer, EnhancerConfig, EnhancerConfigPresets
from ..settings import EPOCH_NUM, SAMPLE_PER_BATCH, DATASET_DIR

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

class DatasetLoader:
    """
    Dataset loader for weather chart classification.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.label_processor = LabelProcessor(vocabulary.token2idx)

        # Default image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        logger.info("Initialized DatasetLoader")

    def load_dataset(self, dataset_path: str) -> Tuple[WeatherChartDataset, WeatherChartDataset]:
        """
        Load dataset from path and split into train/val sets.
        
        Args:
            dataset_path: Path to dataset directory
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        # Load dataset metadata
        metadata_path = os.path.join(dataset_path, "labels.json")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                all_samples = json.load(f)
        except Exception as e:
            logger.error("Error loading metadata from %s: %s", metadata_path, str(e))
            raise
            
        logger.info("Loaded %d samples from metadata", len(all_samples))
        
        # Process samples to add image paths
        processed_samples = []
        for sample in all_samples:
            # Create sample with image path
            processed_sample = {
                "image_path": os.path.join(dataset_path, "images", f"{sample['index']:04}.webp"),
                "labels": sample["label"],
                "index": sample["index"],
                "en_name": sample["en_name"],
                "zh_name": sample["zh_name"]
            }
            processed_samples.append(processed_sample)
        
        # Calculate label frequencies for ordering
        label_frequencies = {}
        for sample in processed_samples:
            for label in sample["labels"]:
                label_frequencies[label] = label_frequencies.get(label, 0) + 1
        
        # Split into train/val
        random.seed(42)  # For reproducibility
        random.shuffle(processed_samples)
        
        split_idx = int(len(processed_samples) * 0.9)  # 90% train, 10% val
        train_samples = processed_samples[:split_idx]
        val_samples = processed_samples[split_idx:]
        
        # Create datasets
        train_dataset = WeatherChartDataset(
            train_samples,
            self.label_processor,
            self.image_transform,
            order_strategy="frequency",
            label_frequencies=label_frequencies
        )
        
        val_dataset = WeatherChartDataset(
            val_samples,
            self.label_processor,
            self.image_transform,
            order_strategy="fixed"  # Fixed order for validation
        )
        
        logger.info("Split dataset into %d train and %d validation samples", len(train_dataset), len(val_dataset))
        return train_dataset, val_dataset
    
    def create_data_loaders(
        self,
        train_dataset: WeatherChartDataset,
        val_dataset: WeatherChartDataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create data loaders for training and validation.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            batch_size: Batch size
            num_workers: Number of workers for data loading
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        def collate_fn(batch):
            # Extract components
            images = torch.stack([item["image"] for item in batch])
            input_sequences = [item["input_sequence"] for item in batch]
            multi_hot_vectors = torch.stack([item["multi_hot"] for item in batch])
            
            # Pad sequences and create attention masks
            padded_sequences, attention_masks = self.label_processor.pad_sequences(input_sequences)
            
            # Create target sequences for teacher forcing (input shifted by 1)
            target_sequences = torch.zeros_like(padded_sequences)
            for i, seq in enumerate(input_sequences):
                target_seq = seq[1:]  # Remove first token (BOS)
                target_sequences[i, :len(target_seq)] = target_seq
            
            return {
                "images": images,
                "input_sequences": padded_sequences,
                "target_sequences": target_sequences,
                "attention_masks": attention_masks,
                "multi_hot_vectors": multi_hot_vectors,
                "labels": [item["labels"] for item in batch],
                "image_paths": [item["image_path"] for item in batch]
            }
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
        
        return train_loader, val_loader
