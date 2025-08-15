"""
Dataset classes for weather chart classification with CNN-RNN framework.
"""
import logging
from typing import List, Dict, Tuple, Optional, Any
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from .config import ModelConfig
from .utils import LabelProcessor
from .vocab import vocabulary

logger = logging.getLogger(__name__)

class WeatherChartDataset(Dataset):
    """
    Dataset class for weather chart classification.
    """

    def __init__(
        self,
        data_samples: List[Dict[str, Any]],
        label_processor: LabelProcessor,
        image_transform=None,
        order_strategy: str = "frequency",
        label_frequencies: Optional[Dict[str, int]] = None
    ):
        self.data_samples = data_samples
        self.label_processor = label_processor
        self.image_transform = image_transform
        self.order_strategy = order_strategy
        self.label_frequencies = label_frequencies

        logger.info("Initialized WeatherChartDataset with %d samples", len(self.data_samples))

    def __len__(self):
        return len(self.data_samples)

    def __getitem__(self, idx):
        sample = self.data_samples[idx]
        
        # Load image
        image_path = sample["image_path"]
        try:
            image = Image.open(image_path).convert("RGB")
            if self.image_transform:
                image = self.image_transform(image)
        except Exception as e:
            logger.error("Error loading image %s: %s", image_path, str(e))
            # Return a placeholder image if loading fails
            image = torch.zeros((3, 224, 224))
        
        # Process labels
        label_names = sample["labels"]
        
        # Create input sequence for training
        input_sequence = self.label_processor.create_sequence(
            label_names,
            self.order_strategy,
            self.label_frequencies
        )
        
        # Create multi-hot vector for parallel BCE loss
        multi_hot = self.label_processor.create_multi_hot_vector(label_names)
        
        return {
            "image": image,
            "input_sequence": input_sequence,
            "multi_hot": multi_hot,
            "labels": label_names,
            "image_path": image_path
        }


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