"""
Dataset classes for weather chart classification with CNN-RNN framework.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional, Any
import os
import json
from PIL import Image
import logging
import pandas as pd
import random

from .config import DataConfig
from .utils import ImageTransforms, LabelProcessor

logger = logging.getLogger(__name__)


class WeatherChartDataset(Dataset):
    """
    Dataset for weather chart multi-label classification.

    Supports both sequence and parallel label formats for CNN-RNN training.
    """

    def __init__(
        self,
        data_path: str,
        label_processor: LabelProcessor,
        transform: Optional[Any] = None,
        is_training: bool = True,
        config: Optional[DataConfig] = None
    ):
        self.data_path = data_path
        self.label_processor = label_processor
        self.transform = transform
        self.is_training = is_training
        self.config = config or DataConfig()

        # Load data
        self.samples = self._load_samples()

        # Calculate label frequencies for curriculum learning
        if is_training:
            self.label_frequencies = self._calculate_label_frequencies()
        else:
            self.label_frequencies = {}

        logger.info(f"Loaded {len(self.samples)} samples from {data_path}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples from data path."""
        samples = []

        # Support different data formats
        if os.path.isfile(self.data_path):
            # Single file format (JSON or CSV)
            if self.data_path.endswith('.json'):
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if isinstance(data, list):
                    samples = data
                elif isinstance(data, dict) and 'samples' in data:
                    samples = data['samples']
                else:
                    raise ValueError(f"Unsupported JSON format in {self.data_path}")

            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
                samples = df.to_dict('records')

        elif os.path.isdir(self.data_path):
            # Directory format - load from file structure
            samples = self._load_from_directory(self.data_path)

        else:
            raise FileNotFoundError(f"Data path not found: {self.data_path}")

        # Validate and filter samples
        valid_samples = []
        for sample in samples:
            if self._validate_sample(sample):
                valid_samples.append(sample)

        logger.info(f"Validated {len(valid_samples)}/{len(samples)} samples")
        return valid_samples

    def _load_from_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """Load samples from directory structure."""
        samples = []

        # Look for images and corresponding label files
        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(root, file)

                    # Look for corresponding label file
                    base_name = os.path.splitext(file)[0]
                    label_file = os.path.join(root, f"{base_name}.json")

                    if os.path.exists(label_file):
                        with open(label_file, 'r', encoding='utf-8') as f:
                            label_data = json.load(f)

                        samples.append({
                            'image_path': image_path,
                            'labels': label_data.get('labels', []),
                            'metadata': label_data.get('metadata', {})
                        })

        return samples

    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """Validate a single sample."""
        # Check required fields
        if 'image_path' not in sample or 'labels' not in sample:
            return False

        # Check image file exists
        if not os.path.exists(sample['image_path']):
            logger.warning(f"Image not found: {sample['image_path']}")
            return False

        # Check labels format
        labels = sample['labels']
        if not isinstance(labels, list):
            return False

        # Filter out unknown labels
        valid_labels = [
            label for label in labels 
            if label in self.label_processor.label_mapping
        ]

        # Check minimum label frequency if configured
        if self.config.min_label_frequency > 0:
            # This would require pre-computed frequencies
            pass

        # Check maximum labels per sample
        if len(valid_labels) > self.config.max_labels_per_sample:
            valid_labels = valid_labels[:self.config.max_labels_per_sample]

        # Update sample with valid labels
        sample['labels'] = valid_labels

        return len(valid_labels) > 0

    def _calculate_label_frequencies(self) -> Dict[str, int]:
        """Calculate label frequencies for curriculum learning."""
        frequencies = {}

        for sample in self.samples:
            for label in sample['labels']:
                frequencies[label] = frequencies.get(label, 0) + 1

        return frequencies

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]

        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # Process labels
        labels = sample['labels']

        # Create sequence for RNN input (with BOS token)
        order_strategy = "frequency" if self.is_training else "fixed"
        if self.is_training and random.random() < self.config.random_order_ratio:
            order_strategy = "random"

        input_sequence = self.label_processor.create_sequence(
            labels, 
            order_strategy=order_strategy,
            label_frequencies=self.label_frequencies
        )

        # Create target sequence (input shifted by 1)
        target_sequence = self.label_processor.create_target_sequence(input_sequence)

        # Create multi-hot vector for parallel BCE loss
        multi_hot_labels = self.label_processor.create_multi_hot_vector(labels)

        return {
            'images': image,
            'input_sequences': input_sequence,
            'target_sequences': target_sequence,
            'target_labels': multi_hot_labels,
            'label_names': labels,  # For debugging/analysis
            'image_path': sample['image_path']
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching samples with variable sequence lengths.
    """
    # Separate different types of data
    images = torch.stack([item['images'] for item in batch])
    target_labels = torch.stack([item['target_labels'] for item in batch])

    # Handle variable length sequences
    input_sequences = [item['input_sequences'] for item in batch]
    target_sequences = [item['target_sequences'] for item in batch]

    # Create label processor instance (assuming it's available globally or passed)
    # For now, use simple padding
    max_input_len = max(len(seq) for seq in input_sequences)
    max_target_len = max(len(seq) for seq in target_sequences)

    # Pad sequences
    padded_input_sequences = []
    padded_target_sequences = []
    attention_masks = []

    for i, (input_seq, target_seq) in enumerate(zip(input_sequences, target_sequences)):
        # Pad input sequence
        input_len = len(input_seq)
        if input_len < max_input_len:
            padding = torch.full((max_input_len - input_len,), 2, dtype=torch.long)  # PAD=2
            padded_input = torch.cat([input_seq, padding])
        else:
            padded_input = input_seq
        padded_input_sequences.append(padded_input)

        # Pad target sequence
        target_len = len(target_seq)
        if target_len < max_target_len:
            padding = torch.full((max_target_len - target_len,), 2, dtype=torch.long)  # PAD=2
            padded_target = torch.cat([target_seq, padding])
        else:
            padded_target = target_seq
        padded_target_sequences.append(padded_target)

        # Create attention mask (True for actual tokens, False for padding)
        mask = torch.cat([
            torch.ones(input_len, dtype=torch.bool),
            torch.zeros(max_input_len - input_len, dtype=torch.bool)
        ])
        attention_masks.append(mask)

    return {
        'images': images,
        'input_sequences': torch.stack(padded_input_sequences),
        'target_sequences': torch.stack(padded_target_sequences),
        'target_labels': target_labels,
        'attention_mask': torch.stack(attention_masks),
        # Keep metadata for debugging
        'label_names': [item['label_names'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }


def create_dataloaders(
    config: DataConfig,
    label_processor: LabelProcessor,
    train_data_path: Optional[str] = None,
    val_data_path: Optional[str] = None,
    test_data_path: Optional[str] = None
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """
    Create data loaders for training, validation, and testing.

    Args:
        config: Data configuration
        label_processor: Label processor instance
        train_data_path: Override training data path
        val_data_path: Override validation data path
        test_data_path: Override test data path

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Image transforms
    image_transforms = ImageTransforms(
        image_size=config.image_size,
        mean=config.image_mean,
        std=config.image_std,
        use_augmentation=config.use_data_augmentation
    )

    train_transform = image_transforms.get_train_transforms()
    eval_transform = image_transforms.get_eval_transforms()

    dataloaders = []

    # Training data loader
    train_path = train_data_path or config.train_data_path
    if train_path and os.path.exists(train_path):
        train_dataset = WeatherChartDataset(
            data_path=train_path,
            label_processor=label_processor,
            transform=train_transform,
            is_training=True,
            config=config
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=32,  # Will be overridden by training config
            shuffle=config.shuffle_train,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=config.drop_last,
            collate_fn=collate_fn
        )
        dataloaders.append(train_loader)
        logger.info(f"Created training dataloader with {len(train_dataset)} samples")
    else:
        dataloaders.append(None)
        logger.warning(f"Training data not found: {train_path}")

    # Validation data loader
    val_path = val_data_path or config.val_data_path
    if val_path and os.path.exists(val_path):
        val_dataset = WeatherChartDataset(
            data_path=val_path,
            label_processor=label_processor,
            transform=eval_transform,
            is_training=False,
            config=config
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
        dataloaders.append(val_loader)
        logger.info(f"Created validation dataloader with {len(val_dataset)} samples")
    else:
        dataloaders.append(None)
        logger.warning(f"Validation data not found: {val_path}")

    # Test data loader
    test_path = test_data_path or config.test_data_path
    if test_path and os.path.exists(test_path):
        test_dataset = WeatherChartDataset(
            data_path=test_path,
            label_processor=label_processor,
            transform=eval_transform,
            is_training=False,
            config=config
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
            collate_fn=collate_fn
        )
        dataloaders.append(test_loader)
        logger.info(f"Created test dataloader with {len(test_dataset)} samples")
    else:
        dataloaders.append(None)
        logger.warning(f"Test data not found: {test_path}")

    return tuple(dataloaders)
