"""
Dataset classes for weather chart classification with CNN-RNN framework.
"""
import logging
from typing import List, Dict, Tuple, Optional, Any
from enum import Enum
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

class ProgressStatus(str, Enum):
    """
    status of the data batch
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

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
