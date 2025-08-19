"""
Dataset and DataLoader implementations for weather chart classification.
Includes dataset reading, batch processing, and training state tracking.
"""

import os
import logging
import json
import time
from enum import Enum
import ast
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from ..settings import EPOCH_NUM, SAMPLE_PER_BATCH, CURRENT_DATASET_DIR, NUM_WORKERS, SAVE_FREQUENCY

logger = logging.getLogger(__name__)

class ProgressStatus(str, Enum):
    """Processing status for data batches"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"

class DatasetReader(Dataset):
    """Dataset reader for weather chart images and their metadata"""

    def __init__(self, csv_file: str):
        """
        Initialize dataset reader.

        Args:
            csv_file: Path to CSV file containing image metadata and labels
        """
        self.csv_file = csv_file
        self.dataset_type = csv_file.split("/")[-1].split(".")[0] # train, validation, test
        self.images_dir = f"{CURRENT_DATASET_DIR}/images/{self.dataset_type}"
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.df = pd.read_csv(csv_file)
        logger.info("Loaded %d samples from %s", len(self.df), csv_file)

        self._preprocess_labels()

    def _preprocess_labels(self):
        """Process label column and create image paths"""
        def parse_labels(label_str):
            """Convert label string to list, handling various input formats"""
            if pd.isna(label_str) or label_str == '':
                return []
            try:
                if isinstance(label_str, str):
                    return ast.literal_eval(label_str)
                return label_str if isinstance(label_str, list) else []
            except (ValueError, SyntaxError):
                logger.warning("Failed to parse labels: %s", label_str)
                return []

        self.df['parsed_labels'] = self.df['label'].apply(parse_labels)
        self.df['image_path'] = self.df['index'].apply(
            lambda idx: os.path.join(self.images_dir, f"{int(float(idx)):06d}.png")
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get dataset item by index, including image and metadata"""
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
        """Get dataset statistics including sample counts and label distribution"""
        stats = {
            'total_samples': len(self.df),
            'unique_labels': self.df['parsed_labels'].apply(len).sum(),
            'avg_labels_per_sample': self.df['parsed_labels'].apply(len).mean(),
            'sample_info': self.df[['en_name', 'zh_name', 'index']].head(5).to_dict('records')
        }
        return stats

class TrainingState:
    """Training state tracking and checkpointing for DatasetLoader"""

    def __init__(self, experiment_dir: str):
        self.experiment_dir = Path(experiment_dir)
        self.checkpoint_dir = self.experiment_dir / "checkpoints"
        self.logs_dir = self.experiment_dir / "logs"
        self.eval_results_dir = self.experiment_dir / "eval_results"

        # Create directories if not exist
        for dir_path in [self.checkpoint_dir, self.logs_dir, self.eval_results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Training state variables
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.start_time = None
        self.epoch_start_time = None
        self.batch_times = []
        self.epoch_metrics = []
        self.best_metrics = {}

    def save_state(self, additional_data: Optional[Dict[str, Any]] = None):
        """Save current training state to checkpoint"""
        state_data = {
            "current_epoch": self.current_epoch,
            "current_batch": self.current_batch,
            "total_batches": self.total_batches,
            "start_time": self.start_time,
            "epoch_start_time": self.epoch_start_time,
            "batch_times": self.batch_times[-100:],  # Keep last 100 batch times
            "epoch_metrics": self.epoch_metrics,
            "best_metrics": self.best_metrics,
            "timestamp": datetime.now().isoformat()
        }

        if additional_data:
            state_data.update(additional_data)

        state_file = self.checkpoint_dir / "training_state.json"
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state_data, f, indent=2, ensure_ascii=False)

        logger.info("Saved training state to %s", state_file)

    def load_state(self) -> Dict[str, Any]:
        """Load training state from checkpoint"""
        state_file = self.checkpoint_dir / "training_state.json"
        if state_file.exists():
            with open(state_file, 'r', encoding='utf-8') as f:
                state_data = json.load(f)

            self.current_epoch = state_data.get("current_epoch", 0)
            self.current_batch = state_data.get("current_batch", 0)
            self.total_batches = state_data.get("total_batches", 0)
            self.start_time = state_data.get("start_time")
            self.epoch_start_time = state_data.get("epoch_start_time")
            self.batch_times = state_data.get("batch_times", [])
            self.epoch_metrics = state_data.get("epoch_metrics", [])
            self.best_metrics = state_data.get("best_metrics", {})

            logger.info("Loaded training state from %s", state_file)
            return state_data
        return {}

    def log_epoch_start(self, epoch: int, total_batches: int):
        """Log epoch start"""
        self.current_epoch = epoch
        self.total_batches = total_batches
        self.current_batch = 0
        self.epoch_start_time = time.time()

        if self.start_time is None:
            self.start_time = self.epoch_start_time

        logger.info("="*60)
        logger.info("Starting Epoch %d/%d", epoch + 1, EPOCH_NUM)
        logger.info("Total batches in epoch: %d", total_batches)
        logger.info("="*60)

    def log_batch_progress(self, batch_idx: int, batch_loss: float, lr: float = None):
        """Log batch progress"""
        self.current_batch = batch_idx + 1
        batch_time = time.time()

        # Calculate time for progress tracking
        if len(self.batch_times) == 0:
            # First batch of epoch - start timing from epoch start
            pass

        self.batch_times.append(batch_time)

        # Calculate ETA
        if len(self.batch_times) > 1:
            avg_batch_time = (batch_time - self.epoch_start_time) / self.current_batch
            remaining_batches = self.total_batches - self.current_batch
            eta_seconds = remaining_batches * avg_batch_time
            eta_str = f"{int(eta_seconds//60):02d}:{int(eta_seconds%60):02d}"
        else:
            eta_str = "N/A"

        # Log every 10 batches or at epoch end
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == self.total_batches:
            progress_pct = (batch_idx + 1) / self.total_batches * 100
            log_msg = f"Batch {batch_idx + 1}/{self.total_batches} ({progress_pct:.1f}%) | Loss: {batch_loss:.4f} | ETA: {eta_str}"
            if lr is not None:
                log_msg += f" | LR: {lr:.2e}"
            logger.info(log_msg)

    def log_epoch_end(self, epoch_metrics: Dict[str, float]):
        """Log epoch end with metrics"""
        epoch_time = time.time() - self.epoch_start_time
        self.epoch_metrics.append({
            "epoch": self.current_epoch,
            "metrics": epoch_metrics,
            "epoch_time": epoch_time,
            "timestamp": datetime.now().isoformat()
        })

        logger.info("-"*60)
        logger.info("Epoch %d completed in %.2f seconds", self.current_epoch + 1, epoch_time)

        # Log metrics
        for metric_name, metric_value in epoch_metrics.items():
            logger.info("%s: %.4f", metric_name, metric_value)

            # Track best metrics
            if metric_name.startswith('val_'):
                if metric_name not in self.best_metrics or metric_value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = metric_value
                    logger.info("New best %s: %.4f", metric_name, metric_value)

        logger.info("-"*60)

        # Save epoch metrics to file
        metrics_file = self.logs_dir / f"epoch_{self.current_epoch:03d}_metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump({
                "epoch": self.current_epoch,
                "metrics": epoch_metrics,
                "epoch_time": epoch_time,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2, ensure_ascii=False)


class DatasetLoader(DataLoader):
    """Enhanced DataLoader with training state tracking and checkpointing"""

    def __init__(
        self,
        dataset,
        epochs: int = EPOCH_NUM,
        experiment_dir: Optional[str] = None,
        save_frequency: int = 5,  # Save state every N epochs
        **kwargs
    ):
        """
        Initialize DatasetLoader

        Args:
            dataset: Dataset to load from
            epochs: Number of epochs to iterate
            experiment_dir: Directory for experiment logs and checkpoints
            save_frequency: How often to save checkpoints (in epochs)
            **kwargs: Additional arguments for DataLoader
        """
        super().__init__(dataset, **kwargs)
        self.epochs = epochs
        self.save_frequency = save_frequency

        # Setup training state tracking
        if experiment_dir:
            self.state = TrainingState(experiment_dir)
            self.has_state_tracking = True
        else:
            self.state = None
            self.has_state_tracking = False
            logger.warning("No experiment_dir provided, state tracking disabled")

    def __iter__(self):
        """Enhanced iteration with state tracking and checkpointing"""

        # Load previous state if available
        if self.has_state_tracking:
            self.state.load_state()  # Load state but don't need to store return value
            start_epoch = self.state.current_epoch
        else:
            start_epoch = 0

        for epoch in range(start_epoch, self.epochs):
            # Start epoch logging
            if self.has_state_tracking:
                total_batches = len(self)
                self.state.log_epoch_start(epoch, total_batches)
            else:
                logger.info("Starting epoch %d/%d", epoch + 1, self.epochs)

            # Get iterator for current epoch
            epoch_iterator = super().__iter__()

            batch_idx = 0
            for batch in epoch_iterator:
                # Add batch metadata
                if isinstance(batch, dict):
                    batch['epoch'] = epoch
                    batch['batch_idx'] = batch_idx
                    batch['global_step'] = epoch * len(self) + batch_idx

                yield batch
                batch_idx += 1

            # End epoch logging and save checkpoint
            if self.has_state_tracking:
                # Save checkpoint periodically
                if (epoch + 1) % self.save_frequency == 0:
                    self.save_checkpoint(epoch)

    def log_batch_metrics(self, batch_idx: int, loss: float, lr: float = None):
        """Log batch-level metrics"""
        if self.has_state_tracking:
            self.state.log_batch_progress(batch_idx, loss, lr)

    def log_epoch_metrics(self, metrics: Dict[str, float]):
        """Log epoch-level metrics"""
        if self.has_state_tracking:
            self.state.log_epoch_end(metrics)

    def save_checkpoint(self, epoch: int, additional_data: Optional[Dict[str, Any]] = None):
        """Save training checkpoint"""
        if self.has_state_tracking:
            checkpoint_data = {
                "epoch": epoch,
                "dataset_info": {
                    "total_samples": len(self.dataset),
                    "batch_size": self.batch_size,
                    "num_workers": self.num_workers
                }
            }
            if additional_data:
                checkpoint_data.update(additional_data)

            self.state.save_state(checkpoint_data)
            logger.info("Checkpoint saved at epoch %d", epoch + 1)

    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary"""
        if not self.has_state_tracking:
            return {"error": "No state tracking available"}

        total_time = time.time() - self.state.start_time if self.state.start_time else 0

        return {
            "total_epochs": self.epochs,
            "completed_epochs": self.state.current_epoch + 1,
            "total_time_seconds": total_time,
            "total_time_formatted": f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}",
            "avg_epoch_time": total_time / max(1, self.state.current_epoch + 1),
            "best_metrics": self.state.best_metrics,
            "recent_metrics": self.state.epoch_metrics[-5:] if self.state.epoch_metrics else []
        }

    def __repr__(self) -> str:
        if self.has_state_tracking:
            return f"DatasetLoader(epochs={self.epochs}, current_epoch={self.state.current_epoch}, tracking=enabled)"
        else:
            return f"DatasetLoader(epochs={self.epochs}, tracking=disabled)"


class DatasetFactory:
    """Factory class for creating datasets and data loaders with consistent configuration"""

    def __init__(self, dataset_base_path: str = None):
        """
        Initialize dataset factory.

        Args:
            dataset_base_path: Base directory containing metadata and images folders
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
            return dataset

        except Exception as e:
            logger.error("Error loading CSV datasets: %s", str(e))
            raise

    def create_dataloader(
        self,
        dataset: DatasetReader,
        batch_size: int = SAMPLE_PER_BATCH,
        num_workers: int = NUM_WORKERS,
        experiment_dir: Optional[str] = None,
        save_frequency: int = SAVE_FREQUENCY
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
            epochs=EPOCH_NUM,
            experiment_dir=experiment_dir,
            save_frequency=save_frequency
        )

def create_dataloaders(dataset_root_dir: str, experiment_dir: str) -> tuple[DatasetLoader, DatasetLoader, DatasetLoader]:
    """Create enhanced data loaders with state tracking"""
    factory = DatasetFactory(dataset_base_path=dataset_root_dir)
    train_dataset = None
    val_dataset = None
    test_dataset = None

    metadata_dir = Path(dataset_root_dir) / "metadata"
    train_file_path = metadata_dir / "train.csv"
    if train_file_path.exists():
        train_dataset = factory.load_dataset("train")
    else:
        logger.warning("Training dataset not found at: %s", train_file_path)

    val_file_path = metadata_dir / "validation.csv"
    if val_file_path.exists():
        val_dataset = factory.load_dataset("validation")
    else:
        logger.warning("Validation dataset not found at: %s", val_file_path)

    test_file_path = metadata_dir / "test.csv"
    if test_file_path.exists():
        test_dataset = factory.load_dataset("test")
    else:
        logger.warning("Test dataset not found at: %s", test_file_path)

    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset:
        train_loader = factory.create_dataloader(
            dataset=train_dataset,
            batch_size=SAMPLE_PER_BATCH,
            num_workers=NUM_WORKERS,
            experiment_dir=experiment_dir,
            save_frequency=SAVE_FREQUENCY
        )

    if val_dataset:
        val_loader = factory.create_dataloader(
            dataset=val_dataset,
            batch_size=SAMPLE_PER_BATCH,
            num_workers=NUM_WORKERS,
            experiment_dir=experiment_dir,
            save_frequency=SAVE_FREQUENCY
        )

    if test_dataset:
        test_loader = factory.create_dataloader(
            dataset=test_dataset,
            batch_size=SAMPLE_PER_BATCH,
            num_workers=NUM_WORKERS,
            experiment_dir=experiment_dir,
            save_frequency=SAVE_FREQUENCY
        )

    return train_loader, val_loader, test_loader
