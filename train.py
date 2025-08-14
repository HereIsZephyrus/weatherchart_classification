"""
Training script for CNN-RNN unified framework.
Based on the training strategy from docs/train.md section 3.2.

Usage:
    python -m traniner.train --config config.json
    python -m traniner.train --data_path /path/to/data --output_dir ./outputs
"""
import argparse
import logging
import os
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from traniner import (
    ExperimentConfig,
    create_default_config,
    WeatherChartModel,
    WeatherChartConfig,
    WeatherChartTrainer,
    LabelProcessor,
    create_dataloaders,
    load_label_mapping,
    set_seed
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train CNN-RNN unified framework for weather chart classification"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )

    # Data paths
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to training data"
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        help="Path to validation data"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data"
    )
    parser.add_argument(
        "--label_mapping_path",
        type=str,
        help="Path to label mapping JSON file"
    )

    # Training parameters
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="Learning rate"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    # Model parameters
    parser.add_argument(
        "--num_labels",
        type=int,
        help="Number of label classes"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        help="Maximum sequence length"
    )

    # Training modes
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--evaluate_only",
        action="store_true",
        help="Only evaluate model without training"
    )
    parser.add_argument(
        "--predict_only",
        action="store_true",
        help="Only generate predictions without training"
    )

    # Distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )

    # Logging
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="weather_chart_classification",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        help="Wandb entity name"
    )

    return parser.parse_args()


def load_config(args) -> ExperimentConfig:
    """Load configuration from file and command line arguments."""
    # Start with default configuration
    config = create_default_config()

    # Load from config file if provided
    if args.config and os.path.exists(args.config):
        logger.info(f"Loading configuration from {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        config = ExperimentConfig.from_dict(config_dict)

    # Override with command line arguments
    if args.train_data_path:
        config.data.train_data_path = args.train_data_path
    if args.val_data_path:
        config.data.val_data_path = args.val_data_path
    if args.test_data_path:
        config.data.test_data_path = args.test_data_path
    if args.label_mapping_path:
        config.data.label_mapping_path = args.label_mapping_path

    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.rnn_learning_rate = args.learning_rate
        config.training.cnn_learning_rate = args.learning_rate * 0.1
    if args.seed:
        config.seed = args.seed

    if args.num_labels:
        config.model.num_labels = args.num_labels
    if args.max_sequence_length:
        config.model.max_sequence_length = args.max_sequence_length

    if args.local_rank != -1:
        config.local_rank = args.local_rank

    if args.use_wandb:
        config.use_wandb = True
        if args.wandb_project:
            config.wandb_project = args.wandb_project
        if args.wandb_entity:
            config.wandb_entity = args.wandb_entity

    return config


def setup_distributed_training(local_rank: int):
    """Setup distributed training if needed."""
    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
        logger.info(f"Initialized distributed training on rank {local_rank}")


def create_model(config: ExperimentConfig) -> WeatherChartModel:
    """Create and initialize model."""
    # Create model configuration
    model_config = WeatherChartConfig(config.model)

    # Create model
    model = WeatherChartModel(model_config)

    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    return model


def setup_label_processor(config: ExperimentConfig) -> LabelProcessor:
    """Setup label processor."""
    # Load label mapping
    if config.data.label_mapping_path and os.path.exists(config.data.label_mapping_path):
        label_mapping = load_label_mapping(config.data.label_mapping_path)
    else:
        # Create dummy mapping for testing
        logger.warning("No label mapping provided, creating dummy mapping")
        label_mapping = {f"label_{i}": i for i in range(config.model.num_labels)}

    # Update num_labels in config
    config.model.num_labels = len(label_mapping)

    # Create label processor
    label_processor = LabelProcessor(
        label_mapping=label_mapping,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        pad_token_id=config.model.pad_token_id,
        max_sequence_length=config.model.max_sequence_length
    )

    return label_processor


def main():
    """Main training function."""
    # Parse arguments
    args = parse_arguments()

    # Setup distributed training
    setup_distributed_training(args.local_rank)

    # Load configuration
    config = load_config(args)

    # Set seed for reproducibility
    set_seed(config.seed)

    # Setup output directory
    os.makedirs(config.output_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(config.output_dir, "config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved configuration to {config_path}")

    # Setup label processor
    label_processor = setup_label_processor(config)

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        config=config.data,
        label_processor=label_processor,
        train_data_path=config.data.train_data_path,
        val_data_path=config.data.val_data_path,
        test_data_path=config.data.test_data_path
    )

    # Override batch size in data loaders
    if train_loader:
        train_loader.batch_size = config.training.batch_size
    if val_loader:
        val_loader.batch_size = config.training.batch_size
    if test_loader:
        test_loader.batch_size = config.training.batch_size

    # Create model
    model = create_model(config)

    # Wrap model for distributed training
    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=config.ddp_find_unused_parameters
        )

    # Create trainer
    trainer = WeatherChartTrainer(
        config=config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        label_processor=label_processor
    )

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        trainer.load_checkpoint(args.resume_from_checkpoint)
        logger.info(f"Resumed training from {args.resume_from_checkpoint}")

    try:
        if args.evaluate_only:
            # Evaluation mode
            logger.info("Running evaluation only...")
            if val_loader:
                eval_metrics = trainer._evaluate()
                logger.info(f"Evaluation results: {eval_metrics}")
            else:
                logger.error("No validation data provided for evaluation")

        elif args.predict_only:
            # Prediction mode
            logger.info("Running prediction only...")
            if test_loader:
                predictions = trainer.predict(
                    dataloader=test_loader,
                    save_predictions=True,
                    output_path=os.path.join(config.output_dir, "predictions.json")
                )
                logger.info("Predictions saved successfully")
            else:
                logger.error("No test data provided for prediction")

        else:
            # Training mode
            logger.info("Starting training...")
            trainer.train()

            # Final evaluation on test set if available
            if test_loader:
                logger.info("Running final evaluation on test set...")
                test_predictions = trainer.predict(
                    dataloader=test_loader,
                    save_predictions=True,
                    output_path=os.path.join(config.output_dir, "test_predictions.json")
                )
                logger.info("Test predictions saved successfully")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info("Training completed successfully")


if __name__ == "__main__":
    main()
