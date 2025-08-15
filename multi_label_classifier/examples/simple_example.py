"""
Simple example demonstrating how to use the CNN-RNN unified framework.
基于 docs/train.md 第3.2节设计的简单使用示例。
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import logging
from traniner import (
    ExperimentConfig,
    create_default_config,
    WeatherChartModel,
    WeatherChartConfig,
    WeatherChartTrainer,
    LabelProcessor,
    load_label_mapping,
    set_seed
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_dummy_data(num_samples: int = 100):
    """Create dummy data for testing."""
    import json
    import numpy as np
    from PIL import Image

    # Create dummy images and labels
    data_dir = "./dummy_data"
    os.makedirs(data_dir, exist_ok=True)

    # Create label mapping
    label_mapping = {
        "temperature": 0,
        "pressure": 1, 
        "wind_speed": 2,
        "humidity": 3,
        "precipitation": 4
    }

    with open(f"{data_dir}/label_mapping.json", "w") as f:
        json.dump(label_mapping, f, indent=2)

    # Create dummy dataset
    samples = []
    for i in range(num_samples):
        # Create dummy image
        image = Image.fromarray(
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        )
        image_path = f"{data_dir}/image_{i:03d}.jpg"
        image.save(image_path)

        # Create random labels
        all_labels = list(label_mapping.keys())
        num_labels = np.random.randint(1, 4)  # 1-3 labels per sample
        labels = np.random.choice(all_labels, num_labels, replace=False).tolist()

        samples.append({
            "image_path": image_path,
            "labels": labels,
            "metadata": {}
        })

    # Save dataset
    train_samples = samples[:80]
    val_samples = samples[80:]

    with open(f"{data_dir}/train.json", "w") as f:
        json.dump(train_samples, f, indent=2)

    with open(f"{data_dir}/val.json", "w") as f:
        json.dump(val_samples, f, indent=2)

    logger.info(f"Created dummy dataset with {len(samples)} samples")
    return data_dir


def simple_training_example():
    """Simple training example with dummy data."""
    logger.info("Starting simple training example...")

    # Set seed for reproducibility
    set_seed(42)

    # Create dummy data
    data_dir = create_dummy_data(num_samples=50)

    # Create configuration
    config = create_default_config()

    # Update configuration for small example
    config.experiment_name = "simple_example"
    config.output_dir = "./outputs/simple_example"
    config.data.train_data_path = f"{data_dir}/train.json"
    config.data.val_data_path = f"{data_dir}/val.json"
    config.data.label_mapping_path = f"{data_dir}/label_mapping.json"

    # Reduce for quick testing
    config.training.num_epochs = 3
    config.training.batch_size = 4
    config.training.warmup_epochs = 1
    config.training.logging_steps = 5
    config.training.eval_steps = 10
    config.model.num_labels = 5

    # Disable wandb for example
    config.use_wandb = False

    # Setup label processor
    label_mapping = load_label_mapping(config.data.label_mapping_path)
    label_processor = LabelProcessor(
        label_mapping=label_mapping,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        pad_token_id=config.model.pad_token_id,
        max_sequence_length=config.model.max_sequence_length
    )

    # Create data loaders
    from traniner.dataset import create_dataloaders
    train_loader, val_loader, _ = create_dataloaders(
        config=config.data,
        label_processor=label_processor
    )

    # Create model
    model_config = WeatherChartConfig(config.model)
    model = WeatherChartModel(model_config)

    logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = WeatherChartTrainer(
        config=config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader,
        label_processor=label_processor
    )

    # Start training
    try:
        trainer.train()
        logger.info("Training completed successfully!")

        # Test prediction
        logger.info("Testing prediction...")
        predictions = trainer.predict(
            dataloader=val_loader,
            return_predictions=True,
            save_predictions=True
        )

        logger.info(f"Generated {len(predictions['parallel_predictions'])} predictions")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


def inference_example():
    """Example of loading trained model for inference."""
    logger.info("Starting inference example...")

    # This assumes you have a trained model
    model_path = "./outputs/simple_example/best_model"

    if not os.path.exists(model_path):
        logger.warning(f"No trained model found at {model_path}")
        logger.info("Please run simple_training_example() first")
        return

    try:
        # Load trained model
        model = WeatherChartModel.from_pretrained(model_path)
        model.eval()

        logger.info("Loaded trained model successfully")

        # Create dummy input
        dummy_images = torch.randn(2, 3, 224, 224)  # Batch of 2 images

        with torch.no_grad():
            # Generate predictions using beam search
            generation_outputs = model.generate(
                images=dummy_images,
                max_length=10,
                beam_width=3,
                early_stopping=True
            )

            # Get parallel predictions
            parallel_outputs = model(images=dummy_images)
            parallel_probs = torch.sigmoid(parallel_outputs["parallel_logits"])

            logger.info("Generated predictions:")
            logger.info(f"Sequences shape: {generation_outputs['sequences'].shape}")
            logger.info(f"Parallel probs shape: {parallel_probs.shape}")

            # Print some results
            for i in range(len(dummy_images)):
                seq = generation_outputs['sequences'][i].tolist()
                probs = parallel_probs[i].tolist()
                logger.info(f"Sample {i}: sequence={seq}, probs={probs[:5]}...")

        logger.info("Inference completed successfully!")

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    print("CNN-RNN Unified Framework Simple Example")
    print("========================================")

    choice = input("Choose mode: (1) Training (2) Inference (3) Both [1]: ").strip()
    if not choice:
        choice = "1"

    if choice in ["1", "3"]:
        simple_training_example()

    if choice in ["2", "3"]:
        inference_example()

    print("Example completed!")
