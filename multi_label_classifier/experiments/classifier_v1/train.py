"""
Training script for classifier_v1
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging
from multi_label_classifier.core import (
    DatasetFactory,
    ExperimentManager,
    ModelConfig,
    WeatherChartModel,
    WeatherChartTrainer
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def create_dataloaders(config: Dict[str, Any], experiment_dir: str) -> tuple:
    """Create enhanced data loaders with state tracking"""
    dataset_config = config["dataset"]

    # Initialize dataset factory
    factory = DatasetFactory(dataset_base_path=dataset_config["base_path"])

    # Load datasets
    train_dataset = None
    val_dataset = None
    test_dataset = None

    # Build full paths for dataset files
    base_path = Path(dataset_config["base_path"]) / "metadata"

    # Load training dataset
    train_file_path = base_path / dataset_config["train_file"]
    if train_file_path.exists():
        train_dataset = factory.load_dataset("train")
        logger.info("Loaded training dataset: %d samples", len(train_dataset))
    else:
        logger.warning("Training dataset not found at: %s", train_file_path)

    # Load validation dataset  
    val_file_path = base_path / dataset_config["val_file"]
    if val_file_path.exists():
        val_dataset = factory.load_dataset("validation")
        logger.info("Loaded validation dataset: %d samples", len(val_dataset))
    else:
        logger.warning("Validation dataset not found at: %s", val_file_path)

    # Load test dataset
    test_file_path = base_path / dataset_config["test_file"]
    if test_file_path.exists():
        test_dataset = factory.load_dataset("test")
        logger.info("Loaded test dataset: %d samples", len(test_dataset))
    else:
        logger.warning("Test dataset not found at: %s", test_file_path)

    # Create data loaders with state tracking
    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset:
        train_loader = factory.create_dataloader(
            dataset=train_dataset,
            batch_size=dataset_config["batch_size"],
            num_workers=dataset_config["num_workers"],
            experiment_dir=experiment_dir,
            save_frequency=dataset_config["save_frequency"]
        )

    if val_dataset:
        val_loader = factory.create_dataloader(
            dataset=val_dataset,
            batch_size=dataset_config["batch_size"],
            num_workers=dataset_config["num_workers"],
            experiment_dir=experiment_dir,
            save_frequency=dataset_config["save_frequency"]
        )

    if test_dataset:
        test_loader = factory.create_dataloader(
            dataset=test_dataset,
            batch_size=dataset_config["batch_size"],
            num_workers=dataset_config["num_workers"],
            experiment_dir=experiment_dir,
            save_frequency=dataset_config["save_frequency"]
        )

    return train_loader, val_loader, test_loader


def create_model_and_trainer(
    config: Dict[str, Any],
    train_loader,
    val_loader,
    experiment_dir: str
) -> tuple[WeatherChartModel, WeatherChartTrainer]:
    """Create model and trainer with enhanced configuration"""

    # Extract unique labels from training data for label mapping
    all_labels = set()
    if train_loader:
        logger.info("Extracting label vocabulary from training data...")
        sample_count = 0
        for batch in train_loader:
            if 'labels' in batch:
                for label_list in batch['labels']:
                    all_labels.update(label_list)
            sample_count += 1
            if sample_count >= 10:  # Sample first 10 batches for speed
                break

    # Create label mapping
    label_mapping = {label: idx for idx, label in enumerate(sorted(all_labels))}
    config["model"]["num_labels"] = len(label_mapping)

    logger.info("Created label mapping with %d unique labels", len(label_mapping))

    # Save label mapping to experiment
    label_mapping_file = Path(experiment_dir) / "configs" / "label_mapping.json"
    with open(label_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)

    # Create model configuration object
    model_config = ModelConfig(**config["model"])

    # Create model
    model = WeatherChartModel(model_config)

    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Created model with %d total parameters (%d trainable)", total_params, trainable_params)

    # Create trainer configuration by combining all config sections
    trainer_config = ModelConfig(**{
        **config["model"],
        **config["training"],
        **config["learning_rates"],
        **config["optimizer"], 
        **config["loss_weights"],
        **config["validation"],
        "device": config["device"],
        "output_dir": experiment_dir
    })

    # Create trainer
    trainer = WeatherChartTrainer(
        config=trainer_config,
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=val_loader
    )

    return model, trainer

def main():
    """Enhanced main training function with experiment management"""
    try:
        # Parse arguments
        args = parse_arguments()

        # Setup debug logging if requested
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")

        # Setup distributed training
        setup_distributed_training(args.local_rank)

        # Setup experiment and load configuration
        experiment_dir, config, exp_manager = setup_experiment(args)

        # Set seed for reproducibility
        torch.manual_seed(config["training"]["seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config["training"]["seed"])
        logger.info("Set random seed to %d", config["training"]["seed"])

        # Create data loaders with enhanced tracking
        logger.info("Creating data loaders...")
        train_loader, val_loader, test_loader = create_dataloaders(config, experiment_dir)

        # Create model and trainer
        logger.info("Creating model and trainer...")
        model, trainer = create_model_and_trainer(config, train_loader, val_loader, experiment_dir)

        # Wrap model for distributed training if needed
        if args.local_rank != -1:
            model = DDP(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=True
            )
            logger.info("Wrapped model with DistributedDataParallel")

        # Handle different execution modes
        if args.resume:
            # Resume training from latest checkpoint
            checkpoint_dir = Path(experiment_dir) / "checkpoints"
            latest_checkpoint = None

            # Find latest checkpoint
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*"))
                if checkpoints:
                    latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                elif (checkpoint_dir / "best_model").exists():
                    latest_checkpoint = checkpoint_dir / "best_model"

            if latest_checkpoint:
                trainer.load_checkpoint(str(latest_checkpoint))
                logger.info("Resumed training from checkpoint: %s", latest_checkpoint)
                exp_manager.update_experiment_status(experiment_dir, "resumed")
            else:
                logger.warning("No checkpoint found for resume, starting fresh training")
                exp_manager.update_experiment_status(experiment_dir, "training")

        elif args.evaluate_only:
            # Evaluation mode
            logger.info("Running evaluation only...")
            exp_manager.update_experiment_status(experiment_dir, "evaluating")

            if val_loader:
                eval_metrics = trainer._evaluate()  # Access to protected member needed here
                logger.info("Evaluation results:")
                for metric, value in eval_metrics.items():
                    logger.info("  %s: %.4f", metric, value)

                # Save evaluation results
                eval_results_file = Path(experiment_dir) / "eval_results" / "validation_results.json"
                with open(eval_results_file, 'w', encoding='utf-8') as f:
                    json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
                logger.info("Evaluation results saved to %s", eval_results_file)

                exp_manager.update_experiment_status(experiment_dir, "completed", {
                    "evaluation_metrics": eval_metrics
                })
            else:
                logger.error("No validation data provided for evaluation")
                exp_manager.update_experiment_status(experiment_dir, "failed", {
                    "error": "No validation data"
                })

        elif args.predict_only:
            # Prediction mode
            logger.info("Running prediction only...")
            exp_manager.update_experiment_status(experiment_dir, "predicting")

            if test_loader:
                predictions_file = Path(experiment_dir) / "eval_results" / "test_predictions.json"
                trainer.predict(
                    dataloader=test_loader,
                    save_predictions=True,
                    output_path=str(predictions_file)
                )
                logger.info("Predictions saved to %s", predictions_file)

                exp_manager.update_experiment_status(experiment_dir, "completed", {
                    "predictions_file": str(predictions_file)
                })
            else:
                logger.error("No test data provided for prediction")
                exp_manager.update_experiment_status(experiment_dir, "failed", {
                    "error": "No test data"
                })

        else:
            # Training mode
            logger.info("Starting training...")
            exp_manager.update_experiment_status(experiment_dir, "training")

            # Run training with enhanced tracking
            trainer.train()

            # Final evaluation on test set if available
            if test_loader:
                logger.info("Running final evaluation on test set...")
                test_predictions_file = Path(experiment_dir) / "eval_results" / "final_test_predictions.json"
                trainer.predict(
                    dataloader=test_loader,
                    save_predictions=True,
                    output_path=str(test_predictions_file)
                )
                logger.info("Final test predictions saved to %s", test_predictions_file)

            # Get training summary from dataloader
            if train_loader and train_loader.has_state_tracking:
                training_summary = train_loader.get_training_summary()
                logger.info("Training Summary:")
                logger.info("  Total time: %s", training_summary.get("total_time_formatted", "N/A"))
                logger.info("  Completed epochs: %d/%d", 
                           training_summary.get("completed_epochs", 0),
                           training_summary.get("total_epochs", 0))
                logger.info("  Best metrics: %s", training_summary.get("best_metrics", {}))

                # Save training summary
                summary_file = Path(experiment_dir) / "logs" / "training_summary.json"
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(training_summary, f, indent=2, ensure_ascii=False)

            exp_manager.update_experiment_status(experiment_dir, "completed")
            logger.info("Training completed successfully")

        # Print experiment summary
        experiment_summary = exp_manager.get_experiment_summary(experiment_dir)
        logger.info("="*60)
        logger.info("EXPERIMENT SUMMARY")
        logger.info("="*60)
        logger.info("Experiment: %s", experiment_summary["metadata"].get("full_name", "Unknown"))
        logger.info("Status: %s", experiment_summary["metadata"].get("status", "Unknown"))
        logger.info("Directory: %s", experiment_dir)
        logger.info("Total size: %.2f MB", experiment_summary.get("total_size_mb", 0))
        logger.info("Files created:")
        for subdir, count in experiment_summary.get("file_counts", {}).items():
            logger.info("  %s: %d files", subdir, count)
        logger.info("="*60)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        if 'experiment_dir' in locals() and 'exp_manager' in locals():
            exp_manager.update_experiment_status(experiment_dir, "interrupted")
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        if 'experiment_dir' in locals() and 'exp_manager' in locals():
            exp_manager.update_experiment_status(experiment_dir, "failed", {
                "error": str(e)
            })
        raise
