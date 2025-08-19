"""
Experiment management utilities for organizing training runs
"""

import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
import yaml
from pydantic import BaseModel
from .trainer import WeatherChartTrainer
from .model import WeatherChartModel
from .config import ModelConfig, Hyperparameter
from .dataset import create_dataloaders

logger = logging.getLogger(__name__)

class TrainingConfig(BaseModel):
    """Training configuration"""
    trainer_name: str
    dataset_root: str
    experiments_root: str
    training_mode: Literal["predict", "resume", "evaluate", "train"]
    description: str
    tags: List[str]

class ExperimentMetadata(BaseModel):
    """Experiment metadata"""
    experiment_name: str
    created_at: str
    description: str
    directory_structure: List[str]
    status: str
    last_updated: Optional[str] = None

class ExperimentManager:
    """
    Manage experiments with standardized directory structure and configuration
    """

    def __init__(self, training_config: TrainingConfig):
        """
        Initialize experiment manager

        Args:
            training_config: Training configuration
        """
        self.training_config = training_config
        self.experiment_dir = Path(self.training_config.experiments_root) / self.training_config.trainer_name
        self.experiment_dir.mkdir(exist_ok=True)
        self.metadata = self.create_experiment()
        self.model_config = self.read_model_config()
        logger.info("Experiment manager initialized at: %s", self.experiment_dir)

    def read_model_config(self) -> ModelConfig:
        """
        Read model configuration from model_config.json
        """
        with open(self.experiment_dir / "config.yaml", 'r', encoding='utf-8') as f:
            hyper_params = yaml.safe_load(f)
            return ModelConfig(config_list=Hyperparameter(**hyper_params))

    def create_experiment(self) -> ExperimentMetadata:
        """
        Create a new experiment with standardized directory structure

        Returns:
            Path to experiment directory
        """
        # Load metadata if experiment is being resumed
        if self.training_config.training_mode == "resume":
            logger.info("Resuming experiment: %s", self.training_config.trainer_name)
            return self.load_metadata()
        
        # Create directory structure if experiment is being created
        subdir_names = ["logs", "checkpoints", "eval_results", "configs"]
        for subdir in subdir_names:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        metadata = ExperimentMetadata(
            experiment_name=self.training_config.trainer_name,
            created_at=datetime.now().isoformat(),
            description=self.training_config.description,
            directory_structure=subdir_names,
            status="created"
        )

        # Save metadata
        with open(self.experiment_dir / "experiment_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

        logger.info("Created experiment: %s at %s", self.training_config.trainer_name, self.experiment_dir)
        return metadata

    def load_metadata(self) -> ExperimentMetadata:
        """Load experiment metadata"""
        metadata_file = self.experiment_dir / "experiment_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                self.metadata = ExperimentMetadata(**json.load(f))
            return self.metadata
        else:
            logger.error("Metadata file not found: %s", metadata_file)
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    def update_experiment_status(self, status: str, additional_info: Optional[Dict[str, Any]] = None):
        """Update experiment status and metadata"""
        metadata_file = self.experiment_dir / "experiment_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = ExperimentMetadata(**json.load(f))

            metadata.status = status
            metadata.last_updated = datetime.now().isoformat()

            if additional_info:
                metadata.model_dump().update(additional_info)

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.model_dump(), f, indent=2, ensure_ascii=False)

            logger.info("Updated experiment status to: %s", status)

    def get_experiment_summary(self) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""

        # Load metadata
        metadata_file = self.experiment_dir / "experiment_metadata.json"
        metadata = None
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = ExperimentMetadata(**json.load(f))

        # Check for training state
        training_state_file = self.experiment_dir / "checkpoints" / "training_state.json"
        training_state = {}
        if training_state_file.exists():
            with open(training_state_file, 'r', encoding='utf-8') as f:
                training_state = json.load(f)

        # Count files in each directory
        file_counts = {}
        for subdir in ["checkpoints", "eval_results", "configs"]:
            subdir_path = self.experiment_dir / subdir
            if subdir_path.exists():
                file_counts[subdir] = len([f for f in subdir_path.iterdir() if f.is_file()])
            else:
                file_counts[subdir] = 0

        # Check for latest metrics
        latest_metrics = {}
        results_dir = self.experiment_dir / "eval_results"
        if results_dir.exists():
            metric_files = list(results_dir.glob("epoch_*_metrics.json"))
            if metric_files:
                # Get latest metrics file
                latest_file = max(metric_files, key=lambda x: x.stat().st_mtime)
                with open(latest_file, 'r', encoding='utf-8') as f:
                    latest_metrics = json.load(f)

        return {
            "metadata": metadata,
            "training_state": training_state,
            "file_counts": file_counts,
            "latest_metrics": latest_metrics,
            "total_size_mb": self._get_directory_size(self.experiment_dir)
        }

    def archive_experiment(self, archive_dir: Optional[str] = None):
        """Archive an experiment by moving it to archive directory"""

        if archive_dir is None:
            archive_dir = self.training_config.experiments_root / "archived"
        else:
            archive_dir = Path(archive_dir)

        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / self.training_config.trainer_name
        shutil.move(str(self.experiment_dir), str(archive_path))
        logger.info("Archived experiment %s to %s", self.experiment_dir.name, archive_path)
        return str(archive_path)

    def _get_directory_size(self, directory: Path) -> float:
        """Get directory size in MB"""
        total_size = 0
        try:
            for file_path in directory.rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            pass
        return total_size / (1024 * 1024)  # Convert to MB

    def create_trainer(
        self,
        train_loader,
        val_loader
    ) -> WeatherChartTrainer:
        """Create model and trainer with enhanced configuration"""

        # Create trainer
        trainer = WeatherChartTrainer(
            config=self.model_config,
            model=WeatherChartModel(self.model_config),
            train_dataloader=train_loader,
            eval_dataloader=val_loader,
            output_dir=str(self.experiment_dir),  # Convert Path to string
            label_processor=None  # Will be auto-created in trainer
        )
        return trainer

    def execute(self):
        """Enhanced main training function with experiment management"""
        try:
            logger.info("Creating data loaders...")
            train_loader, val_loader, test_loader = create_dataloaders(self.training_config.dataset_root, self.experiment_dir)
            logger.info("Creating model and trainer...")
            trainer = self.create_trainer(train_loader, val_loader)

            if self.training_config.training_mode == "resume":
                # Resume training from latest checkpoint
                checkpoint_dir = self.experiment_dir / "checkpoints"
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
                    self.update_experiment_status("resumed")
                else:
                    logger.warning("No checkpoint found for resume, starting fresh training")
                    self.update_experiment_status("training")

            elif self.training_config.training_mode == "evaluate":
                # Evaluation mode
                logger.info("Running evaluation only...")
                self.update_experiment_status("evaluating")

                if val_loader:
                    eval_metrics = trainer._evaluate()  # Access to protected member needed here
                    logger.info("Evaluation results:")
                    for metric, value in eval_metrics.items():
                        logger.info("  %s: %.4f", metric, value)

                    # Save evaluation results
                    eval_results_file = self.experiment_dir / "eval_results" / "validation_results.json"
                    with open(eval_results_file, 'w', encoding='utf-8') as f:
                        json.dump(eval_metrics, f, indent=2, ensure_ascii=False)
                    logger.info("Evaluation results saved to %s", eval_results_file)

                    self.update_experiment_status("completed", {
                        "evaluation_metrics": eval_metrics
                    })
                else:
                    logger.error("No validation data provided for evaluation")
                    self.update_experiment_status("failed", {
                        "error": "No validation data"
                    })

            elif self.training_config.training_mode == "predict":
                # Prediction mode
                logger.info("Running prediction only...")
                self.update_experiment_status("predicting")

                if test_loader:
                    predictions_file = self.experiment_dir / "eval_results" / "test_predictions.json"
                    trainer.predict(
                        dataloader=test_loader,
                        save_predictions=True,
                        output_path=str(predictions_file)
                    )
                    logger.info("Predictions saved to %s", predictions_file)

                    self.update_experiment_status("completed", {
                        "predictions_file": str(predictions_file)
                    })
                else:
                    logger.error("No test data provided for prediction")
                    self.update_experiment_status("failed", {
                        "error": "No test data"
                    })

            else:
                # Training mode
                logger.info("Starting training...")
                self.update_experiment_status("training")

                # Run training with enhanced tracking
                trainer.train()

                # Final evaluation on test set if available
                if test_loader:
                    logger.info("Running final evaluation on test set...")
                    test_predictions_file = self.experiment_dir / "eval_results" / "final_test_predictions.json"
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
                    summary_file = self.experiment_dir / "logs" / "training_summary.json"
                    with open(summary_file, 'w', encoding='utf-8') as f:
                        json.dump(training_summary, f, indent=2, ensure_ascii=False)

                self.update_experiment_status("completed")
                logger.info("Training completed successfully")

            # Print experiment summary
            experiment_summary = self.get_experiment_summary()
            logger.info("="*60)
            logger.info("EXPERIMENT SUMMARY")
            logger.info("="*60)
            logger.info("Experiment: %s", experiment_summary["metadata"].get("full_name", "Unknown"))
            logger.info("Status: %s", experiment_summary["metadata"].get("status", "Unknown"))
            logger.info("Directory: %s", self.experiment_dir)
            logger.info("Total size: %.2f MB", experiment_summary.get("total_size_mb", 0))
            logger.info("Files created:")
            for subdir, count in experiment_summary.get("file_counts", {}).items():
                logger.info("  %s: %d files", subdir, count)
            logger.info("="*60)

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
            self.update_experiment_status("interrupted")
        except Exception as e:
            logger.error("Training failed with error: %s", e, exc_info=True)
            self.update_experiment_status("failed", {
                    "error": str(e)
                })
            raise
