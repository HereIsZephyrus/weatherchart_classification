"""
Experiment management utilities for organizing training runs
"""

import json
import shutil
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Literal
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class TrainingConfig(BaseModel):
    """Training configuration"""
    trainer_name: str
    experiments_root: str
    training_mode: Literal["predict", "resume", "train"]
    description: str
    tags: List[str]

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
        self.config = training_config
        self.experiment_dir = Path(self.config.experiments_root) / self.config.trainer_name
        self.experiment_dir.mkdir(exist_ok=True)
        logger.info("Experiment manager initialized at: %s", self.experiment_dir)

    def create_experiment(self) -> str:
        """
        Create a new experiment with standardized directory structure

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
            description: Experiment description
            tags: List of tags for categorization

        Returns:
            Path to experiment directory
        """
        # Create timestamped experiment name if needed

        # Create directory structure
        subdir_names = ["checkpoints", "eval_results", "configs", "plots"]
        for subdir in subdir_names:
            (self.experiment_dir / subdir).mkdir(parents=True, exist_ok=True)

        # Create experiment metadata
        metadata = {
            "experiment_name": self.config.trainer_name,
            "created_at": datetime.now().isoformat(),
            "description": self.config.description,
            "status": "created",
            "directory_structure": subdir_names
        }

        # Save metadata
        with open(self.experiment_dir / "experiment_metadata.json", 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Created experiment: %s at %s", self.config.trainer_name, self.experiment_dir)
        return str(self.experiment_dir)

    def save_config(self, config_name: str = "config.yaml"):
        """Save experiment configuration"""
        config_file = self.experiment_dir / "configs" / config_name

        # Save as YAML for better readability
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)

        # Also save as JSON for programmatic access
        json_config_file = self.experiment_dir / "configs" / config_name.replace('.yaml', '.json')
        with open(json_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        logger.info("Saved configuration to %s", config_file)

    def load_config(self, experiment_dir: str, config_name: str = "config.yaml") -> Dict[str, Any]:
        """Load experiment configuration"""
        experiment_path = Path(experiment_dir)
        config_file = experiment_path / "configs" / config_name

        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix == '.yaml':
                    return yaml.safe_load(f)
                else:
                    return json.load(f)
        else:
            logger.warning("Configuration file not found: %s", config_file)
            return {}

    def update_experiment_status(self, experiment_dir: str, status: str, additional_info: Optional[Dict[str, Any]] = None):
        """Update experiment status and metadata"""
        experiment_path = Path(experiment_dir)
        metadata_file = experiment_path / "experiment_metadata.json"

        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            metadata["status"] = status
            metadata["last_updated"] = datetime.now().isoformat()

            if additional_info:
                metadata.update(additional_info)

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info("Updated experiment status to: %s", status)

    def list_experiments(self, status_filter: Optional[str] = None, tag_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all experiments with optional filters"""
        experiments = []

        for experiment_dir in self.experiments_root.iterdir():
            if experiment_dir.is_dir():
                metadata_file = experiment_dir / "experiment_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)

                    # Apply filters
                    if status_filter and metadata.get("status") != status_filter:
                        continue
                    if tag_filter and tag_filter not in metadata.get("tags", []):
                        continue

                    metadata["path"] = str(experiment_dir)
                    experiments.append(metadata)

        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return experiments

    def get_experiment_summary(self, experiment_dir: str) -> Dict[str, Any]:
        """Get comprehensive experiment summary"""
        experiment_path = Path(experiment_dir)

        # Load metadata
        metadata_file = experiment_path / "experiment_metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

        # Check for training state
        training_state_file = experiment_path / "checkpoints" / "training_state.json"
        training_state = {}
        if training_state_file.exists():
            with open(training_state_file, 'r', encoding='utf-8') as f:
                training_state = json.load(f)

        # Count files in each directory
        file_counts = {}
        for subdir in ["checkpoints", "eval_results", "configs"]:
            subdir_path = experiment_path / subdir
            if subdir_path.exists():
                file_counts[subdir] = len([f for f in subdir_path.iterdir() if f.is_file()])
            else:
                file_counts[subdir] = 0

        # Check for latest metrics
        latest_metrics = {}
        results_dir = experiment_path / "eval_results"
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
            "total_size_mb": self._get_directory_size(experiment_path)
        }

    def archive_experiment(self, archive_dir: Optional[str] = None):
        """Archive an experiment by moving it to archive directory"""

        if archive_dir is None:
            archive_dir = self.config.experiments_root / "archived"
        else:
            archive_dir = Path(archive_dir)

        archive_dir.mkdir(exist_ok=True)
        archive_path = archive_dir / self.config.trainer_name
        shutil.move(str(self.experiment_dir), str(archive_path))
        logger.info("Archived experiment %s to %s", self.experiment_dir.name, archive_path)
        return str(archive_path)

    def execute(self):
        """
        main function to execute the training process
        """
        pass

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
