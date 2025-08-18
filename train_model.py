"""
entry point for the multi-label classifier trainning
Uses improved DatasetLoader with state tracking and core training modules.

Usage:
    python train_model.py
"""
import os
import logging
import torch
from .multi_label_classifier import(
    ExperimentManager,
    TrainingConfig,
    DataSpliter,
    SplitConfig,
    CURRENT_DATASET_DIR
)
from .constants import MULTI_LABEL_EXPERIMENTS_DIR

config = TrainingConfig(
    trainer_name="classifier_v1",
    dataset_root=CURRENT_DATASET_DIR,
    experiments_root=MULTI_LABEL_EXPERIMENTS_DIR,
    training_mode="train",
    description="Training classifier_v1",
    tags=["classifier_v1"]
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'{MULTI_LABEL_EXPERIMENTS_DIR}/{config.trainer_name}/training.log')
    ]
)
logger = logging.getLogger(__name__)


def train_model():
    """
    train the model
    """ 
    if not os.path.exists(f"{CURRENT_DATASET_DIR}/metadata"):
        logger.info("No dataset found, splitting the dataset")
        spliter = DataSpliter(SplitConfig())
        spliter.split()
    os.makedirs(MULTI_LABEL_EXPERIMENTS_DIR, exist_ok=True)
    logger.info("Creating experiment manager")
    exp_manager = ExperimentManager(config)
    logger.info("Executing training")
    exp_manager.execute()
    logger.info("Training completed")

if __name__ == "__main__":
    print("PyTorch 版本:", torch.__version__)
    print("CUDA 版本:", torch.version.cuda)
    train_model()
