"""
Trainer package for CNN-RNN unified framework.
Based on the design from docs/train.md section 3.2.
"""
# Import main components
from .config import (
    ModelConfig,
    TrainingConfig, 
    DataConfig,
    ExperimentConfig,
    create_default_config
)

from .model import (
    WeatherChartModel,
    WeatherChartConfig,
    CNNEncoder,
    RNNDecoder,
    DualPredictionHead
)

from .trainer import WeatherChartTrainer

from .utils import (
    ImageTransforms,
    LabelProcessor,
    MetricsCalculator,
    LossCalculator,
    TeacherForcingScheduler,
    set_seed,
    load_label_mapping,
    save_predictions
)

from .dataset import (
    WeatherChartDataset,
    create_dataloaders,
    collate_fn
)

__all__ = [
    # Configuration
    'ModelConfig',
    'TrainingConfig',
    'DataConfig', 
    'ExperimentConfig',
    'create_default_config',

    # Model components
    'WeatherChartModel',
    'WeatherChartConfig',
    'CNNEncoder',
    'RNNDecoder',
    'DualPredictionHead',

    # Training
    'WeatherChartTrainer',

    # Utilities
    'ImageTransforms',
    'LabelProcessor',
    'MetricsCalculator',
    'LossCalculator',
    'TeacherForcingScheduler',
    'set_seed',
    'load_label_mapping',
    'save_predictions',

    # Dataset
    'WeatherChartDataset',
    'create_dataloaders',
    'collate_fn',
]
