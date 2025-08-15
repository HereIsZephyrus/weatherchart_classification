"""
Configuration classes for the CNN-RNN unified framework training.
"""
from typing import ClassVar
import logging
import json
from transformers import PretrainedConfig
from pydantic import BaseModel
from ..constants import EPOCH_NUM, SINGLE_EXPOCH_SIZE

logger = logging.getLogger(__name__)

class CNNconfig(BaseModel):
    """
    CNN configuration
    Args:
        cnn_backbone: CNN backbone model
        cnn_feature_dim: CNN feature dimension
        cnn_dropout: CNN dropout rate
    """
    cnn_backbone: ClassVar[str] = "resnet50"
    cnn_feature_dim: ClassVar[int] = 2048
    cnn_dropout: float
    cnn_output_dim: int
    
class RNNconfig(BaseModel):
    """
    RNN configuration
    Args:
        rnn_type: RNN type
        rnn_num_layers: RNN number of layers
        rnn_hidden_dim: RNN hidden dimension
        rnn_dropout: RNN dropout rate
        rnn_bidirectional: RNN bidirectional
    """
    rnn_type:ClassVar[str] = "LSTM"
    rnn_num_layers: int
    rnn_input_dim: int
    rnn_hidden_dim: int
    rnn_dropout: float
    rnn_bidirectional:ClassVar[bool] = False

class UnifiedConfig(BaseModel):
    """
    Unified configuration
    Args:
        beam_width: Beam width
        beam_max_length: Beam max length
        beam_early_stopping: Beam early stopping
        joint_embedding_dim: The CNN and RNN joint embedding dimension
    """
    beam_width: int
    beam_max_length: int
    beam_early_stopping: bool
    joint_embedding_dim: int

class BasicTrainingConfig(BaseModel):
    """
    Basic training configuration
    Args:
        num_epochs: Number of epochs
        batch_size: Batch size
        gradient_accumulation_steps: Gradient accumulation steps
        max_grad_norm: Max gradient norm
    """
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float

class LearningStrategyConfig(BaseModel):
    """
    Learning strategy configuration
    Args:
        warmup_epochs: Warmup epochs
        freeze_cnn_during_warmup: Freeze CNN during warmup
        cnn_learning_rate: CNN learning rate
        rnn_learning_rate: RNN learning rate
        warmup_learning_rate: Warmup learning rate
    """
    # learning rate
    warmup_epochs: ClassVar[int] = 5
    freeze_cnn_during_warmup: ClassVar[bool] = True
    cnn_learning_rate: float
    rnn_learning_rate: float
    warmup_learning_rate: float
    # Teacher Forcing Schedule
    teacher_forcing_start: ClassVar[float] = 1.0
    teacher_forcing_end: ClassVar[float] = 0.7
    teacher_forcing_decay: ClassVar[str] = "linear"  # linear, exponential
    # Focal Loss for Class Imbalance
    use_focal_loss: ClassVar[bool] = True
    focal_alpha: ClassVar[float] = 0.25
    focal_gamma: ClassVar[float] = 2.0
    # Label Order Strategy
    label_order_strategy: ClassVar[str] = "frequency"  # frequency, random, fixed
    random_order_ratio: ClassVar[float] = 0.2  # 20% samples use random order

class OptimizerConfig(BaseModel):
    """
    Optimizer configuration
    Args:
        optimizer: Optimizer
        weight_decay: Weight decay
        adam_beta1: Adam beta1
        adam_beta2: Adam beta2
        adam_epsilon: Adam epsilon
    """
    optimizer: str = "AdamW"
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float

class LossWeightConfig(BaseModel):
    """
    Loss weight configuration
    Args:
        bce_loss_weight: BCE loss weight
        sequence_loss_weight: Sequence loss weight
        coverage_loss_weight: Coverage loss weight
    """
    bce_loss_weight: float
    sequence_loss_weight: float
    coverage_loss_weight: float

class ValidationConfig(BaseModel):
    """
    Validation configuration
    Args:
        eval_steps: Evaluation steps
        save_steps: Save steps
        save_total_limit: Save total limit
        metric_for_best_model: Metric for best model
        greater_is_better: Greater is better
        early_stopping: Early stopping
        early_stopping_patience: Early stopping patience
        early_stopping_threshold: Early stopping threshold
    """
    # Validation and Saving
    eval_steps: ClassVar[int] = 500
    save_steps: ClassVar[int] = 1000
    save_total_limit: ClassVar[int] = 3
    metric_for_best_model: ClassVar[str] = "eval_f1_macro"
    greater_is_better: ClassVar[bool] = True
    # Early Stopping
    early_stopping: ClassVar[bool] = True
    early_stopping_patience: int
    early_stopping_threshold: float

class Hyperparameter(BaseModel):
    """
    Hyperparameter list that can be configured
    """
    # CNN Parameters
    cnn_dropout: float
    # RNN Parameters
    rnn_num_layers: int
    rnn_hidden_dim: int
    rnn_dropout: float
    # Beam Search Parameters
    beam_width: int
    beam_max_length: int
    beam_early_stopping: bool
    # Model Architecture Parameters
    joint_embedding_dim: int
    # Training Control Parameters
    gradient_accumulation_steps: int
    max_grad_norm: float
    # Learning Rate Parameters
    cnn_learning_rate: float
    rnn_learning_rate: float
    warmup_learning_rate: float
    # Optimizer Parameters
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    # Loss Weight Parameters
    bce_loss_weight: float
    sequence_loss_weight: float
    coverage_loss_weight: float
    # Early Stopping Parameters
    early_stopping_patience: int
    early_stopping_threshold: float

class ModelConfig(PretrainedConfig):
    """
    Model configuration class, compatible with Hugging Face.
    """

    model_type = "weather_chart_cnn_rnn"

    def __init__(self,parameter: Hyperparameter, **kwargs):
        self.seed = 473066198
        self.cnn_config = CNNconfig(
            cnn_dropout=parameter.cnn_dropout,
            cnn_output_dim=parameter.joint_embedding_dim
        )
        self.rnn_config = RNNconfig(
            rnn_num_layers=parameter.rnn_num_layers,
            rnn_input_dim=parameter.joint_embedding_dim,
            rnn_hidden_dim=parameter.rnn_hidden_dim,
            rnn_dropout=parameter.rnn_dropout
        )
        self.unified_config = UnifiedConfig(
            beam_width=parameter.beam_width,
            beam_max_length=parameter.beam_max_length,
            beam_early_stopping=parameter.beam_early_stopping,
            joint_embedding_dim=parameter.joint_embedding_dim
        )
        self.basic_config = BasicTrainingConfig(
            num_epochs=EPOCH_NUM,
            batch_size=SINGLE_EXPOCH_SIZE,
            gradient_accumulation_steps=parameter.gradient_accumulation_steps,
            max_grad_norm=parameter.max_grad_norm
        )
        self.learning_strategy_config = LearningStrategyConfig(
            cnn_learning_rate=parameter.cnn_learning_rate,
            rnn_learning_rate=parameter.rnn_learning_rate,
            warmup_learning_rate=parameter.warmup_learning_rate
        )
        self.optimizer_config = OptimizerConfig(
            weight_decay=parameter.weight_decay,
            adam_beta1=parameter.adam_beta1,
            adam_beta2=parameter.adam_beta2,
            adam_epsilon=parameter.adam_epsilon
        )
        self.loss_weight_config = LossWeightConfig(
            bce_loss_weight=parameter.bce_loss_weight,
            sequence_loss_weight=parameter.sequence_loss_weight,
            coverage_loss_weight=parameter.coverage_loss_weight
        )
        self.validation_config = ValidationConfig(
            early_stopping_patience=parameter.early_stopping_patience,
            early_stopping_threshold=parameter.early_stopping_threshold
        )
        super().__init__(**kwargs)

    def get_learning_rate_for_stage(self, stage: str, component: str) -> float:
        """
        Get learning rate for specific training stage and component.

        Args:
            stage: "warmup" or "finetune"
            component: "cnn" or "rnn"

        Returns:
            Learning rate for the specified stage and component
        """
        if stage == "warmup":
            return self.learning_strategy_config.warmup_learning_rate
        elif stage == "finetune":
            if component == "cnn":
                return self.learning_strategy_config.cnn_learning_rate
            elif component == "rnn":
                return self.learning_strategy_config.rnn_learning_rate
            else:
                raise ValueError(f"Unknown component: {component}")
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def print_config(self):
        """
        Print the configuration
        """
        total_config = {
            "cnn_config": self.cnn_config.model_dump(),
            "rnn_config": self.rnn_config.model_dump(),
            "unified_config": self.unified_config.model_dump(),
            "basic_learning_config": self.basic_config.model_dump(),
            "learning_strategy_config": self.learning_strategy_config.model_dump(),
            "optimizer_config": self.optimizer_config.model_dump(),
            "loss_weight_config": self.loss_weight_config.model_dump(),
            "validation_config": self.validation_config.model_dump()
        }
        with open("config.json", "w", encoding="utf-8") as f:
            json.dump(total_config, f, indent=2)

default_hyperparameter = Hyperparameter(
    # CNN Parameters
    cnn_dropout = 0.1,
    # RNN Parameters
    rnn_num_layers = 1,
    rnn_hidden_dim = 64,
    rnn_dropout = 0.2,
    # Beam Search Parameters
    beam_width = 2,
    beam_max_length = 10,
    beam_early_stopping = True,
    # Model Architecture Parameters
    joint_embedding_dim = 128,
    # Training Control Parameters
    gradient_accumulation_steps = 1,
    max_grad_norm = 1.0,
    # Learning Rate Parameters
    cnn_learning_rate = 1e-3,
    rnn_learning_rate = 5e-3,
    warmup_learning_rate = 2e-4,
    # Optimizer Parameters
    weight_decay = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-8,
    # Loss Weight Parameters
    bce_loss_weight = 1.0,
    sequence_loss_weight = 0.5,
    coverage_loss_weight = 0.1,
    # Early Stopping Parameters
    early_stopping_patience = 5,
    early_stopping_threshold = 0.001
)
