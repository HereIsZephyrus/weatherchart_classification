"""
Configuration classes for the CNN-RNN unified model training.
Includes model architecture, training strategy, and optimization parameters.
"""
from typing import ClassVar
import logging
import json
from transformers import PretrainedConfig
from pydantic import BaseModel
from ..settings import EPOCH_NUM, SAMPLE_PER_BATCH

logger = logging.getLogger(__name__)

class CNNconfig(BaseModel):
    """CNN backbone configuration parameters"""
    cnn_backbone: ClassVar[str] = "resnet50"
    cnn_feature_dim: ClassVar[int] = 2048
    cnn_dropout: float
    cnn_output_dim: int

class RNNconfig(BaseModel):
    """RNN architecture configuration parameters"""
    rnn_type:ClassVar[str] = "LSTM"
    rnn_num_layers: int
    rnn_input_dim: int
    rnn_hidden_dim: int
    rnn_dropout: float
    rnn_bidirectional:ClassVar[bool] = False

class UnifiedConfig(BaseModel):
    """Configuration for beam search and joint embedding"""
    beam_width: int
    beam_max_length: int
    beam_early_stopping: bool
    joint_embedding_dim: int

class BasicTrainingConfig(BaseModel):
    """Basic training hyperparameters"""
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    max_grad_norm: float

class LearningStrategyConfig(BaseModel):
    """Training strategy parameters including learning rates and teacher forcing"""
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
    """Optimizer configuration and hyperparameters"""
    optimizer: str = "AdamW"
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float

class LossWeightConfig(BaseModel):
    """Loss weights for multi-task learning"""
    bce_loss_weight: float
    sequence_loss_weight: float

class ValidationConfig(BaseModel):
    """Validation and early stopping configuration"""
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
    """Configurable hyperparameters for model training"""
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
    # Early Stopping Parameters
    early_stopping_patience: int
    early_stopping_threshold: float

class ModelConfig(PretrainedConfig):
    """Model configuration class compatible with Transformers"""

    model_type = "weather_chart_cnn_rnn"

    def __init__(self, config_list: Hyperparameter = None, **kwargs):
        self.seed = 473066198
        self.device = "auto"

        # Create default configuration if config_list is None (for Transformers compatibility)
        if config_list is None:
            self.cnn_config = CNNconfig(
                cnn_dropout=0.1,
                cnn_output_dim=256
            )
            self.rnn_config = RNNconfig(
                rnn_num_layers=2,
                rnn_input_dim=256,
                rnn_hidden_dim=256,
                rnn_dropout=0.1
            )
            self.unified_config = UnifiedConfig(
                beam_width=5,
                beam_max_length=20,
                beam_early_stopping=True,
                joint_embedding_dim=256
            )
            self.basic_config = BasicTrainingConfig(
                num_epochs=10,
                batch_size=8,
                gradient_accumulation_steps=1,
                max_grad_norm=1.0
            )
            self.learning_strategy_config = LearningStrategyConfig(
                cnn_learning_rate=1e-5,
                rnn_learning_rate=1e-4,
                warmup_learning_rate=1e-4
            )
            self.optimizer_config = OptimizerConfig(
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1e-8
            )
            self.loss_weight_config = LossWeightConfig(
                bce_loss_weight=1.0,
                sequence_loss_weight=0.5
            )
            self.validation_config = ValidationConfig(
                early_stopping_patience=3,
                early_stopping_threshold=0.001
            )
        else:
            # Use provided configuration
            self.cnn_config = CNNconfig(
                cnn_dropout=config_list.cnn_dropout,
                cnn_output_dim=config_list.joint_embedding_dim
            )
            self.rnn_config = RNNconfig(
                rnn_num_layers=config_list.rnn_num_layers,
                rnn_input_dim=config_list.joint_embedding_dim,
                rnn_hidden_dim=config_list.rnn_hidden_dim,
                rnn_dropout=config_list.rnn_dropout
            )
            self.unified_config = UnifiedConfig(
                beam_width=config_list.beam_width,
                beam_max_length=config_list.beam_max_length,
                beam_early_stopping=config_list.beam_early_stopping,
                joint_embedding_dim=config_list.joint_embedding_dim
            )
            self.basic_config = BasicTrainingConfig(
                num_epochs=EPOCH_NUM,
                batch_size=SAMPLE_PER_BATCH,
                gradient_accumulation_steps=config_list.gradient_accumulation_steps,
                max_grad_norm=config_list.max_grad_norm
            )
            self.learning_strategy_config = LearningStrategyConfig(
                cnn_learning_rate=config_list.cnn_learning_rate,
                rnn_learning_rate=config_list.rnn_learning_rate,
                warmup_learning_rate=config_list.warmup_learning_rate
            )
            self.optimizer_config = OptimizerConfig(
                weight_decay=config_list.weight_decay,
                adam_beta1=config_list.adam_beta1,
                adam_beta2=config_list.adam_beta2,
                adam_epsilon=config_list.adam_epsilon
            )
            self.loss_weight_config = LossWeightConfig(
                bce_loss_weight=config_list.bce_loss_weight,
                sequence_loss_weight=config_list.sequence_loss_weight
            )
            self.validation_config = ValidationConfig(
                early_stopping_patience=config_list.early_stopping_patience,
                early_stopping_threshold=config_list.early_stopping_threshold
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

    def to_dict(self):
        """
        Convert the configuration to a dictionary.
        """
        config_dict = super().to_dict()
        if config_dict is None:
            config_dict = {}
        config_dict.update({
            "cnn_config": self.cnn_config.model_dump(),
            "rnn_config": self.rnn_config.model_dump(),
            "unified_config": self.unified_config.model_dump(),
            "basic_learning_config": self.basic_config.model_dump(),
            "learning_strategy_config": self.learning_strategy_config.model_dump(),
            "optimizer_config": self.optimizer_config.model_dump(),
            "loss_weight_config": self.loss_weight_config.model_dump(),
            "validation_config": self.validation_config.model_dump()
        })
        return config_dict
