"""
Configuration classes for the CNN-RNN unified framework training.
"""
import logging
from pydantic import BaseModel
from inspector.dataset_maker import DatasetConfig
from ..constants import DATASET_DIR, IMAGE_SIZE

logger = logging.getLogger(__name__)

class CNNconfig(BaseModel):
    """
    CNN configuration
    Args:
        cnn_backbone: CNN backbone model
        cnn_feature_dim: CNN feature dimension
        cnn_dropout: CNN dropout rate
    """
    cnn_backbone = "resnet50"
    cnn_feature_dim = 2048
    cnn_dropout: float
    
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
    rnn_type = "LSTM"
    rnn_num_layers: int
    rnn_hidden_dim: int
    rnn_dropout: float
    rnn_bidirectional = False

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

class TokenConfig(BaseModel):
    """
    Token configuration
    Args:
        bos_token_id: BOS token id
        eos_token_id: EOS token id
        pad_token_id: PAD token id
    """
    bos_token_id = 0
    eos_token_id = 1
    pad_token_id = 2

class LabelConfig(BaseModel):
    """
    Label configuration
    Args:
        num_labels: Number of labels
        label_embedding_dim: Label embedding dimension
        max_sequence_length: Max sequence length
    """
    _num_labels = 0
    label_embedding_dim = 256
    max_sequence_length = 5

    @property
    def num_labels(self) -> int:
        if self._num_labels == 0:
            self._num_labels = self.count_label()
        return self._num_labels

    @classmethod
    def count_label(cls) -> int:
        return 0

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
    warmup_epochs = 5
    freeze_cnn_during_warmup = True
    cnn_learning_rate: float
    rnn_learning_rate: float
    warmup_learning_rate: float
    # Teacher Forcing Schedule
    teacher_forcing_start = 1.0
    teacher_forcing_end = 0.7
    teacher_forcing_decay = "linear"  # linear, exponential
    # Focal Loss for Class Imbalance
    use_focal_loss = True
    focal_alpha = 0.25
    focal_gamma = 2.0
    # Label Order Strategy
    label_order_strategy = "frequency"  # frequency, random, fixed
    random_order_ratio = 0.2  # 20% samples use random order

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
    optimizer = "AdamW"
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
    eval_steps = 500
    save_steps = 1000
    save_total_limit = 3
    metric_for_best_model = "eval_f1_macro"
    greater_is_better = True
    # Early Stopping
    early_stopping = True
    early_stopping_patience: int
    early_stopping_threshold: float

class Hyperparameter(BaseModel):
    """
    Hyperparameter list that can be configured
    """
    cnn_dropout : float
    rnn_num_layers : int
    rnn_hidden_dim : int
    rnn_dropout : float
    beam_width : int
    beam_max_length : int
    beam_early_stopping : bool
    joint_embedding_dim: int
    early_stopping_patience: int
    early_stopping_threshold: float
    gradient_accumulation_steps: int
    max_grad_norm: float
    cnn_learning_rate: float
    rnn_learning_rate: float
    warmup_learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    bce_loss_weight: float
    sequence_loss_weight: float
    coverage_loss_weight: float

class ModelConfig:
    """
    Model configuration class
    """
    def __init__(self,parameter: Hyperparameter):
        self.cnn_config = CNNconfig(
            cnn_dropout=parameter.cnn_dropout
        )
        self.rnn_config = RNNconfig(
            rnn_num_layers=parameter.rnn_num_layers,
            rnn_hidden_dim=parameter.rnn_hidden_dim,
            rnn_dropout=parameter.rnn_dropout
        )
        self.unified_config = UnifiedConfig(
            beam_width=parameter.beam_width,
            beam_max_length=parameter.beam_max_length,
            beam_early_stopping=parameter.beam_early_stopping,
            joint_embedding_dim=parameter.joint_embedding_dim
        )
        self.token_config = TokenConfig()
        self.label_config = LabelConfig()
        self.basic_config = BasicTrainingConfig(
            num_epochs=parameter.num_epochs,
            batch_size=parameter.batch_size,
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
            eval_steps=parameter.eval_steps,
            save_steps=parameter.save_steps,
            save_total_limit=parameter.save_total_limit,
            metric_for_best_model=parameter.metric_for_best_model,
            greater_is_better=parameter.greater_is_better,
            early_stopping=parameter.early_stopping,
            early_stopping_patience=parameter.early_stopping_patience,
            early_stopping_threshold=parameter.early_stopping_threshold
        )

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

default_hyperparameter = Hyperparameter(
    cnn_dropout = 0.1,
    rnn_num_layers = 2,
    rnn_hidden_dim = 384,
    rnn_dropout = 0.2,
    beam_width = 2,
    beam_max_length = 10,
    beam_early_stopping = True,
    joint_embedding_dim = 256,
    early_stopping_patience = 5,
    early_stopping_threshold = 0.001,
    gradient_accumulation_steps = 1,
    max_grad_norm = 1.0,
    cnn_learning_rate = 1e-4,
    rnn_learning_rate = 5e-4,
    warmup_learning_rate = 2e-3,
    weight_decay = 0.01,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-8,
    bce_loss_weight = 1.0,
    sequence_loss_weight = 0.5,
    coverage_loss_weight = 0.1
)
