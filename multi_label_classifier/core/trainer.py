"""
Trainer for CNN-RNN unified model implementing a two-stage training strategy:
1. Warmup: Train RNN with frozen CNN
2. Fine-tuning: End-to-end training with differential learning rates
"""
import logging
import os
import json
from typing import Dict, Optional, Any
from enum import Enum
import torch
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm.auto import tqdm
from .config import ModelConfig
from .model import WeatherChartModel
from .utils import (
    LossCalculator, MetricsCalculator, TeacherForcingScheduler,
    LabelProcessor, set_seed
)
from .vocab import vocabulary
from .utils import save_predictions as save_pred_util

logger = logging.getLogger(__name__)
__all__ = ["WeatherChartTrainer"]

class TrainingStage(Enum):
    """Training stages for the model"""
    WARMUP = "warmup"
    FINE_TUNE = "finetune"

class WeatherChartTrainer:
    """
    Trainer for CNN-RNN unified framework with two-stage training strategy.

    Training Stages:
    1. Warmup (Epochs 1-5): Freeze CNN, optimize RNN and projection layers
    2. End-to-end (Epochs 6+): Unfreeze CNN with differential learning rates
    """

    def __init__(
        self,
        config: ModelConfig,
        model: WeatherChartModel,
        train_dataloader: DataLoader,
        output_dir: str,
        eval_dataloader: Optional[DataLoader] = None,
        label_processor: Optional[LabelProcessor] = None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        # Initialize label processor if not provided
        if label_processor is None:
            # Create label mapping from vocabulary
            label_mapping = {vocabulary.idx2token[i]: i for i in range(len(vocabulary.idx2token)) 
                           if vocabulary.idx2token[i] not in [vocabulary.unk, vocabulary.bos, vocabulary.eos]}
            label_processor = LabelProcessor(label_mapping=label_mapping)
            logger.info("Created label processor with %d labels from vocabulary", len(label_mapping))

        self.label_processor = label_processor

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.training_stage = TrainingStage.WARMUP

        # Early stopping config
        self.no_improvement_count = 0
        self.early_stopping_patience = self.config.validation_config.early_stopping_patience
        self.early_stopping_threshold = self.config.validation_config.early_stopping_threshold

        # Initialize components
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        self.loss_calculator = LossCalculator(
            bce_weight=config.loss_weight_config.bce_loss_weight,
            sequence_weight=config.loss_weight_config.sequence_loss_weight,
            use_focal_loss=config.learning_strategy_config.use_focal_loss,
            focal_alpha=config.learning_strategy_config.focal_alpha,
            focal_gamma=config.learning_strategy_config.focal_gamma
        )

        if label_processor:
            self.metrics_calculator = MetricsCalculator(
                label_names=list(label_processor.label_mapping.keys())
            )
        else:
            self.metrics_calculator = None

        self.teacher_forcing_scheduler = TeacherForcingScheduler(
            start_ratio=config.learning_strategy_config.teacher_forcing_start,
            end_ratio=config.learning_strategy_config.teacher_forcing_end,
            total_steps=len(train_dataloader) * config.basic_config.num_epochs,
            schedule_type=config.learning_strategy_config.teacher_forcing_decay
        )

        # Setup optimizers and schedulers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()

        # Output directories
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info("Initialized WeatherChartTrainer")

    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.device)

        logger.info("Using device: %s", device)
        return device

    def _setup_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Setup optimizers for different training stages."""
        optimizers = {}

        # Warmup optimizer (only RNN components)
        rnn_params = []
        for name, param in self.model.named_parameters():
            if 'cnn_encoder' not in name and param.requires_grad:
                rnn_params.append(param)

        optimizers['warmup'] = optim.AdamW(
            rnn_params,
            lr=self.config.learning_strategy_config.warmup_learning_rate,
            weight_decay=self.config.optimizer_config.weight_decay,
            betas=(self.config.optimizer_config.adam_beta1, self.config.optimizer_config.adam_beta2),
            eps=self.config.optimizer_config.adam_epsilon
        )

        # Fine-tuning optimizers with differential learning rates
        cnn_params = []
        rnn_params = []

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'cnn_encoder' in name:
                    cnn_params.append(param)
                else:
                    rnn_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': cnn_params,
                'lr': self.config.learning_strategy_config.cnn_learning_rate,
                'name': 'cnn'
            },
            {
                'params': rnn_params,
                'lr': self.config.learning_strategy_config.rnn_learning_rate,
                'name': 'rnn'
            }
        ]

        optimizers['finetune'] = optim.AdamW(
            param_groups,
            weight_decay=self.config.optimizer_config.weight_decay,
            betas=(self.config.optimizer_config.adam_beta1, self.config.optimizer_config.adam_beta2),
            eps=self.config.optimizer_config.adam_epsilon
        )

        logger.info("Setup optimizers for warmup and fine-tuning stages")
        return optimizers

    def _setup_schedulers(self) -> Dict[str, Any]:
        """Setup learning rate schedulers."""
        schedulers = {}

        # Simple step schedulers for both stages
        for stage, optimizer in self.optimizers.items():
            schedulers[stage] = optim.lr_scheduler.StepLR(
                optimizer, step_size=10, gamma=0.8
            )

        return schedulers

    def train(self):
        """Main training loop with two-stage strategy."""
        logger.info("Starting training...")

        # Set seed for reproducibility
        set_seed(self.config.seed)

        try:
            for epoch in range(self.config.basic_config.num_epochs):
                self.current_epoch = epoch

                # Determine training stage
                if epoch < self.config.learning_strategy_config.warmup_epochs:
                    self.training_stage = TrainingStage.WARMUP
                    if epoch == 0:  # First warmup epoch
                        self.model.freeze_cnn()
                        logger.info("Stage 1: Warmup training (CNN frozen)")
                else:
                    if self.training_stage == TrainingStage.WARMUP:  # First fine-tuning epoch
                        self.training_stage = TrainingStage.FINE_TUNE
                        self.model.unfreeze_cnn()
                        logger.info("Stage 2: End-to-end fine-tuning (CNN unfrozen)")

                # Train for one epoch
                train_metrics = self._train_epoch()

                # Evaluate
                if self.eval_dataloader and (epoch + 1) % 2 == 0:
                    eval_metrics = self._evaluate()

                    # Check for best model
                    current_metric = eval_metrics.get(
                        self.config.validation_config.metric_for_best_model, 0.0
                    )
                    # Check if the current metric is better than the best metric
                    is_improved = False
                    if current_metric > self.best_metric + self.early_stopping_threshold:
                        self.best_metric = current_metric
                        self.best_metric = current_metric
                        self._save_model("best_model")
                        logger.info("New best model saved with %s: %.4f", self.config.validation_config.metric_for_best_model, current_metric)
                        self.no_improvement_count = 0
                        is_improved = True

                    if not is_improved:
                        self.no_improvement_count += 1
                        logger.info("No improvement for %d epochs", self.no_improvement_count)

                        # Reach early stopping patience
                        if self.config.validation_config.early_stopping and self.no_improvement_count >= self.early_stopping_patience:
                            logger.info("Early stopping triggered after %d epochs without improvement", self.no_improvement_count)
                            break
                else:
                    eval_metrics = {}

                # Log metrics
                all_metrics = {**train_metrics, **eval_metrics}
                all_metrics["epoch"] = epoch
                all_metrics["learning_rate"] = self._get_current_lr()
                all_metrics["training_stage"] = self.training_stage.value

                self._log_metrics(all_metrics)

                # Save checkpoint
                if (epoch + 1) % self.config.validation_config.save_steps == 0:
                    self._save_model(f"checkpoint_epoch_{epoch}")

                # Update learning rate
                stage_key = self.training_stage.value
                if stage_key in self.schedulers:
                    self.schedulers[stage_key].step()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error("Training failed with error: %s", e)
            raise
        #finally:
            # Save final model
        #    self._save_model("final_model")

        logger.info("Training completed")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Get current optimizer
        stage_key = self.training_stage.value
        optimizer = self.optimizers[stage_key]

        # Metrics tracking
        epoch_losses = []
        epoch_metrics = {}

        # Progress bar
        pbar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.current_epoch} ({self.training_stage})",
            leave=False
        )

        for batch in pbar:
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass
            outputs = self._forward_step(batch)

            # Calculate loss
            loss_dict = self._calculate_loss(outputs, batch)
            total_loss = loss_dict["total_loss"]

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()

            # Gradient clipping
            if self.config.basic_config.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.basic_config.max_grad_norm
                )

            optimizer.step()

            # Update metrics
            epoch_losses.append(total_loss.item())
            for key, value in loss_dict.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value.item())

            # Update progress bar
            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "lr": f"{self._get_current_lr():.2e}"
            })

            # Global step tracking
            self.global_step += 1
            self.global_step = self.global_step

            # Log step metrics
            if self.global_step % 100 == 0:
                step_metrics = {
                    "step_loss": total_loss.item(),
                    "step": self.global_step,
                    "teacher_forcing_ratio": self.teacher_forcing_scheduler.get_ratio(self.global_step)
                }
                self._log_metrics(step_metrics, step=self.global_step)

        # Calculate epoch averages
        train_metrics = {
            f"train_{key}": np.mean(values)
            for key, values in epoch_metrics.items()
        }
        train_metrics["train_loss"] = np.mean(epoch_losses)

        return train_metrics

    def _forward_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass for one step."""
        images = batch["images"]

        # Process labels to create input sequences for teacher forcing
        input_sequences = None

        if "labels" in batch and self.model.training:
            # Convert label names to sequences for teacher forcing
            batch_sequences = []
            max_seq_len = 0

            for label_list in batch["labels"]:
                if label_list:  # Non-empty label list
                    sequence = self.label_processor.create_sequence(label_list)
                    batch_sequences.append(sequence)
                    max_seq_len = max(max_seq_len, len(sequence))
                else:
                    # Empty label case - create sequence with just BOS/EOS
                    sequence = torch.tensor([vocabulary.bos, vocabulary.eos], dtype=torch.long)
                    batch_sequences.append(sequence)
                    max_seq_len = max(max_seq_len, len(sequence))

            # Pad sequences to same length
            if batch_sequences:
                padded_sequences = []

                for sequence in batch_sequences:
                    if len(sequence) < max_seq_len:
                        # Pad with UNK tokens
                        padded_seq = torch.cat([
                            sequence, 
                            torch.full((max_seq_len - len(sequence),), vocabulary.unk, dtype=torch.long)
                        ])
                    else:
                        padded_seq = sequence

                    padded_sequences.append(padded_seq)

                input_sequences = torch.stack(padded_sequences).to(self.device)

        # Teacher forcing during training
        if input_sequences is not None and self.model.training:
            # Apply teacher forcing ratio
            teacher_forcing_ratio = self.teacher_forcing_scheduler.get_ratio(self.global_step)

            # Implement scheduled sampling
            use_teacher_forcing = True
            if teacher_forcing_ratio < 1.0: # determine whether to use teacher forcing
                use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio

            outputs = self.model(
                images=images,
                input_labels=input_sequences,
                use_teacher_forcing=use_teacher_forcing
            )
        else:
            # Inference mode
            outputs = self.model(images=images)

        return outputs

    def _calculate_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate multi-task loss."""
        sequential_logits = outputs.get("sequential_logits")
        parallel_logits = outputs.get("parallel_logits")
        # No attention weights in new implementation

        # Create target sequences and labels from batch
        target_sequences = None
        target_labels = None
        sequence_mask = None

        if "labels" in batch:
            # Create target sequences (shifted input sequences for next token prediction)
            batch_target_sequences = []
            batch_target_labels = []
            max_seq_len = 0

            for label_list in batch["labels"]:
                if label_list:  # Non-empty label list
                    # Create input sequence with BOS/EOS
                    sequence = self.label_processor.create_sequence(label_list)
                    # Target sequence is input sequence shifted by 1 (for next token prediction)
                    target_seq = sequence[1:]  # Remove BOS, keep EOS
                    batch_target_sequences.append(target_seq)
                    max_seq_len = max(max_seq_len, len(target_seq))

                    # Create binary label vector for parallel prediction using vocabulary size
                    label_vector = torch.zeros(len(vocabulary))
                    for label_name in label_list:
                        if label_name in vocabulary.token2idx:
                            label_idx = vocabulary.token2idx[label_name]
                            # Skip special tokens for multi-hot vector
                            if label_idx not in [vocabulary.bos, vocabulary.eos, vocabulary.unk]:
                                label_vector[label_idx] = 1.0
                    batch_target_labels.append(label_vector)
                else:
                    # Empty label case
                    target_seq = torch.tensor([vocabulary.eos], dtype=torch.long)
                    batch_target_sequences.append(target_seq)
                    max_seq_len = max(max_seq_len, len(target_seq))

                    # Empty label vector using vocabulary size
                    label_vector = torch.zeros(len(vocabulary))
                    batch_target_labels.append(label_vector)

            # Pad target sequences
            if batch_target_sequences:
                padded_target_sequences = []
                sequence_masks = []

                for target_seq in batch_target_sequences:
                    seq_len = len(target_seq)
                    if seq_len < max_seq_len:
                        # Pad with UNK tokens
                        padded_seq = torch.cat([
                            target_seq, 
                            torch.full((max_seq_len - seq_len,), vocabulary.unk, dtype=torch.long)
                        ])
                    else:
                        padded_seq = target_seq

                    # Create sequence mask (1 for real tokens, 0 for padding)
                    mask = torch.cat([
                        torch.ones(seq_len, dtype=torch.long),
                        torch.zeros(max_seq_len - seq_len, dtype=torch.long)
                    ])

                    padded_target_sequences.append(padded_seq)
                    sequence_masks.append(mask)

                target_sequences = torch.stack(padded_target_sequences).to(self.device)
                sequence_mask = torch.stack(sequence_masks).to(self.device)
                target_labels = torch.stack(batch_target_labels).to(self.device)
        else:
            logger.warning("No labels provided for training")
            return {}
        
        return self.loss_calculator.calculate_loss(
            sequential_logits=sequential_logits,
            parallel_logits=parallel_logits,
            target_sequence=target_sequences,
            target_labels=target_labels,
            sequence_mask=sequence_mask
        )

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on validation set."""
        self.model.eval()

        all_predictions = []
        all_targets = []
        eval_losses = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}

                # Forward pass
                outputs = self._forward_step(batch)

                # Calculate loss
                if "target_sequences" in batch and "target_labels" in batch:
                    loss_dict = self._calculate_loss(outputs, batch)
                    eval_losses.append(loss_dict["total_loss"].item())

                # Collect predictions for metrics
                if "parallel_logits" in outputs and "labels" in batch:
                    predictions = torch.sigmoid(outputs["parallel_logits"])

                    # Create target labels from batch labels using vocabulary
                    batch_target_labels = []
                    for label_list in batch["labels"]:
                        label_vector = torch.zeros(len(vocabulary))
                        for label_name in label_list:
                            if label_name in vocabulary.token2idx:
                                label_idx = vocabulary.token2idx[label_name]
                                # Skip special tokens for multi-hot vector
                                if label_idx not in [vocabulary.bos, vocabulary.eos, vocabulary.unk]:
                                    label_vector[label_idx] = 1.0
                        batch_target_labels.append(label_vector)

                    if batch_target_labels:
                        targets = torch.stack(batch_target_labels).to(self.device)
                        all_predictions.append(predictions.cpu().numpy())
                        all_targets.append(targets.cpu().numpy())

        # Calculate metrics
        eval_metrics = {}

        if eval_losses:
            eval_metrics["eval_loss"] = np.mean(eval_losses)

        if all_predictions and all_targets and self.metrics_calculator:
            y_pred = np.vstack(all_predictions)
            y_true = np.vstack(all_targets)

            metrics = self.metrics_calculator.calculate_multilabel_metrics(
                y_true, y_pred
            )
            eval_metrics.update({f"eval_{k}": v for k, v in metrics.items()})

        return eval_metrics

    def _get_current_lr(self) -> float:
        """Get current learning rate."""
        stage_key = self.training_stage.value
        optimizer = self.optimizers[stage_key]
        return optimizer.param_groups[0]['lr']

    def _log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to console and wandb."""
        # Console logging
        if step is None:
            step = self.global_step

        log_str = f"Step {step}"
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                log_str += f" | {key}: {value:.4f}"
        logger.info(log_str)

    def _save_model(self, checkpoint_name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save model
        self.model.save_pretrained(save_directory=checkpoint_dir)

        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "training_stage": self.training_stage.value,
            "optimizer_state": self.optimizers[self.training_stage.value].state_dict(),
            "config": self.config.to_dict()
        }

        with open(os.path.join(checkpoint_dir, "training_state.json"), "w", encoding="utf-8") as f:
            json.dump(training_state, f, indent=2)

        logger.info("Saved checkpoint: %s", checkpoint_dir)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        # Load model
        self.model = WeatherChartModel.from_pretrained(checkpoint_path)
        self.model = self.model.to(self.device)

        # Load training state
        training_state_path = os.path.join(checkpoint_path, "training_state.json")
        if os.path.exists(training_state_path):
            with open(training_state_path, "r", encoding="utf-8") as f:
                training_state = json.load(f)

            self.current_epoch = training_state.get("epoch", 0)
            self.global_step = training_state.get("global_step", 0)
            self.best_metric = training_state.get("best_metric", 0.0)
            self.training_stage = training_state.get("training_stage", "warmup")

            # Load optimizer state
            if "optimizer_state" in training_state:
                self.optimizers[self.training_stage].load_state_dict(
                    training_state["optimizer_state"]
                )

        logger.info("Loaded checkpoint from: %s", checkpoint_path)

    def predict(
        self,
        dataloader: DataLoader,
        return_predictions: bool = True,
        save_predictions: bool = False,
        output_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Generate predictions on a dataset."""
        self.model.eval()

        all_predictions = []
        all_sequences = []
        all_scores = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                images = batch["images"].to(self.device)

                # Generate sequences using beam search
                generation_outputs = self.model.generate(
                    images=images,
                    early_stopping=self.config.unified_config.beam_early_stopping
                )

                # Get parallel predictions
                parallel_outputs = self.model(images=images)
                parallel_predictions = torch.sigmoid(
                    parallel_outputs["parallel_logits"]
                ).cpu().numpy()

                # Convert sequences to label names
                sequences = generation_outputs["sequences"].cpu().numpy()
                scores = generation_outputs["scores"].cpu().numpy()

                if self.label_processor:
                    decoded_sequences = []
                    for seq in sequences:
                        # Remove special tokens and decode
                        seq_list = seq.tolist()
                        labels = self.label_processor.decode_labels(seq_list)
                        decoded_sequences.append(labels)

                    all_sequences.extend(decoded_sequences)
                else:
                    all_sequences.extend(sequences.tolist())

                all_predictions.extend(parallel_predictions)
                all_scores.extend(scores)

        if return_predictions or save_predictions:
            predictions = {
                "parallel_predictions": all_predictions,
                "sequence_predictions": all_sequences,
                "sequence_scores": all_scores,
                "config": self.config.to_dict()
            }

            if save_predictions:
                if output_path is None:
                    output_path = os.path.join(self.output_dir, "predictions.json")

                save_pred_util(predictions, output_path)

            if return_predictions:
                return predictions

        return None
