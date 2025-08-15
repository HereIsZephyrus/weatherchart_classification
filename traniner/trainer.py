"""
Trainer class for CNN-RNN unified framework with two-stage training strategy.
Based on the training strategy from docs/train.md section 3.2.
"""
import logging
import os
import json
from typing import Dict, Optional, Any
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

logger = logging.getLogger(__name__)
__all__ = ["WeatherChartTrainer"]

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
        eval_dataloader: Optional[DataLoader] = None,
        label_processor: Optional[LabelProcessor] = None
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.label_processor = label_processor

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.training_stage = "warmup"  # "warmup" or "finetune"

        # Initialize components
        self.device = self._setup_device()
        self.model = self.model.to(self.device)

        self.loss_calculator = LossCalculator(
            bce_weight=config.training.bce_loss_weight,
            sequence_weight=config.training.sequence_loss_weight,
            coverage_weight=config.training.coverage_loss_weight,
            use_focal_loss=config.training.use_focal_loss,
            focal_alpha=config.training.focal_alpha,
            focal_gamma=config.training.focal_gamma
        )

        if label_processor:
            self.metrics_calculator = MetricsCalculator(
                label_names=list(label_processor.label_mapping.keys())
            )
        else:
            self.metrics_calculator = None

        self.teacher_forcing_scheduler = TeacherForcingScheduler(
            start_ratio=config.training.teacher_forcing_start,
            end_ratio=config.training.teacher_forcing_end,
            total_steps=len(train_dataloader) * config.training.num_epochs,
            schedule_type=config.training.teacher_forcing_decay
        )

        # Setup optimizers and schedulers
        self.optimizers = self._setup_optimizers()
        self.schedulers = self._setup_schedulers()

        # Output directories
        self.output_dir = config.output_dir
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
            lr=self.config.training.warmup_learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
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
                'lr': self.config.training.cnn_learning_rate,
                'name': 'cnn'
            },
            {
                'params': rnn_params,
                'lr': self.config.training.rnn_learning_rate,
                'name': 'rnn'
            }
        ]

        optimizers['finetune'] = optim.AdamW(
            param_groups,
            weight_decay=self.config.training.weight_decay,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            eps=self.config.training.adam_epsilon
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
            for epoch in range(self.config.training.num_epochs):
                self.current_epoch = epoch

                # Determine training stage
                if epoch < self.config.training.warmup_epochs:
                    self.training_stage = "warmup"
                    if epoch == 0:  # First warmup epoch
                        self.model.freeze_cnn()
                        logger.info("Stage 1: Warmup training (CNN frozen)")
                else:
                    if self.training_stage == "warmup":  # First fine-tuning epoch
                        self.training_stage = "finetune"
                        self.model.unfreeze_cnn()
                        logger.info("Stage 2: End-to-end fine-tuning (CNN unfrozen)")

                # Train for one epoch
                train_metrics = self._train_epoch()

                # Evaluate
                if self.eval_dataloader and (epoch + 1) % 2 == 0:
                    eval_metrics = self._evaluate()

                    # Check for best model
                    current_metric = eval_metrics.get(
                        self.config.training.metric_for_best_model, 0.0
                    )
                    if current_metric > self.best_metric:
                        self.best_metric = current_metric
                        self._save_model("best_model")
                        logger.info("New best model saved with %s: %.4f", self.config.training.metric_for_best_model, current_metric)
                else:
                    eval_metrics = {}

                # Log metrics
                all_metrics = {**train_metrics, **eval_metrics}
                all_metrics["epoch"] = epoch
                all_metrics["learning_rate"] = self._get_current_lr()
                all_metrics["training_stage"] = self.training_stage

                self._log_metrics(all_metrics)

                # Save checkpoint
                if (epoch + 1) % self.config.training.save_steps == 0:
                    self._save_model(f"checkpoint_epoch_{epoch}")

                # Update learning rate
                if self.training_stage in self.schedulers:
                    self.schedulers[self.training_stage].step()

        except KeyboardInterrupt:
            logger.info("Training interrupted by user")
        except Exception as e:
            logger.error("Training failed with error: %s", e)
            raise
        finally:
            # Save final model
            self._save_model("final_model")

        logger.info("Training completed")

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Get current optimizer
        optimizer = self.optimizers[self.training_stage]

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
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.max_grad_norm
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

            # Log step metrics
            if self.global_step % self.config.training.logging_steps == 0:
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
        input_sequences = batch.get("input_sequences")
        attention_mask = batch.get("attention_mask")

        # Teacher forcing during training
        if input_sequences is not None and self.model.training:
            # Apply teacher forcing ratio
            tf_ratio = self.teacher_forcing_scheduler.get_ratio(self.global_step)

            # For simplicity, we'll use teacher forcing for now
            # In practice, you might want to implement scheduled sampling
            outputs = self.model(
                images=images,
                input_labels=input_sequences,
                attention_mask=attention_mask
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
        attention_weights = outputs.get("attention_weights")

        target_sequences = batch.get("target_sequences")
        target_labels = batch.get("target_labels")
        sequence_mask = batch.get("attention_mask")

        return self.loss_calculator.calculate_loss(
            sequential_logits=sequential_logits,
            parallel_logits=parallel_logits,
            target_sequence=target_sequences,
            target_labels=target_labels,
            attention_weights=attention_weights,
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
                if "parallel_logits" in outputs:
                    predictions = torch.sigmoid(outputs["parallel_logits"])
                    targets = batch.get("target_labels")

                    if targets is not None:
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
        optimizer = self.optimizers[self.training_stage]
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
        self.model.save_pretrained(checkpoint_dir)

        # Save training state
        training_state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "training_stage": self.training_stage,
            "optimizer_state": self.optimizers[self.training_stage].state_dict(),
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
                    max_length=self.config.model.max_sequence_length,
                    beam_width=self.config.model.beam_width,
                    early_stopping=self.config.model.beam_early_stopping
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

                from .utils import save_predictions as save_pred_util
                save_pred_util(predictions, output_path)

            if return_predictions:
                return predictions

        return None
