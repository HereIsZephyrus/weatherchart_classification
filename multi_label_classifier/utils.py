"""
Utility functions for the CNN-RNN unified framework training.
Includes data preprocessing, sequence processing, metrics calculation, etc.
"""
import logging
import json
import random
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .vocab import vocabulary

logger = logging.getLogger(__name__)

class LabelProcessor:
    """
    Label processing utilities for sequence generation and multi-label handling.
    """

    def __init__(
        self,
        label_mapping: Dict[str, int],
    ):
        self.label_mapping = label_mapping
        self.reverse_mapping = {v: k for k, v in label_mapping.items()}
        self.num_labels = len(label_mapping)

        logger.info("Initialized LabelProcessor with %d labels", self.num_labels)

    def encode_labels(self, label_names: List[str]) -> List[int]:
        """
        Convert label names to IDs.

        Args:
            label_names: List of label names

        Returns:
            List of label IDs
        """
        # Use vocabulary directly to encode labels
        return vocabulary.embedding(label_names, add_boseos=False)

    def decode_labels(self, label_ids: List[int]) -> List[str]:
        """
        Convert label IDs back to names.

        Args:
            label_ids: List of label IDs

        Returns:
            List of label names
        """
        # Use vocabulary directly to decode labels
        return vocabulary.detokenize(label_ids, keep_bos_eos=False)

    def create_sequence(
        self,
        label_names: List[str],
        order_strategy: str = "frequency",
        label_frequencies: Optional[Dict[str, int]] = None
    ) -> torch.Tensor:
        """
        Create input sequence for training with BOS/EOS tokens.

        Args:
            label_names: List of label names
            order_strategy: How to order labels ("frequency", "random", "fixed")
            label_frequencies: Label frequency statistics for ordering

        Returns:
            Input sequence tensor
        """
        # Encode labels (without BOS/EOS)
        label_ids = vocabulary.embedding(label_names, add_boseos=False)

        # Order labels based on strategy
        if order_strategy == "frequency" and label_frequencies:
            # Sort by frequency (high to low)
            label_ids.sort(
                key=lambda x: label_frequencies.get(
                    vocabulary.idx2token[x] if x < len(vocabulary.idx2token) else "", 0
                ),
                reverse=True
            )
        elif order_strategy == "random":
            random.shuffle(label_ids)
        # "fixed" order keeps original order

        # Create sequence: [BOS, label1, label2, ..., EOS]
        sequence = [vocabulary.bos] + label_ids + [vocabulary.eos]

        # Truncate if too long
        if len(sequence) > vocabulary.max_sequence_length:
            sequence = sequence[:vocabulary.max_sequence_length-1] + [vocabulary.eos]

        return torch.tensor(sequence, dtype=torch.long)

    def create_target_sequence(self, input_sequence: torch.Tensor) -> torch.Tensor:
        """
        Create target sequence for teacher forcing (input shifted by 1).

        Args:
            input_sequence: Input sequence with BOS token

        Returns:
            Target sequence for loss calculation
        """
        # Target is input sequence shifted by 1 (remove BOS, keep EOS)
        return input_sequence[1:]

    def create_multi_hot_vector(self, label_names: List[str]) -> torch.Tensor:
        """
        Create multi-hot vector for parallel BCE loss.

        Args:
            label_names: List of label names

        Returns:
            Multi-hot vector tensor
        """
        multi_hot = torch.zeros(len(vocabulary), dtype=torch.float)
        
        # Get token IDs for each label name
        for name in label_names:
            if name in vocabulary.token2idx:
                multi_hot[vocabulary.token2idx[name]] = 1.0

        return multi_hot

    def pad_sequences(
        self, 
        sequences: List[torch.Tensor],
        max_length: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pad sequences to same length and create attention mask.

        Args:
            sequences: List of sequence tensors
            max_length: Maximum length (if None, use longest sequence)

        Returns:
            Padded sequences and attention mask
        """
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        padded_sequences = []
        attention_masks = []

        for seq in sequences:
            seq_len = len(seq)

            if seq_len >= max_length:
                # Truncate if necessary
                padded_seq = seq[:max_length]
                mask = torch.ones(max_length, dtype=torch.bool)
            else:
                # Pad with PAD token
                padding = torch.full(
                    (max_length - seq_len,), 
                    vocabulary.pad, 
                    dtype=torch.long
                )
                padded_seq = torch.cat([seq, padding])
                mask = torch.cat([
                    torch.ones(seq_len, dtype=torch.bool),
                    torch.zeros(max_length - seq_len, dtype=torch.bool)
                ])

            padded_sequences.append(padded_seq)
            attention_masks.append(mask)

        return torch.stack(padded_sequences), torch.stack(attention_masks)


class MetricsCalculator:
    """
    Calculate various metrics for multi-label classification.
    """

    def __init__(self, label_names: List[str]):
        self.label_names = label_names
        self.num_labels = len(label_names)

    def calculate_multilabel_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate comprehensive multi-label classification metrics.

        Args:
            y_true: True labels [batch_size, num_labels]
            y_pred: Predicted probabilities [batch_size, num_labels]
            threshold: Classification threshold

        Returns:
            Dictionary of metrics
        """
        # Convert predictions to binary
        y_pred_binary = (y_pred > threshold).astype(int)

        # Calculate metrics
        metrics = {}

        # Overall metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='macro', zero_division=0
        )
        metrics.update({
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred_binary, average='micro', zero_division=0
        )
        metrics.update({
            "precision_micro": precision,
            "recall_micro": recall,
            "f1_micro": f1
        })

        # Subset accuracy (exact match)
        subset_accuracy = accuracy_score(y_true, y_pred_binary)
        metrics["subset_accuracy"] = subset_accuracy

        # Hamming loss
        hamming_loss = np.mean(y_true != y_pred_binary)
        metrics["hamming_loss"] = hamming_loss

        # Per-label metrics
        for i, label_name in enumerate(self.label_names):
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i], y_pred_binary[:, i], average='binary', zero_division=0
            )
            metrics[f"{label_name}_precision"] = precision
            metrics[f"{label_name}_recall"] = recall
            metrics[f"{label_name}_f1"] = f1

        return metrics

    def calculate_sequence_metrics(
        self,
        y_true_sequences: List[List[int]],
        y_pred_sequences: List[List[int]]
    ) -> Dict[str, float]:
        """
        Calculate sequence-specific metrics.

        Args:
            y_true_sequences: True label sequences
            y_pred_sequences: Predicted label sequences

        Returns:
            Dictionary of sequence metrics
        """
        metrics = {}

        # Exact sequence match
        exact_matches = sum(
            true_seq == pred_seq 
            for true_seq, pred_seq in zip(y_true_sequences, y_pred_sequences)
        )
        metrics["sequence_accuracy"] = exact_matches / len(y_true_sequences)

        # Average sequence length
        true_lengths = [len(seq) for seq in y_true_sequences]
        pred_lengths = [len(seq) for seq in y_pred_sequences]
        metrics["avg_true_length"] = np.mean(true_lengths)
        metrics["avg_pred_length"] = np.mean(pred_lengths)

        # Length difference
        length_diffs = [abs(len(t) - len(p)) for t, p in zip(y_true_sequences, y_pred_sequences)]
        metrics["avg_length_diff"] = np.mean(length_diffs)

        return metrics


class LossCalculator:
    """
    Calculate the multi-task loss combining BCE, sequence, and coverage components.
    """

    def __init__(
        self,
        bce_weight: float = 1.0,
        sequence_weight: float = 0.5,
        coverage_weight: float = 0.1,
        use_focal_loss: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        self.bce_weight = bce_weight
        self.sequence_weight = sequence_weight
        self.coverage_weight = coverage_weight
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        logger.info("Initialized loss calculator with weights: BCE=%f, Seq=%f, Coverage=%f", bce_weight, sequence_weight, coverage_weight)

    def focal_loss(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2.0
    ) -> torch.Tensor:
        """
        Calculate focal loss for addressing class imbalance.

        Args:
            inputs: Predicted logits
            targets: Target labels
            alpha: Weighting factor for rare class
            gamma: Focusing parameter

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()

    def coverage_loss(
        self,
        attention_weights: torch.Tensor,
        sequence_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate coverage loss to prevent repetitive attention.

        Args:
            attention_weights: Attention weights [batch_size, seq_len, seq_len]
            sequence_mask: Sequence mask [batch_size, seq_len]

        Returns:
            Coverage loss value
        """
        # Sum attention weights over time steps
        coverage = attention_weights.sum(dim=1)  # [batch_size, seq_len]

        # Penalize when coverage exceeds 1.0
        penalty = torch.clamp(coverage - 1.0, min=0.0)

        # Apply sequence mask to ignore padding
        if sequence_mask is not None:
            penalty = penalty * sequence_mask.float()
            return penalty.sum() / sequence_mask.sum()
        else:
            return penalty.mean()

    def calculate_loss(
        self,
        sequential_logits: torch.Tensor,
        parallel_logits: torch.Tensor,
        target_sequence: torch.Tensor,
        target_labels: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
        sequence_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.

        Args:
            sequential_logits: Sequential prediction logits [batch_size, seq_len, vocab_size]
            parallel_logits: Parallel prediction logits [batch_size, num_labels]
            target_sequence: Target sequence [batch_size, seq_len]
            target_labels: Target multi-hot labels [batch_size, num_labels]
            attention_weights: Attention weights for coverage loss
            sequence_mask: Sequence mask for padding

        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}

        # 1. Parallel BCE Loss
        if self.use_focal_loss:
            # Apply focal loss to parallel predictions
            bce_loss = F.binary_cross_entropy_with_logits(
                parallel_logits, target_labels, reduction='none'
            )
            pt = torch.sigmoid(parallel_logits)
            pt = torch.where(target_labels == 1, pt, 1 - pt)
            focal_weight = self.focal_alpha * (1 - pt) ** self.focal_gamma
            bce_loss = (focal_weight * bce_loss).mean()
        else:
            bce_loss = F.binary_cross_entropy_with_logits(
                parallel_logits, target_labels
            )
        losses["bce_loss"] = bce_loss

        # 2. Sequential Cross-Entropy Loss
        batch_size, seq_len, vocab_size = sequential_logits.shape
        sequential_logits_flat = sequential_logits.view(-1, vocab_size)
        target_sequence_flat = target_sequence.view(-1)

        if self.use_focal_loss:
            sequence_loss = self.focal_loss(
                sequential_logits_flat, 
                target_sequence_flat,
                self.focal_alpha,
                self.focal_gamma
            )
        else:
            sequence_loss = F.cross_entropy(
                sequential_logits_flat, 
                target_sequence_flat,
                ignore_index=-100  # Ignore padding tokens if using -100
            )
        losses["sequence_loss"] = sequence_loss

        # 3. Coverage Loss
        if attention_weights is not None and self.coverage_weight > 0:
            coverage_loss = self.coverage_loss(attention_weights, sequence_mask)
            losses["coverage_loss"] = coverage_loss
        else:
            losses["coverage_loss"] = torch.tensor(0.0, device=parallel_logits.device)

        # 4. Total Loss
        total_loss = (
            self.bce_weight * losses["bce_loss"] +
            self.sequence_weight * losses["sequence_loss"] +
            self.coverage_weight * losses["coverage_loss"]
        )
        losses["total_loss"] = total_loss

        return losses


class TeacherForcingScheduler:
    """
    Scheduler for teacher forcing ratio during training.
    """

    def __init__(
        self,
        start_ratio: float = 1.0,
        end_ratio: float = 0.7,
        total_steps: int = 10000,
        schedule_type: str = "linear"
    ):
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.total_steps = total_steps
        self.schedule_type = schedule_type

        logger.info("Initialized teacher forcing scheduler: %f -> %f", start_ratio, end_ratio)

    def get_ratio(self, current_step: int) -> float:
        """
        Get teacher forcing ratio for current step.

        Args:
            current_step: Current training step

        Returns:
            Teacher forcing ratio
        """
        if current_step >= self.total_steps:
            return self.end_ratio

        progress = current_step / self.total_steps

        if self.schedule_type == "linear":
            ratio = self.start_ratio - (self.start_ratio - self.end_ratio) * progress
        elif self.schedule_type == "exponential":
            ratio = self.end_ratio + (self.start_ratio - self.end_ratio) * (0.5 ** progress)
        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        return ratio


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info("Set random seed to %d", seed)


def load_label_mapping(mapping_path: str) -> Dict[str, int]:
    """
    Load label mapping from JSON file.

    Args:
        mapping_path: Path to label mapping file

    Returns:
        Dictionary mapping label names to IDs
    """
    try:
        with open(mapping_path, 'r', encoding='utf-8') as f:
            mapping = json.load(f)
        logger.info("Loaded label mapping with %d labels from %s", len(mapping), mapping_path)
        return mapping
    except Exception as e:
        logger.error("Failed to load label mapping from %s: %s", mapping_path, e)
        raise


def save_predictions(
    predictions: Dict[str, Any],
    output_path: str
):
    """
    Save model predictions to file.

    Args:
        predictions: Dictionary containing predictions and metadata
        output_path: Output file path
    """
    try:
        # Convert tensors to lists for JSON serialization
        serializable_predictions = {}
        for key, value in predictions.items():
            if isinstance(value, torch.Tensor):
                serializable_predictions[key] = value.cpu().numpy().tolist()
            else:
                serializable_predictions[key] = value

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_predictions, f, indent=2, ensure_ascii=False)

        logger.info("Saved predictions to %s", output_path)
    except Exception as e:
        logger.error("Failed to save predictions to %s: %s", output_path, e)
        raise


def create_label_frequency_stats(
    dataset_path: str
) -> Dict[str, int]:
    """
    Calculate label frequency statistics from dataset.

    Args:
        dataset_path: Path to dataset

    Returns:
        Dictionary with label frequencies
    """
    # This is a placeholder - implement based on your dataset format
    frequencies = defaultdict(int)

    # Example implementation - adapt to your data format
    # for sample in dataset:
    #     for label in sample.labels:
    #         frequencies[label] += 1

    # 如果可用，直接使用vocabulary中的统计信息
    if hasattr(vocabulary, '_token_freqs'):
        for token, freq in vocabulary._token_freqs:
            frequencies[token] = freq

    logger.info("Calculated label frequencies for %d labels", len(frequencies))
    return dict(frequencies)
