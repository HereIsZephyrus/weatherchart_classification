"""
CNN-RNN Unified Framework Model Implementation.

Architecture:
Image → CNN Encoder → Feature Projection → Joint Embedding Space ← Label Embedding
                                     ↓
              RNN Sequence Decoder → {Sequential Head, Parallel Head} → Element Prediction
"""
import logging
from typing import Tuple, Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from torchvision.models import resnet50, ResNet50_Weights
from .config import ModelConfig, UnifiedConfig, CNNconfig, RNNconfig
from .vocab import vocabulary

logger = logging.getLogger(__name__)
__all__ = ["WeatherChartModel"]

class CNNEncoder(nn.Module):
    """
    CNN encoder using ResNet-50 with ImageNet pretraining.
    Outputs global features through Global Average Pooling.
    """

    def __init__(self, config: CNNconfig):
        super().__init__()
        self.config = config

        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.dropout = nn.Dropout(config.cnn_dropout)

        # Feature projection to joint embedding space
        self.feature_projection = nn.Linear(
            config.cnn_feature_dim,
            config.cnn_output_dim
        )

        logger.info("Initialized CNN encoder with %s", config.cnn_backbone)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through CNN encoder.

        Args:
            images: Input images [batch_size, 3, height, width]

        Returns:
            Projected features [batch_size, joint_embedding_dim]
        """
        # Extract features through ResNet backbone
        features = self.backbone(images)  # [batch_size, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, 2048]

        # Apply dropout and projection
        features = self.dropout(features)
        projected_features = self.feature_projection(features)

        return projected_features


class LabelEmbedding(nn.Module):
    """
    Learnable label embedding matrix for weather elements.
    """

    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(len(vocabulary), self.config.joint_embedding_dim)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        logger.info("Initialized label embedding")

    def forward(self, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through label embedding.

        Args:
            label_ids: Label indices [batch_size, sequence_length]

        Returns:
            Label embeddings [batch_size, sequence_length, label_embedding_dim]
        """
        return self.embedding(label_ids)


class RNNDecoder(nn.Module):
    """
    RNN sequence decoder with dual-layer LSTM architecture.
    Implements joint feature fusion and attention mechanism.
    """

    def __init__(self, config: RNNconfig, cnn_feature_dim: int):
        super().__init__()
        self.config = config

        # RNN layers
        self.rnn = nn.LSTM(
            input_size=config.rnn_input_dim,
            hidden_size=config.rnn_hidden_dim,
            num_layers=config.rnn_num_layers,
            dropout=config.rnn_dropout if config.rnn_num_layers > 1 else 0,
            bidirectional=config.rnn_bidirectional,
            batch_first=True
        )

        # Joint feature fusion layers
        self.feature_fusion = nn.Linear(
            config.rnn_hidden_dim + config.rnn_input_dim,
            cnn_feature_dim
        )

        # Attention mechanism for coverage
        self.attention = nn.MultiheadAttention(
            embed_dim=config.rnn_input_dim,
            num_heads=8,
            dropout=config.rnn_dropout,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(config.rnn_input_dim)
        self.dropout = nn.Dropout(config.rnn_dropout)

        logger.info("Initialized RNN decoder")

    def forward(
        self,
        label_embeddings: torch.Tensor,
        image_features: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Forward pass through RNN decoder.

        Args:
            label_embeddings: Input label embeddings [batch_size, seq_len, embed_dim]
            image_features: CNN image features [batch_size, joint_embedding_dim]
            hidden_state: Initial hidden state for RNN
            attention_mask: Attention mask for sequence [batch_size, seq_len]

        Returns:
            fused_features: Joint features [batch_size, seq_len, joint_embedding_dim]
            hidden_state: Final RNN hidden state
            attention_weights: Attention weights for coverage loss
        """
        batch_size, seq_len = label_embeddings.shape[:2]

        # RNN forward pass
        rnn_output, hidden_state = self.rnn(label_embeddings, hidden_state)
        # rnn_output: [batch_size, seq_len, rnn_hidden_dim]

        # Expand image features to match sequence length
        image_features_expanded = image_features.unsqueeze(1).expand(
            batch_size, seq_len, -1
        )  # [batch_size, seq_len, joint_embedding_dim]

        # Concatenate RNN output with image features
        concat_features = torch.cat([rnn_output, image_features_expanded], dim=-1)
        # concat_features: [batch_size, seq_len, rnn_hidden_dim + joint_embedding_dim]

        # Joint feature fusion
        fused_features = torch.tanh(self.feature_fusion(concat_features))
        # fused_features: [batch_size, seq_len, joint_embedding_dim]

        # Apply attention mechanism
        attended_features, attention_weights = self.attention(
            fused_features, fused_features, fused_features,
            key_padding_mask=attention_mask
        )

        # Residual connection and layer normalization
        fused_features = self.layer_norm(fused_features + self.dropout(attended_features))

        return fused_features, hidden_state, attention_weights


class DualPredictionHead(nn.Module):
    """
    Dual prediction mechanism with sequential and parallel heads.

    - Sequential head: Models conditional probability P(l_t | I, l_{<t})
    - Parallel BCE head: Direct multi-hot vector prediction
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Sequential prediction head
        self.sequential_head = nn.Linear(
            config.unified_config.joint_embedding_dim,
            len(vocabulary)
        )

        # Parallel BCE prediction head
        self.parallel_head = nn.Linear(
            config.unified_config.joint_embedding_dim,
            len(vocabulary)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(config.rnn_config.rnn_dropout)

        logger.info("Initialized dual prediction heads")

    def forward(
        self,
        fused_features: torch.Tensor,
        pooled_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through dual prediction heads.

        Args:
            fused_features: Joint features [batch_size, seq_len, joint_embedding_dim]
            pooled_features: Pooled features for parallel head [batch_size, joint_embedding_dim]

        Returns:
            sequential_logits: Sequential prediction logits [batch_size, seq_len, vocab_size]
            parallel_logits: Parallel prediction logits [batch_size, num_labels]
        """
        # Sequential prediction
        sequential_logits = self.sequential_head(self.dropout(fused_features))

        # Parallel prediction using mean pooling if pooled_features not provided
        if pooled_features is None:
            pooled_features = fused_features.mean(dim=1)  # [batch_size, joint_embedding_dim]

        parallel_logits = self.parallel_head(self.dropout(pooled_features))

        return sequential_logits, parallel_logits


class WeatherChartModel(PreTrainedModel):
    """
    CNN-RNN Unified Framework for Weather Chart Multi-label Classification.

    This model implements the architecture described in docs/train.md section 3.2,
    combining CNN feature extraction with RNN sequence modeling for weather element
    prediction with label dependency modeling.
    """

    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        # Initialize model components
        self.cnn_encoder = CNNEncoder(config.cnn_config)
        self.label_embedding = LabelEmbedding(config.unified_config)
        self.rnn_decoder = RNNDecoder(config.rnn_config, config.cnn_config.cnn_feature_dim)
        self.prediction_head = DualPredictionHead(config)

        # Initialize weights
        self.init_weights()

        logger.info("Initialized WeatherChartModel with CNN-RNN unified framework")

    def get_position_embeddings(self):
        """
        Return None since this model doesn't use position embeddings.
        Required by PreTrainedModel abstract method.
        """
        return None

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        No-op since this model doesn't use position embeddings.
        Required by PreTrainedModel abstract method.
        """
        pass

    def init_weights(self):
        """Initialize model weights."""
        # CNN encoder uses pretrained weights, only initialize projection layer
        nn.init.xavier_uniform_(self.cnn_encoder.feature_projection.weight)

        # Initialize RNN decoder
        for name, param in self.rnn_decoder.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize prediction heads
        nn.init.xavier_uniform_(self.prediction_head.sequential_head.weight)
        nn.init.xavier_uniform_(self.prediction_head.parallel_head.weight)
        nn.init.zeros_(self.prediction_head.sequential_head.bias)
        nn.init.zeros_(self.prediction_head.parallel_head.bias)

    def forward(
        self,
        images: torch.Tensor,
        input_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the unified framework.

        Args:
            images: Input images [batch_size, 3, height, width]
            input_labels: Label sequences for teacher forcing [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_dict: Whether to return ModelOutput or dict

        Returns:
            Dictionary containing:
            - sequential_logits: Sequential prediction logits
            - parallel_logits: Parallel prediction logits  
            - attention_weights: Attention weights for coverage loss
            - image_features: Extracted image features
        """
        # Extract image features through CNN
        image_features = self.cnn_encoder(images)
        # image_features: [batch_size, joint_embedding_dim]

        outputs = {"image_features": image_features}

        # If input_labels provided (training mode)
        if input_labels is not None:
            # Get label embeddings
            label_embeddings = self.label_embedding(input_labels)
            # label_embeddings: [batch_size, seq_len, label_embedding_dim]

            # RNN decoder forward pass
            fused_features, hidden_state, attention_weights = self.rnn_decoder(
                label_embeddings=label_embeddings,
                image_features=image_features,
                attention_mask=attention_mask
            )

            # Dual prediction
            sequential_logits, parallel_logits = self.prediction_head(
                fused_features=fused_features,
                pooled_features=image_features
            )

            outputs.update({
                "sequential_logits": sequential_logits,
                "parallel_logits": parallel_logits,
                "attention_weights": attention_weights,
                "hidden_state": hidden_state,
            })

        return outputs

    def generate(
        self,
        images: torch.Tensor,
        early_stopping: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Generate label sequences using beam search decoding.

        Args:
            images: Input images [batch_size, 3, height, width]
            early_stopping: Whether to use early stopping

        Returns:
            Dictionary containing generated sequences and scores
        """

        batch_size = images.shape[0]
        device = images.device

        # Extract image features
        image_features = self.cnn_encoder(images)

        # Initialize beam search
        
        beam_sequences = torch.full(
            (batch_size, self.config.beam_width, 1),
            vocabulary.bos,
            dtype=torch.long,
            device=device
        )
        beam_scores = torch.zeros(batch_size, self.config.beam_width, device=device)
        beam_scores[:, 1:] = float('-inf')  # Only first beam is active initially

        # Track completed sequences
        completed_sequences = []
        completed_scores = []

        # Add safety check to prevent infinite loop
        max_iterations = min(vocabulary.max_sequence_length - 1, 50)  # Safety limit
        for step in range(max_iterations):
            # Current sequence length
            current_length = beam_sequences.shape[-1]

            # Reshape for batch processing
            current_sequences = beam_sequences.view(batch_size * self.config.beam_width, current_length)
            expanded_image_features = image_features.unsqueeze(1).expand(
                batch_size, self.config.beam_width, -1
            ).contiguous().view(batch_size * self.config.beam_width, -1)

            # Get label embeddings for current sequences
            label_embeddings = self.label_embedding(current_sequences)

            # RNN decoder forward pass
            fused_features, _, _ = self.rnn_decoder(
                label_embeddings=label_embeddings,
                image_features=expanded_image_features
            )

            # Get sequential logits for next token prediction
            sequential_logits, _ = self.prediction_head(fused_features)
            next_token_logits = sequential_logits[:, -1, :]  # Last time step

            # Reshape back to beam format
            next_token_logits = next_token_logits.view(
                batch_size, self.config.beam_width, -1
            )

            # Calculate scores for all possible next tokens
            vocab_size = next_token_logits.shape[-1]
            next_token_scores = F.log_softmax(next_token_logits, dim=-1)

            # Add to beam scores
            candidate_scores = beam_scores.unsqueeze(-1) + next_token_scores
            candidate_scores = candidate_scores.view(batch_size, -1)

            # Select top beam_width candidates
            top_scores, top_indices = torch.topk(
                candidate_scores, self.config.beam_width, dim=-1
            )

            # Convert flat indices back to beam and token indices
            beam_indices = top_indices // vocab_size
            token_indices = top_indices % vocab_size

            # Update beam sequences
            new_sequences = []
            new_scores = []

            for batch_idx in range(batch_size):
                batch_sequences = []
                batch_scores = []

                for beam_idx in range(self.config.beam_width):
                    # Get the parent beam and next token
                    parent_beam = beam_indices[batch_idx, beam_idx]
                    next_token = token_indices[batch_idx, beam_idx]
                    score = top_scores[batch_idx, beam_idx]

                    # Get parent sequence
                    parent_sequence = beam_sequences[batch_idx, parent_beam]

                    # Create new sequence
                    new_sequence = torch.cat([
                        parent_sequence,
                        next_token.unsqueeze(0)
                    ])

                    # Check if sequence is completed (EOS token)
                    if next_token == vocabulary.eos:
                        if batch_idx >= len(completed_sequences):
                            completed_sequences.extend([[] for _ in range(batch_idx + 1 - len(completed_sequences))])
                            completed_scores.extend([[] for _ in range(batch_idx + 1 - len(completed_scores))])

                        completed_sequences[batch_idx].append(new_sequence)
                        completed_scores[batch_idx].append(score)
                    else:
                        batch_sequences.append(new_sequence)
                        batch_scores.append(score)

                # Pad to beam_width if needed
                while len(batch_sequences) < self.config.beam_width:
                    batch_sequences.append(beam_sequences[batch_idx, 0])  # Duplicate first beam
                    batch_scores.append(float('-inf'))

                new_sequences.append(torch.stack(batch_sequences[:self.config.beam_width]))
                new_scores.append(torch.stack(batch_scores[:self.config.beam_width]))

            beam_sequences = torch.stack(new_sequences)
            beam_scores = torch.stack(new_scores)

            # Early stopping check
            if early_stopping:
                all_completed = True
                for batch_idx in range(batch_size):
                    if batch_idx < len(completed_sequences):
                        if len(completed_sequences[batch_idx]) == 0:
                            all_completed = False
                            break
                        # Check if best completed score is better than best active score
                        best_completed = max(completed_scores[batch_idx])
                        best_active = beam_scores[batch_idx].max()
                        if best_active > best_completed:
                            all_completed = False
                            break
                    else:
                        all_completed = False
                        break

                if all_completed:
                    break

        # Collect final results
        final_sequences = []
        final_scores = []

        for batch_idx in range(batch_size):
            if batch_idx < len(completed_sequences) and completed_sequences[batch_idx]:
                # Use best completed sequence
                best_idx = torch.tensor(completed_scores[batch_idx]).argmax()
                final_sequences.append(completed_sequences[batch_idx][best_idx])
                final_scores.append(completed_scores[batch_idx][best_idx])
            else:
                # Use best active sequence
                best_idx = beam_scores[batch_idx].argmax()
                final_sequences.append(beam_sequences[batch_idx, best_idx])
                final_scores.append(beam_scores[batch_idx, best_idx])

        # Pad sequences to same length
        max_seq_length = max(seq.shape[0] for seq in final_sequences)
        padded_sequences = torch.full(
            (batch_size, max_seq_length),
            vocabulary.unk,
            dtype=torch.long,
            device=device
        )

        for i, seq in enumerate(final_sequences):
            padded_sequences[i, :len(seq)] = seq

        return {
            "sequences": padded_sequences,
            "scores": torch.stack(final_scores)
        }

    def freeze_cnn(self):
        """Freeze CNN encoder parameters for warmup training."""
        for param in self.cnn_encoder.parameters():
            param.requires_grad = False
        logger.info("Frozen CNN encoder parameters")

    def unfreeze_cnn(self):
        """Unfreeze CNN encoder parameters for end-to-end training."""
        for param in self.cnn_encoder.parameters():
            param.requires_grad = True
        logger.info("Unfrozen CNN encoder parameters")
