"""
CNN-RNN Unified Framework Model Implementation.

Architecture:
Image → CNN Encoder → Feature Projection → Joint Embedding Space ← Label Embedding
                                     ↓
              RNN Sequence Decoder → Sequential Head → Element Prediction
"""
import logging
from typing import Optional, Dict
import torch
from torch import nn
import torch.nn.functional as F
from torch.ao.quantization import QuantStub, DeQuantStub
from transformers import PreTrainedModel
from torchvision.models import resnet50, ResNet50_Weights
from .config import ModelConfig, CNNconfig, RNNconfig
from .vocab import vocabulary

logger = logging.getLogger(__name__)
__all__ = ["WeatherChartModel"]

class CNNEncoder(nn.Module):
    """
    CNN encoder using ResNet-50 with ImageNet pretraining.
    Outputs global features through Global Average Pooling.
    Supports 8-bit quantization for reduced memory and faster inference.
    """

    def __init__(self, config: CNNconfig):
        super().__init__()
        self.config = config

        # Initialize ResNet backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

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
        # Quantize input
        x = self.quant(images)

        # Extract features through ResNet backbone
        features = self.backbone(x)  # [batch_size, 2048, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [batch_size, 2048]

        # Dequantize before dropout and projection
        features = self.dequant(features)

        # Apply dropout and projection
        features = self.dropout(features)
        projected_features = self.feature_projection(features)

        return projected_features

    def prepare_for_quantization(self):
        """
        Prepare the ResNet50 backbone for quantization while preserving pretrained weights.
        """
        # Use qnnpack backend for better mobile/CPU performance
        backend = 'qnnpack' if torch.backends.quantized.engine == 'qnnpack' else 'fbgemm'
        qconfig = torch.quantization.get_default_qconfig(backend)

        # Set qconfig for the entire backbone
        self.backbone.qconfig = qconfig

        # Manually fuse layers in ResNet50 for better quantization
        # ResNet50 structure: conv1, bn1, relu -> layer1,2,3,4 -> avgpool
        self._fuse_resnet_layers()

        # Set qconfig for feature projection
        self.feature_projection.qconfig = qconfig

    def _fuse_resnet_layers(self):
        """
        Fuse Conv-BN-ReLU layers in ResNet50 for quantization.
        This preserves the pretrained weights while enabling quantization.
        """
        # Get the original ResNet model from backbone
        if hasattr(self.backbone, '0'):  # Sequential wrapper
            resnet = self.backbone[0] if len(self.backbone) == 1 else self.backbone
        else:
            resnet = self.backbone

        # Fuse initial conv-bn-relu block
        if hasattr(resnet, 'conv1') and hasattr(resnet, 'bn1'):
            torch.quantization.fuse_modules(
                resnet,
                [['conv1', 'bn1', 'relu']],
                inplace=True
            )

        # Fuse layers in each residual block
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            if hasattr(resnet, layer_name):
                layer = getattr(resnet, layer_name)
                self._fuse_residual_blocks(layer)

    def _fuse_residual_blocks(self, layer):
        """
        Fuse Conv-BN-ReLU in residual blocks.
        """
        for i, block in enumerate(layer):
            # BasicBlock or Bottleneck block
            if hasattr(block, 'conv1') and hasattr(block, 'bn1'):
                # Fuse conv1-bn1-relu
                torch.quantization.fuse_modules(
                    block,
                    [['conv1', 'bn1', 'relu']],
                    inplace=True
                )

            if hasattr(block, 'conv2') and hasattr(block, 'bn2'):
                # For Bottleneck: fuse conv2-bn2-relu
                if hasattr(block, 'relu'):
                    torch.quantization.fuse_modules(
                        block,
                        [['conv2', 'bn2', 'relu']],
                        inplace=True
                    )
                else:
                    # For BasicBlock: fuse conv2-bn2 (no relu after conv2)
                    torch.quantization.fuse_modules(
                        block,
                        [['conv2', 'bn2']],
                        inplace=True
                    )

            # For Bottleneck blocks, also fuse conv3-bn3
            if hasattr(block, 'conv3') and hasattr(block, 'bn3'):
                torch.quantization.fuse_modules(
                    block,
                    [['conv3', 'bn3']],
                    inplace=True
                )

            # Fuse downsample layers if present
            if hasattr(block, 'downsample') and block.downsample is not None:
                if len(block.downsample) >= 2:
                    torch.quantization.fuse_modules(
                        block.downsample,
                        [['0', '1']],  # conv, bn
                        inplace=True
                    )

# LabelEmbedding class removed - now handled inside RNNDecoder
class RNNDecoder(nn.Module):
    def __init__(self, config: RNNconfig, img_feat_dim: int):
        """
        Initialize RNN decoder with quantization support.

        Args:
            config: RNN configuration parameters
            img_feat_dim: Dimension of image features from CNN encoder
        """
        super(RNNDecoder, self).__init__()

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Label embedding matrix U_l (Eq. 2): Convert one-hot labels to low-dim embeddings
        self.label_embedding = nn.Embedding(
            num_embeddings=len(vocabulary),
            embedding_dim=config.rnn_hidden_dim
        )

        # Forget gate f_t: f_t = δ(U_f_r · r(t-1) + U_f_w · w_k(t))
        self.f_gate_r = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_f_r
        self.f_gate_w = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_f_w

        # Input gate i_t: i_t = δ(U_i_r · r(t-1) + U_i_w · w_k(t))
        self.i_gate_r = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_i_r
        self.i_gate_w = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_i_w

        # Candidate state x_t: x_t = δ(U_r · r(t-1) + U_w · w_k(t))
        self.candidate_r = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_r
        self.candidate_w = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_w

        # Output gate o_t: o_t = δ(U_o_r · r(t-1) + U_o_w · w_k(t))
        self.o_gate_r = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_o_r
        self.o_gate_w = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_o_w

        # Projection matrices (Eq. 4): Project RNN output o(t) and image features I to embedding space
        self.proj_o = nn.Linear(config.rnn_hidden_dim, config.rnn_hidden_dim)  # U_o^x
        self.proj_img = nn.Linear(img_feat_dim, config.rnn_hidden_dim)  # U_I^x

        # Activation function: ReLU (as specified in Section 3.1)
        self.activation = nn.ReLU()

    def forward(self, prev_label_idx, prev_hidden, img_feat):
        """
        Forward pass for a single time step with quantization support.

        Args:
            prev_label_idx: Previous predicted label indices [batch_size]
            prev_hidden: Previous hidden state r(t-1) [batch_size, hidden_dim]
            img_feat: Image features I [batch_size, img_feat_dim]

        Returns:
            current_score: Current label scores s(t) [batch_size, num_labels]
            current_hidden: Current hidden state r(t) [batch_size, hidden_dim]
        """
        # Quantize inputs
        prev_hidden = self.quant(prev_hidden)
        img_feat = self.quant(img_feat)

        # 1. Compute label embedding w_k(t) (Eq. 2)
        w_k = self.label_embedding(prev_label_idx)  # [batch_size, embed_dim]
        w_k = self.quant(w_k)

        # 2. LSTM gate computations (Eq. 3.1)
        # Forget gate
        f_t = self.activation(self.f_gate_r(prev_hidden) + self.f_gate_w(w_k))  # [batch_size, hidden_dim]
        # Input gate
        i_t = self.activation(self.i_gate_r(prev_hidden) + self.i_gate_w(w_k))  # [batch_size, hidden_dim]
        # Candidate state
        x_t = self.activation(self.candidate_r(prev_hidden) + self.candidate_w(w_k))  # [batch_size, hidden_dim]
        # Update hidden state r(t)
        current_hidden = f_t * prev_hidden + i_t * x_t  # [batch_size, hidden_dim]
        # Output gate
        o_t = self.activation(self.o_gate_r(current_hidden) + self.o_gate_w(w_k))  # [batch_size, hidden_dim]
        # RNN layer output o(t)
        o_t = current_hidden * o_t  # [batch_size, hidden_dim]

        # 3. Joint projection of image features and RNN output (Eq. 4)
        proj_o = self.proj_o(o_t)  # [batch_size, embed_dim]
        proj_img = self.proj_img(img_feat)  # [batch_size, embed_dim]
        x_t_proj = self.activation(proj_o + proj_img)  # [batch_size, embed_dim]

        # 4. Compute label scores s(t) (Eq. 5): s(t) = U_l^T · x_t_proj
        # Note: label_embedding.weight is U_l, transposed for dot product
        current_score = torch.matmul(x_t_proj, self.label_embedding.weight.t())  # (batch_size, num_labels)

        # Dequantize outputs
        current_score = self.dequant(current_score)
        current_hidden = self.dequant(current_hidden)

        return current_score, current_hidden

    def prepare_for_quantization(self):
        """
        Prepare the model for quantization by setting qconfig for all linear layers.
        """
        qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # Set qconfig for all linear layers
        self.f_gate_r.qconfig = qconfig
        self.f_gate_w.qconfig = qconfig
        self.i_gate_r.qconfig = qconfig
        self.i_gate_w.qconfig = qconfig
        self.candidate_r.qconfig = qconfig
        self.candidate_w.qconfig = qconfig
        self.o_gate_r.qconfig = qconfig
        self.o_gate_w.qconfig = qconfig
        self.proj_o.qconfig = qconfig
        self.proj_img.qconfig = qconfig
        self.label_embedding.qconfig = qconfig

    def compute_loss(self, score, target_label):
        return F.cross_entropy(score, target_label)

class ParallelPredictionHead(nn.Module):
    """
    Parallel prediction head for direct multi-hot vector prediction.
    Sequential prediction is now handled directly by RNNDecoder.
    Supports 8-bit quantization for reduced memory and faster inference.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Quantization stubs
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

        # Parallel BCE prediction head
        self.parallel_head = nn.Linear(
            config.cnn_config.cnn_output_dim,  # Use CNN output dimension
            len(vocabulary)
        )

        # Dropout for regularization
        self.dropout = nn.Dropout(config.rnn_config.rnn_dropout)

        logger.info("Initialized parallel prediction head with quantization support")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through parallel prediction head with quantization support.

        Args:
            image_features: Image features [batch_size, joint_embedding_dim]

        Returns:
            parallel_logits: Parallel prediction logits [batch_size, vocab_size]
        """
        # Quantize input
        x = self.quant(image_features)

        # Apply dropout and prediction head
        x = self.dropout(x)
        parallel_logits = self.parallel_head(x)

        # Dequantize output
        parallel_logits = self.dequant(parallel_logits)

        return parallel_logits

    def prepare_for_quantization(self):
        """
        Prepare the model for quantization by setting qconfig.
        """
        # Set qconfig for linear layer
        self.parallel_head.qconfig = torch.quantization.get_default_qconfig('fbgemm')


class WeatherChartModel(PreTrainedModel):
    """
    CNN-RNN Unified Framework for Weather Chart Multi-label Classification.

    This model implements the architecture described in docs/train.md section 3.2,
    combining CNN feature extraction with RNN sequence modeling for weather element
    prediction with label dependency modeling.

    Supports 8-bit quantization for reduced memory and faster inference.
    """

    config_class = ModelConfig

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        # Initialize model components
        self.cnn_encoder = CNNEncoder(config.cnn_config)
        self.rnn_decoder = RNNDecoder(config.rnn_config, config.cnn_config.cnn_output_dim)
        self.prediction_head = ParallelPredictionHead(config)

        # Initialize weights
        self.init_weights()

        # Note: Quantization should be done after training/loading pretrained weights
        # Don't quantize during initialization to preserve pretrained weights
        self._quantization_prepared = False

        logger.info("Initialized WeatherChartModel with CNN-RNN unified framework and quantization support")

    def get_position_embeddings(self):
        """
        Return None since this model doesn't use position embeddings.
        Required by PreTrainedModel abstract method.
        """
        return None

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

        # Initialize parallel prediction head
        nn.init.xavier_uniform_(self.prediction_head.parallel_head.weight)
        nn.init.zeros_(self.prediction_head.parallel_head.bias)

    def forward(
        self,
        images: torch.Tensor,
        input_labels: Optional[torch.Tensor] = None,
        use_teacher_forcing: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the CNN-RNN unified framework.

        Args:
            images: Input images [batch_size, 3, height, width]
            input_labels: Label sequences for teacher forcing [batch_size, seq_len]
            use_teacher_forcing: Whether to use teacher forcing

        Returns:
            Dictionary containing:
            - sequential_logits: Sequential prediction logits [batch_size, seq_len-1, vocab_size]
            - parallel_logits: Parallel prediction logits [batch_size, vocab_size]  
            - image_features: Extracted image features [batch_size, joint_embedding_dim]
            - hidden_state: Final hidden state [batch_size, hidden_dim]
        """
        # Extract image features through CNN
        image_features = self.cnn_encoder(images)
        # image_features: [batch_size, joint_embedding_dim]

        outputs = {"image_features": image_features}
        if input_labels is not None:
            batch_size, seq_len = input_labels.shape
        else:
            batch_size = images.shape[0]
            seq_len = vocabulary.max_sequence_length
            use_teacher_forcing = False
        device = images.device
        # Initialize hidden state (batch_size, hidden_dim)
        hidden_state = torch.zeros(batch_size, self.config.rnn_config.rnn_hidden_dim, device=device)
        # For parallel prediction, use image features
        parallel_logits = self.prediction_head(image_features)

        sequential_logits_list = []
        # If input_labels provided (training mode with teacher forcing)
        for t in range(seq_len - 1):  # -1 because we predict next token
            if use_teacher_forcing:
                # Current input label index
                current_label_idx = input_labels[:, t]  # [batch_size]
            else:
                # Use previous predicted label as input
                if t == 0:
                    current_label_idx = torch.tensor([vocabulary.bos]*batch_size, device=device)
                else:
                    # Get previous logits and argmax to get predicted label
                    prev_logit = sequential_logits_list[t-1]  # Previous logits
                    current_label_idx = torch.argmax(prev_logit, dim=-1)  # [batch_size]

            # RNN decoder forward for current step
            current_score, hidden_state = self.rnn_decoder(
                prev_label_idx=current_label_idx,
                prev_hidden=hidden_state,
                img_feat=image_features
            )
            # current_score: [batch_size, vocab_size]
            # hidden_state: [batch_size, hidden_dim]

            sequential_logits_list.append(current_score)

        # Stack logits: [batch_size, seq_len-1, vocab_size]
        sequential_logits = torch.stack(sequential_logits_list, dim=1)
        outputs.update({
            "sequential_logits": sequential_logits,
            "parallel_logits": parallel_logits,
            "hidden_state": hidden_state,
        })
        return outputs

    def generate(
        self,
        images: torch.Tensor,
        early_stopping: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate label sequences using beam search with the new RNN decoder.

        Args:
            images: Input images [batch_size, 3, height, width]
            early_stopping: Whether to stop when all beams complete

        Returns:
            Dictionary containing generated sequences and scores
        """
        batch_size = images.shape[0]
        device = images.device
        max_len = vocabulary.max_sequence_length
        beam_width = self.config.unified_config.beam_width

        # Extract image features (shared across all beams)
        image_features = self.cnn_encoder(images)  # [batch_size, joint_embedding_dim]

        # Initialize beam search
        # Start with BOS token for all beams
        beam_sequences = torch.full(
            (batch_size, beam_width, 1),
            vocabulary.bos,
            dtype=torch.long,
            device=device
        )  # [B, K, 1]

        # Initialize hidden states for each beam
        beam_hiddens = torch.zeros(
            batch_size, beam_width, self.config.rnn_config.rnn_hidden_dim,
            device=device
        )  # [B, K, hidden_dim]

        # Initialize scores: first beam has 0, others -inf
        beam_scores = torch.full((batch_size, beam_width), -float('inf'), device=device)
        beam_scores[:, 0] = 0.0  # Only first beam is active initially

        # Track completion status
        beam_completed = torch.zeros((batch_size, beam_width), dtype=torch.bool, device=device)

        for step in range(max_len - 1):  # -1 because we start with BOS
            # Get current input tokens (last token in each sequence)
            current_tokens = beam_sequences[:, :, -1]  # [B, K]

            # Flatten for batch processing
            flat_tokens = current_tokens.view(batch_size * beam_width)  # [B*K]
            flat_hiddens = beam_hiddens.view(batch_size * beam_width, -1)  # [B*K, hidden_dim]
            flat_image_feats = image_features.unsqueeze(1).expand(
                batch_size, beam_width, -1
            ).contiguous().view(batch_size * beam_width, -1)  # [B*K, joint_dim]

            # RNN forward pass for all beams
            next_scores, new_hiddens = self.rnn_decoder(
                prev_label_idx=flat_tokens,
                prev_hidden=flat_hiddens,
                img_feat=flat_image_feats
            )
            # next_scores: [B*K, vocab_size]
            # new_hiddens: [B*K, hidden_dim]

            # Reshape back to beam structure
            next_scores = next_scores.view(batch_size, beam_width, -1)  # [B, K, vocab_size]
            new_hiddens = new_hiddens.view(batch_size, beam_width, -1)  # [B, K, hidden_dim]

            # Convert to log probabilities
            next_log_probs = F.log_softmax(next_scores, dim=-1)  # [B, K, vocab_size]

            # Calculate candidate scores: add current beam score to next token probabilities
            candidate_scores = beam_scores.unsqueeze(-1) + next_log_probs

            # Mask completed beams (their scores remain -inf)
            candidate_scores = torch.where(
                beam_completed.unsqueeze(-1),
                -float('inf'),
                candidate_scores
            )  # [B, K, vocab_size]

            # Flatten candidates to select top-K
            flat_candidates = candidate_scores.view(batch_size, -1)  # [B, K*vocab_size]
            top_scores, top_indices = torch.topk(flat_candidates, beam_width, dim=-1)  # [B, K]

            # Convert flat indices to beam and token indices
            beam_indices = top_indices // len(vocabulary)  # [B, K]
            token_indices = top_indices % len(vocabulary)   # [B, K]

            # Update sequences: append new tokens to selected beams
            gather_indices = beam_indices.unsqueeze(-1).expand(-1, -1, step + 1)  # [B, K, seq_len]
            previous_sequences = torch.gather(beam_sequences, 1, gather_indices)  # [B, K, seq_len]
            new_sequences = torch.cat([
                previous_sequences,
                token_indices.unsqueeze(-1)  # [B, K, 1]
            ], dim=-1)  # [B, K, seq_len+1]

            # Update hidden states: gather from previous hidden states using beam_indices
            gather_hidden_indices = beam_indices.unsqueeze(-1).expand(
                -1, -1, self.config.rnn_config.rnn_hidden_dim
            )  # [B, K, hidden_dim]
            new_beam_hiddens = torch.gather(new_hiddens, 1, gather_hidden_indices)  # [B, K, hidden_dim]

            # Update completion status
            new_completed = beam_completed.gather(1, beam_indices) | (token_indices == vocabulary.eos)

            # Update beam variables for next step
            beam_sequences = new_sequences
            beam_hiddens = new_beam_hiddens
            beam_scores = top_scores
            beam_completed = new_completed

            # Early stopping check: stop if all beams are completed
            if early_stopping and beam_completed.all():
                break

        # Post-process: select best sequence per batch
        final_sequences = []
        final_scores = []

        for b in range(batch_size):
            sequences = beam_sequences[b]  # [K, seq_len]
            scores = beam_scores[b]        # [K]
            completed = beam_completed[b]  # [K]

            # Apply length normalization to scores
            seq_lengths = sequences.shape[1]
            normalized_scores = scores / seq_lengths

            # Prioritize completed sequences
            if completed.any():
                completed_indices = torch.nonzero(completed).squeeze(-1)
                best_completed_idx = torch.argmax(normalized_scores[completed_indices])
                best_idx = completed_indices[best_completed_idx]
            else:
                # No completed sequences: pick best normalized score
                best_idx = torch.argmax(normalized_scores)

            final_sequences.append(sequences[best_idx])
            final_scores.append(scores[best_idx])

        # Pad sequences to uniform length
        max_seq_len = max(seq.shape[0] for seq in final_sequences)
        padded_sequences = torch.full(
            (batch_size, max_seq_len),
            vocabulary.unk,  # Use UNK token for padding
            dtype=torch.long,
            device=device
        )
        for i, seq in enumerate(final_sequences):
            padded_sequences[i, :len(seq)] = seq

        return {
            "sequences": padded_sequences,
            "scores": torch.stack(final_scores),
            "lengths": torch.tensor([len(seq) for seq in final_sequences], device=device)
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

    def prepare_for_quantization(self):
        """
        Prepare the model for quantization, specifically handling pretrained ResNet50.
        This should be called after loading pretrained weights but before quantization.
        """
        if not self.config.quantization_config.enable_quantization:
            return

        if self._quantization_prepared:
            logger.warning("Model already prepared for quantization")
            return

        logger.info("Preparing pretrained model for quantization...")

        # Choose backend based on deployment target
        backend = 'qnnpack' if torch.backends.quantized.engine == 'qnnpack' else 'fbgemm'

        # Set quantization scheme
        if self.config.quantization_config.quantization_scheme == "symmetric":
            qconfig = torch.quantization.get_default_qconfig(backend)
        else:  # asymmetric
            qconfig = torch.quantization.QConfig(
                activation=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.quint8,
                    qscheme=torch.per_tensor_affine
                ),
                weight=torch.quantization.MinMaxObserver.with_args(
                    dtype=torch.qint8,
                    qscheme=torch.per_tensor_symmetric
                )
            )

        # Set qconfig for the model
        self.qconfig = qconfig

        # Prepare submodules in correct order
        logger.info("Preparing CNN encoder (ResNet50) for quantization...")
        self.cnn_encoder.prepare_for_quantization()

        logger.info("Preparing RNN decoder for quantization...")
        self.rnn_decoder.prepare_for_quantization()

        logger.info("Preparing prediction head for quantization...")
        self.prediction_head.prepare_for_quantization()

        # Propagate qconfig to all modules
        torch.quantization.propagate_qconfig_(self)

        # Set model to eval mode for quantization preparation
        # This is important for pretrained models
        self.eval()

        # Prepare the model for quantization-aware training or post-training quantization
        torch.quantization.prepare(self, inplace=True)

        self._quantization_prepared = True
        logger.info("Model prepared for quantization with pretrained ResNet50")

    def quantize(self):
        """
        Convert the pretrained model to quantized version.
        This should be called after prepare_for_quantization() and calibration.
        """
        if not self.config.quantization_config.enable_quantization:
            return

        if not self._quantization_prepared:
            logger.error("Model must be prepared for quantization first. Call prepare_for_quantization()")
            return

        logger.info("Converting pretrained model to quantized version...")

        # Ensure model is in eval mode for quantization
        self.eval()

        # Convert to quantized model
        torch.quantization.convert(self, inplace=True)

        logger.info("Pretrained model successfully converted to 8-bit quantized version")

    def calibrate_quantization(self, dataloader, num_batches=100):
        """
        Calibrate quantization observers using sample data.
        This is required for post-training quantization of pretrained models.

        Args:
            dataloader: DataLoader containing calibration data
            num_batches: Number of batches to use for calibration
        """
        if not self._quantization_prepared:
            logger.error("Model must be prepared for quantization first")
            return

        logger.info(f"Calibrating quantization with {num_batches} batches...")

        self.eval()
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                images = batch.get('images') or batch.get('image') or batch[0]
                if torch.cuda.is_available() and next(self.parameters()).is_cuda:
                    images = images.cuda()

                # Forward pass to collect statistics
                _ = self.cnn_encoder(images)

                if i % 20 == 0:
                    logger.info(f"Calibration progress: {i+1}/{num_batches}")

        logger.info("Quantization calibration completed")
