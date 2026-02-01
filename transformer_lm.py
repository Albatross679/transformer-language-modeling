# transformer_lm.py

import math
import time
import random
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
from utils import Indexer
from config_loader import ConfigLoader
from src.utils.metrics_logger import MetricsLogger


def create_optimizer(model, optimizer_cfg):
    """Factory function to create optimizer based on config."""
    opt_type = optimizer_cfg.get('type', 'Adam')
    lr = optimizer_cfg.get('learning_rate', 0.001)
    weight_decay = optimizer_cfg.get('weight_decay', 0.0)

    if opt_type == 'Adam':
        betas = tuple(optimizer_cfg.get('betas', [0.9, 0.999]))
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    elif opt_type == 'SGD':
        momentum = optimizer_cfg.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_type == 'AdamW':
        betas = tuple(optimizer_cfg.get('betas', [0.9, 0.999]))
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    else:
        return optim.Adam(model.parameters(), lr=lr)


def create_scheduler(optimizer, scheduler_cfg, num_epochs):
    """Factory function to create learning rate scheduler based on config."""
    if not scheduler_cfg.get('enabled', False):
        return None
    sched_type = scheduler_cfg.get('type', 'cosine')
    if sched_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=scheduler_cfg.get('min_lr', 1e-6)
        )
    elif sched_type == 'step':
        return StepLR(
            optimizer,
            step_size=scheduler_cfg.get('step_size', 10),
            gamma=scheduler_cfg.get('gamma', 0.1)
        )
    elif sched_type == 'exponential':
        return ExponentialLR(
            optimizer,
            gamma=scheduler_cfg.get('gamma', 0.95)
        )
    return None


def init_weights(model, weight_init_cfg):
    """Apply weight initialization based on config."""
    if not weight_init_cfg.get('enabled', False):
        return
    method = weight_init_cfg.get('method', 'uniform')
    init_range = weight_init_cfg.get('range', {})
    min_val = init_range.get('min', -0.1)
    max_val = init_range.get('max', 0.1)

    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # Skip biases and 1D params
        if method == 'uniform':
            nn.init.uniform_(param, min_val, max_val)
        elif method == 'xavier_uniform':
            nn.init.xavier_uniform_(param)
        elif method == 'xavier_normal':
            nn.init.xavier_normal_(param)
        elif method == 'kaiming_uniform':
            nn.init.kaiming_uniform_(param, nonlinearity='relu')
        elif method == 'kaiming_normal':
            nn.init.kaiming_normal_(param, nonlinearity='relu')


class LanguageModel(object):

    def get_next_char_log_probs(self, context) -> np.ndarray:
        """
        Returns a log probability distribution over the next characters given a context.
        The log should be base e

        NOTE: You should make sure you call model.eval() to determinize inference here (turns off dropout
        layers in TransformerEncoder).
        :param context: the string context that the LM conditions on
        :return: A numpy vector log P(y | context) where y ranges over the output vocabulary.
        """
        raise Exception("Only implemented in subclasses")


class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0/self.voc_size)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.emb = nn.Embedding(num_positions, d_model)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, d_model] or [seq_len, d_model]
        :return: tensor with positional embeddings added
        """
        if x.dim() == 3:
            seq_len = x.shape[1]
            indices = torch.arange(seq_len, device=x.device)
            pos_emb = self.emb(indices).unsqueeze(0)  # [1, seq_len, d_model]
            return self.dropout(x + pos_emb)
        else:
            seq_len = x.shape[0]
            indices = torch.arange(seq_len, device=x.device)
            pos_emb = self.emb(indices)  # [seq_len, d_model]
            return self.dropout(x + pos_emb)


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_layers, num_heads, dropout):
        super().__init__()
        self.d_model = d_model
        self.num_positions = num_positions

        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, num_positions, dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_internal,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len] input token indices
        :return: [batch_size, seq_len, vocab_size] log probabilities
        """
        seq_len = x.shape[1]

        # Embed tokens
        x = self.embedding(x) * math.sqrt(self.d_model)  # [batch, seq_len, d_model]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Generate causal mask (upper triangular)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)

        # Apply transformer with causal mask
        x = self.transformer(x, mask=causal_mask, is_causal=True)

        # Project to vocabulary
        logits = self.output_proj(x)  # [batch, seq_len, vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        return log_probs


class NeuralLanguageModel(LanguageModel):
    def __init__(self, model, vocab_index, device, num_positions):
        self.model = model
        self.vocab_index = vocab_index
        self.device = device
        self.num_positions = num_positions

    def get_next_char_log_probs(self, context):
        self.model.eval()
        with torch.no_grad():
            # Use space as start token, then add context
            # Truncate context if too long (keep last num_positions-1 chars)
            if len(context) >= self.num_positions:
                context = context[-(self.num_positions - 1):]

            # Convert context to indices (prepend space as start token)
            full_context = ' ' + context
            indices = [self.vocab_index.index_of(c) for c in full_context]
            input_tensor = torch.LongTensor(indices).unsqueeze(0).to(self.device)

            # Forward pass
            log_probs = self.model(input_tensor)

            # Return log probs for next character (after last position)
            return log_probs[0, -1, :].cpu().numpy()


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def evaluate_perplexity(model, text, vocab_index, device, seq_length):
    """Compute perplexity on text."""
    model.eval()
    total_loss = 0.0
    total_chars = 0

    loss_fn = nn.NLLLoss(reduction='sum')

    with torch.no_grad():
        # Process text in chunks
        for i in range(0, len(text) - seq_length, seq_length):
            chunk = ' ' + text[i:i + seq_length]  # prepend space as start
            if len(chunk) < 2:
                continue

            indices = [vocab_index.index_of(c) for c in chunk]
            input_tensor = torch.LongTensor(indices[:-1]).unsqueeze(0).to(device)
            target_tensor = torch.LongTensor(indices[1:]).to(device)

            log_probs = model(input_tensor)
            loss = loss_fn(log_probs[0], target_tensor)

            total_loss += loss.item()
            total_chars += len(target_tensor)

    model.train()
    avg_loss = total_loss / total_chars if total_chars > 0 else float('inf')
    return math.exp(avg_loss)


def train_lm(args, train_text, dev_text, vocab_index):
    """
    :param args: command-line args, passed through here for your convenience
    :param train_text: train text as a sequence of characters
    :param dev_text: dev text as a sequence of characters
    :param vocab_index: an Indexer of the character vocabulary (27 characters)
    :return: a NeuralLanguageModel instance trained on the given data
    """
    # Load config using ConfigLoader
    loader = ConfigLoader()
    cfg = loader.get_experiment_config('part2')

    # Set random seed for reproducibility
    seed = cfg.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Device setup from config
    device_cfg = cfg.get('device', {})
    device_type = device_cfg.get('type', 'auto')
    if device_type == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_type)
    print(f"Using device: {device}")

    # Model hyperparameters from merged config
    model_cfg = cfg['model']
    arch_cfg = cfg['architecture']
    vocab_size = model_cfg.get('vocab_size', 27)
    num_positions = model_cfg.get('num_positions', 128)
    d_model = arch_cfg.get('d_model', 128)
    d_internal = arch_cfg.get('d_internal', 128)
    num_layers = arch_cfg.get('num_layers', 2)
    num_heads = arch_cfg.get('num_heads', 4)

    # Get dropout from model config (with nested structure support)
    dropout_cfg = model_cfg.get('dropout', {})
    if isinstance(dropout_cfg, dict):
        dropout = dropout_cfg.get('rate', 0.1) if dropout_cfg.get('enabled', True) else 0.0
    else:
        dropout = dropout_cfg

    # Training hyperparameters from merged config
    training_cfg = cfg.get('training', {})
    num_epochs = training_cfg.get('num_epochs', 20)
    batch_size = training_cfg.get('batch_size', 64)
    seq_length = training_cfg.get('seq_length', 64)

    # Optimizer config (now at root level)
    optimizer_cfg = cfg.get('optimizer', {})

    # Scheduler config (now at root level)
    scheduler_cfg = cfg.get('scheduler', {})

    # Gradient clipping config (now at root level)
    grad_clip_cfg = cfg.get('gradient_clipping', {})
    use_grad_clip = grad_clip_cfg.get('enabled', False)
    grad_clip_max_norm = grad_clip_cfg.get('max_norm', 1.0)

    # Early stopping config (now at root level)
    early_stop_cfg = cfg.get('early_stopping', {})
    use_early_stopping = early_stop_cfg.get('enabled', False)
    patience = early_stop_cfg.get('patience', 5)
    min_delta = early_stop_cfg.get('min_delta', 0.001)

    # Weight initialization config
    weight_init_cfg = cfg.get('weight_init', {})

    # Create run directory for all outputs
    run_dir = loader.create_run_directory(cfg, base_dir=Path(__file__).parent)
    print(f"Experiment output directory: {run_dir}")

    # TensorBoard config and MetricsLogger (now at root level)
    tb_cfg = cfg.get('tensorboard', {})
    metrics_logger = MetricsLogger(
        config=tb_cfg,
        experiment_name='part2',
        task_type='language_modeling',
        base_dir=Path(__file__).parent,
        output_dir=run_dir
    )

    # Create model
    model = TransformerLM(vocab_size, num_positions, d_model, d_internal, num_layers, num_heads, dropout)

    # Apply weight initialization
    init_weights(model, weight_init_cfg)

    model = model.to(device)
    model.train()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, optimizer_cfg)
    scheduler = create_scheduler(optimizer, scheduler_cfg, num_epochs)

    loss_fn = nn.NLLLoss()

    # Prepare training data - create sequences
    train_indices = [vocab_index.index_of(c) for c in train_text]

    start_time = time.time()

    # Create batches of sequences
    num_sequences = (len(train_indices) - 1) // seq_length

    # Early stopping state
    best_dev_perplexity = float('inf')
    epochs_without_improvement = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_chars = 0
        grad_norm_sum = 0.0
        num_batches = 0

        # Shuffle starting positions for diversity (use seed for reproducibility)
        positions = list(range(0, len(train_indices) - seq_length - 1, seq_length // 2))
        np.random.seed(seed + epoch)
        np.random.shuffle(positions)

        # Process in batches
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start:batch_start + batch_size]
            if len(batch_positions) == 0:
                continue

            # Build batch
            batch_inputs = []
            batch_targets = []

            for pos in batch_positions:
                # Input: positions 0 to seq_length-1
                # Target: positions 1 to seq_length
                seq = train_indices[pos:pos + seq_length + 1]
                if len(seq) < seq_length + 1:
                    continue
                batch_inputs.append(seq[:-1])
                batch_targets.append(seq[1:])

            if len(batch_inputs) == 0:
                continue

            input_tensor = torch.LongTensor(batch_inputs).to(device)
            target_tensor = torch.LongTensor(batch_targets).to(device)

            optimizer.zero_grad()

            # Forward pass
            log_probs = model(input_tensor)

            # Compute loss
            # log_probs: [batch, seq_len, vocab_size]
            # target: [batch, seq_len]
            loss = loss_fn(log_probs.view(-1, vocab_size), target_tensor.view(-1))

            # Backward pass
            loss.backward()

            # Apply gradient clipping if enabled
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            # Track gradient norm
            batch_grad_norm = compute_gradient_norm(model)
            grad_norm_sum += batch_grad_norm
            num_batches += 1

            # Per-batch logging
            metrics_logger.log_batch(loss=loss.item(), gradient_norm=batch_grad_norm)

            optimizer.step()

            total_loss += loss.item() * target_tensor.numel()
            total_chars += target_tensor.numel()

        # Step scheduler if enabled
        if scheduler is not None:
            scheduler.step()

        # Compute epoch metrics
        avg_loss = total_loss / total_chars if total_chars > 0 else float('inf')
        train_perplexity = math.exp(avg_loss)
        avg_grad_norm = grad_norm_sum / num_batches if num_batches > 0 else 0.0
        dev_perplexity = evaluate_perplexity(model, dev_text, vocab_index, device, seq_length)
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train PPL: {train_perplexity:.2f}, Dev PPL: {dev_perplexity:.2f}, LR: {current_lr:.6f}")

        # Log epoch metrics via MetricsLogger
        metrics_logger.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            train_perplexity=train_perplexity,
            dev_perplexity=dev_perplexity,
            gradient_norm=avg_grad_norm,
            learning_rate=current_lr,
            elapsed_time=elapsed_time
        )

        # Early stopping check (for perplexity, lower is better)
        if use_early_stopping:
            if dev_perplexity < best_dev_perplexity - min_delta:
                best_dev_perplexity = dev_perplexity
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break

    # Final inference logging - sample text generations
    final_inference_cfg = tb_cfg.get('final_inference', {})
    pred_cfg = final_inference_cfg.get('predictions', {})
    num_samples = pred_cfg.get('num_samples', 10)

    model.eval()
    with torch.no_grad():
        for sample_idx in range(num_samples):
            # Generate sample text starting from space
            context = ' '
            generated = ''
            for _ in range(50):  # Generate 50 characters
                indices = [vocab_index.index_of(c) for c in context]
                # Truncate if context too long
                if len(indices) > num_positions:
                    indices = indices[-num_positions:]
                input_tensor = torch.LongTensor(indices).unsqueeze(0).to(device)
                log_probs = model(input_tensor)
                # Sample from distribution
                probs = torch.exp(log_probs[0, -1, :])
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = vocab_index.get_object(next_char_idx)
                generated += next_char
                context += next_char

            metrics_logger.log_inference(
                predictions=None,
                input_text=f'Sample {sample_idx}',
                gold_labels=None,
                sample_index=sample_idx,
                generated_text=generated
            )

    # Close MetricsLogger
    metrics_logger.close()

    model.eval()
    return NeuralLanguageModel(model, vocab_index, device, num_positions)
