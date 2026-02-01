# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
from pathlib import Path
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ExponentialLR
import matplotlib.pyplot as plt
from typing import List
from utils import *
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


# Wraps an example: stores the raw input string (input), the indexed form of the string (input_indexed),
# a tensorized version of that (input_tensor), the raw outputs (output; a numpy array) and a tensorized version
# of it (output_tensor).
# Per the task definition, the outputs are 0, 1, or 2 based on whether the character occurs 0, 1, or 2 or more
# times previously in the input sequence (not counting the current occurrence).
class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)


# Should contain your overall Transformer implementation. You will want to use Transformer layer to implement
# a single layer of the Transformer; this Module will take the raw words as input and do all of the steps necessary
# to return distributions over the labels (0, 1, or 2).
class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        """
        :param vocab_size: vocabulary size of the embedding layer
        :param num_positions: max sequence length that will be fed to the model; should be 20
        :param d_model: see TransformerLayer
        :param d_internal: see TransformerLayer
        :param num_classes: number of classes predicted at the output layer; should be 3
        :param num_layers: number of TransformerLayers to use; can be whatever you want
        """
        super().__init__()
        # Character embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, num_positions)

        # Stack of TransformerLayers using nn.ModuleList
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, d_internal) for _ in range(num_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(d_model, num_classes)

        # Store num_positions for causal mask
        self.num_positions = num_positions

    def forward(self, indices):
        """

        :param indices: list of input indices
        :return: A tuple of the softmax log probabilities (should be a 20x3 matrix) and a list of the attention
        maps you use in your layers (can be variable length, but each should be a 20x20 matrix)
        """
        # Embed input indices
        x = self.embedding(indices)  # [seq_len, d_model]

        # Add positional encoding
        x = self.positional_encoding(x)  # [seq_len, d_model]

        # Pass through transformer layers
        attn_maps = []
        for layer in self.layers:
            x, attn_weights = layer(x)
            attn_maps.append(attn_weights)

        # Output projection and log softmax
        logits = self.output_proj(x)  # [seq_len, num_classes]
        log_probs = torch.log_softmax(logits, dim=-1)  # [seq_len, num_classes]

        return log_probs, attn_maps


# Your implementation of the Transformer layer goes here. It should take vectors and return the same number of vectors
# of the same length, applying self-attention, the feedforward layer, etc.
class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        """
        :param d_model: The dimension of the inputs and outputs of the layer (note that the inputs and outputs
        have to be the same size for the residual connection to work)
        :param d_internal: The "internal" dimension used in the self-attention computation. Your keys and queries
        should both be of this length.
        """
        super().__init__()
        # Attention projections
        self.W_q = nn.Linear(d_model, d_internal)
        self.W_k = nn.Linear(d_model, d_internal)
        self.W_v = nn.Linear(d_model, d_internal)
        self.W_o = nn.Linear(d_internal, d_model)

        # Feed-forward network
        self.ff1 = nn.Linear(d_model, d_internal)
        self.ff2 = nn.Linear(d_internal, d_model)

        # Store d_internal for scaling
        self.d_internal = d_internal

    def forward(self, input_vecs, mask=None):
        """
        :param input_vecs: tensor of shape [seq_len, d_model]
        :param mask: optional attention mask of shape [seq_len, seq_len]
        :return: tuple of (output_vecs, attention_weights)
        """
        # Self-attention
        Q = self.W_q(input_vecs)  # [seq_len, d_internal]
        K = self.W_k(input_vecs)  # [seq_len, d_internal]
        V = self.W_v(input_vecs)  # [seq_len, d_internal]

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_internal)  # [seq_len, seq_len]

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores + mask

        attn_weights = torch.softmax(scores, dim=-1)  # [seq_len, seq_len]
        attn_output = torch.matmul(attn_weights, V)  # [seq_len, d_internal]
        attn_output = self.W_o(attn_output)  # [seq_len, d_model]

        # First residual connection
        x = input_vecs + attn_output

        # Feed-forward network
        ff_output = self.ff2(torch.relu(self.ff1(x)))

        # Second residual connection
        output = x + ff_output

        return output, attn_weights


# Implementation of positional encoding that you can use in your network
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, num_positions: int=20, batched=False):
        """
        :param d_model: dimensionality of the embedding layer to your model; since the position encodings are being
        added to character encodings, these need to match (and will match the dimension of the subsequent Transformer
        layer inputs/outputs)
        :param num_positions: the number of positions that need to be encoded; the maximum sequence length this
        module will see
        :param batched: True if you are using batching, False otherwise
        """
        super().__init__()
        # Dict size
        self.emb = nn.Embedding(num_positions, d_model)
        self.batched = batched

    def forward(self, x):
        """
        :param x: If using batching, should be [batch size, seq len, embedding dim]. Otherwise, [seq len, embedding dim]
        :return: a tensor of the same size with positional embeddings added in
        """
        # Second-to-last dimension will always be sequence length
        input_size = x.shape[-2]
        indices_to_embed = torch.arange(input_size, dtype=torch.long, device=x.device)
        if self.batched:
            # Use unsqueeze to form a [1, seq len, embedding dim] tensor -- broadcasting will ensure that this
            # gets added correctly across the batch
            emb_unsq = self.emb(indices_to_embed).unsqueeze(0)
            return x + emb_unsq
        else:
            return x + self.emb(indices_to_embed)


def compute_gradient_norm(model):
    """Compute total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


def evaluate_accuracy(model, examples, device):
    """Compute accuracy on a set of examples."""
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for ex in examples:
            input_tensor = ex.input_tensor.to(device)
            log_probs, _ = model.forward(input_tensor)
            predictions = torch.argmax(log_probs, dim=-1)
            correct += (predictions.cpu() == ex.output_tensor).sum().item()
            total += len(ex.output_tensor)
    model.train()
    return correct / total


# This is a skeleton for train_classifier: you can implement this however you want
def train_classifier(args, train, dev):
    # Load config using ConfigLoader
    loader = ConfigLoader()
    cfg = loader.get_experiment_config('part1')

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
    num_positions = model_cfg.get('num_positions', 20)
    d_model = arch_cfg.get('d_model', 64)
    d_internal = arch_cfg.get('d_internal', 64)
    num_classes = model_cfg.get('num_classes', 3)
    num_layers = arch_cfg.get('num_layers', 1)

    # Training hyperparameters from merged config
    training_cfg = cfg.get('training', {})
    num_epochs = training_cfg.get('num_epochs', 10)

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
        experiment_name='part1',
        task_type='classification',
        base_dir=Path(__file__).parent,
        output_dir=run_dir
    )

    # Create model
    model = Transformer(vocab_size, num_positions, d_model, d_internal, num_classes, num_layers)

    # Apply weight initialization
    init_weights(model, weight_init_cfg)

    model = model.to(device)
    model.zero_grad()
    model.train()

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, optimizer_cfg)
    scheduler = create_scheduler(optimizer, scheduler_cfg, num_epochs)

    loss_fcn = nn.NLLLoss()

    start_time = time.time()

    # Early stopping state
    best_dev_accuracy = 0.0
    epochs_without_improvement = 0

    for t in range(0, num_epochs):
        loss_this_epoch = 0.0
        correct_this_epoch = 0
        total_this_epoch = 0
        grad_norm_sum = 0.0
        random.seed(seed + t)  # Use seed for reproducibility
        ex_idxs = [i for i in range(0, len(train))]
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            ex = train[ex_idx]
            model.zero_grad()

            # Move tensors to device
            input_tensor = ex.input_tensor.to(device)
            output_tensor = ex.output_tensor.to(device)

            # Forward pass
            log_probs, _ = model.forward(input_tensor)

            # Track training accuracy
            predictions = torch.argmax(log_probs, dim=-1)
            correct_this_epoch += (predictions == output_tensor).sum().item()
            total_this_epoch += len(output_tensor)

            # Compute loss over all positions
            loss = loss_fcn(log_probs, output_tensor)

            # Backward pass
            loss.backward()

            # Apply gradient clipping if enabled
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_max_norm)

            # Track gradient norm (after backward, before optimizer step)
            batch_grad_norm = compute_gradient_norm(model)
            grad_norm_sum += batch_grad_norm

            # Per-batch logging
            metrics_logger.log_batch(loss=loss.item(), gradient_norm=batch_grad_norm)

            optimizer.step()

            loss_this_epoch += loss.item()

        # Step scheduler if enabled
        if scheduler is not None:
            scheduler.step()

        # Compute epoch metrics
        avg_loss = loss_this_epoch / len(train)
        train_accuracy = correct_this_epoch / total_this_epoch
        avg_grad_norm = grad_norm_sum / len(train)
        dev_accuracy = evaluate_accuracy(model, dev, device)
        elapsed_time = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {t+1}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Dev Acc: {dev_accuracy:.4f}, LR: {current_lr:.6f}")

        # Log epoch metrics via MetricsLogger
        metrics_logger.log_epoch(
            epoch=t + 1,
            train_loss=avg_loss,
            train_accuracy=train_accuracy,
            dev_accuracy=dev_accuracy,
            gradient_norm=avg_grad_norm,
            learning_rate=current_lr,
            elapsed_time=elapsed_time
        )

        # Early stopping check
        if use_early_stopping:
            if dev_accuracy > best_dev_accuracy + min_delta:
                best_dev_accuracy = dev_accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {t+1} epochs")
                    break

    # Final inference logging
    final_inference_cfg = tb_cfg.get('final_inference', {})
    num_samples = final_inference_cfg.get('attention_maps', {}).get('num_samples', 4)

    model.eval()
    with torch.no_grad():
        for sample_idx in range(min(num_samples, len(dev))):
            ex = dev[sample_idx]
            input_tensor = ex.input_tensor.to(device)
            log_probs, attn_maps = model.forward(input_tensor)
            predictions = torch.argmax(log_probs, dim=-1).cpu().numpy()

            metrics_logger.log_inference(
                attention_maps=attn_maps,
                log_probs=log_probs,
                predictions=str(predictions),
                input_text=ex.input,
                gold_labels=str(ex.output),
                sample_index=sample_idx
            )

    # Close MetricsLogger
    metrics_logger.close()

    # Move model back to CPU for decode function compatibility
    model = model.to('cpu')
    model.eval()
    return model


####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################
def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False, do_attention_normalization_test=False):
    """
    Decodes the given dataset, does plotting and printing of examples, and prints the final accuracy.
    :param model: your Transformer that returns log probabilities at each position in the input
    :param dev_examples: the list of LetterCountingExample
    :param do_print: True if you want to print the input/gold/predictions for the examples, false otherwise
    :param do_plot_attn: True if you want to write out plots for each example, false otherwise
    :return:
    """
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
        do_attention_normalization_test = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (log_probs, attn_maps) = model.forward(ex.input_tensor)
        predictions = np.argmax(log_probs.detach().numpy(), axis=1)
        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))
        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j]
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map.detach().numpy(), cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                # plt.show()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        if do_attention_normalization_test:
            normalizes = attention_normalization_test(attn_maps)
            print("%s normalization test on attention maps" % ("Passed" if normalizes else "Failed"))
        acc = sum([predictions[i] == ex.output[i] for i in range(0, len(predictions))])
        num_correct += acc
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))


def attention_normalization_test(attn_maps):
    """
    Tests that the attention maps sum to one over rows
    :param attn_maps: the list of attention maps
    :return:
    """
    for attn_map in attn_maps:
        total_prob_over_rows = torch.sum(attn_map, dim=1)
        if torch.any(total_prob_over_rows < 0.99).item() or torch.any(total_prob_over_rows > 1.01).item():
            print("Failed normalization test: probabilities not sum to 1.0 over rows")
            print("Total probability over rows:", total_prob_over_rows)
            return False
    return True
