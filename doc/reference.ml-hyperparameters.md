---
id: 581829e4-91e7-4d54-a95e-e3a3466bb4ca
name: ML Hyperparameters Reference
description: Comprehensive guide to popular machine learning hyperparameters for debugging model issues.
type: reference
created: 2026-01-29T21:12:49
updated: 2026-01-29T21:12:49
tags: [machine-learning, hyperparameters, debugging, reference]
aliases: [hyperparameter-guide, ml-tuning]
---

# ML Hyperparameters Reference

A quick reference for when you're asking "what's wrong with my model?"

## Learning Rate

The most critical hyperparameter. Controls step size during optimization.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss not decreasing | LR too low | Increase by 10x |
| Loss exploding/NaN | LR too high | Decrease by 10x |
| Loss oscillating wildly | LR too high | Decrease by 2-5x |
| Converges then gets worse | LR too high for fine-tuning | Use LR decay/scheduler |
| Stuck at plateau | LR too low or stuck in local min | Try LR warmup or restart |

**Typical ranges:**
- SGD: 0.01 - 0.1
- Adam/AdamW: 1e-4 - 1e-3
- Fine-tuning pretrained: 1e-5 - 1e-4
- Transformers: 1e-5 - 5e-5

## Batch Size

Number of samples per gradient update. Affects training dynamics and memory.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Out of memory (OOM) | Batch too large | Reduce batch size |
| Noisy/unstable training | Batch too small | Increase batch size |
| Poor generalization | Batch too large | Reduce batch size |
| Slow convergence | Batch too small | Increase batch + scale LR |

**Typical ranges:**
- Small datasets: 16-64
- Large datasets: 64-256
- Transformers/LLMs: 8-32 (memory limited)

> **Note:** When scaling batch size, scale learning rate proportionally (linear scaling rule).

## Number of Epochs

How many times to iterate through the full dataset.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Training loss high | Underfitting, too few epochs | Train longer |
| Val loss increasing while train decreases | Overfitting | Early stopping, fewer epochs |
| Both losses plateau early | Model capacity or data issue | Check architecture/data |

**Typical ranges:**
- Simple models: 10-50
- Deep networks: 50-200
- Fine-tuning: 2-10
- Large datasets: 1-5

## Weight Decay (L2 Regularization)

Penalizes large weights to prevent overfitting.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Overfitting (val >> train loss) | Regularization too weak | Increase weight decay |
| Underfitting (both losses high) | Regularization too strong | Decrease weight decay |
| Weights becoming very large | No regularization | Add weight decay |

**Typical ranges:**
- Standard: 1e-4 - 1e-2
- Adam/AdamW: 0.01 - 0.1
- Fine-tuning: 0.01

## Dropout Rate

Randomly zeros activations during training to prevent overfitting.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Overfitting | Dropout too low | Increase (0.3-0.5) |
| Underfitting | Dropout too high | Decrease or remove |
| Train loss much higher than expected | Forgot to call `model.eval()` | Set eval mode at inference |

**Typical ranges:**
- Hidden layers: 0.1 - 0.5
- Input layer: 0.0 - 0.2
- Attention: 0.1 - 0.3
- After embeddings: 0.1 - 0.2

## Hidden Layer Size / Width

Number of neurons per layer.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Underfitting | Model too small | Increase width |
| Overfitting | Model too large | Decrease width, add regularization |
| Slow training | Model too large | Reduce size or use GPU |

**Typical ranges:**
- Small tasks: 32-128
- Medium tasks: 128-512
- Large tasks: 512-2048
- Transformers: 768-4096

## Number of Layers / Depth

How many layers in the network.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Underfitting | Too shallow | Add layers |
| Vanishing gradients | Too deep without residuals | Add skip connections, use LayerNorm |
| Overfitting | Too deep for data size | Reduce layers |

**Typical ranges:**
- MLPs: 2-4 layers
- CNNs: 5-50+ (with residuals)
- Transformers: 6-24 layers

## Optimizer Choice

| Optimizer | When to Use | Watch Out For |
|-----------|-------------|---------------|
| SGD + Momentum | CNNs, when you have time to tune | Sensitive to LR |
| Adam | Default choice, fast convergence | May generalize worse |
| AdamW | Transformers, when using weight decay | Slightly more memory |
| RMSprop | RNNs, non-stationary problems | Less common now |

## Momentum (for SGD)

Accelerates convergence by accumulating gradient direction.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Slow convergence | No momentum | Add momentum (0.9) |
| Oscillating around minimum | Momentum too high | Reduce to 0.5-0.8 |

**Typical values:** 0.9, 0.99

## Learning Rate Schedule

How LR changes during training.

| Schedule | Use Case |
|----------|----------|
| Constant | Simple baselines |
| Step decay | Standard CNNs |
| Cosine annealing | Modern training |
| Warmup + decay | Transformers |
| ReduceLROnPlateau | When unsure |

## Embedding Dimension

Size of embedding vectors for discrete inputs.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Poor representation | Dimension too small | Increase dimension |
| Overfitting | Dimension too large | Reduce or use pretrained |

**Typical ranges:**
- Word embeddings: 100-300
- Transformer tokens: 256-1024
- Categorical features: 8-64

## Sequence Length / Context Window

Maximum input length for sequential models.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| OOM errors | Sequence too long | Truncate or use chunking |
| Missing context | Sequence too short | Increase length |
| Slow training | Sequence too long | Use efficient attention |

## Gradient Clipping

Prevents exploding gradients by capping gradient norm.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss becomes NaN | Exploding gradients | Add clipping (1.0-5.0) |
| Training unstable | Gradient spikes | Add clipping |

**Typical values:** 0.5, 1.0, 5.0

## Temperature (for sampling/softmax)

Controls randomness in output distributions.

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Outputs too random | Temperature too high | Decrease toward 1.0 |
| Outputs too deterministic | Temperature too low | Increase toward 1.0 |
| Knowledge distillation not working | Wrong temperature | Try 2.0-20.0 |

## Quick Debugging Checklist

When your model isn't working, check in this order:

1. **Data issues first**
   - Are labels correct?
   - Is data normalized/standardized?
   - Any data leakage?
   - Class imbalance?

2. **Learning rate**
   - Try 0.1, 0.01, 0.001, 0.0001
   - Plot loss curves

3. **Architecture**
   - Can it overfit a tiny subset? (If no, bug in code)
   - Is model appropriate for task?

4. **Regularization**
   - Remove all regularization first
   - Add back gradually

5. **Batch size**
   - Try 32 as default
   - Adjust based on memory/stability

6. **Training length**
   - Use early stopping
   - Monitor validation loss

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Forgot `model.eval()` | Always set eval mode for inference |
| Forgot `optimizer.zero_grad()` | Call before each backward pass |
| Wrong loss function | Classification: CrossEntropy; Regression: MSE/MAE |
| Data not shuffled | Shuffle training data |
| Preprocessing mismatch train/test | Use same pipeline |
| Random seed not set | Set for reproducibility |
| GPU/CPU tensor mismatch | Move all to same device |

## Hyperparameter Search Strategies

| Strategy | When to Use |
|----------|-------------|
| Grid search | Few hyperparameters, small ranges |
| Random search | Many hyperparameters, initial exploration |
| Bayesian optimization | Expensive evaluations, refined tuning |
| Population-based | Large compute budget |

> **Note:** Start with published baselines and defaults. Tune one thing at a time.
