---
id: 7e3a9f12-4b8c-4d6e-a1f0-2c5e8b9d3a7f
name: Assignment 2 Report
description: Report template for CSE 5525 Assignment 2 Transformer Language Modeling.
type: log
created: 2026-01-30T18:00:00
updated: 2026-01-30T22:20:00
tags: [assignment, report, transformer, cse5525]
aliases: [hw2-report, assignment2-report]
---

# Assignment 2 Report

> Template: [ACL Conference](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)
> Format: 1 page text + 1 page plots (PDF)

---

## 1. Part 0: BEFOREAFTER Task (Ungraded)

### 1.1 Hyperparameters and Structure

#### 1.1.1 Attention Mechanism
- [ ] Query projection (Linear)
- [ ] Key projection (Linear)
- [ ] Value projection (Linear)
- [ ] Scaled dot-product attention (divide by sqrt(d_k))
- [ ] Bidirectional attention (attend to all positions)

#### 1.1.2 First Residual Block
- [ ] Residual connection after attention

#### 1.1.3 Feed-Forward Network
- [ ] Linear layer
- [ ] Nonlinearity activation
- [ ] Linear layer

#### 1.1.4 Second Residual Block
- [ ] Residual connection after feed-forward

### 1.2 Transformer

- [ ] Character embedding layer
- [ ] Stack TransformerLayer(s) using `nn.ModuleList`
- [ ] Output projection (Linear -> log_softmax)
- [ ] Return log probs (20x3) AND attention weights

### 1.3 Training

- [ ] Implement `train_classifier` function
- [ ] Use NLLLoss computed over all timesteps
- [ ] Verify accuracy >= 85%

### 1.4 Restrictions

- [ ] **NO** `nn.TransformerEncoder`
- [ ] **NO** `nn.TransformerDecoder`
- [ ] **NO** off-the-shelf attention modules
- [ ] **ONLY** allowed: `Linear`, `softmax`, standard nonlinearities

---

## 2. Part 1: BEFORE Task (35 points)

> Command: `python letter_counting.py`
> Target: > 95% accuracy (ref: > 98%)

### 2.1 Add Positional Encoding

#### 2.1.1 Implementation
- [ ] Use `PositionalEncoding` class
- [ ] Initialize `nn.Embedding` layer for position indices

#### 2.1.2 Embedding Formula
- [ ] First token: `embed_char(token) + embed_pos(0)`
- [ ] Second token: `embed_char(token) + embed_pos(1)`
- [ ] General: `embed_char(token) + embed_pos(index)`

### 2.2 Add Causal Masking

#### 2.2.1 Masking Logic
- [ ] Tokens can **ONLY** attend to previous positions
- [ ] Tokens can attend to themselves (current position)
- [ ] Mask future positions with `-inf` before softmax

#### 2.2.2 Mask Shape
- [ ] Create triangular mask matrix (seq_len × seq_len)
- [ ] Upper triangle = `-inf` (blocked)
- [ ] Lower triangle + diagonal = `0` (allowed)

### 2.3 Verification

#### 2.3.1 Accuracy Check
- [ ] Train model for 5-10 epochs
- [ ] Verify accuracy > 95%
- [ ] Reference achieves > 98%

#### 2.3.2 Visualization
- [ ] Attention weights automatically saved to `plots/`
- [ ] Inspect attention patterns for correctness

### 2.4 Exploration 1: Attention Analysis

#### 2.4.1 Visualization
- [ ] Include at least one attention chart
- [ ] Save attention weights from `plots/` directory

#### 2.4.2 Single Layer Analysis
- [ ] Describe what the model appears to be doing
- [ ] Compare observed behavior to expected behavior
- [ ] Explain how attention enables letter counting

#### 2.4.3 Multi-Layer Analysis
- [ ] Test with 3-4 Transformer layers
- [ ] Analyze attention patterns at each layer
- [ ] Note: Do all layers fit expected patterns?

### 2.5 Exploration 2: Alternative Attention

#### 2.5.1 Choose One Implementation
- [ ] **Option A**: Relative Position (Shaw et al., 2018)
- [ ] **Option B**: ALiBi (Press et al.)
- [ ] **Option C**: Other published technique

#### 2.5.2 Comparison
- [ ] Report training differences (convergence speed)
- [ ] Report performance differences (accuracy/perplexity)
- [ ] Describe any observed behavioral differences

#### 2.5.3 ALiBi-Specific (if chosen)
- [ ] Test sequence length extrapolation
- [ ] Train on length 20, test on longer sequences
- [ ] Report extrapolation performance

---

## 3. Part 2: Language Modeling (35 points)

> Command: `python lm.py`
> Target: Perplexity <= 7 (ref: 6.3), < 10 min training

### 3.1 Network Architecture

#### 3.1.1 Input Layer
- [ ] Character embedding layer
- [ ] Positional encoding (reuse from Part 1)

#### 3.1.2 Transformer
- [ ] Can use `nn.TransformerEncoder` (allowed for Part 2)
- [ ] Or reuse custom Transformer from Part 1

#### 3.1.3 Output Layer
- [ ] Linear projection to vocab size (27 characters)
- [ ] `log_softmax` activation

### 3.2 Causal Masking (CRITICAL)

#### 3.2.1 Why It's Critical
- [ ] Tokens **CANNOT** attend to future tokens
- [ ] Without mask: model "cheats" by looking ahead
- [ ] **Symptom**: Perplexity → 1 very quickly = mask is broken

#### 3.2.2 Implementation
- [ ] Create triangular mask (seq_len × seq_len)
- [ ] Upper triangle = `-inf` (future blocked)
- [ ] Lower triangle + diagonal = `0` (past + current allowed)
- [ ] Pass mask to `TransformerEncoder` via `mask` argument

### 3.3 Start-of-Sequence

#### 3.3.1 SOS Token
- [ ] Use **space** character as start-of-sequence token

#### 3.3.2 Input/Output Alignment
- [ ] Given 20 characters to predict
- [ ] Feed: `[space] + first_19_chars` (20 tokens)
- [ ] Predict: all 20 characters at each position

### 3.4 Training

#### 3.4.1 Data Processing
- [ ] Chunk continuous text into fixed-size sequences
- [ ] Create input sequences (with SOS prepended)
- [ ] Create target sequences (characters to predict)

#### 3.4.2 Training Loop
- [ ] Implement `train_lm` function
- [ ] Compute loss over **all** positions simultaneously
- [ ] Use NLLLoss for sequence prediction

### 3.5 NeuralLanguageModel

#### 3.5.1 Interface
- [ ] Implement `get_next_char_log_probs(context)` method

#### 3.5.2 Input/Output
- **Input**: context string (any length)
- **Output**: numpy vector of length 27 (log probabilities)

#### 3.5.3 Implementation
- [ ] Convert context string to indices
- [ ] Run through trained model
- [ ] Return log probs for next character

### 3.6 Validation

#### 3.6.1 Performance Targets
- [ ] Perplexity <= 7 (reference: 6.3)
- [ ] Training time < 10 minutes

#### 3.6.2 Correctness Checks
- [ ] Pass sanity check
- [ ] Pass normalization check
- [ ] Probabilities sum to 1 (valid distribution)

### 3.7 Optional Enhancements

#### 3.7.1 Batching
- [ ] Add batch dimension to tensors
- [ ] Speeds up training significantly

#### 3.7.2 Context Overlap
- [ ] Include extra tokens for context
- [ ] Don't compute loss on context tokens
- [ ] Improves predictions at chunk boundaries

### 3.8 Exploration 3: Custom Transformer (Optional)

#### 3.8.1 Custom Transformer for Part 2
- [ ] Replace `nn.TransformerEncoder` with Part 1 implementation
- [ ] Add multi-head self-attention (likely required)

#### 3.8.2 Multi-Head Attention
- [ ] Split Q, K, V into multiple heads
- [ ] Compute attention per head
- [ ] Concatenate and project outputs

---

## 4. Training Results

> Training completed on 2026-01-30 with GPU acceleration (CUDA).
> Metrics saved to `output/part1_training_metrics.json` and `output/part2_training_metrics.json`.

### 4.1 Part 1: Letter Counting (BEFORE Task)

#### 4.1.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| vocab_size | 27 |
| num_positions | 20 |
| d_model | 64 |
| d_internal | 64 |
| num_classes | 3 |
| num_layers | 1 |
| learning_rate | 0.001 |
| num_epochs | 10 |

#### 4.1.2 Per-Epoch Metrics

| Epoch | Train Loss | Train Acc | Dev Acc | Grad Norm | Time (s) |
|-------|------------|-----------|---------|-----------|----------|
| 1 | 0.2291 | 0.9037 | 0.9886 | 2.2122 | 41.61 |
| 2 | 0.0519 | 0.9838 | 0.9872 | 1.6335 | 82.49 |
| 3 | 0.0403 | 0.9878 | 0.9841 | 1.2773 | 123.41 |
| 4 | 0.0331 | 0.9898 | 0.9916 | 1.1184 | 164.10 |
| 5 | 0.0285 | 0.9914 | 0.9928 | 0.9816 | 205.25 |
| 6 | 0.0247 | 0.9923 | 0.9930 | 0.9115 | 245.98 |
| 7 | 0.0221 | 0.9932 | 0.9962 | 0.8337 | 286.70 |
| 8 | 0.0195 | 0.9940 | 0.9931 | 0.7275 | 327.60 |
| 9 | 0.0186 | 0.9944 | 0.9954 | 0.6763 | 368.21 |
| 10 | 0.0168 | 0.9947 | 0.9954 | 0.6439 | 408.94 |

#### 4.1.3 Final Results

| Metric | Value | Target |
|--------|-------|--------|
| Train Accuracy | **99.63%** | - |
| Dev Accuracy | **99.54%** | > 95% |
| Total Time | 417.13s | - |

> **Status**: PASSED (99.54% > 95% target)

---

### 4.2 Part 2: Language Modeling

#### 4.2.1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| vocab_size | 27 |
| num_positions | 128 |
| d_model | 128 |
| d_internal | 128 |
| num_layers | 2 |
| num_heads | 4 |
| dropout | 0.1 |
| learning_rate | 0.001 |
| batch_size | 64 |
| seq_length | 64 |
| num_epochs | 20 |

#### 4.2.2 Per-Epoch Metrics

| Epoch | Train Loss | Train PPL | Dev PPL | Grad Norm | Time (s) |
|-------|------------|-----------|---------|-----------|----------|
| 1 | 2.5505 | 12.81 | 10.30 | 0.5003 | 0.99 |
| 2 | 2.3868 | 10.88 | 10.25 | 0.3204 | 1.45 |
| 3 | 2.3617 | 10.61 | 10.10 | 0.3311 | 1.91 |
| 4 | 2.3438 | 10.42 | 9.97 | 0.3319 | 2.36 |
| 5 | 2.3307 | 10.29 | 10.07 | 0.3411 | 2.83 |
| 6 | 2.3185 | 10.16 | 9.94 | 0.3401 | 3.30 |
| 7 | 2.3097 | 10.07 | 10.03 | 0.3514 | 3.84 |
| 8 | 2.3005 | 9.98 | 9.95 | 0.3450 | 4.38 |
| 9 | 2.2964 | 9.94 | 9.78 | 0.3742 | 4.91 |
| 10 | 2.2875 | 9.85 | 9.76 | 0.3342 | 5.40 |
| 11 | 2.2811 | 9.79 | 9.92 | 0.3485 | 5.87 |
| 12 | 2.2779 | 9.76 | 9.68 | 0.3736 | 6.34 |
| 13 | 2.2749 | 9.73 | 9.68 | 0.3922 | 6.81 |
| 14 | 2.2684 | 9.66 | 9.50 | 0.3581 | 7.26 |
| 15 | 2.2641 | 9.62 | 9.52 | 0.3773 | 7.74 |
| 16 | 2.2604 | 9.59 | 9.62 | 0.3644 | 8.22 |
| 17 | 2.2555 | 9.54 | 9.40 | 0.3756 | 8.73 |
| 18 | 2.2532 | 9.52 | 9.56 | 0.3949 | 9.26 |
| 19 | 2.2480 | 9.47 | 9.68 | 0.3822 | 9.77 |
| 20 | 2.2438 | 9.43 | 9.73 | 0.3651 | 10.24 |

#### 4.2.3 Final Results

| Metric | Value | Target |
|--------|-------|--------|
| Train Perplexity | 9.43 | - |
| Dev Perplexity | **8.89** | <= 7.0 |
| Total Time | 10.25s | < 600s |
| Sanity Check | PASSED | - |
| Normalization | PASSED | - |
| Range Check | PASSED | - |

> **Status**: NEEDS IMPROVEMENT (8.89 > 7.0 target). Consider:
> - Increasing model capacity (d_model, num_layers)
> - Training for more epochs
> - Adjusting learning rate schedule

---

### 4.3 Training Configuration

#### 4.3.1 Device
- **GPU**: CUDA enabled
- **Acceleration**: GPU used for both Part 1 and Part 2

#### 4.3.2 Metrics Tracked
- `train_loss`: Average NLL loss per epoch
- `train_accuracy` / `train_perplexity`: Training performance
- `dev_accuracy` / `dev_perplexity`: Validation performance
- `gradient_norm`: Average gradient L2 norm per epoch
- `elapsed_seconds`: Cumulative training time

#### 4.3.3 Output Files
- `output/part1_training_metrics.json`: Part 1 metrics with hyperparameters
- `output/part2_training_metrics.json`: Part 2 metrics with hyperparameters
