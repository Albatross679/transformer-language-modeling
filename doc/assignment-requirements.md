---
id: f3909ce6-645c-4ccf-827e-e9e0449de7fc
name: Assignment Requirements
description: Sequential checklist for CSE 5525 Assignment 2 Transformer Language Modeling.
type: reference
created: 2026-01-30T17:42:45
updated: 2026-01-30T17:50:00
tags: [assignment, requirements, transformer, cse5525]
aliases: [hw2-requirements, assignment2-requirements]
---

# Assignment Requirements

**Due: February 11th, 2026 at 11:59 PM**

---

## 1. Part 0: BEFOREAFTER Task (Ungraded)

> Command: `python letter_counting.py --task BEFOREAFTER`
> Target: >= 85% accuracy

### 1.1 TransformerLayer

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

1. [ ] Character embedding layer
2. [ ] Stack TransformerLayer(s) using `nn.ModuleList`
3. [ ] Output projection (Linear -> log_softmax)
4. [ ] Return log probs (20x3) AND attention weights

### 1.3 Training

1. [ ] Implement `train_classifier` function
2. [ ] Use NLLLoss computed over all timesteps
3. [ ] Verify accuracy >= 85%

### 1.4 Restrictions (Must Follow)

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
1. [ ] Use `PositionalEncoding` class
2. [ ] Initialize `nn.Embedding` layer for position indices

#### 2.1.2 Embedding Formula
- [ ] First token: `embed_char(token) + embed_pos(0)`
- [ ] Second token: `embed_char(token) + embed_pos(1)`
- [ ] General: `embed_char(token) + embed_pos(index)`

### 2.2 Add Causal Masking

#### 2.2.1 Masking Logic
1. [ ] Tokens can **ONLY** attend to previous positions
2. [ ] Tokens can attend to themselves (current position)
3. [ ] Mask future positions with `-inf` before softmax

#### 2.2.2 Mask Shape
- [ ] Create triangular mask matrix (seq_len × seq_len)
- [ ] Upper triangle = `-inf` (blocked)
- [ ] Lower triangle + diagonal = `0` (allowed)

### 2.3 Verification

#### 2.3.1 Accuracy Check
1. [ ] Train model for 5-10 epochs
2. [ ] Verify accuracy > 95%
3. [ ] Reference achieves > 98%

#### 2.3.2 Visualization
- [ ] Attention weights automatically saved to `plots/`
- [ ] Inspect attention patterns for correctness

> **Warning**: Autograder tests on hidden task. **NO** hardcoding allowed.

### 2.4 Exploration 1: Attention Analysis

#### 2.4.1 Visualization
1. [ ] Include at least one attention chart
2. [ ] Save attention weights from `plots/` directory

#### 2.4.2 Single Layer Analysis
1. [ ] Describe what the model appears to be doing
2. [ ] Compare observed behavior to expected behavior
3. [ ] Explain how attention enables letter counting

#### 2.4.3 Multi-Layer Analysis
1. [ ] Test with 3-4 Transformer layers
2. [ ] Analyze attention patterns at each layer
3. [ ] Note: Do all layers fit expected patterns?

### 2.5 Exploration 2: Alternative Attention

#### 2.5.1 Choose One Implementation
- [ ] **Option A**: Relative Position (Shaw et al., 2018)
- [ ] **Option B**: ALiBi (Press et al.)
- [ ] **Option C**: Other published technique

#### 2.5.2 Comparison
1. [ ] Report training differences (convergence speed)
2. [ ] Report performance differences (accuracy/perplexity)
3. [ ] Describe any observed behavioral differences

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
1. [ ] Character embedding layer
2. [ ] Positional encoding (reuse from Part 1)

#### 3.1.2 Transformer
- [ ] Can use `nn.TransformerEncoder` (allowed for Part 2)
- [ ] Or reuse custom Transformer from Part 1

#### 3.1.3 Output Layer
1. [ ] Linear projection to vocab size (27 characters)
2. [ ] `log_softmax` activation

### 3.2 Causal Masking (CRITICAL)

#### 3.2.1 Why It's Critical
- [ ] Tokens **CANNOT** attend to future tokens
- [ ] Without mask: model "cheats" by looking ahead
- [ ] **Symptom**: Perplexity → 1 very quickly = mask is broken

#### 3.2.2 Implementation
1. [ ] Create triangular mask (seq_len × seq_len)
2. [ ] Upper triangle = `-inf` (future blocked)
3. [ ] Lower triangle + diagonal = `0` (past + current allowed)
4. [ ] Pass mask to `TransformerEncoder` via `mask` argument

### 3.3 Start-of-Sequence

#### 3.3.1 SOS Token
- [ ] Use **space** character as start-of-sequence token

#### 3.3.2 Input/Output Alignment
1. [ ] Given 20 characters to predict
2. [ ] Feed: `[space] + first_19_chars` (20 tokens)
3. [ ] Predict: all 20 characters at each position

### 3.4 Training

#### 3.4.1 Data Processing
1. [ ] Chunk continuous text into fixed-size sequences
2. [ ] Create input sequences (with SOS prepended)
3. [ ] Create target sequences (characters to predict)

#### 3.4.2 Training Loop
1. [ ] Implement `train_lm` function
2. [ ] Compute loss over **all** positions simultaneously
3. [ ] Use NLLLoss for sequence prediction

### 3.5 NeuralLanguageModel

#### 3.5.1 Interface
- [ ] Implement `get_next_char_log_probs(context)` method

#### 3.5.2 Input/Output
- **Input**: context string (any length)
- **Output**: numpy vector of length 27 (log probabilities)

#### 3.5.3 Implementation
1. [ ] Convert context string to indices
2. [ ] Run through trained model
3. [ ] Return log probs for next character

### 3.6 Validation

#### 3.6.1 Performance Targets
1. [ ] Perplexity <= 7 (reference: 6.3)
2. [ ] Training time < 10 minutes

#### 3.6.2 Correctness Checks
1. [ ] Pass sanity check
2. [ ] Pass normalization check
3. [ ] Probabilities sum to 1 (valid distribution)

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
1. [ ] Replace `nn.TransformerEncoder` with Part 1 implementation
2. [ ] Add multi-head self-attention (likely required)

#### 3.8.2 Multi-Head Attention
1. [ ] Split Q, K, V into multiple heads
2. [ ] Compute attention per head
3. [ ] Concatenate and project outputs

---

## 4. Writeup (30 points)

> Template: [ACL Conference](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)
> Format: 1 page text + 1 page plots (PDF)

### 4.1 Results Section

#### 4.1.1 Performance Metrics
1. [ ] Report Part 1 accuracy (target: > 95%)
2. [ ] Report Part 2 perplexity (target: <= 7)

#### 4.1.2 Implementation Details
- [ ] Describe architecture choices (layers, dimensions)
- [ ] Describe training setup (epochs, learning rate, optimizer)
- [ ] Note any distinctive implementation decisions

### 4.2 Exploration Writeup

#### 4.2.1 Exploration 1 Results
- [ ] Include attention visualization(s)
- [ ] Describe findings from attention analysis (see 2.4)

#### 4.2.2 Exploration 2 Results
- [ ] Report alternative attention implementation results (see 2.5)
- [ ] Compare performance with baseline

#### 4.2.3 Exploration 3 Results (Optional)
- [ ] Report custom Transformer results for Part 2 (see 3.8)

### 4.3 Submission

#### 4.3.1 Report Format
1. [ ] Maximum 1 page of text
2. [ ] Additional page for plots/graphs
3. [ ] Use ACL conference template
4. [ ] Export as PDF

#### 4.3.2 Code Submission
1. [ ] Create .zip file with all files
2. [ ] Include: data, code, report PDF
3. [ ] **Important**: Zip files directly, not the folder
4. [ ] Upload to Gradescope "HW2: Transformer Language Modeling"
