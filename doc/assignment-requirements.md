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

- [ ] Query projection (Linear)
- [ ] Key projection (Linear)
- [ ] Value projection (Linear)
- [ ] Scaled dot-product attention (divide by sqrt(d_k))
- [ ] Bidirectional attention (attend to all positions)
- [ ] Residual connection after attention
- [ ] Feed-forward: Linear -> nonlinearity -> Linear
- [ ] Residual connection after feed-forward

### 1.2 Transformer

- [ ] Character embedding
- [ ] TransformerLayer(s) via `nn.ModuleList`
- [ ] Output projection (Linear -> log_softmax)
- [ ] Return log probs (20x3) AND attention weights

### 1.3 Training

- [ ] Implement `train_classifier`
- [ ] Use NLLLoss over all timesteps
- [ ] Verify >= 85% accuracy

### 1.4 Restrictions

- [ ] NO `nn.TransformerEncoder`
- [ ] NO `nn.TransformerDecoder`
- [ ] NO off-the-shelf attention
- [ ] ONLY: Linear, softmax, standard nonlinearities

---

## 2. Part 1: BEFORE Task (35 points)

> Command: `python letter_counting.py`
> Target: > 95% accuracy (ref: > 98%)

### 2.1 Add Positional Encoding

- [ ] Use `PositionalEncoding` class
- [ ] Formula: `embed_char(token) + embed_pos(index)`

### 2.2 Add Causal Masking

- [ ] Tokens can only attend to previous positions
- [ ] Mask future positions with -inf

### 2.3 Verify

- [ ] Accuracy > 95%
- [ ] Attention weights saved to `plots/`

> **Warning**: Autograder tests on hidden task. No hardcoding.

---

## 3. Part 2: Language Modeling (35 points)

> Command: `python lm.py`
> Target: Perplexity <= 7 (ref: 6.3), < 10 min training

### 3.1 Network Architecture

- [ ] Character embedding
- [ ] Positional encoding
- [ ] Transformer (can use `nn.TransformerEncoder`)
- [ ] Output projection to vocab size (27)
- [ ] log_softmax activation

### 3.2 Causal Masking (CRITICAL)

- [ ] Tokens CANNOT attend to future
- [ ] Use triangular mask (zeros / -inf)
- [ ] If perplexity -> 1 quickly, mask is broken

### 3.3 Start-of-Sequence

- [ ] Use space as SOS token
- [ ] Feed `[space] + first_19_chars` -> predict all 20

### 3.4 Training

- [ ] Implement `train_lm`
- [ ] Chunk data into fixed-size sequences
- [ ] Compute loss over all positions

### 3.5 NeuralLanguageModel

- [ ] Implement `get_next_char_log_probs(context)`
  - Input: context string
  - Output: numpy vector length 27

### 3.6 Validation

- [ ] Perplexity <= 7
- [ ] Training < 10 minutes
- [ ] Pass sanity check
- [ ] Pass normalization check
- [ ] Probabilities sum to 1

### 3.7 Optional

- [ ] Batching (add batch dimension)
- [ ] Context overlap for better predictions

---

## 4. Writeup (30 points)

> Template: [ACL Conference](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)
> Format: 1 page text + 1 page plots (PDF)

### 4.1 Results

- [ ] Report Part 1 accuracy
- [ ] Report Part 2 perplexity
- [ ] Describe implementation details

### 4.2 Exploration 1: Attention Analysis

- [ ] Include attention chart(s)
- [ ] Describe model behavior
- [ ] Compare to expected behavior
- [ ] Test with 3-4 layers
- [ ] Analyze attention patterns

### 4.3 Exploration 2: Alternative Attention

- [ ] Implement one: Relative Position / ALiBi / Other
- [ ] Report training/performance differences
- [ ] (ALiBi) Test sequence length extrapolation

### 4.4 Exploration 3 (Optional)

- [ ] Use custom Transformer from Part 1 for Part 2
- [ ] May need multi-head attention
