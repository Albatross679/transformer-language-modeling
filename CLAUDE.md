# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Conventions

Preferred project hierarchy:
```
project/
├── .claude/        # Claude Code settings and memory
├── .git/           # Git version control
├── .gitignore
├── README.md
├── CLAUDE.md
├── requirements.txt  # (optional) pinned dependencies
├── pyproject.toml    # (optional) alternative to requirements.txt
├── setup.sh          # (optional) environment setup script
├── data/           # datasets and data files
├── doc/            # documentation (markdown, PDFs, LaTeX)
├── media/          # images, videos, and GIFs
├── model/          # saved model weights
├── output/         # outputs and errors from running
├── script/         # standalone scripts (shell, examples)
├── src/            # core source code
│   └── utils/      # generic reusable helper modules
└── tests/          # unit and integration tests
```

**Note**: Create any new files following the structure above, but do not move files that are already in the root directory, unless the user specifies.

## Do Not Modify

The following files are drivers/framework and should not be modified:
- `letter_counting.py` - Driver for Part 1, imports transformer.py
- `lm.py` - Driver for Part 2, imports transformer_lm.py
- `utils.py` - Helper functions (Indexer class)

## Files You Can Modify

The following files contain your implementation:
- `transformer.py` - Implement Transformer and TransformerLayer for Part 1
- `transformer_lm.py` - Implement Transformer language model for Part 2

## Auto-grader Testing

| Part | Driver (auto-grader runs) | Your code (tested) |
|------|---------------------------|-------------------|
| Part 1 | `letter_counting.py` | `transformer.py` |
| Part 2 | `lm.py` | `transformer_lm.py` |

**Note**: The autograder also tests Part 1 on a hidden task. Do not hardcode anything specific to the letter counting labels.

## Project Overview

**Assignment 2: Transformer Language Modeling** (CSE 5525 - Spring 2026)

Transformer-based language models for:
- **Part 1** (35 pts): Letter counting task using a Transformer encoder
- **Part 2** (35 pts): Character-level language modeling on text8 dataset
- **Writeup** (30 pts): Report with results and exploration

## Commands

### Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

### Running
```bash
source .venv/bin/activate

# Part 1: Letter counting (main task - count preceding occurrences)
python letter_counting.py

# Part 1: BEFOREAFTER variant (count all occurrences in sequence)
python letter_counting.py --task BEFOREAFTER

# Part 2: Language modeling
python lm.py
```

## Architecture

### File Roles
- `transformer.py` - Transformer/TransformerLayer implementation (Part 1)
- `transformer_lm.py` - Language model implementation (Part 2)
- `letter_counting.py` - Driver for Part 1 (DO NOT MODIFY)
- `lm.py` - Driver for Part 2 (DO NOT MODIFY)
- `utils.py` - Indexer class for mapping objects to integers

### Data Files
- `data/lettercounting-train.txt` - Part 1 training data (length-20 strings)
- `data/lettercounting-dev.txt` - Part 1 dev set
- `data/text8-100k.txt` - Part 2 training data (100k characters)
- `data/text8-dev.txt` - Part 2 dev set (500 characters)
- `data/text8-test.txt` - Part 2 test set

## Part 1: Letter Counting Task

**Task**: Predict for each position how many times that character occurred before (max 2). 3-class classification (0, 1, 2+).

Example:
```
i like movies a lot
00010010002102021102
```

### Requirements
- Implement `Transformer` and `TransformerLayer` from scratch
- Do NOT use `nn.TransformerEncoder`, `nn.TransformerDecoder`, or off-the-shelf attention
- Use only `Linear`, `softmax`, and standard nonlinearities
- Target: >95% accuracy (reference achieves >98% in 5-10 epochs)

### TransformerLayer Structure
1. Self-attention (single-headed, causal mask for main task)
2. Residual connection
3. Linear → nonlinearity → Linear
4. Final residual connection

## Part 2: Language Modeling

**Task**: Character-level language model on text8. Predict next character at each position.

### Requirements
- Perplexity <= 7 (reference achieves 6.3)
- Train in < 10 minutes
- Must pass sanity and normalization checks
- Can use `nn.TransformerEncoder` for Part 2
- Must use **causal mask** (tokens cannot attend to future)
- Use space as start-of-sequence token

## Implementation Notes

- Use `PositionalEncoding` class (adds learned position embeddings)
- Divide attention scores by sqrt(d_k) for stability
- Small hyperparameters work: embedding ~100, hidden ~100
- Use `nn.ModuleList` not Python lists (or weights won't update!)
- Perplexity near 1 = causal mask broken (model cheating)
