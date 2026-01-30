# Assignment 2: Transformer Language Modeling

CSE 5525 - Spring 2026

## Overview

This project implements transformer-based language models for:
- Character-level language modeling on text8 dataset
- Letter counting task

## Project Structure

```
assignment2/
├── p2-distrib/           # Distribution code
│   ├── transformer.py    # Transformer architecture
│   ├── transformer_lm.py # Language model training
│   ├── letter_counting.py# Letter counting task
│   ├── lm.py            # Language model utilities
│   ├── utils.py         # Helper functions
│   └── data/            # Training/dev/test data
├── doc/                 # Documentation and references
└── CLAUDE.md            # Project conventions
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch numpy matplotlib
```

## Usage

```bash
# Activate virtual environment
source .venv/bin/activate

# Run transformer language model
python p2-distrib/transformer_lm.py

# Run letter counting task
python p2-distrib/letter_counting.py
```

## Data

- `text8-100k.txt` - Training data (100k characters from text8)
- `text8-dev.txt` - Development set
- `text8-test.txt` - Test set
- `lettercounting-train.txt` - Letter counting training data
- `lettercounting-dev.txt` - Letter counting dev data
