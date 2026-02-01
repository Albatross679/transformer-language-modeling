# ML Configuration File

Generate comprehensive JSON configuration files for machine learning projects with all necessary hyperparameters and settings using a modular split-file structure.

## Trigger

Use when the user asks to:
- Create a configuration file for a machine learning project
- Set up training configuration for a neural network
- Generate config for deep learning experiments
- Create a JSON config for model training

## Input

The user provides:
- Type of ML task (classification, language modeling, etc.)
- Model architecture preference (transformer, mlp, rnn, cnn), OR
- Specific parameters they want to configure

## Output Format

A modular configuration structure with the following layout:

```
config/
├── config.json              # Main file that references others
├── global.json              # Seed, device, logging, checkpoint
├── tensorboard.json         # TensorBoard settings
├── architectures/
│   ├── transformer.json
│   ├── mlp.json
│   ├── rnn.json
│   └── cnn.json
├── training/
│   ├── optimizer.json
│   ├── scheduler.json
│   ├── early_stopping.json
│   └── gradient_clipping.json
└── experiments/
    ├── part1.json           # Task-specific overrides
    └── part2.json
```

## Configuration Files

### 1. Main Config (config.json)

```json
{
    "_description": "Main configuration file - references split config files",
    "includes": {
        "global": "global.json",
        "tensorboard": "tensorboard.json",
        "architectures": {
            "transformer": "architectures/transformer.json",
            "mlp": "architectures/mlp.json",
            "rnn": "architectures/rnn.json",
            "cnn": "architectures/cnn.json"
        },
        "training": {
            "optimizer": "training/optimizer.json",
            "scheduler": "training/scheduler.json",
            "early_stopping": "training/early_stopping.json",
            "gradient_clipping": "training/gradient_clipping.json"
        },
        "experiments": {
            "part1": "experiments/part1.json",
            "part2": "experiments/part2.json"
        }
    },
    "active_experiment": "part1"
}
```

### 2. Global Settings (global.json)

```json
{
    "seed": 42,
    "device": { "type": "auto", "mixed_precision": false },
    "logging": { "log_frequency": 100, "verbose": true },
    "checkpoint": { "save_best": true, "save_frequency": 5, "path": "model/" }
}
```

### 3. TensorBoard Settings (tensorboard.json)

```json
{
    "enabled": true,
    "log_dir": "runs/",
    "run_name": "experiment",
    "flush_secs": 120,
    "log_scalars": { "enabled": true, "frequency": 10, "metrics": ["loss", "accuracy", "perplexity", "learning_rate"] },
    "log_histograms": { "enabled": true, "frequency": 100, "track_weights": true, "track_gradients": true, "track_activations": false },
    "log_graph": { "enabled": true },
    "log_embeddings": { "enabled": false, "frequency": 500, "num_samples": 1000 },
    "log_images": { "enabled": false, "frequency": 100, "max_images": 8 },
    "log_hparams": { "enabled": true, "metrics_to_track": ["val_loss", "val_accuracy"] },
    "log_attention": { "enabled": false, "frequency": 100, "num_samples": 4 },
    "profiler": { "enabled": false, "wait": 1, "warmup": 1, "active": 3, "repeat": 1 }
}
```

### 4. Architecture Files (architectures/)

**transformer.json**
```json
{
    "d_model": 128,
    "d_internal": 128,
    "num_layers": 2,
    "num_heads": 4,
    "activation": "relu",
    "positional_encoding": { "type": "learned", "max_length": 512 },
    "layer_norm": { "type": "post", "epsilon": 1e-6 },
    "attention": { "scaled": true, "dropout": 0.1, "causal_mask": true }
}
```

**mlp.json**
```json
{
    "hidden_layers": [256, 128, 64],
    "activation": "relu",
    "batch_norm": { "enabled": false, "momentum": 0.1, "epsilon": 1e-5 },
    "dropout_per_layer": [0.1, 0.1, 0.1]
}
```

**rnn.json**
```json
{
    "type": "lstm",
    "hidden_size": 128,
    "num_layers": 2,
    "bidirectional": false,
    "dropout": 0.1
}
```

**cnn.json**
```json
{
    "channels": [32, 64, 128],
    "kernel_sizes": [3, 3, 3],
    "strides": [1, 1, 1],
    "padding": "same",
    "pooling": { "type": "max", "kernel_size": 2 },
    "batch_norm": true,
    "activation": "relu"
}
```

### 5. Training Files (training/)

**optimizer.json**
```json
{
    "type": "Adam",
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "momentum": 0.9,
    "betas": [0.9, 0.999]
}
```

**scheduler.json**
```json
{
    "enabled": true,
    "type": "cosine",
    "warmup_steps": 100,
    "min_lr": 1e-6
}
```

**early_stopping.json**
```json
{
    "enabled": true,
    "patience": 5,
    "min_delta": 0.001,
    "monitor": "val_loss"
}
```

**gradient_clipping.json**
```json
{
    "enabled": true,
    "max_norm": 1.0
}
```

### 6. Experiment Files (experiments/)

Experiment files contain task-specific settings and can override base configs:

```json
{
    "name": "part1",
    "description": "Letter counting task using Transformer encoder",
    "model": {
        "vocab_size": 27,
        "num_positions": 20,
        "num_classes": 3,
        "embedding_dim": 64,
        "dropout": { "enabled": false, "rate": 0.1 }
    },
    "architecture": {
        "type": "transformer",
        "overrides": {
            "transformer": { "d_model": 64, "d_internal": 64, "num_layers": 1, "num_heads": 1 }
        }
    },
    "training": { "num_epochs": 10, "batch_size": 32 },
    "data": { "shuffle": true, "num_workers": 4, "pin_memory": true },
    "weight_init": { "enabled": true, "method": "uniform", "range": { "min": -0.1, "max": 0.1 } }
}
```

## Rules

1. Always use the modular split-file structure
2. Main config.json references other files via `includes`
3. Use `active_experiment` to select which experiment config to use
4. Experiment files can override base architecture/training configs via `overrides`
5. Always include all four architecture files (transformer, mlp, rnn, cnn)
6. Use `enabled` flags for optional features (dropout, batch_norm, scheduler, etc.)
7. Include configurable ranges for weight initialization with min/max
8. Do NOT include `label_smoothing` parameter
9. TensorBoard config is a separate file (shared across experiments)
10. Training components are split into separate files (optimizer, scheduler, etc.)

## Workflow

1. **Create directory structure**: Set up config/, architectures/, training/, experiments/
2. **Create global.json**: Add seed, device, logging, checkpoint settings
3. **Create tensorboard.json**: Add all TensorBoard logging options
4. **Create architecture files**: One file per architecture type
5. **Create training files**: Separate files for optimizer, scheduler, early stopping, gradient clipping
6. **Create experiment files**: Task-specific configs with overrides
7. **Create main config.json**: Reference all files and set active experiment

## Benefits of Split Structure

| Benefit | Description |
|---------|-------------|
| **Reusability** | Share optimizer.json across experiments |
| **Readability** | Smaller, focused files are easier to scan |
| **Version control** | Isolated changes, cleaner diffs |
| **Experiment management** | Swap architecture files without touching training config |
| **Team collaboration** | Different people own different configs |

## Example

**Input:**
> Create a config structure for a text classification task

**Output:**
Complete modular config structure with:
- config.json (main reference file)
- global.json, tensorboard.json
- architectures/ folder with all 4 architecture types
- training/ folder with optimizer, scheduler, early_stopping, gradient_clipping
- experiments/ folder with task-specific config
