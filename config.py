# config.py

from dataclasses import dataclass, field, asdict, replace
from typing import Dict, Any, Tuple


# === Nested Config Classes ===

@dataclass
class DeviceConfig:
    type: str = "auto"
    mixed_precision: bool = False


@dataclass
class OutputConfig:
    base_dir: str = "output"
    save_config: bool = True
    subdirs: Dict[str, str] = field(default_factory=lambda: {
        "tensorboard": "tensorboard",
        "checkpoints": "checkpoints",
        "plots": "plots"
    })


@dataclass
class TensorboardPerBatchMetrics:
    loss: bool = True
    gradient_norm: bool = True


@dataclass
class TensorboardPerBatch:
    enabled: bool = True
    frequency: int = 10
    metrics: TensorboardPerBatchMetrics = field(default_factory=TensorboardPerBatchMetrics)


@dataclass
class TensorboardPerEpochMetrics:
    train_loss: bool = True
    train_accuracy: bool = True
    train_perplexity: bool = True
    dev_accuracy: bool = True
    dev_perplexity: bool = True
    gradient_norm: bool = True
    learning_rate: bool = True
    elapsed_time: bool = True


@dataclass
class TensorboardPerEpoch:
    enabled: bool = True
    metrics: TensorboardPerEpochMetrics = field(default_factory=TensorboardPerEpochMetrics)


@dataclass
class AttentionMapsConfig:
    enabled: bool = True
    num_samples: int = 4
    colormap: str = "viridis"


@dataclass
class LogProbsConfig:
    enabled: bool = True
    as_heatmap: bool = True


@dataclass
class PredictionsConfig:
    enabled: bool = True
    num_samples: int = 10


@dataclass
class TensorboardFinalInference:
    enabled: bool = True
    attention_maps: AttentionMapsConfig = field(default_factory=AttentionMapsConfig)
    log_probs: LogProbsConfig = field(default_factory=LogProbsConfig)
    predictions: PredictionsConfig = field(default_factory=PredictionsConfig)


@dataclass
class TensorboardHistograms:
    enabled: bool = False
    frequency: int = 100
    track_weights: bool = True
    track_gradients: bool = True


@dataclass
class TensorboardModelGraph:
    enabled: bool = False


@dataclass
class TensorboardConfig:
    enabled: bool = True
    flush_secs: int = 120
    per_batch: TensorboardPerBatch = field(default_factory=TensorboardPerBatch)
    per_epoch: TensorboardPerEpoch = field(default_factory=TensorboardPerEpoch)
    final_inference: TensorboardFinalInference = field(default_factory=TensorboardFinalInference)
    histograms: TensorboardHistograms = field(default_factory=TensorboardHistograms)
    model_graph: TensorboardModelGraph = field(default_factory=TensorboardModelGraph)


@dataclass
class ModelConfig:
    vocab_size: int = 27
    num_positions: int = 20
    num_classes: int = 3
    dropout: float = 0.0


@dataclass
class ArchitectureConfig:
    d_model: int = 64
    d_internal: int = 64
    num_layers: int = 1
    num_heads: int = 1
    causal_mask: bool = False
    positional_encoding: bool = False


@dataclass
class TrainingConfig:
    num_epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001
    seq_length: int = 64


@dataclass
class WeightInitRange:
    min: float = -0.1
    max: float = 0.1


@dataclass
class WeightInitConfig:
    enabled: bool = True
    method: str = "uniform"
    range: WeightInitRange = field(default_factory=WeightInitRange)


@dataclass
class OptimizerConfig:
    type: str = "Adam"
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class SchedulerConfig:
    enabled: bool = False
    type: str = "cosine"
    min_lr: float = 1e-6


@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 5
    min_delta: float = 0.001


@dataclass
class GradientClippingConfig:
    enabled: bool = False
    max_norm: float = 1.0


# === Base & Experiment Configs ===

@dataclass
class BaseConfig:
    """Base configuration - all experiments inherit from this."""
    name: str = "base"
    description: str = ""
    seed: int = 42
    device: DeviceConfig = field(default_factory=DeviceConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    weight_init: WeightInitConfig = field(default_factory=WeightInitConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    gradient_clipping: GradientClippingConfig = field(default_factory=GradientClippingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for JSON serialization."""
        d = asdict(self)
        # Convert betas tuple to list for JSON compatibility
        d['optimizer']['betas'] = list(d['optimizer']['betas'])
        return d


@dataclass
class Part0Config(BaseConfig):
    """BEFOREAFTER task - bidirectional attention."""
    name: str = "part0"
    description: str = "BEFOREAFTER task - count all occurrences (bidirectional attention)"

    def __post_init__(self):
        # Bidirectional attention, no positional encoding needed
        self.architecture = replace(self.architecture,
            causal_mask=False,
            positional_encoding=False
        )


@dataclass
class Part1Config(BaseConfig):
    """BEFORE task - causal attention."""
    name: str = "part1"
    description: str = "BEFORE task - count preceding occurrences (causal attention)"

    def __post_init__(self):
        # Causal attention with positional encoding
        self.architecture = replace(self.architecture,
            causal_mask=True,
            positional_encoding=True
        )


@dataclass
class Part2Config(BaseConfig):
    """Character-level language modeling."""
    name: str = "part2"
    description: str = "Character-level language modeling on text8"

    def __post_init__(self):
        # Larger model for language modeling
        self.model = replace(self.model,
            num_positions=128,
            dropout=0.1
        )
        self.architecture = replace(self.architecture,
            d_model=128,
            d_internal=128,
            num_layers=2,
            num_heads=4
        )
        self.training = replace(self.training,
            num_epochs=20,
            batch_size=64
        )
        self.weight_init = replace(self.weight_init,
            method="xavier_uniform"
        )


# Registry of available configs
CONFIGS = {
    "base": BaseConfig,
    "part0": Part0Config,
    "part1": Part1Config,
    "part2": Part2Config,
}


def get_config(name: str = "part1") -> BaseConfig:
    """Get an experiment configuration by name."""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()
