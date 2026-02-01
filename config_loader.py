# config_loader.py

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from config import BaseConfig, Part0Config, Part1Config, Part2Config, get_config


CONFIGS = {
    "part0": Part0Config,
    "part1": Part1Config,
    "part2": Part2Config,
}


class ConfigLoader:
    """
    Configuration loader using dataclass-based configs.

    Provides experiment configurations via Python dataclasses with inheritance.
    Base defaults are defined in BaseConfig, experiment-specific overrides
    in Part0Config, Part1Config, Part2Config.
    """

    def __init__(self, config_dir: str = None):
        # config_dir is ignored but kept for backwards compatibility
        pass

    def get_experiment_config(self, name: str = 'part1') -> dict:
        """
        Load an experiment configuration.

        Args:
            name: Experiment name ('part0', 'part1', or 'part2')

        Returns:
            Fully resolved configuration dict.
        """
        cfg = get_config(name)
        merged = cfg.to_dict()

        # Sync training.learning_rate to optimizer.learning_rate
        training = merged.get('training', {})
        if 'learning_rate' in training:
            merged['optimizer']['learning_rate'] = training['learning_rate']

        return merged

    def create_run_directory(self, cfg: dict, base_dir: Path = None) -> Path:
        """
        Create a timestamped run directory for experiment outputs.

        Args:
            cfg: Experiment config dict (from get_experiment_config or to_dict)
            base_dir: Base directory (defaults to project root)

        Returns:
            Path to the created run directory
        """
        if base_dir is None:
            base_dir = Path(__file__).parent

        output_cfg = cfg.get('output', {})
        output_base = base_dir / output_cfg.get('base_dir', 'output')

        # Create run directory with experiment name and timestamp
        exp_name = cfg.get('name', 'experiment')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = output_base / f"{exp_name}_{timestamp}"

        # Create subdirectories
        subdirs = output_cfg.get('subdirs', {})
        for subdir_name in subdirs.values():
            (run_dir / subdir_name).mkdir(parents=True, exist_ok=True)

        # Save config snapshot if enabled
        if output_cfg.get('save_config', True):
            config_path = run_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(cfg, f, indent=2)

        return run_dir


# Convenience function for simple usage
def load_experiment_config(experiment_name: str = 'part1', config_dir: str = None) -> dict:
    """
    Load a fully merged experiment configuration.

    Args:
        experiment_name: Name of experiment ('part0', 'part1', 'part2').
        config_dir: Ignored (kept for backwards compatibility).

    Returns:
        Fully resolved configuration dict.
    """
    loader = ConfigLoader(config_dir)
    return loader.get_experiment_config(experiment_name)
