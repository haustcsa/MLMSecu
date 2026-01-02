"""
Configuration Management
========================

Handles loading, validation, and seed management for CM-PIUG.
"""

import os
import random
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import yaml
import numpy as np


@dataclass
class SeedConfig:
    """Random seed configuration."""
    global_seed: int = 42
    numpy_seed: int = 42
    torch_seed: int = 42


@dataclass
class DetectionConfig:
    """Detection module configuration (Algorithm 1)."""
    risk_threshold: float = 0.5
    lambda_weight: float = 0.3
    max_path_depth: int = 20
    bfs_max_iterations: int = 1000
    bfs_early_stop: bool = True


@dataclass
class SemanticConfig:
    """Semantic equivalence configuration."""
    entailment_threshold: float = 0.7
    similarity_threshold: float = 0.85
    entropy_threshold: float = 0.5
    use_real_model: bool = False
    encoder_model: str = "all-MiniLM-L6-v2"
    nli_model: str = "facebook/bart-large-mnli"


@dataclass
class MFGConfig:
    """Mean Field Game configuration."""
    inner_tolerance: float = 1e-4
    max_inner_iterations: int = 100
    gamma: float = 0.95
    temperature: float = 0.5
    update_rate: float = 0.5


@dataclass
class StackelbergConfig:
    """Stackelberg game configuration."""
    outer_tolerance: float = 1e-3
    max_outer_iterations: int = 100
    learning_rate: float = 0.01


@dataclass
class DefenseConfig:
    """Defense module configuration (Algorithm 2)."""
    mfg: MFGConfig = field(default_factory=MFGConfig)
    stackelberg: StackelbergConfig = field(default_factory=StackelbergConfig)
    risk_weight: float = 1.0
    cost_weight: float = 0.3


@dataclass
class CMPIUGConfig:
    """Complete CM-PIUG configuration."""
    seed: SeedConfig = field(default_factory=SeedConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    defense: DefenseConfig = field(default_factory=DefenseConfig)
    system_prompt: str = ""
    system_facts: list = field(default_factory=list)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError otherwise
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required sections
    required_sections = ['seed', 'detection', 'semantic', 'defense']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate detection parameters
    det = config.get('detection', {})
    if not 0 <= det.get('risk_threshold', 0.5) <= 1:
        raise ValueError("risk_threshold must be in [0, 1]")
    if not 0 <= det.get('lambda_weight', 0.3) <= 1:
        raise ValueError("lambda_weight must be in [0, 1]")
    
    # Validate semantic parameters
    sem = config.get('semantic', {})
    if not 0 <= sem.get('entailment_threshold', 0.7) <= 1:
        raise ValueError("entailment_threshold must be in [0, 1]")
    
    # Validate defense parameters
    defense = config.get('defense', {})
    mfg = defense.get('mfg', {})
    if mfg.get('gamma', 0.95) <= 0 or mfg.get('gamma', 0.95) > 1:
        raise ValueError("gamma must be in (0, 1]")
    
    return True


def set_seeds(config: Dict[str, Any]) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        config: Configuration dictionary containing seed values
        
    Notes:
        Sets seeds for: Python random, NumPy, PyTorch (if available)
    """
    seed_config = config.get('seed', {})
    global_seed = seed_config.get('global_seed', 42)
    numpy_seed = seed_config.get('numpy_seed', 42)
    torch_seed = seed_config.get('torch_seed', 42)
    
    # Python random
    random.seed(global_seed)
    
    # NumPy
    np.random.seed(numpy_seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    print(f"[Config] Seeds set: global={global_seed}, numpy={numpy_seed}, torch={torch_seed}")


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration.
    
    Returns:
        Default configuration dictionary
    """
    return {
        'seed': {
            'global_seed': 42,
            'numpy_seed': 42,
            'torch_seed': 42,
        },
        'detection': {
            'risk_threshold': 0.5,
            'lambda_weight': 0.3,
            'max_path_depth': 20,
            'bfs': {
                'max_iterations': 1000,
                'early_stop': True,
            },
        },
        'semantic': {
            'entailment_threshold': 0.7,
            'similarity_threshold': 0.85,
            'entropy_threshold': 0.5,
            'use_real_model': False,
        },
        'defense': {
            'mfg': {
                'inner_tolerance': 1e-4,
                'max_inner_iterations': 100,
                'gamma': 0.95,
                'temperature': 0.5,
            },
            'stackelberg': {
                'outer_tolerance': 1e-3,
                'max_outer_iterations': 100,
                'learning_rate': 0.01,
            },
        },
        'system_prompt': 'You are a helpful AI assistant.',
        'system_facts': ['text_input_enabled'],
    }


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configurations.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
