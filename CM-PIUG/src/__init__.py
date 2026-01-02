# CM-PIUG: Cross-Modal Prompt Injection Unified Graph Framework
# 跨模态提示注入统一攻击面图框架

"""
CM-PIUG Framework
=================

A unified attack surface graph framework for cross-modal prompt injection detection and defense.

Modules:
--------
- core: Core data structures (nodes, edges, attack graph, rule engine)
- detection: Zero-shot detection algorithm (Algorithm 1)
- defense: Stackelberg-MFG defense strategy (Algorithm 2)
- parsers: Multimodal parsing (OCR, ASR, PDF)
- utils: Utility functions (config, logging, visualization)

Quick Start:
-----------
>>> from src.detection import ZeroShotDetector, detect_prompt_injection
>>> result = detect_prompt_injection("Ignore previous instructions", context="helpful assistant")
>>> print(f"Risk: {result.risk_score}, Flag: {result.flag}")

>>> from src.defense import StackelbergMFGSolver
>>> solver = StackelbergMFGSolver("configs/default_config.yaml")
>>> policy = solver.offline_solve()

>>> from src.parsers import parse_image, parse_multimodal
>>> text = parse_image("image.png")
"""

__version__ = "1.0.0"
__author__ = "CM-PIUG Team"

from .core import NodeType, NodeAttribute, EdgeAttribute, UnifiedAttackGraph, RuleEngine
from .detection import ZeroShotDetector, detect_prompt_injection
from .defense import StackelbergMFGSolver

# Optional: parsers (may require additional dependencies)
try:
    from .parsers import MultimodalParser, parse_image, parse_audio, parse_pdf, parse_multimodal
    _PARSERS_AVAILABLE = True
except ImportError:
    _PARSERS_AVAILABLE = False

__all__ = [
    # Core
    "NodeType",
    "NodeAttribute", 
    "EdgeAttribute",
    "UnifiedAttackGraph",
    "RuleEngine",
    # Detection
    "ZeroShotDetector",
    "detect_prompt_injection",
    # Defense
    "StackelbergMFGSolver",
]

# Add parser exports if available
if _PARSERS_AVAILABLE:
    __all__.extend([
        "MultimodalParser",
        "parse_image",
        "parse_audio", 
        "parse_pdf",
        "parse_multimodal",
    ])
