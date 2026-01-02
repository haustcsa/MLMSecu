# Detection module - Algorithm 1: Zero-shot detection
"""
Detection Module (Algorithm 1)
==============================

Implements the zero-shot cross-modal prompt injection detection algorithm.

Key Components:
- SemanticEquivalenceChecker: Checks bidirectional entailment (u ≈_C v)
- SemanticEntropyCalculator: Computes semantic entropy SE(x)
- CrossModalAligner: Aligns text/image/audio modalities
- ZeroShotDetector: Main detection pipeline
- CrossModalInjectionDetector: Real multimodal detection with OCR/ASR/Semantic models

Algorithm Steps:
1. Multimodal parsing (OCR/ASR)
2. Semantic reduction via Φ(u)
3. Rule closure: Closure(F(x) ∪ F_sys, R)
4. Edge confidence: c(u,v) = (1-λ)·c_rule + λ·c_sem
5. Risk calculation via BFS
6. Evidence chain backtracking

Complexity: O(|F|·|R|·k + |V|·|E|)
"""

from .semantic_equiv import (
    SemanticEquivalenceChecker,
    SemanticEntropyCalculator,
    CrossModalAligner,
)
from .zero_shot import ZeroShotDetector, detect_prompt_injection, DetectionResult

# 新增：真正的跨模态检测器
try:
    from .multimodal_detector import (
        CrossModalInjectionDetector,
        detect_multimodal,
        MultimodalDetectionResult,
        OCREngine,
        ASREngine,
        SemanticDetector,
    )
    _MULTIMODAL_AVAILABLE = True
except ImportError as e:
    _MULTIMODAL_AVAILABLE = False
    import logging
    logging.getLogger(__name__).debug(f"Multimodal detector not available: {e}")

__all__ = [
    "SemanticEquivalenceChecker",
    "SemanticEntropyCalculator", 
    "CrossModalAligner",
    "ZeroShotDetector",
    "detect_prompt_injection",
    "DetectionResult",
    # Multimodal
    "CrossModalInjectionDetector",
    "detect_multimodal",
    "MultimodalDetectionResult",
    "OCREngine",
    "ASREngine",
    "SemanticDetector",
    # Pretrained
    "PretrainedInjectionDetector",
    "HybridInjectionDetector",
]

# 预训练模型检测器
try:
    from .pretrained_detector import (
        PretrainedInjectionDetector,
        HybridInjectionDetector,
        PretrainedDetectionResult,
    )
    _PRETRAINED_AVAILABLE = True
except ImportError:
    _PRETRAINED_AVAILABLE = False
    PretrainedInjectionDetector = None
    HybridInjectionDetector = None
