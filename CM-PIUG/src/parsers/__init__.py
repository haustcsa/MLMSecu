"""
CM-PIUG 解析器模块
==================
提供多模态解析功能
"""

from .multimodal import (
    Modality,
    ParseResult,
    ImageOCRParser,
    AudioASRParser,
    PDFParser,
    MultimodalParser,
    parse_image,
    parse_audio,
    parse_pdf,
    parse_multimodal
)

__all__ = [
    "Modality",
    "ParseResult",
    "ImageOCRParser",
    "AudioASRParser",
    "PDFParser",
    "MultimodalParser",
    "parse_image",
    "parse_audio",
    "parse_pdf",
    "parse_multimodal"
]
