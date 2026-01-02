"""
CM-PIUG 多模态解析器
====================
提供图像OCR和音频ASR解析功能，支持从非文本模态中提取潜在的注入内容

支持的模态:
- 图像: PNG, JPG, JPEG, GIF, BMP, WEBP
- 音频: WAV, MP3, FLAC, OGG
- PDF文档: PDF
"""

import os
import io
import base64
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class Modality(Enum):
    """模态类型"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    PDF = "pdf"
    VIDEO = "video"


@dataclass
class ParseResult:
    """解析结果"""
    modality: Modality
    text: str
    confidence: float
    metadata: Dict[str, Any]
    raw_data: Optional[bytes] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality.value,
            "text": self.text,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class BaseParser(ABC):
    """解析器基类"""
    
    @abstractmethod
    def parse(self, data: Union[bytes, str, np.ndarray]) -> ParseResult:
        """解析输入数据"""
        pass
    
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """返回支持的格式列表"""
        pass


class ImageOCRParser(BaseParser):
    """
    图像OCR解析器
    
    使用pytesseract进行OCR识别，支持多语言
    """
    
    def __init__(
        self,
        lang: str = "chi_sim+eng",
        use_gpu: bool = False,
        confidence_threshold: float = 0.5
    ):
        self.lang = lang
        self.use_gpu = use_gpu
        self.confidence_threshold = confidence_threshold
        self._tesseract_available = self._check_tesseract()
        self._easyocr_available = self._check_easyocr()
    
    def _check_tesseract(self) -> bool:
        """检查tesseract是否可用"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _check_easyocr(self) -> bool:
        """检查easyocr是否可用"""
        try:
            import easyocr
            return True
        except ImportError:
            return False
    
    def supported_formats(self) -> List[str]:
        return ["png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"]
    
    def parse(self, data: Union[bytes, str, np.ndarray, Path]) -> ParseResult:
        """
        解析图像并提取文本
        
        Args:
            data: 图像数据 (bytes, 文件路径, 或numpy数组)
            
        Returns:
            ParseResult: 解析结果
        """
        # 加载图像
        image = self._load_image(data)
        
        # 预处理
        processed = self._preprocess(image)
        
        # OCR识别
        text, confidence, details = self._ocr(processed)
        
        return ParseResult(
            modality=Modality.IMAGE,
            text=text,
            confidence=confidence,
            metadata={
                "image_size": image.shape[:2] if hasattr(image, 'shape') else None,
                "ocr_engine": "tesseract" if self._tesseract_available else "easyocr",
                "language": self.lang,
                "details": details
            }
        )
    
    def _load_image(self, data: Union[bytes, str, np.ndarray, Path]) -> np.ndarray:
        """加载图像"""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("请安装Pillow: pip install Pillow")
        
        if isinstance(data, np.ndarray):
            return data
        elif isinstance(data, (str, Path)):
            img = Image.open(data)
            return np.array(img)
        elif isinstance(data, bytes):
            img = Image.open(io.BytesIO(data))
            return np.array(img)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """图像预处理"""
        try:
            import cv2
        except ImportError:
            # 如果没有cv2，返回原图
            return image
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 降噪
        denoised = cv2.fastNlMeansDenoising(binary)
        
        return denoised
    
    def _ocr(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """执行OCR"""
        if self._tesseract_available:
            return self._ocr_tesseract(image)
        elif self._easyocr_available:
            return self._ocr_easyocr(image)
        else:
            # 回退到模拟OCR
            return self._ocr_fallback(image)
    
    def _ocr_tesseract(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """使用tesseract进行OCR"""
        import pytesseract
        from PIL import Image
        
        pil_image = Image.fromarray(image)
        
        # 获取详细结果
        data = pytesseract.image_to_data(
            pil_image, 
            lang=self.lang, 
            output_type=pytesseract.Output.DICT
        )
        
        # 提取文本和置信度
        texts = []
        confidences = []
        
        for i, conf in enumerate(data['conf']):
            if conf > 0:  # -1 表示无效
                text = data['text'][i].strip()
                if text:
                    texts.append(text)
                    confidences.append(conf / 100.0)
        
        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_conf, {
            "word_count": len(texts),
            "word_confidences": confidences[:10]  # 只保留前10个
        }
    
    def _ocr_easyocr(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """使用easyocr进行OCR"""
        import easyocr
        
        reader = easyocr.Reader(['ch_sim', 'en'], gpu=self.use_gpu)
        results = reader.readtext(image)
        
        texts = []
        confidences = []
        
        for bbox, text, conf in results:
            if conf >= self.confidence_threshold:
                texts.append(text)
                confidences.append(conf)
        
        full_text = " ".join(texts)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        
        return full_text, avg_conf, {
            "detection_count": len(results),
            "filtered_count": len(texts)
        }
    
    def _ocr_fallback(self, image: np.ndarray) -> Tuple[str, float, Dict]:
        """回退OCR (模拟)"""
        logger.warning("OCR引擎不可用，使用模拟模式")
        return "", 0.0, {"engine": "fallback", "message": "No OCR engine available"}


class AudioASRParser(BaseParser):
    """
    音频ASR解析器
    
    使用whisper或其他ASR引擎进行语音识别
    """
    
    def __init__(
        self,
        model_name: str = "base",
        language: str = "zh",
        device: str = "cpu"
    ):
        self.model_name = model_name
        self.language = language
        self.device = device
        self._whisper_available = self._check_whisper()
        self._model = None
    
    def _check_whisper(self) -> bool:
        """检查whisper是否可用"""
        try:
            import whisper
            return True
        except ImportError:
            return False
    
    def supported_formats(self) -> List[str]:
        return ["wav", "mp3", "flac", "ogg", "m4a"]
    
    def parse(self, data: Union[bytes, str, Path]) -> ParseResult:
        """
        解析音频并提取文本
        
        Args:
            data: 音频数据 (bytes或文件路径)
            
        Returns:
            ParseResult: 解析结果
        """
        # 加载音频
        audio_path = self._prepare_audio(data)
        
        # ASR识别
        text, confidence, details = self._transcribe(audio_path)
        
        return ParseResult(
            modality=Modality.AUDIO,
            text=text,
            confidence=confidence,
            metadata={
                "asr_engine": "whisper" if self._whisper_available else "fallback",
                "model": self.model_name,
                "language": self.language,
                "details": details
            }
        )
    
    def _prepare_audio(self, data: Union[bytes, str, Path]) -> str:
        """准备音频文件"""
        if isinstance(data, (str, Path)):
            return str(data)
        elif isinstance(data, bytes):
            # 保存到临时文件
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(data)
                return f.name
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def _transcribe(self, audio_path: str) -> Tuple[str, float, Dict]:
        """执行语音识别"""
        if self._whisper_available:
            return self._transcribe_whisper(audio_path)
        else:
            return self._transcribe_fallback(audio_path)
    
    def _transcribe_whisper(self, audio_path: str) -> Tuple[str, float, Dict]:
        """使用whisper进行语音识别"""
        import whisper
        
        if self._model is None:
            self._model = whisper.load_model(self.model_name, device=self.device)
        
        result = self._model.transcribe(
            audio_path,
            language=self.language,
            fp16=False
        )
        
        text = result["text"]
        
        # 计算平均置信度
        segments = result.get("segments", [])
        if segments:
            avg_prob = sum(s.get("avg_logprob", 0) for s in segments) / len(segments)
            confidence = min(1.0, max(0.0, 1.0 + avg_prob / 5))  # 转换为0-1范围
        else:
            confidence = 0.5
        
        return text, confidence, {
            "segments": len(segments),
            "duration": result.get("duration", 0)
        }
    
    def _transcribe_fallback(self, audio_path: str) -> Tuple[str, float, Dict]:
        """回退ASR (模拟)"""
        logger.warning("ASR引擎不可用，使用模拟模式")
        return "", 0.0, {"engine": "fallback", "message": "No ASR engine available"}


class PDFParser(BaseParser):
    """
    PDF解析器
    
    提取PDF中的文本内容
    """
    
    def __init__(self, extract_images: bool = True):
        self.extract_images = extract_images
        self._ocr_parser = ImageOCRParser() if extract_images else None
    
    def supported_formats(self) -> List[str]:
        return ["pdf"]
    
    def parse(self, data: Union[bytes, str, Path]) -> ParseResult:
        """
        解析PDF并提取文本
        
        Args:
            data: PDF数据 (bytes或文件路径)
            
        Returns:
            ParseResult: 解析结果
        """
        try:
            import fitz  # PyMuPDF
            return self._parse_pymupdf(data)
        except ImportError:
            pass
        
        try:
            import pdfplumber
            return self._parse_pdfplumber(data)
        except ImportError:
            pass
        
        return self._parse_fallback(data)
    
    def _parse_pymupdf(self, data: Union[bytes, str, Path]) -> ParseResult:
        """使用PyMuPDF解析"""
        import fitz
        
        if isinstance(data, bytes):
            doc = fitz.open(stream=data, filetype="pdf")
        else:
            doc = fitz.open(str(data))
        
        texts = []
        image_texts = []
        
        for page_num, page in enumerate(doc):
            # 提取文本
            text = page.get_text()
            texts.append(text)
            
            # 提取图像中的文本
            if self.extract_images and self._ocr_parser:
                for img_index, img in enumerate(page.get_images()):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        if pix.n >= 4:  # RGBA
                            pix = fitz.Pixmap(fitz.csRGB, pix)
                        img_data = pix.tobytes("png")
                        ocr_result = self._ocr_parser.parse(img_data)
                        if ocr_result.text:
                            image_texts.append(ocr_result.text)
                    except Exception as e:
                        logger.warning(f"图像OCR失败: {e}")
        
        doc.close()
        
        full_text = "\n".join(texts)
        if image_texts:
            full_text += "\n[IMAGE_TEXT]\n" + "\n".join(image_texts)
        
        return ParseResult(
            modality=Modality.PDF,
            text=full_text,
            confidence=0.9,
            metadata={
                "engine": "pymupdf",
                "page_count": len(texts),
                "image_count": len(image_texts)
            }
        )
    
    def _parse_pdfplumber(self, data: Union[bytes, str, Path]) -> ParseResult:
        """使用pdfplumber解析"""
        import pdfplumber
        
        if isinstance(data, bytes):
            pdf = pdfplumber.open(io.BytesIO(data))
        else:
            pdf = pdfplumber.open(str(data))
        
        texts = []
        for page in pdf.pages:
            text = page.extract_text() or ""
            texts.append(text)
        
        pdf.close()
        
        full_text = "\n".join(texts)
        
        return ParseResult(
            modality=Modality.PDF,
            text=full_text,
            confidence=0.85,
            metadata={
                "engine": "pdfplumber",
                "page_count": len(texts)
            }
        )
    
    def _parse_fallback(self, data: Union[bytes, str, Path]) -> ParseResult:
        """回退解析"""
        logger.warning("PDF解析器不可用")
        return ParseResult(
            modality=Modality.PDF,
            text="",
            confidence=0.0,
            metadata={"engine": "fallback", "message": "No PDF parser available"}
        )


class MultimodalParser:
    """
    统一多模态解析器
    
    自动检测输入模态并调用相应的解析器
    """
    
    def __init__(self):
        self.parsers = {
            Modality.IMAGE: ImageOCRParser(),
            Modality.AUDIO: AudioASRParser(),
            Modality.PDF: PDFParser()
        }
    
    def parse(
        self,
        data: Union[bytes, str, Path, np.ndarray],
        modality: Optional[Modality] = None
    ) -> ParseResult:
        """
        解析输入数据
        
        Args:
            data: 输入数据
            modality: 模态类型 (可选，自动检测)
            
        Returns:
            ParseResult: 解析结果
        """
        if modality is None:
            modality = self._detect_modality(data)
        
        if modality == Modality.TEXT:
            return ParseResult(
                modality=Modality.TEXT,
                text=str(data) if not isinstance(data, bytes) else data.decode('utf-8', errors='ignore'),
                confidence=1.0,
                metadata={"source": "direct"}
            )
        
        parser = self.parsers.get(modality)
        if parser is None:
            raise ValueError(f"不支持的模态: {modality}")
        
        return parser.parse(data)
    
    def _detect_modality(self, data: Union[bytes, str, Path, np.ndarray]) -> Modality:
        """自动检测模态"""
        if isinstance(data, np.ndarray):
            return Modality.IMAGE
        
        if isinstance(data, (str, Path)):
            path = Path(data)
            if path.exists():
                ext = path.suffix.lower().lstrip('.')
                return self._ext_to_modality(ext)
            else:
                return Modality.TEXT
        
        if isinstance(data, bytes):
            # 通过魔数检测
            return self._detect_by_magic(data)
        
        return Modality.TEXT
    
    def _ext_to_modality(self, ext: str) -> Modality:
        """扩展名转模态"""
        image_exts = {"png", "jpg", "jpeg", "gif", "bmp", "webp", "tiff"}
        audio_exts = {"wav", "mp3", "flac", "ogg", "m4a"}
        
        if ext in image_exts:
            return Modality.IMAGE
        elif ext in audio_exts:
            return Modality.AUDIO
        elif ext == "pdf":
            return Modality.PDF
        else:
            return Modality.TEXT
    
    def _detect_by_magic(self, data: bytes) -> Modality:
        """通过文件魔数检测"""
        # PNG
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            return Modality.IMAGE
        # JPEG
        if data[:2] == b'\xff\xd8':
            return Modality.IMAGE
        # GIF
        if data[:6] in (b'GIF87a', b'GIF89a'):
            return Modality.IMAGE
        # PDF
        if data[:4] == b'%PDF':
            return Modality.PDF
        # WAV
        if data[:4] == b'RIFF' and data[8:12] == b'WAVE':
            return Modality.AUDIO
        # MP3
        if data[:3] == b'ID3' or data[:2] == b'\xff\xfb':
            return Modality.AUDIO
        
        return Modality.TEXT
    
    def parse_multimodal_input(
        self,
        text: Optional[str] = None,
        image: Optional[Union[bytes, str, Path, np.ndarray]] = None,
        audio: Optional[Union[bytes, str, Path]] = None,
        pdf: Optional[Union[bytes, str, Path]] = None
    ) -> Dict[str, ParseResult]:
        """
        解析多模态输入
        
        Args:
            text: 文本输入
            image: 图像输入
            audio: 音频输入
            pdf: PDF输入
            
        Returns:
            Dict[str, ParseResult]: 各模态的解析结果
        """
        results = {}
        
        if text:
            results["text"] = ParseResult(
                modality=Modality.TEXT,
                text=text,
                confidence=1.0,
                metadata={"source": "direct"}
            )
        
        if image is not None:
            results["image"] = self.parsers[Modality.IMAGE].parse(image)
        
        if audio is not None:
            results["audio"] = self.parsers[Modality.AUDIO].parse(audio)
        
        if pdf is not None:
            results["pdf"] = self.parsers[Modality.PDF].parse(pdf)
        
        return results
    
    def merge_results(self, results: Dict[str, ParseResult]) -> str:
        """
        合并多模态解析结果
        
        Args:
            results: 各模态的解析结果
            
        Returns:
            str: 合并后的文本
        """
        texts = []
        
        # 按优先级排序: text > image > audio > pdf
        priority = ["text", "image", "audio", "pdf"]
        
        for key in priority:
            if key in results and results[key].text:
                result = results[key]
                if key == "text":
                    texts.append(result.text)
                else:
                    texts.append(f"[{key.upper()}_CONTENT]: {result.text}")
        
        return "\n".join(texts)


# 便捷函数
def parse_image(image_data: Union[bytes, str, Path, np.ndarray], lang: str = "chi_sim+eng") -> str:
    """解析图像中的文本"""
    parser = ImageOCRParser(lang=lang)
    result = parser.parse(image_data)
    return result.text


def parse_audio(audio_data: Union[bytes, str, Path], language: str = "zh") -> str:
    """解析音频中的文本"""
    parser = AudioASRParser(language=language)
    result = parser.parse(audio_data)
    return result.text


def parse_pdf(pdf_data: Union[bytes, str, Path], extract_images: bool = True) -> str:
    """解析PDF中的文本"""
    parser = PDFParser(extract_images=extract_images)
    result = parser.parse(pdf_data)
    return result.text


def parse_multimodal(
    text: Optional[str] = None,
    image: Optional[Union[bytes, str, Path]] = None,
    audio: Optional[Union[bytes, str, Path]] = None,
    pdf: Optional[Union[bytes, str, Path]] = None
) -> str:
    """解析多模态输入并合并"""
    parser = MultimodalParser()
    results = parser.parse_multimodal_input(text=text, image=image, audio=audio, pdf=pdf)
    return parser.merge_results(results)
