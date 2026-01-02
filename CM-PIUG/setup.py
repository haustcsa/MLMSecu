#!/usr/bin/env python3
"""
CM-PIUG 安装配置
================
Cross-Modal Prompt Injection Unified Graph Framework
"""

from setuptools import setup, find_packages
from pathlib import Path

# 读取README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# 读取版本
version = "1.0.0"

# 核心依赖
install_requires = [
    "numpy>=1.21.0",
    "scipy>=1.7.0",
    "networkx>=2.6.0",
    "pyyaml>=6.0",
    "omegaconf>=2.1.0",
    "tqdm>=4.62.0",
]

# 可选依赖
extras_require = {
    # NLP功能
    "nlp": [
        "transformers>=4.20.0",
        "sentence-transformers>=2.2.0",
        "nltk>=3.6.0",
        "jieba>=0.42.0",
    ],
    # 深度学习
    "torch": [
        "torch>=1.10.0",
    ],
    # 多模态解析
    "multimodal": [
        "Pillow>=8.0.0",
        "pytesseract>=0.3.8",
        "opencv-python>=4.5.0",
        "librosa>=0.9.0",
        "soundfile>=0.10.0",
        "PyMuPDF>=1.20.0",
    ],
    # 优化求解
    "optimization": [
        "cvxpy>=1.2.0",
    ],
    # API服务
    "api": [
        "fastapi>=0.85.0",
        "uvicorn[standard]>=0.18.0",
        "python-multipart>=0.0.5",
        "aiofiles>=0.8.0",
        "pydantic>=1.10.0",
    ],
    # 可视化
    "viz": [
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "rich>=10.0.0",
    ],
    # 开发工具
    "dev": [
        "pytest>=7.0.0",
        "pytest-cov>=3.0.0",
        "pytest-asyncio>=0.18.0",
        "black>=22.0.0",
        "isort>=5.10.0",
        "flake8>=4.0.0",
        "mypy>=0.950",
    ],
}

# 完整安装
extras_require["all"] = list(set(
    dep for deps in extras_require.values() for dep in deps
))

setup(
    name="cm-piug",
    version=version,
    author="CM-PIUG Team",
    author_email="cmpiug@example.com",
    description="Cross-Modal Prompt Injection Unified Graph Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/cm-piug",
    project_urls={
        "Documentation": "https://cm-piug.readthedocs.io/",
        "Bug Tracker": "https://github.com/example/cm-piug/issues",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "examples.*"]),
    package_data={
        "cm_piug": [
            "configs/*.yaml",
            "data/rules/*.json",
            "data/examples/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points={
        "console_scripts": [
            "cmpiug=cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Security",
    ],
    keywords=[
        "prompt-injection",
        "llm-security",
        "multimodal",
        "attack-detection",
        "defense",
        "mean-field-game",
    ],
)
