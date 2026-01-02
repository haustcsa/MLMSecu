# CM-PIUG Dockerfile
# 跨模态提示注入统一图框架
#
# 构建: docker build -t cm-piug .
# 运行: docker run -p 8000:8000 cm-piug

FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OCR依赖
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    tesseract-ocr-eng \
    # 图像处理依赖
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # 音频处理依赖
    ffmpeg \
    libsndfile1 \
    # 构建依赖
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt requirements-api.txt* ./

# 安装Python依赖
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install fastapi uvicorn python-multipart aiofiles

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 cmpiug && \
    chown -R cmpiug:cmpiug /app
USER cmpiug

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动API服务
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
