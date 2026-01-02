#!/usr/bin/env python3
"""
CM-PIUG API 服务器
==================
基于FastAPI的RESTful API服务，提供检测和防御功能

启动方式:
    uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload
    
或者:
    python -m api.server
"""

import os
import sys
import time
import uuid
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
except ImportError:
    print("请安装FastAPI: pip install fastapi uvicorn")
    sys.exit(1)

from src.detection.zero_shot import ZeroShotDetector
from src.defense.stackelberg_mfg import StackelbergMFGSolver, DefenseActionLibrary
from src.utils.logging import setup_logger

# 全局变量
detector: Optional[ZeroShotDetector] = None
solver: Optional[StackelbergMFGSolver] = None
logger = setup_logger("cm-piug-api")

# ==================== 数据模型 ====================

class DetectionRequest(BaseModel):
    """检测请求"""
    text: str = Field(..., description="要检测的文本内容")
    context: str = Field(default="你是一个AI助手", description="系统上下文")
    image_text: Optional[str] = Field(None, description="图像中提取的文本（OCR结果）")
    audio_text: Optional[str] = Field(None, description="音频中提取的文本（ASR结果）")
    threshold: float = Field(default=0.5, ge=0, le=1, description="风险阈值")
    include_defense: bool = Field(default=True, description="是否返回防御建议")

class DetectionResponse(BaseModel):
    """检测响应"""
    request_id: str
    timestamp: str
    is_attack: bool
    risk_score: float
    risk_level: str
    evidence_chain: List[Dict[str, Any]]
    fired_rules: List[str]
    defense_action: Optional[Dict[str, Any]] = None
    processing_time_ms: float

class BatchDetectionRequest(BaseModel):
    """批量检测请求"""
    items: List[DetectionRequest]
    parallel: bool = Field(default=True, description="是否并行处理")

class BatchDetectionResponse(BaseModel):
    """批量检测响应"""
    total: int
    attack_count: int
    results: List[DetectionResponse]
    total_time_ms: float

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    uptime_seconds: float
    detector_ready: bool
    solver_ready: bool

class DefenseActionResponse(BaseModel):
    """防御动作响应"""
    actions: List[Dict[str, Any]]
    categories: List[str]

class AnalysisRequest(BaseModel):
    """分析请求"""
    text: str
    context: str = "你是一个AI助手"
    detailed: bool = Field(default=False, description="是否返回详细分析")

class AnalysisResponse(BaseModel):
    """分析响应"""
    request_id: str
    semantic_features: Dict[str, Any]
    rule_analysis: Dict[str, Any]
    graph_info: Dict[str, Any]
    recommendations: List[str]

# ==================== 应用初始化 ====================

start_time = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global detector, solver
    
    # 启动时初始化
    logger.info("正在初始化CM-PIUG服务...")
    
    try:
        detector = ZeroShotDetector()
        solver = StackelbergMFGSolver()
        logger.info("服务初始化完成")
    except Exception as e:
        logger.error(f"初始化失败: {e}")
        raise
    
    yield
    
    # 关闭时清理
    logger.info("服务正在关闭...")
    detector = None
    solver = None

app = FastAPI(
    title="CM-PIUG API",
    description="跨模态提示注入统一图框架 - RESTful API服务",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 辅助函数 ====================

def get_risk_level(score: float) -> str:
    """获取风险等级"""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MEDIUM"
    elif score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def format_evidence_chain(chain) -> List[Dict[str, Any]]:
    """格式化证据链"""
    if not chain:
        return []
    return [
        {
            "source": edge.source,
            "target": edge.target,
            "relation": edge.relation,
            "confidence": edge.confidence,
            "rule_id": edge.rule_id
        }
        for edge in chain
    ]

async def process_detection(request: DetectionRequest) -> DetectionResponse:
    """处理单个检测请求"""
    request_id = str(uuid.uuid4())[:8]
    start = time.time()
    
    # 合并多模态文本
    combined_text = request.text
    if request.image_text:
        combined_text += f"\n[IMAGE_TEXT]: {request.image_text}"
    if request.audio_text:
        combined_text += f"\n[AUDIO_TEXT]: {request.audio_text}"
    
    # 执行检测
    result = detector.detect({
        "text": combined_text,
        "context": request.context
    })
    
    # 获取防御建议
    defense_action = None
    if request.include_defense and (result.flag or result.risk_score >= request.threshold):
        action_id, prob = solver.online_match(
            evidence_chain=result.evidence_chain,
            risk_score=result.risk_score,
            fired_rules=result.fired_rules
        )
        action_library = DefenseActionLibrary()
        action = action_library.get_action(action_id)
        defense_action = {
            "action_id": action_id,
            "probability": prob,
            **action
        }
    
    processing_time = (time.time() - start) * 1000
    
    return DetectionResponse(
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        is_attack=result.flag or result.risk_score >= request.threshold,
        risk_score=result.risk_score,
        risk_level=get_risk_level(result.risk_score),
        evidence_chain=format_evidence_chain(result.evidence_chain),
        fired_rules=result.fired_rules or [],
        defense_action=defense_action,
        processing_time_ms=processing_time
    )

# ==================== API路由 ====================

@app.get("/", response_model=Dict[str, str])
async def root():
    """根路径"""
    return {
        "service": "CM-PIUG API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse(
        status="healthy" if detector and solver else "degraded",
        version="1.0.0",
        uptime_seconds=time.time() - start_time,
        detector_ready=detector is not None,
        solver_ready=solver is not None
    )

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """
    检测单个输入
    
    对输入文本进行提示注入检测，返回风险评估和防御建议
    """
    if not detector:
        raise HTTPException(status_code=503, detail="检测器未就绪")
    
    try:
        return await process_detection(request)
    except Exception as e:
        logger.error(f"检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def batch_detect(request: BatchDetectionRequest):
    """
    批量检测
    
    并行处理多个检测请求
    """
    if not detector:
        raise HTTPException(status_code=503, detail="检测器未就绪")
    
    start = time.time()
    
    try:
        if request.parallel:
            # 并行处理
            tasks = [process_detection(item) for item in request.items]
            results = await asyncio.gather(*tasks)
        else:
            # 串行处理
            results = []
            for item in request.items:
                result = await process_detection(item)
                results.append(result)
        
        attack_count = sum(1 for r in results if r.is_attack)
        total_time = (time.time() - start) * 1000
        
        return BatchDetectionResponse(
            total=len(results),
            attack_count=attack_count,
            results=results,
            total_time_ms=total_time
        )
    except Exception as e:
        logger.error(f"批量检测失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(request: AnalysisRequest):
    """
    详细分析
    
    对输入进行深度分析，返回语义特征、规则分析和图信息
    """
    if not detector:
        raise HTTPException(status_code=503, detail="检测器未就绪")
    
    request_id = str(uuid.uuid4())[:8]
    
    try:
        # 执行检测
        result = detector.detect({
            "text": request.text,
            "context": request.context
        })
        
        # 构建分析响应
        semantic_features = {
            "text_length": len(request.text),
            "has_instruction_markers": any(
                marker in request.text.lower() 
                for marker in ["ignore", "忽略", "system", "系统", "admin", "管理员"]
            ),
            "special_tokens": [
                token for token in ["[", "]", "<", ">", "{{", "}}"]
                if token in request.text
            ],
            "risk_indicators": []
        }
        
        # 添加风险指标
        if "忽略" in request.text or "ignore" in request.text.lower():
            semantic_features["risk_indicators"].append("instruction_override")
        if "系统" in request.text or "system" in request.text.lower():
            semantic_features["risk_indicators"].append("system_reference")
        if "执行" in request.text or "execute" in request.text.lower():
            semantic_features["risk_indicators"].append("execution_intent")
        
        rule_analysis = {
            "total_rules_checked": len(detector.rule_engine.rules) if hasattr(detector, 'rule_engine') else 0,
            "rules_fired": len(result.fired_rules) if result.fired_rules else 0,
            "fired_rule_ids": result.fired_rules or [],
            "rule_categories": list(set(
                rule_id.split("_")[1] if "_" in rule_id else "unknown"
                for rule_id in (result.fired_rules or [])
            ))
        }
        
        graph_info = {
            "nodes_created": len(result.evidence_chain) + 1 if result.evidence_chain else 0,
            "edges_created": len(result.evidence_chain) if result.evidence_chain else 0,
            "max_path_strength": result.risk_score,
            "goal_reached": result.flag
        }
        
        # 生成建议
        recommendations = []
        if result.flag:
            recommendations.append("建议拒绝或过滤此输入")
            if result.risk_score >= 0.8:
                recommendations.append("高风险输入，建议启用隔离模式")
            if "tool" in str(result.fired_rules).lower():
                recommendations.append("检测到工具调用风险，建议启用沙箱执行")
        else:
            recommendations.append("输入看起来是安全的")
            if result.risk_score > 0.3:
                recommendations.append("建议持续监控此类输入")
        
        return AnalysisResponse(
            request_id=request_id,
            semantic_features=semantic_features,
            rule_analysis=rule_analysis,
            graph_info=graph_info,
            recommendations=recommendations
        )
    except Exception as e:
        logger.error(f"分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/defense/actions", response_model=DefenseActionResponse)
async def get_defense_actions():
    """
    获取所有防御动作
    
    返回可用的防御动作库
    """
    library = DefenseActionLibrary()
    actions = library.get_all_actions()
    categories = list(set(a.get("category", "other") for a in actions))
    
    return DefenseActionResponse(
        actions=actions,
        categories=categories
    )

@app.get("/defense/actions/{category}")
async def get_actions_by_category(category: str):
    """
    按类别获取防御动作
    """
    library = DefenseActionLibrary()
    actions = library.get_actions_by_category(category)
    
    if not actions:
        raise HTTPException(status_code=404, detail=f"类别 '{category}' 不存在")
    
    return {"category": category, "actions": actions}

@app.post("/defense/recommend")
async def recommend_defense(
    risk_score: float = 0.5,
    attack_type: Optional[str] = None
):
    """
    推荐防御策略
    
    根据风险分数和攻击类型推荐防御动作
    """
    if not solver:
        raise HTTPException(status_code=503, detail="求解器未就绪")
    
    # 模拟证据链和触发规则
    fired_rules = []
    if attack_type:
        fired_rules.append(f"R_{attack_type.upper()}_001")
    
    action_id, prob = solver.online_match(
        evidence_chain=[],
        risk_score=risk_score,
        fired_rules=fired_rules
    )
    
    library = DefenseActionLibrary()
    action = library.get_action(action_id)
    
    return {
        "recommended_action": action_id,
        "confidence": prob,
        "action_details": action,
        "risk_level": get_risk_level(risk_score)
    }

@app.get("/stats")
async def get_stats():
    """
    获取服务统计信息
    """
    return {
        "uptime_seconds": time.time() - start_time,
        "version": "1.0.0",
        "detector_status": "ready" if detector else "not_ready",
        "solver_status": "ready" if solver else "not_ready",
        "supported_modalities": ["text", "image_ocr", "audio_asr"],
        "api_endpoints": [
            "/detect",
            "/detect/batch",
            "/analyze",
            "/defense/actions",
            "/defense/recommend"
        ]
    }

# ==================== 错误处理 ====================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )

# ==================== 主入口 ====================

if __name__ == "__main__":
    import uvicorn
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                  CM-PIUG API Server                       ║
    ║                                                           ║
    ║   启动中...                                               ║
    ║   API文档: http://localhost:8000/docs                     ║
    ║   ReDoc: http://localhost:8000/redoc                      ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
