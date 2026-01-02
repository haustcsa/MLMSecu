"""
CM-PIUG Node Types Definition
节点类型定义模块

定义统一攻击面图中的节点类型τ: V → Types
"""

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set


class NodeType(Enum):
    """
    节点类型枚举
    用于区分输入语义单元、提示解析中间量、任务执行/工具调用、关键控制点与安全目标
    """
    # 输入层节点
    INPUT_TEXT = auto()           # 原生文本输入
    INPUT_IMAGE = auto()          # 图像输入(含OCR提取文本)
    INPUT_AUDIO = auto()          # 音频输入(含ASR转写)
    INPUT_STRUCTURED = auto()     # 结构化输入(JSON/XML等)
    
    # 解析层节点
    PARSE_SYSTEM_PROMPT = auto()  # 系统提示解析结果
    PARSE_USER_QUERY = auto()     # 用户查询解析
    PARSE_CONTEXT = auto()        # 上下文信息
    PARSE_TOOL_SCHEMA = auto()    # 工具Schema解析
    
    # 控制流节点
    CONTROL_INSTRUCTION = auto()  # 指令控制点
    CONTROL_PRIVILEGE = auto()    # 权限控制点
    CONTROL_BOUNDARY = auto()     # 边界控制点
    
    # 执行层节点
    EXEC_TOOL_CALL = auto()       # 工具调用
    EXEC_API_REQUEST = auto()     # API请求
    EXEC_FILE_ACCESS = auto()     # 文件访问
    EXEC_NETWORK = auto()         # 网络操作
    
    # 安全目标节点 (G_goal)
    GOAL_PRIVILEGE_ESCALATION = auto()  # 越权
    GOAL_DATA_EXFILTRATION = auto()     # 数据泄露
    GOAL_POLICY_BYPASS = auto()         # 策略绕过
    GOAL_TASK_HIJACKING = auto()        # 任务劫持
    GOAL_UNSAFE_CONTENT = auto()        # 不安全内容生成


@dataclass
class NodeAttribute:
    """节点属性"""
    # 基础属性
    node_id: str
    node_type: NodeType
    content: str = ""
    
    # 语义属性
    semantic_embedding: Optional[List[float]] = None
    semantic_cluster_id: Optional[int] = None
    
    # 置信度属性
    confidence: float = 1.0          # 节点置信度
    semantic_reliability: float = 1.0  # 语义归约可靠度 ρ_sem
    
    # 来源属性
    source_modality: str = "text"    # 来源模态
    extraction_method: str = "direct" # 提取方法
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.node_id)
    
    def __eq__(self, other):
        if isinstance(other, NodeAttribute):
            return self.node_id == other.node_id
        return False


@dataclass
class EdgeAttribute:
    """边属性"""
    # 基础属性
    source_id: str
    target_id: str
    rule_id: str  # 触发该边的规则ID
    
    # 置信度属性
    rule_confidence: float = 1.0      # 规则证据强度 c_rule
    semantic_confidence: float = 1.0  # 语义归约置信度 c_sem
    weight_lambda: float = 0.3        # 权重参数 λ
    
    # 计算属性
    @property
    def edge_confidence(self) -> float:
        """
        边置信度计算: c(u,v) = (1-λ)·c_rule + λ·c_sem
        """
        return (1 - self.weight_lambda) * self.rule_confidence + \
               self.weight_lambda * self.semantic_confidence
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class NodeTypeChecker:
    """节点类型检查器"""
    
    # 输入层节点类型集合
    INPUT_TYPES: Set[NodeType] = {
        NodeType.INPUT_TEXT,
        NodeType.INPUT_IMAGE,
        NodeType.INPUT_AUDIO,
        NodeType.INPUT_STRUCTURED
    }
    
    # 安全目标节点类型集合
    GOAL_TYPES: Set[NodeType] = {
        NodeType.GOAL_PRIVILEGE_ESCALATION,
        NodeType.GOAL_DATA_EXFILTRATION,
        NodeType.GOAL_POLICY_BYPASS,
        NodeType.GOAL_TASK_HIJACKING,
        NodeType.GOAL_UNSAFE_CONTENT
    }
    
    # 控制节点类型集合
    CONTROL_TYPES: Set[NodeType] = {
        NodeType.CONTROL_INSTRUCTION,
        NodeType.CONTROL_PRIVILEGE,
        NodeType.CONTROL_BOUNDARY
    }
    
    # 执行节点类型集合
    EXEC_TYPES: Set[NodeType] = {
        NodeType.EXEC_TOOL_CALL,
        NodeType.EXEC_API_REQUEST,
        NodeType.EXEC_FILE_ACCESS,
        NodeType.EXEC_NETWORK
    }
    
    @classmethod
    def is_input_node(cls, node_type: NodeType) -> bool:
        return node_type in cls.INPUT_TYPES
    
    @classmethod
    def is_goal_node(cls, node_type: NodeType) -> bool:
        return node_type in cls.GOAL_TYPES
    
    @classmethod
    def is_control_node(cls, node_type: NodeType) -> bool:
        return node_type in cls.CONTROL_TYPES
    
    @classmethod
    def is_exec_node(cls, node_type: NodeType) -> bool:
        return node_type in cls.EXEC_TYPES


# 预定义的安全规则模式
ATTACK_PATTERNS = {
    "instruction_override": {
        "description": "指令覆盖攻击",
        "trigger_types": [NodeType.INPUT_TEXT, NodeType.INPUT_IMAGE],
        "target_type": NodeType.CONTROL_INSTRUCTION,
        "goal_type": NodeType.GOAL_TASK_HIJACKING
    },
    "context_pollution": {
        "description": "上下文污染",
        "trigger_types": [NodeType.INPUT_TEXT, NodeType.INPUT_IMAGE],
        "target_type": NodeType.PARSE_CONTEXT,
        "goal_type": NodeType.GOAL_POLICY_BYPASS
    },
    "tool_param_manipulation": {
        "description": "工具参数操纵",
        "trigger_types": [NodeType.PARSE_USER_QUERY],
        "target_type": NodeType.EXEC_TOOL_CALL,
        "goal_type": NodeType.GOAL_PRIVILEGE_ESCALATION
    },
    "privilege_boundary_bypass": {
        "description": "权限边界绕过",
        "trigger_types": [NodeType.CONTROL_PRIVILEGE],
        "target_type": NodeType.EXEC_API_REQUEST,
        "goal_type": NodeType.GOAL_DATA_EXFILTRATION
    }
}
