"""
CM-PIUG Rule Engine
规则引擎模块

实现基于Horn子句的前向链式推理(Forward Chaining)
用于从系统流水线到依赖结构的规则驱动图生成
"""

import re
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import json
from collections import defaultdict

from .node_types import NodeType, NodeAttribute
from .attack_graph import HornClause, UnifiedAttackGraph, EdgeAttribute


class RuleCategory(Enum):
    """规则类别"""
    INPUT_PROCESSING = auto()      # 输入处理规则
    PROMPT_PARSING = auto()        # 提示解析规则
    CONTEXT_MANIPULATION = auto()  # 上下文操控规则
    TOOL_INVOCATION = auto()       # 工具调用规则
    PRIVILEGE_CONTROL = auto()     # 权限控制规则
    GOAL_ACHIEVEMENT = auto()      # 目标达成规则


@dataclass
class Rule:
    """
    简单规则类
    
    用于基础的前向链推理
    """
    rule_id: str
    preconditions: List[str]
    postcondition: str
    confidence: float = 1.0
    description: str = ""


@dataclass
class ParameterizedRule:
    """
    参数化规则模板
    
    支持参数化的前置条件和后置结论
    例如: input_contains(X, "ignore") ∧ context(X, C) → instruction_override(X, C)
    """
    rule_id: str
    category: RuleCategory
    
    # 参数化前置条件模板
    precondition_templates: List[str]
    
    # 参数化后置结论模板
    postcondition_template: str
    
    # 参数绑定约束
    parameter_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # 规则置信度
    base_confidence: float = 1.0
    
    # 描述
    description: str = ""
    
    def instantiate(self, 
                    bindings: Dict[str, str],
                    confidence_modifier: float = 1.0) -> HornClause:
        """
        用参数绑定实例化规则
        
        Args:
            bindings: 参数名 -> 实际值的映射
            confidence_modifier: 置信度修正因子
        
        Returns:
            实例化的HornClause
        """
        # 替换前置条件中的参数
        instantiated_prec = []
        for template in self.precondition_templates:
            inst = template
            for param, value in bindings.items():
                inst = inst.replace(f"${{{param}}}", value)
            instantiated_prec.append(inst)
        
        # 替换后置结论中的参数
        inst_post = self.postcondition_template
        for param, value in bindings.items():
            inst_post = inst_post.replace(f"${{{param}}}", value)
        
        return HornClause(
            rule_id=f"{self.rule_id}_inst_{hash(tuple(bindings.items())) % 10000}",
            preconditions=instantiated_prec,
            postcondition=inst_post,
            rule_type=self.category.name,
            confidence=self.base_confidence * confidence_modifier,
            description=self.description
        )


class RuleEngine:
    """
    规则引擎
    
    管理规则库并执行前向链式推理
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 原子规则库 (HornClause)
        self._rules: Dict[str, HornClause] = {}
        
        # 简单规则库 (Rule)
        self._simple_rules: List[Rule] = []
        
        # 参数化规则模板库
        self._rule_templates: Dict[str, ParameterizedRule] = {}
        
        # 规则按类别索引
        self._rules_by_category: Dict[RuleCategory, List[str]] = defaultdict(list)
        
        # 推理统计
        self._inference_stats = {
            'total_iterations': 0,
            'rules_fired': 0,
            'facts_derived': 0
        }
        
        # 初始化内置规则
        self._init_builtin_rules()
    
    @property
    def rules(self) -> List[Rule]:
        """返回所有简单规则"""
        return self._simple_rules
    
    def _init_builtin_rules(self):
        """初始化内置规则库"""
        # 内置的跨模态提示注入相关规则
        builtin_rules = [
            # ===== 输入处理规则 =====
            ParameterizedRule(
                rule_id="R_INPUT_001",
                category=RuleCategory.INPUT_PROCESSING,
                precondition_templates=["input_text_${id}"],
                postcondition_template="parsed_content_${id}",
                base_confidence=0.95,
                description="文本输入解析"
            ),
            ParameterizedRule(
                rule_id="R_INPUT_002",
                category=RuleCategory.INPUT_PROCESSING,
                precondition_templates=["input_image_${id}", "ocr_enabled"],
                postcondition_template="extracted_text_${id}",
                base_confidence=0.85,
                description="图像OCR文本提取"
            ),
            ParameterizedRule(
                rule_id="R_INPUT_003",
                category=RuleCategory.INPUT_PROCESSING,
                precondition_templates=["input_audio_${id}", "asr_enabled"],
                postcondition_template="transcribed_text_${id}",
                base_confidence=0.80,
                description="音频ASR转写"
            ),
            
            # ===== 提示解析规则 =====
            ParameterizedRule(
                rule_id="R_PARSE_001",
                category=RuleCategory.PROMPT_PARSING,
                precondition_templates=["parsed_content_${id}", "contains_instruction_marker"],
                postcondition_template="detected_instruction_${id}",
                base_confidence=0.90,
                description="检测指令标记"
            ),
            ParameterizedRule(
                rule_id="R_PARSE_002",
                category=RuleCategory.PROMPT_PARSING,
                precondition_templates=["extracted_text_${id}", "contains_instruction_marker"],
                postcondition_template="detected_instruction_${id}",
                base_confidence=0.85,
                description="从提取文本检测指令"
            ),
            
            # ===== 上下文操控规则 =====
            ParameterizedRule(
                rule_id="R_CTX_001",
                category=RuleCategory.CONTEXT_MANIPULATION,
                precondition_templates=[
                    "detected_instruction_${id}",
                    "instruction_type_override"
                ],
                postcondition_template="context_pollution_${id}",
                base_confidence=0.88,
                description="上下文污染-指令覆盖"
            ),
            ParameterizedRule(
                rule_id="R_CTX_002",
                category=RuleCategory.CONTEXT_MANIPULATION,
                precondition_templates=[
                    "parsed_content_${id}",
                    "contains_jailbreak_pattern"
                ],
                postcondition_template="jailbreak_attempt_${id}",
                base_confidence=0.92,
                description="越狱尝试检测"
            ),
            ParameterizedRule(
                rule_id="R_CTX_003",
                category=RuleCategory.CONTEXT_MANIPULATION,
                precondition_templates=[
                    "detected_instruction_${id}",
                    "instruction_type_roleplay"
                ],
                postcondition_template="roleplay_injection_${id}",
                base_confidence=0.85,
                description="角色扮演注入"
            ),
            
            # ===== 工具调用规则 =====
            ParameterizedRule(
                rule_id="R_TOOL_001",
                category=RuleCategory.TOOL_INVOCATION,
                precondition_templates=[
                    "context_pollution_${id}",
                    "tool_available_${tool}"
                ],
                postcondition_template="unauthorized_tool_call_${id}_${tool}",
                base_confidence=0.80,
                description="未授权工具调用"
            ),
            ParameterizedRule(
                rule_id="R_TOOL_002",
                category=RuleCategory.TOOL_INVOCATION,
                precondition_templates=[
                    "jailbreak_attempt_${id}",
                    "sensitive_api_${api}"
                ],
                postcondition_template="sensitive_api_access_${id}_${api}",
                base_confidence=0.85,
                description="敏感API访问"
            ),
            
            # ===== 权限控制规则 =====
            ParameterizedRule(
                rule_id="R_PRIV_001",
                category=RuleCategory.PRIVILEGE_CONTROL,
                precondition_templates=[
                    "unauthorized_tool_call_${id}_${tool}",
                    "privilege_level_${level}"
                ],
                postcondition_template="privilege_escalation_attempt_${id}",
                base_confidence=0.90,
                description="权限提升尝试"
            ),
            
            # ===== 目标达成规则 =====
            ParameterizedRule(
                rule_id="R_GOAL_001",
                category=RuleCategory.GOAL_ACHIEVEMENT,
                precondition_templates=["privilege_escalation_attempt_${id}"],
                postcondition_template="GOAL_privilege_escalation",
                base_confidence=0.95,
                description="达成目标:权限提升"
            ),
            ParameterizedRule(
                rule_id="R_GOAL_002",
                category=RuleCategory.GOAL_ACHIEVEMENT,
                precondition_templates=["sensitive_api_access_${id}_${api}"],
                postcondition_template="GOAL_data_exfiltration",
                base_confidence=0.92,
                description="达成目标:数据泄露"
            ),
            ParameterizedRule(
                rule_id="R_GOAL_003",
                category=RuleCategory.GOAL_ACHIEVEMENT,
                precondition_templates=["jailbreak_attempt_${id}"],
                postcondition_template="GOAL_policy_bypass",
                base_confidence=0.88,
                description="达成目标:策略绕过"
            ),
            ParameterizedRule(
                rule_id="R_GOAL_004",
                category=RuleCategory.GOAL_ACHIEVEMENT,
                precondition_templates=["roleplay_injection_${id}"],
                postcondition_template="GOAL_task_hijacking",
                base_confidence=0.85,
                description="达成目标:任务劫持"
            ),
        ]
        
        for rule in builtin_rules:
            self.add_rule_template(rule)
        
        # 添加简单规则（不需要参数化）
        simple_rules = [
            # 指令标记检测规则
            Rule(
                rule_id="R_SIMPLE_001",
                preconditions=["contains_instruction_marker"],
                postcondition="instruction_injection_detected",
                confidence=0.85,
                description="检测到指令标记"
            ),
            Rule(
                rule_id="R_SIMPLE_002",
                preconditions=["contains_jailbreak_pattern"],
                postcondition="jailbreak_attempt_detected",
                confidence=0.90,
                description="检测到越狱模式"
            ),
            Rule(
                rule_id="R_SIMPLE_003",
                preconditions=["instruction_override_attempt"],
                postcondition="instruction_injection_detected",
                confidence=0.88,
                description="检测到指令覆盖尝试"
            ),
            Rule(
                rule_id="R_SIMPLE_004",
                preconditions=["prompt_extraction_attempt"],
                postcondition="data_extraction_attempt",
                confidence=0.85,
                description="检测到提示词提取尝试"
            ),
            Rule(
                rule_id="R_SIMPLE_005",
                preconditions=["code_execution_attempt"],
                postcondition="dangerous_operation_detected",
                confidence=0.92,
                description="检测到代码执行尝试"
            ),
            # 组合规则
            Rule(
                rule_id="R_SIMPLE_006",
                preconditions=["instruction_injection_detected", "instruction_type_override"],
                postcondition="confirmed_injection_attack",
                confidence=0.90,
                description="确认注入攻击"
            ),
            Rule(
                rule_id="R_SIMPLE_007",
                preconditions=["jailbreak_attempt_detected"],
                postcondition="confirmed_jailbreak",
                confidence=0.88,
                description="确认越狱攻击"
            ),
            # 目标达成规则
            Rule(
                rule_id="R_GOAL_SIMPLE_001",
                preconditions=["confirmed_injection_attack"],
                postcondition="GOAL_policy_bypass",
                confidence=0.85,
                description="注入攻击达成策略绕过目标"
            ),
            Rule(
                rule_id="R_GOAL_SIMPLE_002",
                preconditions=["confirmed_jailbreak"],
                postcondition="GOAL_policy_bypass",
                confidence=0.90,
                description="越狱攻击达成策略绕过目标"
            ),
            Rule(
                rule_id="R_GOAL_SIMPLE_003",
                preconditions=["data_extraction_attempt"],
                postcondition="GOAL_data_exfiltration",
                confidence=0.82,
                description="数据提取达成目标"
            ),
            Rule(
                rule_id="R_GOAL_SIMPLE_004",
                preconditions=["dangerous_operation_detected"],
                postcondition="GOAL_privilege_escalation",
                confidence=0.88,
                description="危险操作达成权限提升目标"
            ),
        ]
        
        for rule in simple_rules:
            self.add_rule(rule)
    
    def add_rule(self, rule) -> None:
        """
        添加规则
        
        支持 Rule 和 HornClause 类型
        """
        if isinstance(rule, Rule):
            # 简单规则
            self._simple_rules.append(rule)
        elif isinstance(rule, HornClause):
            # HornClause 规则
            self._rules[rule.rule_id] = rule
            try:
                category = RuleCategory[rule.rule_type]
                self._rules_by_category[category].append(rule.rule_id)
            except (KeyError, ValueError):
                pass
        else:
            raise TypeError(f"不支持的规则类型: {type(rule)}")
    
    def get_applicable_rules(self, facts: Set[str]) -> List[Rule]:
        """
        获取可应用的规则
        
        返回所有前提条件被满足的规则
        """
        applicable = []
        for rule in self._simple_rules:
            if all(pre in facts for pre in rule.preconditions):
                applicable.append(rule)
        return applicable
    
    def add_rule_template(self, template: ParameterizedRule) -> None:
        """添加参数化规则模板"""
        self._rule_templates[template.rule_id] = template
        self._rules_by_category[template.category].append(template.rule_id)
    
    def instantiate_rules(self, 
                          input_id: str,
                          context: Dict[str, Any]) -> List[HornClause]:
        """
        根据输入和上下文实例化规则
        
        Args:
            input_id: 输入标识符
            context: 上下文信息(包含可用工具、API、权限等)
        
        Returns:
            实例化的规则列表
        """
        instantiated = []
        
        for template in self._rule_templates.values():
            # 基础绑定
            bindings = {"id": input_id}
            
            # 根据上下文扩展绑定
            if "tools" in context:
                for tool in context["tools"]:
                    tool_bindings = {**bindings, "tool": tool}
                    instantiated.append(template.instantiate(tool_bindings))
            
            if "apis" in context:
                for api in context["apis"]:
                    api_bindings = {**bindings, "api": api}
                    instantiated.append(template.instantiate(api_bindings))
            
            if "privilege_level" in context:
                level_bindings = {**bindings, "level": str(context["privilege_level"])}
                instantiated.append(template.instantiate(level_bindings))
            
            # 默认实例化
            instantiated.append(template.instantiate(bindings))
        
        return instantiated
    
    def forward_chain(self,
                      initial_facts: Set[str],
                      rules = None,
                      max_iterations: int = 100,
                      return_derivations: bool = False
                      ) -> Tuple[Set[str], List]:
        """
        前向链式推理
        
        实现: Closure(F, R) - 迭代应用规则直至不动点
        
        Args:
            initial_facts: 初始事实集合
            rules: 规则列表,默认使用所有已加载规则
            max_iterations: 最大迭代次数
            return_derivations: 是否返回完整推导轨迹
        
        Returns:
            如果 return_derivations=False: (闭包事实集合, 触发的规则ID列表)
            如果 return_derivations=True: (闭包事实集合, [(derived_fact, rule, supporting_facts)])
        """
        # 合并所有可用规则
        all_rules = []
        if rules is not None:
            all_rules = list(rules)
        else:
            # 使用 HornClause 规则
            all_rules.extend(self._rules.values())
            # 使用简单 Rule 规则
            all_rules.extend(self._simple_rules)
        
        current_facts = initial_facts.copy()
        fired_rules = []
        derivation_trace = []
        
        self._inference_stats['total_iterations'] = 0
        
        for iteration in range(max_iterations):
            self._inference_stats['total_iterations'] += 1
            new_facts_found = False
            
            for rule in all_rules:
                # 检查所有前置条件是否满足
                if self._check_preconditions(rule.preconditions, current_facts):
                    post = rule.postcondition
                    if post not in current_facts:
                        current_facts.add(post)
                        
                        # 记录触发的规则ID
                        fired_rules.append(rule.rule_id)
                        
                        # 记录推导轨迹
                        supporting = [p for p in rule.preconditions if p in current_facts]
                        derivation_trace.append((post, rule, supporting))
                        
                        new_facts_found = True
                        self._inference_stats['rules_fired'] += 1
                        self._inference_stats['facts_derived'] += 1
            
            # 不动点:没有新事实被推导
            if not new_facts_found:
                break
        
        if return_derivations:
            return current_facts, derivation_trace
        else:
            return current_facts, fired_rules
    
    def _check_preconditions(self, 
                              preconditions: List[str],
                              facts: Set[str]) -> bool:
        """
        检查前置条件是否满足
        
        支持精确匹配和模式匹配
        """
        for prec in preconditions:
            # 精确匹配
            if prec in facts:
                continue
            
            # 模式匹配(支持通配符)
            pattern = prec.replace("*", ".*")
            matched = any(re.match(f"^{pattern}$", f) for f in facts)
            
            if not matched:
                return False
        
        return True
    
    def build_dependency_graph(self,
                               initial_facts: Set[str],
                               context: Dict[str, Any],
                               input_id: str
                               ) -> UnifiedAttackGraph:
        """
        构建依赖图
        
        从初始事实和规则推理生成攻击图的依赖结构
        
        Args:
            initial_facts: 初始事实集合
            context: 上下文信息
            input_id: 输入标识符
        
        Returns:
            构建的UnifiedAttackGraph
        """
        # 实例化规则
        rules = self.instantiate_rules(input_id, context)
        
        # 添加已有的原子规则 (HornClause)
        rules.extend(self._rules.values())
        
        # 添加简单规则 (Rule)
        rules.extend(self._simple_rules)
        
        # 前向推理
        closure, derivations = self.forward_chain(initial_facts, rules, return_derivations=True)
        
        # 构建图
        graph = UnifiedAttackGraph()
        
        # 添加所有事实节点
        for fact in closure:
            node_type = self._infer_node_type(fact)
            attr = NodeAttribute(
                node_id=fact,
                node_type=node_type,
                content=fact
            )
            graph.add_node(attr)
        
        # 根据推导轨迹添加边
        for derived_fact, rule, supporting_facts in derivations:
            for supp in supporting_facts:
                edge_attr = EdgeAttribute(
                    source_id=supp,
                    target_id=derived_fact,
                    rule_id=rule.rule_id,
                    rule_confidence=rule.confidence
                )
                graph.add_edge(edge_attr)
            
            # 将规则添加到图
            graph.add_rule(rule)
        
        # 设置激活节点
        graph.set_activated_nodes(initial_facts)
        
        return graph
    
    def _infer_node_type(self, fact_id: str) -> NodeType:
        """根据事实ID推断节点类型"""
        fact_lower = fact_id.lower()
        
        # 目标节点
        if fact_lower.startswith("goal_"):
            if "privilege" in fact_lower:
                return NodeType.GOAL_PRIVILEGE_ESCALATION
            elif "data" in fact_lower or "exfiltration" in fact_lower:
                return NodeType.GOAL_DATA_EXFILTRATION
            elif "policy" in fact_lower or "bypass" in fact_lower:
                return NodeType.GOAL_POLICY_BYPASS
            elif "task" in fact_lower or "hijack" in fact_lower:
                return NodeType.GOAL_TASK_HIJACKING
            else:
                return NodeType.GOAL_UNSAFE_CONTENT
        
        # 输入节点
        if "input_" in fact_lower:
            if "image" in fact_lower:
                return NodeType.INPUT_IMAGE
            elif "audio" in fact_lower:
                return NodeType.INPUT_AUDIO
            else:
                return NodeType.INPUT_TEXT
        
        # 解析节点
        if "parsed" in fact_lower or "extracted" in fact_lower or "detected" in fact_lower:
            return NodeType.PARSE_USER_QUERY
        
        # 控制节点
        if "instruction" in fact_lower or "override" in fact_lower:
            return NodeType.CONTROL_INSTRUCTION
        if "privilege" in fact_lower:
            return NodeType.CONTROL_PRIVILEGE
        
        # 执行节点
        if "tool_call" in fact_lower or "unauthorized_tool" in fact_lower:
            return NodeType.EXEC_TOOL_CALL
        if "api" in fact_lower:
            return NodeType.EXEC_API_REQUEST
        
        # 默认
        return NodeType.PARSE_CONTEXT
    
    def load_rules_from_json(self, filepath: str) -> None:
        """从JSON文件加载规则"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for rule_data in data.get('rules', []):
            rule = HornClause(
                rule_id=rule_data['id'],
                preconditions=rule_data['preconditions'],
                postcondition=rule_data['postcondition'],
                rule_type=rule_data.get('type', 'generic'),
                confidence=rule_data.get('confidence', 1.0),
                description=rule_data.get('description', '')
            )
            self.add_rule(rule)
    
    def export_rules_to_json(self, filepath: str) -> None:
        """导出规则到JSON文件"""
        rules_data = {
            'rules': [
                {
                    'id': rule.rule_id,
                    'preconditions': rule.preconditions,
                    'postcondition': rule.postcondition,
                    'type': rule.rule_type,
                    'confidence': rule.confidence,
                    'description': rule.description
                }
                for rule in self._rules.values()
            ],
            'templates': [
                {
                    'id': t.rule_id,
                    'category': t.category.name,
                    'precondition_templates': t.precondition_templates,
                    'postcondition_template': t.postcondition_template,
                    'confidence': t.base_confidence,
                    'description': t.description
                }
                for t in self._rule_templates.values()
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(rules_data, f, indent=2, ensure_ascii=False)
    
    def get_inference_stats(self) -> Dict[str, int]:
        """获取推理统计信息"""
        return self._inference_stats.copy()
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self._inference_stats = {
            'total_iterations': 0,
            'rules_fired': 0,
            'facts_derived': 0
        }
