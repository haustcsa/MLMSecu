"""
CM-PIUG Unified Attack Surface Graph
统一攻击面图核心模块

定义: G = (V, E, τ, R, F_init, G_goal)
- V: 节点集合
- E: 依赖边集合
- τ: 节点类型映射
- R: 原子操控/推理规则集合
- F_init: 初始事实节点集合
- G_goal: 安全违例目标节点集合
"""

import networkx as nx
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import copy

from .node_types import (
    NodeType, NodeAttribute, EdgeAttribute, 
    NodeTypeChecker, ATTACK_PATTERNS
)


@dataclass
class HornClause:
    """
    Horn子句规则定义
    形式: pre_1 ∧ pre_2 ∧ ... ∧ pre_n → post
    """
    rule_id: str
    preconditions: List[str]  # 前置条件节点ID列表
    postcondition: str        # 后置结论节点ID
    rule_type: str           # 规则类型
    confidence: float = 1.0  # 规则固有置信度
    description: str = ""
    
    def __repr__(self):
        prec_str = " ∧ ".join(self.preconditions)
        return f"[{self.rule_id}] {prec_str} → {self.postcondition}"


class UnifiedAttackGraph:
    """
    统一攻击面图 G = (V, E, τ, R, F_init, G_goal)
    
    基于攻击图的经典思想,将提示系统的安全分析对象抽象为统一攻击面图
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化统一攻击面图
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 底层图结构 (使用NetworkX)
        self._graph = nx.DiGraph()
        
        # 节点属性映射 V -> NodeAttribute
        self._node_attrs: Dict[str, NodeAttribute] = {}
        
        # 边属性映射 (u,v) -> EdgeAttribute
        self._edge_attrs: Dict[Tuple[str, str], EdgeAttribute] = {}
        
        # 规则库 R
        self._rules: Dict[str, HornClause] = {}
        
        # 初始事实集合 F_init
        self._initial_facts: Set[str] = set()
        
        # 安全目标集合 G_goal
        self._goal_nodes: Set[str] = set()
        
        # 激活节点集合 (当前输入激活的入口节点)
        self._activated_nodes: Set[str] = set()
        
        # 推理闭包缓存
        self._closure_cache: Optional[Set[str]] = None
        
    # ==================== 节点操作 ====================
    
    def add_node(self, node_attr: NodeAttribute) -> None:
        """添加节点"""
        node_id = node_attr.node_id
        self._graph.add_node(node_id)
        self._node_attrs[node_id] = node_attr
        
        # 自动分类到目标集合
        if NodeTypeChecker.is_goal_node(node_attr.node_type):
            self._goal_nodes.add(node_id)
            
        # 清除缓存
        self._closure_cache = None
    
    def get_node(self, node_id: str) -> Optional[NodeAttribute]:
        """获取节点属性"""
        return self._node_attrs.get(node_id)
    
    def get_nodes_by_type(self, node_type: NodeType) -> List[NodeAttribute]:
        """按类型获取节点"""
        return [attr for attr in self._node_attrs.values() 
                if attr.node_type == node_type]
    
    def remove_node(self, node_id: str) -> None:
        """删除节点"""
        if node_id in self._graph:
            self._graph.remove_node(node_id)
            self._node_attrs.pop(node_id, None)
            self._goal_nodes.discard(node_id)
            self._initial_facts.discard(node_id)
            self._activated_nodes.discard(node_id)
            self._closure_cache = None
    
    # ==================== 边操作 ====================
    
    def add_edge(self, edge_attr: EdgeAttribute) -> None:
        """
        添加依赖边
        若规则实例r以pre_1,...,pre_k推出post, 则添加边(pre_k, post)
        """
        src, tgt = edge_attr.source_id, edge_attr.target_id
        self._graph.add_edge(src, tgt)
        self._edge_attrs[(src, tgt)] = edge_attr
        self._closure_cache = None
    
    def get_edge(self, source: str, target: str) -> Optional[EdgeAttribute]:
        """获取边属性"""
        return self._edge_attrs.get((source, target))
    
    def get_predecessors(self, node_id: str) -> List[str]:
        """获取前驱节点"""
        return list(self._graph.predecessors(node_id))
    
    def get_successors(self, node_id: str) -> List[str]:
        """获取后继节点"""
        return list(self._graph.successors(node_id))
    
    # ==================== 规则操作 ====================
    
    def add_rule(self, rule: HornClause) -> None:
        """添加Horn子句规则"""
        self._rules[rule.rule_id] = rule
        self._closure_cache = None
    
    def load_rules_from_json(self, filepath: str) -> None:
        """从JSON文件加载规则库"""
        with open(filepath, 'r', encoding='utf-8') as f:
            rules_data = json.load(f)
        
        for rule_data in rules_data.get('rules', []):
            rule = HornClause(
                rule_id=rule_data['id'],
                preconditions=rule_data['preconditions'],
                postcondition=rule_data['postcondition'],
                rule_type=rule_data.get('type', 'generic'),
                confidence=rule_data.get('confidence', 1.0),
                description=rule_data.get('description', '')
            )
            self.add_rule(rule)
    
    def get_applicable_rules(self, fact_set: Set[str]) -> List[HornClause]:
        """获取可应用的规则(所有前置条件都在fact_set中)"""
        applicable = []
        for rule in self._rules.values():
            if all(pre in fact_set for pre in rule.preconditions):
                applicable.append(rule)
        return applicable
    
    # ==================== 初始事实与目标 ====================
    
    def set_initial_facts(self, facts: Set[str]) -> None:
        """设置初始事实集合 F_init"""
        self._initial_facts = facts.copy()
        self._closure_cache = None
    
    def add_initial_fact(self, node_id: str) -> None:
        """添加初始事实"""
        self._initial_facts.add(node_id)
        self._closure_cache = None
    
    def set_goal_nodes(self, goals: Set[str]) -> None:
        """设置目标节点集合 G_goal"""
        self._goal_nodes = goals.copy()
    
    def set_activated_nodes(self, activated: Set[str]) -> None:
        """设置当前输入激活的节点"""
        self._activated_nodes = activated.copy()
        self._closure_cache = None
    
    # ==================== 规则推理 (Forward Chaining) ====================
    
    def compute_forward_closure(self, 
                                 initial_facts: Optional[Set[str]] = None
                                 ) -> Tuple[Set[str], List[Tuple[str, HornClause]]]:
        """
        前向链式推理闭包
        
        实现: Closure(F(x) ∪ F_sys, R)
        通过对规则库R进行前向推理直至不动点
        
        Args:
            initial_facts: 初始事实集合,默认使用 F_init ∪ activated_nodes
        
        Returns:
            (闭包事实集合, 推导轨迹列表[(derived_fact, applied_rule)])
        """
        if initial_facts is None:
            initial_facts = self._initial_facts | self._activated_nodes
        
        # 当前事实集合
        current_facts = initial_facts.copy()
        
        # 推导轨迹 (用于生成依赖边)
        derivation_trace: List[Tuple[str, HornClause]] = []
        
        # 迭代直至不动点
        changed = True
        while changed:
            changed = False
            for rule in self._rules.values():
                # 检查所有前置条件是否满足
                if all(pre in current_facts for pre in rule.preconditions):
                    post = rule.postcondition
                    if post not in current_facts:
                        current_facts.add(post)
                        derivation_trace.append((post, rule))
                        changed = True
        
        self._closure_cache = current_facts
        return current_facts, derivation_trace
    
    def instantiate_attack_subgraph(self,
                                    input_facts: Set[str],
                                    system_facts: Set[str]
                                    ) -> 'UnifiedAttackGraph':
        """
        实例化输入对应的攻击子图
        
        G_x = Instantiate(Closure(F(x) ∪ F_sys, R))
        
        Args:
            input_facts: 输入事实集合 F(x)
            system_facts: 系统配置事实集合 F_sys
        
        Returns:
            实例化的攻击子图
        """
        # 合并事实
        all_facts = input_facts | system_facts
        
        # 计算闭包
        closure, derivations = self.compute_forward_closure(all_facts)
        
        # 创建子图
        subgraph = UnifiedAttackGraph(self.config)
        
        # 添加闭包中的节点
        for fact_id in closure:
            if fact_id in self._node_attrs:
                subgraph.add_node(copy.deepcopy(self._node_attrs[fact_id]))
        
        # 根据推导轨迹添加依赖边
        for derived_fact, rule in derivations:
            for pre in rule.preconditions:
                edge_attr = EdgeAttribute(
                    source_id=pre,
                    target_id=derived_fact,
                    rule_id=rule.rule_id,
                    rule_confidence=rule.confidence
                )
                subgraph.add_edge(edge_attr)
        
        # 设置激活节点
        subgraph.set_activated_nodes(input_facts)
        
        # 继承目标节点
        subgraph._goal_nodes = self._goal_nodes & closure
        
        return subgraph
    
    # ==================== 可达性分析 (BFS) ====================
    
    def bfs_reachability(self, 
                         start_nodes: Set[str],
                         target_nodes: Set[str]
                         ) -> Tuple[bool, List[List[str]]]:
        """
        BFS可达性检测
        
        检测从start_nodes是否可达target_nodes中的任意节点
        
        Args:
            start_nodes: 起始节点集合
            target_nodes: 目标节点集合
        
        Returns:
            (是否可达, 所有可达路径列表)
        """
        from collections import deque
        
        reachable_paths = []
        
        for start in start_nodes:
            if start not in self._graph:
                continue
                
            # BFS
            queue = deque([(start, [start])])
            visited = {start}
            
            while queue:
                current, path = queue.popleft()
                
                # 检查是否到达目标
                if current in target_nodes:
                    reachable_paths.append(path)
                    continue  # 继续搜索其他路径
                
                # 扩展后继
                for neighbor in self._graph.successors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, path + [neighbor]))
        
        is_reachable = len(reachable_paths) > 0
        return is_reachable, reachable_paths
    
    def find_all_paths_to_goals(self, max_depth: int = 20) -> List[List[str]]:
        """
        查找从激活节点到目标节点的所有路径
        
        Args:
            max_depth: 最大搜索深度
        
        Returns:
            路径列表
        """
        all_paths = []
        
        for start in self._activated_nodes:
            for goal in self._goal_nodes:
                if start in self._graph and goal in self._graph:
                    try:
                        paths = list(nx.all_simple_paths(
                            self._graph, start, goal, cutoff=max_depth
                        ))
                        all_paths.extend(paths)
                    except nx.NetworkXNoPath:
                        continue
        
        return all_paths
    
    # ==================== 风险计算 ====================
    
    def compute_path_strength(self, path: List[str]) -> float:
        """
        计算路径强度
        
        Strength(π) = ∏_{(u,v)∈π} c(u,v)
        
        Args:
            path: 节点ID路径
        
        Returns:
            路径强度值
        """
        if len(path) < 2:
            return 0.0
        
        strength = 1.0
        for i in range(len(path) - 1):
            edge_attr = self.get_edge(path[i], path[i+1])
            if edge_attr:
                strength *= edge_attr.edge_confidence
            else:
                # 如果边不存在,强度为0
                return 0.0
        
        return strength
    
    def compute_risk_score(self) -> Tuple[float, List[str]]:
        """
        计算风险分数
        
        Risk(x) = max_{t∈G_goal} max_{π:A→t} Strength(π)
        
        Returns:
            (风险分数, 最强证据链)
        """
        all_paths = self.find_all_paths_to_goals()
        
        if not all_paths:
            return 0.0, []
        
        max_strength = 0.0
        best_path = []
        
        for path in all_paths:
            strength = self.compute_path_strength(path)
            if strength > max_strength:
                max_strength = strength
                best_path = path
        
        return max_strength, best_path
    
    # ==================== 图统计与导出 ====================
    
    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()
    
    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()
    
    @property
    def num_rules(self) -> int:
        return len(self._rules)
    
    def get_node_type_distribution(self) -> Dict[NodeType, int]:
        """获取节点类型分布"""
        dist = defaultdict(int)
        for attr in self._node_attrs.values():
            dist[attr.node_type] += 1
        return dict(dist)
    
    def to_dict(self) -> Dict[str, Any]:
        """导出为字典"""
        return {
            'nodes': [
                {
                    'id': attr.node_id,
                    'type': attr.node_type.name,
                    'content': attr.content,
                    'confidence': attr.confidence
                }
                for attr in self._node_attrs.values()
            ],
            'edges': [
                {
                    'source': src,
                    'target': tgt,
                    'rule_id': attr.rule_id,
                    'confidence': attr.edge_confidence
                }
                for (src, tgt), attr in self._edge_attrs.items()
            ],
            'rules': [
                {
                    'id': rule.rule_id,
                    'preconditions': rule.preconditions,
                    'postcondition': rule.postcondition,
                    'type': rule.rule_type
                }
                for rule in self._rules.values()
            ],
            'initial_facts': list(self._initial_facts),
            'goal_nodes': list(self._goal_nodes),
            'activated_nodes': list(self._activated_nodes)
        }
    
    def export_to_json(self, filepath: str) -> None:
        """导出为JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    def __repr__(self):
        return (f"UnifiedAttackGraph(nodes={self.num_nodes}, "
                f"edges={self.num_edges}, rules={self.num_rules})")


class AttackGraphBuilder:
    """
    攻击图构建器
    
    提供便捷的攻击图构建方法
    """
    
    def __init__(self):
        self.graph = UnifiedAttackGraph()
        self._node_counter = 0
    
    def _gen_node_id(self, prefix: str = "node") -> str:
        """生成唯一节点ID"""
        self._node_counter += 1
        return f"{prefix}_{self._node_counter}"
    
    def add_input_node(self, 
                       content: str,
                       modality: str = "text",
                       confidence: float = 1.0) -> str:
        """添加输入节点"""
        type_map = {
            "text": NodeType.INPUT_TEXT,
            "image": NodeType.INPUT_IMAGE,
            "audio": NodeType.INPUT_AUDIO,
            "structured": NodeType.INPUT_STRUCTURED
        }
        
        node_id = self._gen_node_id("input")
        attr = NodeAttribute(
            node_id=node_id,
            node_type=type_map.get(modality, NodeType.INPUT_TEXT),
            content=content,
            confidence=confidence,
            source_modality=modality
        )
        self.graph.add_node(attr)
        return node_id
    
    def add_goal_node(self, 
                      goal_type: str,
                      description: str = "") -> str:
        """添加目标节点"""
        type_map = {
            "privilege_escalation": NodeType.GOAL_PRIVILEGE_ESCALATION,
            "data_exfiltration": NodeType.GOAL_DATA_EXFILTRATION,
            "policy_bypass": NodeType.GOAL_POLICY_BYPASS,
            "task_hijacking": NodeType.GOAL_TASK_HIJACKING,
            "unsafe_content": NodeType.GOAL_UNSAFE_CONTENT
        }
        
        node_id = self._gen_node_id("goal")
        attr = NodeAttribute(
            node_id=node_id,
            node_type=type_map.get(goal_type, NodeType.GOAL_TASK_HIJACKING),
            content=description
        )
        self.graph.add_node(attr)
        return node_id
    
    def add_control_node(self,
                         control_type: str,
                         content: str = "") -> str:
        """添加控制节点"""
        type_map = {
            "instruction": NodeType.CONTROL_INSTRUCTION,
            "privilege": NodeType.CONTROL_PRIVILEGE,
            "boundary": NodeType.CONTROL_BOUNDARY
        }
        
        node_id = self._gen_node_id("control")
        attr = NodeAttribute(
            node_id=node_id,
            node_type=type_map.get(control_type, NodeType.CONTROL_INSTRUCTION),
            content=content
        )
        self.graph.add_node(attr)
        return node_id
    
    def add_dependency(self,
                       source: str,
                       target: str,
                       rule_id: str,
                       confidence: float = 1.0) -> None:
        """添加依赖边"""
        edge_attr = EdgeAttribute(
            source_id=source,
            target_id=target,
            rule_id=rule_id,
            rule_confidence=confidence
        )
        self.graph.add_edge(edge_attr)
    
    def build(self) -> UnifiedAttackGraph:
        """构建并返回攻击图"""
        return self.graph
