#!/usr/bin/env python3
"""
CM-PIUG Core Module
Cross-Modal Prompt Injection Unified Modeling

Core components:
- UnifiedAttackSurfaceGraph: Attack graph G = (V, E, τ, R, F_init, G_goal)
- ZeroShotDetector: Algorithm 1 - Open-world zero-shot detection
- SemanticEquivalenceChecker: Cross-modal semantic alignment
- SemanticEntropyCalculator: Uncertainty quantification for OCR/ASR
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import time


class NodeType(Enum):
    """Node types τ(v) in the unified attack surface graph"""
    # Input semantic units
    INPUT_TEXT = "input_text"
    INPUT_IMAGE = "input_image"
    INPUT_AUDIO = "input_audio"
    INPUT_GRAPH = "input_graph"
    
    # Parsing intermediates
    PARSE_OCR_TEXT = "parse_ocr_text"
    PARSE_ASR_TEXT = "parse_asr_text"
    PARSE_SEMANTIC_UNIT = "parse_semantic_unit"
    PARSE_INSTRUCTION = "parse_instruction"
    
    # Control points
    CONTROL_MAIN_INSTRUCTION = "control_main_instruction"
    CONTROL_CONTEXT_WINDOW = "control_context_window"
    CONTROL_PRIVILEGE_BOUNDARY = "control_privilege_boundary"
    CONTROL_SAFETY_FILTER = "control_safety_filter"
    
    # Execution layer
    EXEC_TOOL_INVOKE = "exec_tool_invoke"
    EXEC_API_CALL = "exec_api_call"
    EXEC_FILE_ACCESS = "exec_file_access"
    EXEC_OUTPUT_GENERATION = "exec_output_generation"
    
    # Security violation goals (G_goal)
    GOAL_PRIVILEGE_ESCALATION = "goal_privilege_escalation"
    GOAL_DATA_EXFILTRATION = "goal_data_exfiltration"
    GOAL_POLICY_BYPASS = "goal_policy_bypass"
    GOAL_TASK_HIJACKING = "goal_task_hijacking"
    GOAL_UNSAFE_CONTENT = "goal_unsafe_content"


@dataclass
class NodeAttribute:
    """Node attributes in the attack graph"""
    node_id: str
    node_type: NodeType
    content: str
    confidence: float = 1.0
    source_modality: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_goal_node(self) -> bool:
        return "goal" in self.node_type.value


@dataclass
class EdgeAttribute:
    """Edge attributes: c(u,v) = α·ρ_rule + (1-α)·ρ_sem"""
    source_id: str
    target_id: str
    rule_id: str
    rule_confidence: float = 1.0
    semantic_confidence: float = 1.0
    alpha: float = 0.7
    
    @property
    def edge_confidence(self) -> float:
        return self.alpha * self.rule_confidence + (1 - self.alpha) * self.semantic_confidence


@dataclass
class HornClause:
    """Horn clause rule: pre_1 ∧ pre_2 ∧ ... ∧ pre_n → post"""
    rule_id: str
    preconditions: List[str]
    postcondition: str
    rule_type: str
    confidence: float = 1.0
    description: str = ""


class UnifiedAttackSurfaceGraph:
    """
    Unified Attack Surface Graph G = (V, E, τ, R, F_init, G_goal)
    
    - V: Node set
    - E: Dependency edge set
    - τ: Node type mapping
    - R: Primitive manipulation/inference rules
    - F_init: Initial fact nodes
    - G_goal: Security violation goal nodes
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._nodes: Dict[str, NodeAttribute] = {}
        self._edges: Dict[Tuple[str, str], EdgeAttribute] = {}
        self._successors: Dict[str, Set[str]] = defaultdict(set)
        self._predecessors: Dict[str, Set[str]] = defaultdict(set)
        self._rules: Dict[str, HornClause] = {}
        self._goal_nodes: Set[str] = set()
        self._init_primitive_rules()
    
    def _init_primitive_rules(self):
        """Initialize primitive manipulation rules R"""
        rules = [
            # Instruction manipulation
            HornClause("R_INST_001", ["text_contains_override_pattern"],
                      "instruction_override_detected", "instruction", 0.9),
            HornClause("R_INST_002", ["ocr_text_contains_instruction", "instruction_override_detected"],
                      "cross_modal_instruction_injection", "instruction", 0.85),
            HornClause("R_INST_003", ["asr_text_contains_instruction", "instruction_override_detected"],
                      "audio_instruction_injection", "instruction", 0.85),
            
            # Context pollution
            HornClause("R_CTX_001", ["context_injection_pattern_detected"],
                      "context_pollution_attempt", "context", 0.85),
            HornClause("R_CTX_002", ["context_pollution_attempt", "context_boundary_weak"],
                      "context_successfully_polluted", "context", 0.8),
            
            # Jailbreak
            HornClause("R_JAIL_001", ["jailbreak_pattern_detected"],
                      "jailbreak_attempt", "jailbreak", 0.9),
            HornClause("R_JAIL_002", ["roleplay_injection_detected"],
                      "roleplay_jailbreak_attempt", "jailbreak", 0.85),
            HornClause("R_JAIL_003", ["jailbreak_attempt", "safety_filter_bypassed"],
                      "policy_bypass_achieved", "jailbreak", 0.85),
            
            # Tool/API exploitation
            HornClause("R_TOOL_001", ["tool_parameter_manipulation_detected"],
                      "unauthorized_tool_usage_attempt", "tool", 0.85),
            HornClause("R_TOOL_002", ["unauthorized_tool_usage_attempt", "tool_access_enabled"],
                      "tool_exploitation_possible", "tool", 0.8),
            HornClause("R_API_001", ["sensitive_api_call_detected", "api_access_enabled"],
                      "data_access_risk", "api", 0.8),
            
            # Privilege escalation
            HornClause("R_PRIV_001", ["context_successfully_polluted", "privilege_boundary_weak"],
                      "privilege_escalation_possible", "privilege", 0.8),
            HornClause("R_PRIV_002", ["tool_exploitation_possible", "privilege_boundary_weak"],
                      "privilege_escalation_via_tool", "privilege", 0.75),
            
            # Data exfiltration
            HornClause("R_DATA_001", ["data_access_risk", "exfiltration_channel_available"],
                      "data_exfiltration_risk", "data", 0.8),
            HornClause("R_DATA_002", ["system_prompt_extraction_attempt"],
                      "system_info_leakage_risk", "data", 0.85),
            
            # Cross-modal combination
            HornClause("R_CROSS_001", ["image_text_semantic_mismatch", "hidden_instruction_in_image"],
                      "steganographic_injection", "cross_modal", 0.8),
            HornClause("R_CROSS_002", ["cross_modal_instruction_injection", "instruction_override_detected"],
                      "coordinated_cross_modal_attack", "cross_modal", 0.85),
            
            # Goal achievement
            HornClause("R_GOAL_001", ["privilege_escalation_possible"],
                      "GOAL_privilege_escalation", "goal", 0.9),
            HornClause("R_GOAL_002", ["data_exfiltration_risk"],
                      "GOAL_data_exfiltration", "goal", 0.9),
            HornClause("R_GOAL_003", ["policy_bypass_achieved"],
                      "GOAL_policy_bypass", "goal", 0.9),
            HornClause("R_GOAL_004", ["coordinated_cross_modal_attack"],
                      "GOAL_task_hijacking", "goal", 0.85),
            HornClause("R_GOAL_005", ["instruction_override_detected", "context_successfully_polluted"],
                      "GOAL_task_hijacking", "goal", 0.8),
            HornClause("R_GOAL_006", ["jailbreak_attempt", "safety_filter_bypassed"],
                      "GOAL_unsafe_content", "goal", 0.85),
        ]
        
        for rule in rules:
            self._rules[rule.rule_id] = rule
    
    def add_node(self, node: NodeAttribute) -> None:
        self._nodes[node.node_id] = node
        if node.is_goal_node():
            self._goal_nodes.add(node.node_id)
    
    def add_fact_node(self, fact_id: str, confidence: float = 1.0) -> None:
        if fact_id not in self._nodes:
            node_type = self._infer_node_type(fact_id)
            self.add_node(NodeAttribute(fact_id, node_type, fact_id, confidence))
    
    def _infer_node_type(self, fact_id: str) -> NodeType:
        fact_lower = fact_id.lower()
        if "goal" in fact_lower:
            if "privilege" in fact_lower:
                return NodeType.GOAL_PRIVILEGE_ESCALATION
            elif "data" in fact_lower:
                return NodeType.GOAL_DATA_EXFILTRATION
            elif "policy" in fact_lower:
                return NodeType.GOAL_POLICY_BYPASS
            elif "task" in fact_lower:
                return NodeType.GOAL_TASK_HIJACKING
            return NodeType.GOAL_UNSAFE_CONTENT
        elif "ocr" in fact_lower:
            return NodeType.PARSE_OCR_TEXT
        elif "asr" in fact_lower:
            return NodeType.PARSE_ASR_TEXT
        elif "instruction" in fact_lower:
            return NodeType.CONTROL_MAIN_INSTRUCTION
        elif "context" in fact_lower:
            return NodeType.CONTROL_CONTEXT_WINDOW
        return NodeType.PARSE_SEMANTIC_UNIT
    
    def add_edge(self, edge: EdgeAttribute) -> None:
        key = (edge.source_id, edge.target_id)
        self._edges[key] = edge
        self._successors[edge.source_id].add(edge.target_id)
        self._predecessors[edge.target_id].add(edge.source_id)
    
    def get_successors(self, node_id: str) -> Set[str]:
        return self._successors.get(node_id, set())
    
    def get_edge(self, src: str, tgt: str) -> Optional[EdgeAttribute]:
        return self._edges.get((src, tgt))
    
    def forward_chaining_closure(self, initial_facts: Set[str], max_iter: int = 100
                                  ) -> Tuple[Set[str], List[Tuple[str, HornClause]]]:
        """Forward chaining rule closure: Closure(F(x) ∪ F_sys, R)"""
        current = initial_facts.copy()
        trace: List[Tuple[str, HornClause]] = []
        
        changed = True
        iteration = 0
        while changed and iteration < max_iter:
            changed = False
            iteration += 1
            for rule in self._rules.values():
                if all(pre in current for pre in rule.preconditions):
                    if rule.postcondition not in current:
                        current.add(rule.postcondition)
                        trace.append((rule.postcondition, rule))
                        changed = True
                        self.add_fact_node(rule.postcondition, rule.confidence)
                        for pre in rule.preconditions:
                            self.add_edge(EdgeAttribute(pre, rule.postcondition, 
                                                       rule.rule_id, rule.confidence))
        return current, trace
    
    def compute_risk_score(self, activated: Set[str], goals: Optional[Set[str]] = None
                           ) -> Tuple[float, List[str]]:
        """Compute risk: Risk(x) = max_{t∈G_goal} max_{π:A→t} Strength(π)"""
        if goals is None:
            goals = self._goal_nodes
        
        max_strength = 0.0
        best_path: List[str] = []
        
        for start in activated:
            if start not in self._nodes:
                continue
            for goal in goals:
                if goal not in self._nodes:
                    continue
                paths = self._bfs_find_paths(start, goal)
                for path in paths:
                    strength = self._compute_path_strength(path)
                    if strength > max_strength:
                        max_strength = strength
                        best_path = path
        
        return max_strength, best_path
    
    def _bfs_find_paths(self, start: str, goal: str, max_depth: int = 15) -> List[List[str]]:
        paths = []
        queue = deque([(start, [start])])
        visited = set()
        
        while queue:
            node, path = queue.popleft()
            if len(path) > max_depth:
                continue
            if tuple(path) in visited:
                continue
            visited.add(tuple(path))
            
            if node == goal:
                paths.append(path)
                continue
            
            for succ in self.get_successors(node):
                if succ not in path:
                    queue.append((succ, path + [succ]))
        return paths
    
    def _compute_path_strength(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        strength = 1.0
        for i in range(len(path) - 1):
            edge = self.get_edge(path[i], path[i + 1])
            if edge:
                strength *= edge.edge_confidence
            else:
                return 0.0
        return strength
    
    @property
    def num_nodes(self) -> int:
        return len(self._nodes)
    
    @property
    def num_edges(self) -> int:
        return len(self._edges)


class SemanticEquivalenceChecker:
    """
    Semantic equivalence checker
    u ≈_C v ⟺ Entail_C(u,v) ∧ Entail_C(v,u)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threshold = self.config.get('entailment_threshold', 0.7)
        self._attack_keywords = {
            'override': ['ignore', 'forget', 'disregard', 'override', 'new instruction'],
            'jailbreak': ['jailbreak', 'dan', 'developer mode', 'no restrictions', 'bypass'],
            'context': ['system:', '[system]', 'new context', 'you are now'],
            'tool': ['execute', 'run command', 'call function', 'api call'],
            'exfiltration': ['reveal', 'show me', 'print', 'output', 'system prompt'],
            'roleplay': ['pretend', 'act as', 'you are', 'imagine you are']
        }
    
    def check_entailment(self, premise: str, hypothesis: str, context: str = ""
                         ) -> Tuple[bool, float]:
        p_lower, h_lower = premise.lower(), hypothesis.lower()
        conf = self._compute_similarity(p_lower, h_lower)
        return conf >= self.threshold, conf
    
    def check_semantic_equivalence(self, text1: str, text2: str, context: str = ""
                                    ) -> Tuple[bool, float]:
        ent_12, c12 = self.check_entailment(text1, text2, context)
        ent_21, c21 = self.check_entailment(text2, text1, context)
        return ent_12 and ent_21, min(c12, c21)
    
    def _compute_similarity(self, text1: str, text2: str) -> float:
        score = 0.0
        for keywords in self._attack_keywords.values():
            t1_has = any(kw in text1 for kw in keywords)
            t2_has = any(kw in text2 for kw in keywords)
            if t1_has and t2_has:
                score += 0.3
        
        words1, words2 = set(text1.split()), set(text2.split())
        if words1 and words2:
            overlap = len(words1 & words2) / len(words1 | words2)
            score += 0.7 * overlap
        return min(score, 1.0)


class SemanticEntropyCalculator:
    """Semantic entropy: SE(x) = -∑_k P(C_k) log P(C_k)"""
    
    def __init__(self, equiv_checker: SemanticEquivalenceChecker, config: Optional[Dict] = None):
        self.equiv_checker = equiv_checker
        self.config = config or {}
        self.low_entropy_threshold = self.config.get('low_entropy_threshold', 0.5)
    
    def compute_semantic_entropy(self, candidates: List[Tuple[str, float]], context: str = ""
                                  ) -> Tuple[float, Dict[int, float]]:
        if not candidates:
            return 0.0, {}
        
        clusters = self._cluster_by_semantic(candidates, context)
        total_weight = sum(w for _, w in candidates) or len(candidates)
        
        cluster_probs = {}
        for cid, members in clusters.items():
            cluster_probs[cid] = sum(w for _, w in members) / total_weight
        
        entropy = -sum(p * np.log2(p) for p in cluster_probs.values() if p > 0)
        return entropy, cluster_probs
    
    def _cluster_by_semantic(self, candidates: List[Tuple[str, float]], context: str
                              ) -> Dict[int, List[Tuple[str, float]]]:
        clusters: Dict[int, List[Tuple[str, float]]] = {}
        reps: Dict[int, str] = {}
        next_cid = 0
        
        for text, weight in candidates:
            assigned = False
            for cid, rep in reps.items():
                is_eq, _ = self.equiv_checker.check_semantic_equivalence(text, rep, context)
                if is_eq:
                    clusters[cid].append((text, weight))
                    assigned = True
                    break
            if not assigned:
                clusters[next_cid] = [(text, weight)]
                reps[next_cid] = text
                next_cid += 1
        return clusters
    
    def is_concentrated(self, candidates: List[Tuple[str, float]], context: str = ""
                        ) -> Tuple[bool, float]:
        entropy, _ = self.compute_semantic_entropy(candidates, context)
        return entropy < self.low_entropy_threshold, entropy


class ZeroShotDetector:
    """
    Algorithm 1: Open-world Zero-shot Detection
    
    Input: G, F_sys, C, x, τ_threshold, α
    Output: Flag(x), Risk(x), evidence chain π*
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
        self.alpha = self.config.get('alpha', 0.7)
        
        self.equiv_checker = SemanticEquivalenceChecker(config)
        self.entropy_calc = SemanticEntropyCalculator(self.equiv_checker, config)
        self._pattern_matchers = self._init_patterns()
        self.timing_stats: Dict[str, List[float]] = defaultdict(list)
    
    def _init_patterns(self) -> Dict[str, Callable[[str], bool]]:
        return {
            'text_contains_override_pattern': lambda t: any(
                kw in t.lower() for kw in ['ignore', 'forget', 'disregard', 'override', 'new instruction']
            ),
            'jailbreak_pattern_detected': lambda t: any(
                kw in t.lower() for kw in ['jailbreak', 'dan', 'developer mode', 'no restrictions']
            ),
            'context_injection_pattern_detected': lambda t: any(
                kw in t.lower() for kw in ['system:', '[system]', 'new context:', 'you are now']
            ),
            'roleplay_injection_detected': lambda t: any(
                kw in t.lower() for kw in ['pretend you', 'act as', 'imagine you are', 'roleplay as']
            ),
            'tool_parameter_manipulation_detected': lambda t: any(
                kw in t.lower() for kw in ['execute', 'run command', 'call function', 'invoke api']
            ),
            'system_prompt_extraction_attempt': lambda t: any(
                kw in t.lower() for kw in ['reveal system', 'show prompt', 'print instructions']
            ),
            'sensitive_api_call_detected': lambda t: any(
                kw in t.lower() for kw in ['api key', 'password', 'secret', 'credential']
            ),
        }
    
    def detect(self, input_data: Dict[str, Any], system_context: str = "",
               system_facts: Optional[Set[str]] = None) -> Dict[str, Any]:
        """
        Execute zero-shot detection
        
        Args:
            input_data: {'text': str, 'image_ocr': [(text, conf), ...], 'audio_asr': [...]}
            system_context: System context string
            system_facts: System configuration facts
        
        Returns:
            {flag, risk_score, evidence_chain, timing_ms, ...}
        """
        total_start = time.time()
        timing = {}
        graph = UnifiedAttackSurfaceGraph(self.config)
        
        # Step 1: Semantic reduction
        t0 = time.time()
        semantic_facts, reliability = self._semantic_reduction(input_data, system_context, graph)
        timing['semantic_reduction'] = (time.time() - t0) * 1000
        
        # Step 2: Rule closure
        t0 = time.time()
        if system_facts is None:
            system_facts = self._get_default_system_facts(system_context)
        
        all_facts = semantic_facts | system_facts
        for fact in all_facts:
            graph.add_fact_node(fact)
        
        closure, derivations = graph.forward_chaining_closure(all_facts)
        timing['rule_closure'] = (time.time() - t0) * 1000
        
        # Step 3: BFS risk propagation
        t0 = time.time()
        goal_facts = {f for f in closure if 'GOAL' in f}
        risk_score, evidence_chain = graph.compute_risk_score(semantic_facts, goal_facts)
        risk_score *= reliability
        timing['bfs_propagation'] = (time.time() - t0) * 1000
        
        timing['total'] = (time.time() - total_start) * 1000
        flag = 1 if risk_score >= self.threshold else 0
        
        return {
            'flag': flag,
            'risk_score': float(risk_score),
            'evidence_chain': evidence_chain,
            'input_facts': list(semantic_facts),
            'closure_size': len(closure),
            'num_goals_reached': len(goal_facts),
            'reliability': float(reliability),
            'timing_ms': timing
        }
    
    def _semantic_reduction(self, input_data: Dict[str, Any], context: str,
                            graph: UnifiedAttackSurfaceGraph) -> Tuple[Set[str], float]:
        facts = set()
        reliabilities = []
        
        if 'text' in input_data and input_data['text']:
            facts.update(self._extract_facts(input_data['text']))
            reliabilities.append(1.0)
        
        if 'image_ocr' in input_data and input_data['image_ocr']:
            ocr_facts, ocr_rel = self._process_candidates(input_data['image_ocr'], context)
            facts.update(ocr_facts)
            reliabilities.append(ocr_rel)
            if ocr_facts:
                facts.add('ocr_text_contains_instruction')
                if 'text_contains_override_pattern' in facts:
                    facts.add('hidden_instruction_in_image')
                    facts.add('image_text_semantic_mismatch')
        
        if 'audio_asr' in input_data and input_data['audio_asr']:
            asr_facts, asr_rel = self._process_candidates(input_data['audio_asr'], context)
            facts.update(asr_facts)
            reliabilities.append(asr_rel)
            if asr_facts:
                facts.add('asr_text_contains_instruction')
        
        reliability = np.mean(reliabilities) if reliabilities else 1.0
        return facts, reliability
    
    def _extract_facts(self, text: str) -> Set[str]:
        return {name for name, matcher in self._pattern_matchers.items() if matcher(text)}
    
    def _process_candidates(self, candidates: List[Tuple[str, float]], context: str
                            ) -> Tuple[Set[str], float]:
        if not candidates:
            return set(), 1.0
        
        is_conc, entropy = self.entropy_calc.is_concentrated(candidates, context)
        facts = set()
        
        if is_conc:
            best_text, _ = max(candidates, key=lambda x: x[1])
            facts = self._extract_facts(best_text)
        else:
            for text, _ in candidates:
                facts.update(self._extract_facts(text))
        
        reliability = np.exp(-entropy) * (1.0 if is_conc else 0.8)
        return facts, reliability
    
    def _get_default_system_facts(self, context: str) -> Set[str]:
        facts = set()
        ctx = context.lower()
        if 'tool' in ctx or 'function' in ctx or not context:
            facts.add('tool_access_enabled')
        if 'api' in ctx or not context:
            facts.update(['api_access_enabled', 'exfiltration_channel_available'])
        if 'weak' in ctx or 'permissive' in ctx or not context:
            facts.update(['safety_filter_bypassed', 'privilege_boundary_weak', 'context_boundary_weak'])
        return facts


__all__ = [
    'NodeType', 'NodeAttribute', 'EdgeAttribute', 'HornClause',
    'UnifiedAttackSurfaceGraph', 'SemanticEquivalenceChecker',
    'SemanticEntropyCalculator', 'ZeroShotDetector',
]
