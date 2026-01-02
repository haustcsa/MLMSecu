# Core module - Data structures and algorithms
"""
Core Module
===========

Contains fundamental data structures for the CM-PIUG framework:

- NodeType: Enumeration of node types (INPUT, PARSE, CONTROL, EXEC, GOAL)
- NodeAttribute: Node properties in attack graph
- EdgeAttribute: Edge properties representing attack dependencies
- UnifiedAttackGraph: The main attack graph structure with BFS traversal
- RuleEngine: Horn clause rule engine for forward chaining inference
"""

from .node_types import NodeType, NodeAttribute, EdgeAttribute
from .attack_graph import UnifiedAttackGraph
from .rule_engine import RuleEngine, Rule, ParameterizedRule

__all__ = [
    "NodeType",
    "NodeAttribute",
    "EdgeAttribute",
    "UnifiedAttackGraph",
    "RuleEngine",
    "Rule",
    "ParameterizedRule",
]
