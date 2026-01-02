"""
Visualization Utilities
=======================

Tools for visualizing attack graphs and detection results.
"""

from typing import Optional, List, Dict, Any
import json


def visualize_attack_graph(
    graph,
    output_path: str,
    highlight_path: Optional[List[str]] = None,
    format: str = "png"
) -> str:
    """
    Visualize attack graph using matplotlib/networkx.
    
    Args:
        graph: UnifiedAttackGraph instance
        output_path: Path to save visualization
        highlight_path: Optional path to highlight (list of node IDs)
        format: Output format (png, pdf, svg)
        
    Returns:
        Path to saved visualization
    """
    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        raise ImportError("matplotlib and networkx required for visualization")
    
    # Create networkx graph
    G = nx.DiGraph()
    
    # Add nodes
    node_colors = []
    node_labels = {}
    
    color_map = {
        'INPUT': '#90EE90',   # Light green
        'PARSE': '#87CEEB',   # Sky blue  
        'CONTROL': '#FFD700', # Gold
        'EXEC': '#FFA500',    # Orange
        'GOAL': '#FF6347',    # Tomato red
    }
    
    for node_id, node_attr in graph.nodes.items():
        G.add_node(node_id)
        node_type = node_attr.node_type.name if hasattr(node_attr.node_type, 'name') else str(node_attr.node_type)
        node_colors.append(color_map.get(node_type, '#CCCCCC'))
        node_labels[node_id] = f"{node_id[:10]}...\n({node_type})"
    
    # Add edges
    edge_colors = []
    edge_labels = {}
    
    for edge in graph.edges:
        G.add_edge(edge.source, edge.target)
        edge_labels[(edge.source, edge.target)] = f"{edge.confidence:.2f}"
        
        # Highlight path edges
        if highlight_path and edge.source in highlight_path and edge.target in highlight_path:
            edge_colors.append('red')
        else:
            edge_colors.append('gray')
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.8, ax=ax)
    nx.draw_networkx_labels(G, pos, node_labels, font_size=8, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True, 
                           arrowsize=15, connectionstyle="arc3,rad=0.1", ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7, ax=ax)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                   markersize=10, label=node_type)
        for node_type, color in color_map.items()
    ]
    ax.legend(handles=legend_elements, loc='upper left', framealpha=0.9)
    
    ax.set_title("CM-PIUG Attack Graph Visualization", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, format=format, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


def export_graph_dot(graph, output_path: str) -> str:
    """
    Export attack graph to DOT format (for Graphviz).
    
    Args:
        graph: UnifiedAttackGraph instance
        output_path: Path to save DOT file
        
    Returns:
        Path to saved DOT file
    """
    lines = ['digraph CM_PIUG_AttackGraph {']
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, style=filled];')
    lines.append('')
    
    # Node style mapping
    style_map = {
        'INPUT': 'fillcolor="#90EE90"',
        'PARSE': 'fillcolor="#87CEEB"',
        'CONTROL': 'fillcolor="#FFD700"',
        'EXEC': 'fillcolor="#FFA500"',
        'GOAL': 'fillcolor="#FF6347"',
    }
    
    # Add nodes
    lines.append('    // Nodes')
    for node_id, node_attr in graph.nodes.items():
        node_type = node_attr.node_type.name if hasattr(node_attr.node_type, 'name') else str(node_attr.node_type)
        style = style_map.get(node_type, 'fillcolor="#CCCCCC"')
        safe_id = node_id.replace('"', '\\"').replace('-', '_')
        label = f"{node_id[:20]}\\n({node_type})"
        lines.append(f'    "{safe_id}" [label="{label}", {style}];')
    
    lines.append('')
    
    # Add edges
    lines.append('    // Edges')
    for edge in graph.edges:
        source = edge.source.replace('"', '\\"').replace('-', '_')
        target = edge.target.replace('"', '\\"').replace('-', '_')
        label = f"{edge.confidence:.2f}"
        lines.append(f'    "{source}" -> "{target}" [label="{label}"];')
    
    lines.append('}')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def export_graph_json(graph, output_path: str) -> str:
    """
    Export attack graph to JSON format.
    
    Args:
        graph: UnifiedAttackGraph instance
        output_path: Path to save JSON file
        
    Returns:
        Path to saved JSON file
    """
    data = {
        'nodes': [],
        'edges': [],
        'metadata': {
            'framework': 'CM-PIUG',
            'version': '1.0.0',
        }
    }
    
    # Export nodes
    for node_id, node_attr in graph.nodes.items():
        node_type = node_attr.node_type.name if hasattr(node_attr.node_type, 'name') else str(node_attr.node_type)
        data['nodes'].append({
            'id': node_id,
            'type': node_type,
            'content': node_attr.content,
            'confidence': node_attr.confidence,
            'metadata': node_attr.metadata,
        })
    
    # Export edges
    for edge in graph.edges:
        data['edges'].append({
            'source': edge.source,
            'target': edge.target,
            'rule_id': edge.rule_id,
            'confidence': edge.confidence,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return output_path


def format_detection_result(result) -> str:
    """
    Format detection result for display.
    
    Args:
        result: DetectionResult instance
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CM-PIUG Detection Result")
    lines.append("=" * 60)
    
    # Overall result
    flag_str = "⚠️  INJECTION DETECTED" if result.flag else "✅ SAFE"
    lines.append(f"Status: {flag_str}")
    lines.append(f"Risk Score: {result.risk_score:.4f}")
    lines.append("")
    
    # Evidence chain
    if result.evidence_chain:
        lines.append("Evidence Chain:")
        lines.append("-" * 40)
        for i, evidence in enumerate(result.evidence_chain):
            if hasattr(evidence, 'node_type'):
                # EdgeAttribute 对象
                lines.append(f"  {i+1}. {evidence.source} -> {evidence.target}")
                lines.append(f"      Relation: {evidence.relation}")
                lines.append(f"      Confidence: {evidence.confidence:.3f}")
                if hasattr(evidence, 'rule_id') and evidence.rule_id:
                    lines.append(f"      Rule: {evidence.rule_id}")
            else:
                # 字符串
                lines.append(f"  {i+1}. {evidence}")
    else:
        lines.append("No evidence chain (no attack path found)")
    
    # Triggered rules
    if hasattr(result, 'triggered_rules') and result.triggered_rules:
        lines.append("")
        lines.append("Triggered Rules:")
        lines.append("-" * 40)
        for rule in result.triggered_rules[:10]:
            lines.append(f"  - {rule}")
        if len(result.triggered_rules) > 10:
            lines.append(f"  ... and {len(result.triggered_rules) - 10} more")
    
    # Fired rules (alternative attribute name)
    if hasattr(result, 'fired_rules') and result.fired_rules:
        lines.append("")
        lines.append("Fired Rules:")
        lines.append("-" * 40)
        for rule in result.fired_rules[:10]:
            lines.append(f"  - {rule}")
        if len(result.fired_rules) > 10:
            lines.append(f"  ... and {len(result.fired_rules) - 10} more")
    
    # Detection time
    if hasattr(result, 'detection_time_ms') and result.detection_time_ms > 0:
        lines.append("")
        lines.append(f"Detection Time: {result.detection_time_ms:.2f} ms")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def format_defense_policy(policy, action_library) -> str:
    """
    Format defense policy for display.
    
    Args:
        policy: LeaderPolicy instance
        action_library: DefenseActionLibrary instance
        
    Returns:
        Formatted string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("CM-PIUG Defense Policy")
    lines.append("=" * 60)
    
    # Risk level policies
    lines.append("Risk Level Policies:")
    lines.append("-" * 40)
    
    for level, probs in policy.action_probs.items():
        lines.append(f"\n[{level.upper()}]")
        for action_id, prob in sorted(probs.items(), key=lambda x: -x[1]):
            if prob > 0.01:
                action = action_library.get_action(action_id)
                if action:
                    lines.append(f"  {action.name}: {prob:.1%}")
                    lines.append(f"    └─ {action.description[:50]}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


def print_graph_stats(graph) -> None:
    """
    Print attack graph statistics.
    
    Args:
        graph: UnifiedAttackGraph instance
    """
    print("=" * 50)
    print("Attack Graph Statistics")
    print("=" * 50)
    print(f"Total Nodes: {len(graph.nodes)}")
    print(f"Total Edges: {len(graph.edges)}")
    
    # Count by type
    type_counts = {}
    for node_attr in graph.nodes.values():
        node_type = node_attr.node_type.name if hasattr(node_attr.node_type, 'name') else str(node_attr.node_type)
        type_counts[node_type] = type_counts.get(node_type, 0) + 1
    
    print("\nNodes by Type:")
    for node_type, count in sorted(type_counts.items()):
        print(f"  {node_type}: {count}")
    
    # Edge confidence distribution
    if graph.edges:
        confs = [e.confidence for e in graph.edges]
        print(f"\nEdge Confidence: min={min(confs):.3f}, max={max(confs):.3f}, "
              f"mean={sum(confs)/len(confs):.3f}")
    
    print("=" * 50)
