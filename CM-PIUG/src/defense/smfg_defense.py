#!/usr/bin/env python3
"""
CM-PIUG Defense Module
Stackelberg Mean-Field Game Defense Strategy

Core components:
- DefenseActionLibrary: 20 defense configurations (D1-D20)
- MeanFieldGameSolver: MFG equilibrium solver
- StackelbergMFGSolver: Algorithm 2 - Offline solving + Online matching
"""

import numpy as np
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class DefenseAction:
    """Defense action configuration"""
    action_id: str
    name: str
    combination_mode: str  # F, R, A, G combinations
    description: str
    applicable_scenario: str
    risk_reduction: float = 0.0
    coverage: Set[str] = field(default_factory=set)
    latency_cost: float = 0.0
    compute_cost: float = 0.0
    utility_cost: float = 0.0
    false_positive_rate: float = 0.0
    
    @property
    def total_cost(self) -> float:
        return self.latency_cost + self.compute_cost + self.utility_cost
    
    def to_dict(self) -> Dict:
        return {
            'action_id': self.action_id,
            'name': self.name,
            'combination_mode': self.combination_mode,
            'risk_reduction': self.risk_reduction,
            'total_cost': self.total_cost
        }


@dataclass
class AttackAction:
    """Attack action representation"""
    action_id: str
    attack_type: str
    modality: str
    success_rate: float = 0.5
    detection_evasion: float = 0.5
    effort_cost: float = 0.1


@dataclass
class MeanFieldState:
    """Mean-field state m(x) aggregating attack statistics"""
    attack_type_distribution: Dict[str, float] = field(default_factory=dict)
    modality_distribution: Dict[str, float] = field(default_factory=dict)
    goal_concentration: Dict[str, float] = field(default_factory=dict)
    mean_risk_score: float = 0.0
    semantic_uncertainty: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        features = []
        for t in ['override', 'jailbreak', 'exfiltration', 'cross_modal', 'roleplay']:
            features.append(self.attack_type_distribution.get(t, 0.0))
        for m in ['text', 'image', 'audio', 'graph']:
            features.append(self.modality_distribution.get(m, 0.0))
        for g in ['privilege', 'data', 'policy', 'task', 'unsafe']:
            features.append(self.goal_concentration.get(g, 0.0))
        features.extend([self.mean_risk_score, self.semantic_uncertainty])
        return np.array(features)


class DefenseActionLibrary:
    """
    Defense Action Library - 20 configurations
    
    F: Filtering (Input filtering)
    R: Rewriting (Prompt rewriting)
    A: Auditing (Output auditing)
    G: Guardrails (Safety guardrails)
    """
    
    def __init__(self):
        self.actions: Dict[str, DefenseAction] = {}
        self._init_configurations()
    
    def _init_configurations(self):
        configs = [
            # D1: No defense (baseline)
            DefenseAction("D1", "∅ (No Defense)", "∅", "Baseline", "-",
                         risk_reduction=0.0, latency_cost=0.0),
            
            # D2-D5: Single actions
            DefenseAction("D2", "F: Filtering Only", "F", "Input Filtering", "Low-latency",
                         risk_reduction=0.50, coverage={'instruction_override', 'jailbreak_pattern'},
                         latency_cost=0.10, compute_cost=0.05, false_positive_rate=0.15),
            DefenseAction("D3", "R: Rewriting Only", "R", "Prompt Rewriting", "Compatibility-first",
                         risk_reduction=0.45, coverage={'context_pollution', 'roleplay_injection'},
                         latency_cost=0.20, compute_cost=0.15, utility_cost=0.10),
            DefenseAction("D4", "A: Auditing Only", "A", "Output Auditing", "Post-hoc Tracing",
                         risk_reduction=0.30, coverage={'unsafe_content', 'data_leakage'},
                         latency_cost=0.15, compute_cost=0.10, false_positive_rate=0.20),
            DefenseAction("D5", "G: Guardrails Only", "G", "Guardrails Detection", "General Protection",
                         risk_reduction=0.55, coverage={'jailbreak_attempt', 'policy_bypass'},
                         latency_cost=0.25, compute_cost=0.20),
            
            # D6-D11: Two combinations
            DefenseAction("D6", "F+R", "FR", "Filtering + Rewriting", "Input Hardening",
                         risk_reduction=0.70, coverage={'instruction_override', 'context_pollution'},
                         latency_cost=0.25, compute_cost=0.18, utility_cost=0.08),
            DefenseAction("D7", "F+A", "FA", "Filtering + Auditing", "Detection Enhanced",
                         risk_reduction=0.65, coverage={'instruction_override', 'unsafe_content'},
                         latency_cost=0.22, compute_cost=0.13, false_positive_rate=0.12),
            DefenseAction("D8", "F+G", "FG", "Filtering + Guardrails", "Dual Verification",
                         risk_reduction=0.75, coverage={'instruction_override', 'jailbreak_attempt'},
                         latency_cost=0.30, compute_cost=0.22),
            DefenseAction("D9", "R+A", "RA", "Rewriting + Auditing", "Sanitization + Tracing",
                         risk_reduction=0.60, coverage={'context_pollution', 'unsafe_content'},
                         latency_cost=0.30, compute_cost=0.22, utility_cost=0.08),
            DefenseAction("D10", "R+G", "RG", "Rewriting + Guardrails", "Deep Protection",
                         risk_reduction=0.72, coverage={'context_pollution', 'jailbreak_attempt'},
                         latency_cost=0.38, compute_cost=0.30, utility_cost=0.10),
            DefenseAction("D11", "A+G", "AG", "Auditing + Guardrails", "Runtime Monitoring",
                         risk_reduction=0.68, coverage={'unsafe_content', 'policy_bypass'},
                         latency_cost=0.35, compute_cost=0.28),
            
            # D12-D15: Three combinations
            DefenseAction("D12", "F+R+A", "FRA", "Filtering + Rewriting + Auditing", "Three-layer Input",
                         risk_reduction=0.80, coverage={'instruction_override', 'context_pollution', 'unsafe_content'},
                         latency_cost=0.42, compute_cost=0.30, utility_cost=0.10),
            DefenseAction("D13", "F+R+G", "FRG", "Filtering + Rewriting + Guardrails", "Full Input Protection",
                         risk_reduction=0.85, coverage={'instruction_override', 'context_pollution', 'jailbreak_attempt'},
                         latency_cost=0.48, compute_cost=0.35, utility_cost=0.12),
            DefenseAction("D14", "F+A+G", "FAG", "Filtering + Auditing + Guardrails", "Detection-first",
                         risk_reduction=0.82, coverage={'instruction_override', 'unsafe_content', 'policy_bypass'},
                         latency_cost=0.45, compute_cost=0.32, false_positive_rate=0.10),
            DefenseAction("D15", "R+A+G", "RAG", "Rewriting + Auditing + Guardrails", "Sanitization-first",
                         risk_reduction=0.78, coverage={'context_pollution', 'unsafe_content', 'policy_bypass'},
                         latency_cost=0.50, compute_cost=0.38, utility_cost=0.12),
            
            # D16-D18: Full combination (parameter variants)
            DefenseAction("D16", "F+R+A+G (Weak)", "FRAG", "Full Combination (Weak)", "Conservative",
                         risk_reduction=0.85, coverage={'all'},
                         latency_cost=0.55, compute_cost=0.40, utility_cost=0.15),
            DefenseAction("D17", "F+R+A+G (Medium)", "FRAG", "Full Combination (Medium)", "Balanced",
                         risk_reduction=0.90, coverage={'all'},
                         latency_cost=0.60, compute_cost=0.45, utility_cost=0.18),
            DefenseAction("D18", "F+R+A+G (Strong)", "FRAG", "Full Combination (Strong)", "Aggressive",
                         risk_reduction=0.95, coverage={'all'},
                         latency_cost=0.70, compute_cost=0.55, utility_cost=0.25),
            
            # D19-D20: Adaptive and game-equilibrium
            DefenseAction("D19", "Adaptive Policy", "ADAPTIVE", "Adaptive", "Dynamic Switching",
                         risk_reduction=0.88, coverage={'all'},
                         latency_cost=0.55, compute_cost=0.42, utility_cost=0.15),
            DefenseAction("D20", "SMFE-Optimal", "SMFE", "SMFE-Optimal", "Game Equilibrium",
                         risk_reduction=0.92, coverage={'all'},
                         latency_cost=0.58, compute_cost=0.45, utility_cost=0.16),
        ]
        for action in configs:
            self.actions[action.action_id] = action
    
    def get_action(self, action_id: str) -> Optional[DefenseAction]:
        return self.actions.get(action_id)
    
    def get_all_actions(self) -> List[DefenseAction]:
        return list(self.actions.values())
    
    @property
    def action_space_size(self) -> int:
        return len(self.actions)


class MeanFieldGameSolver:
    """Mean-Field Game equilibrium solver"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.max_iterations = self.config.get('mfg_iterations', 50)
        self.tolerance = self.config.get('mfg_tolerance', 1e-4)
    
    def solve_mfg_equilibrium(self, defense_policy: np.ndarray, attack_actions: List[AttackAction],
                               defense_library: DefenseActionLibrary, initial_mf: MeanFieldState
                               ) -> Tuple[np.ndarray, MeanFieldState]:
        n = len(attack_actions)
        attack_policy = np.ones(n) / n
        mean_field = initial_mf
        
        for _ in range(self.max_iterations):
            new_policy = self._compute_best_response(
                attack_policy, defense_policy, mean_field, attack_actions, defense_library
            )
            new_mf = self._update_mean_field(new_policy, attack_actions, mean_field)
            
            if np.linalg.norm(new_policy - attack_policy) < self.tolerance:
                break
            
            attack_policy = new_policy
            mean_field = new_mf
        
        return attack_policy, mean_field
    
    def _compute_best_response(self, attack_policy: np.ndarray, defense_policy: np.ndarray,
                                mean_field: MeanFieldState, attack_actions: List[AttackAction],
                                defense_library: DefenseActionLibrary) -> np.ndarray:
        n = len(attack_actions)
        utilities = np.zeros(n)
        defense_actions = defense_library.get_all_actions()
        
        for i, attack in enumerate(attack_actions):
            reward = attack.success_rate * (1 - attack.detection_evasion)
            defense_reduction = sum(
                defense_policy[j] * d.risk_reduction
                for j, d in enumerate(defense_actions)
                if attack.attack_type in str(d.coverage) or 'all' in d.coverage
            )
            congestion = mean_field.attack_type_distribution.get(attack.attack_type, 0.1)
            utilities[i] = reward * (1 - defense_reduction) - 0.3 * congestion - attack.effort_cost
        
        exp_util = np.exp(utilities - np.max(utilities))
        return exp_util / np.sum(exp_util)
    
    def _update_mean_field(self, attack_policy: np.ndarray, attack_actions: List[AttackAction],
                            current_mf: MeanFieldState) -> MeanFieldState:
        new_mf = MeanFieldState()
        for i, attack in enumerate(attack_actions):
            new_mf.attack_type_distribution[attack.attack_type] = \
                new_mf.attack_type_distribution.get(attack.attack_type, 0.0) + attack_policy[i]
            new_mf.modality_distribution[attack.modality] = \
                new_mf.modality_distribution.get(attack.modality, 0.0) + attack_policy[i]
        
        smooth = 0.7
        for key in new_mf.attack_type_distribution:
            old_val = current_mf.attack_type_distribution.get(key, 0.1)
            new_mf.attack_type_distribution[key] = (
                smooth * new_mf.attack_type_distribution[key] + (1 - smooth) * old_val
            )
        return new_mf


class StackelbergMFGSolver:
    """
    Algorithm 2: Stackelberg-MFG Offline Solving + Online Matching
    
    Offline: SMFE equilibrium solving
    Online: Fast policy matching based on evidence chain
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.defense_library = DefenseActionLibrary()
        self.mfg_solver = MeanFieldGameSolver(config)
        self.attack_actions = self._init_attack_actions()
        
        self.outer_iterations = self.config.get('outer_iterations', 100)
        self.inner_iterations = self.config.get('inner_iterations', 30)
        self.learning_rate = self.config.get('learning_rate', 0.05)
        self.convergence_threshold = self.config.get('convergence_threshold', 1e-3)
        
        self.optimal_policy: Optional[np.ndarray] = None
        self.optimal_mf: Optional[MeanFieldState] = None
        self.policy_table: Dict[str, np.ndarray] = {}
        self.solve_time: float = 0.0
        self.convergence_history: List[float] = []
    
    def _init_attack_actions(self) -> List[AttackAction]:
        attacks = []
        types = ['override', 'jailbreak', 'exfiltration', 'cross_modal', 'roleplay']
        modalities = ['text', 'image', 'audio']
        for i, atype in enumerate(types):
            for j, mod in enumerate(modalities):
                attacks.append(AttackAction(
                    f"A_{atype}_{mod}", atype, mod,
                    success_rate=0.6 - 0.05 * i,
                    detection_evasion=0.4 + 0.05 * j,
                    effort_cost=0.1 + 0.02 * i
                ))
        return attacks
    
    def offline_solve(self, verbose: bool = True) -> np.ndarray:
        """Offline SMFE solving"""
        start_time = time.time()
        n_actions = self.defense_library.action_space_size
        defense_policy = np.ones(n_actions) / n_actions
        mean_field = self._init_mean_field()
        
        best_utility = -float('inf')
        best_policy = defense_policy.copy()
        
        if verbose:
            print("=" * 60)
            print("SMFE Offline Solving (Algorithm 2)")
            print(f"Defense Actions: {n_actions}, Attack Actions: {len(self.attack_actions)}")
            print("=" * 60)
        
        prev_policy = defense_policy.copy()
        
        for k in range(self.outer_iterations):
            # Inner: MFG equilibrium
            for _ in range(self.inner_iterations):
                _, mean_field = self.mfg_solver.solve_mfg_equilibrium(
                    defense_policy, self.attack_actions, self.defense_library, mean_field
                )
            
            # Compute utility
            utility = self._compute_defender_utility(defense_policy, mean_field)
            self.convergence_history.append(utility)
            
            if utility > best_utility:
                best_utility = utility
                best_policy = defense_policy.copy()
            
            # Outer: Leader policy update
            gradient = self._estimate_gradient(defense_policy)
            defense_policy = defense_policy + self.learning_rate * gradient
            defense_policy = self._project_to_simplex(defense_policy)
            
            policy_drift = np.linalg.norm(defense_policy - prev_policy)
            prev_policy = defense_policy.copy()
            
            if verbose and k % 20 == 0:
                print(f"Iter {k:3d}: Utility={utility:.4f}, Drift={policy_drift:.6f}")
            
            if policy_drift < self.convergence_threshold:
                if verbose:
                    print(f"Converged at iteration {k}")
                break
        
        self.optimal_policy = best_policy
        self.optimal_mf = mean_field
        self.solve_time = time.time() - start_time
        self._build_policy_table()
        
        if verbose:
            print("=" * 60)
            print(f"Completed in {self.solve_time:.2f}s, Best Utility: {best_utility:.4f}")
            self._print_top_actions(best_policy)
            print("=" * 60)
        
        return best_policy
    
    def _init_mean_field(self) -> MeanFieldState:
        mf = MeanFieldState()
        mf.attack_type_distribution = {
            'override': 0.25, 'jailbreak': 0.25, 'exfiltration': 0.2,
            'cross_modal': 0.15, 'roleplay': 0.15
        }
        mf.modality_distribution = {'text': 0.5, 'image': 0.3, 'audio': 0.2}
        mf.mean_risk_score = 0.5
        return mf
    
    def _compute_defender_utility(self, defense_policy: np.ndarray, mean_field: MeanFieldState) -> float:
        actions = self.defense_library.get_all_actions()
        risk_reduction = sum(defense_policy[i] * a.risk_reduction for i, a in enumerate(actions))
        deploy_cost = sum(defense_policy[i] * a.total_cost for i, a in enumerate(actions))
        fp_cost = sum(defense_policy[i] * a.false_positive_rate * 0.5 for i, a in enumerate(actions))
        entropy = -np.sum(defense_policy * np.log(defense_policy + 1e-10))
        switching_penalty = 0.1 * (np.log(len(actions)) - entropy)
        return risk_reduction - 0.4 * deploy_cost - fp_cost - switching_penalty
    
    def _estimate_gradient(self, defense_policy: np.ndarray) -> np.ndarray:
        actions = self.defense_library.get_all_actions()
        return np.array([
            a.risk_reduction - 0.4 * a.total_cost - a.false_positive_rate * 0.5
            for a in actions
        ])
    
    def _project_to_simplex(self, vec: np.ndarray) -> np.ndarray:
        vec = np.maximum(vec, 1e-8)
        return vec / np.sum(vec)
    
    def _build_policy_table(self):
        risk_levels = ['low', 'medium', 'high', 'critical']
        for i, level in enumerate(risk_levels):
            adjusted = self.optimal_policy.copy()
            risk_factor = (i + 1) / len(risk_levels)
            actions = self.defense_library.get_all_actions()
            for j, action in enumerate(actions):
                if action.risk_reduction > 0.8:
                    adjusted[j] *= (1 + 0.5 * risk_factor)
                elif action.risk_reduction < 0.5:
                    adjusted[j] *= (1 - 0.3 * risk_factor)
            self.policy_table[level] = self._project_to_simplex(adjusted)
    
    def _print_top_actions(self, policy: np.ndarray, top_k: int = 5):
        actions = self.defense_library.get_all_actions()
        sorted_idx = np.argsort(policy)[::-1]
        print("\nTop Defense Actions:")
        for idx in sorted_idx[:top_k]:
            if policy[idx] > 0.01:
                a = actions[idx]
                print(f"  {a.name:<30}: {policy[idx]:.3f} (RR={a.risk_reduction:.2f})")
    
    def online_match(self, risk_score: float, evidence_chain: List[str],
                     triggered_rules: Optional[List[str]] = None) -> Tuple[str, float, Dict[str, Any]]:
        """Online policy matching"""
        if self.optimal_policy is None:
            raise RuntimeError("Must call offline_solve() first")
        
        if risk_score < 0.3:
            level = 'low'
        elif risk_score < 0.5:
            level = 'medium'
        elif risk_score < 0.7:
            level = 'high'
        else:
            level = 'critical'
        
        policy = self.policy_table.get(level, self.optimal_policy).copy()
        
        if evidence_chain:
            policy = self._adjust_by_evidence(policy, evidence_chain)
        if triggered_rules:
            policy = self._adjust_by_rules(policy, triggered_rules)
        
        action_idx = np.argmax(policy)
        action = self.defense_library.get_all_actions()[action_idx]
        
        return action.action_id, float(policy[action_idx]), {
            'risk_level': level,
            'policy_distribution': policy.tolist(),
            'action_details': action.to_dict()
        }
    
    def _adjust_by_evidence(self, policy: np.ndarray, evidence: List[str]) -> np.ndarray:
        adjusted = policy.copy()
        evidence_str = ' '.join(evidence).lower()
        actions = self.defense_library.get_all_actions()
        for i, a in enumerate(actions):
            for c in a.coverage:
                if c.lower() in evidence_str or c == 'all':
                    adjusted[i] *= 1.3
        return self._project_to_simplex(adjusted)
    
    def _adjust_by_rules(self, policy: np.ndarray, rules: List[str]) -> np.ndarray:
        adjusted = policy.copy()
        actions = self.defense_library.get_all_actions()
        for rule in rules:
            rule_lower = rule.lower()
            if 'jailbreak' in rule_lower:
                for i, a in enumerate(actions):
                    if 'G' in a.combination_mode:
                        adjusted[i] *= 1.4
            if 'cross_modal' in rule_lower or 'injection' in rule_lower:
                for i, a in enumerate(actions):
                    if 'F' in a.combination_mode:
                        adjusted[i] *= 1.3
        return self._project_to_simplex(adjusted)
    
    def get_defense_recommendations(self, risk_score: float, evidence_chain: List[str],
                                     top_k: int = 5) -> List[Dict[str, Any]]:
        """Get top-k defense recommendations"""
        if self.optimal_policy is None:
            raise RuntimeError("Must call offline_solve() first")
        
        _, _, details = self.online_match(risk_score, evidence_chain)
        policy = np.array(details['policy_distribution'])
        actions = self.defense_library.get_all_actions()
        sorted_idx = np.argsort(policy)[::-1]
        
        return [
            {
                'action_id': actions[idx].action_id,
                'name': actions[idx].name,
                'probability': float(policy[idx]),
                'risk_reduction': actions[idx].risk_reduction,
                'total_cost': actions[idx].total_cost,
            }
            for idx in sorted_idx[:top_k] if policy[idx] > 0.01
        ]


__all__ = [
    'DefenseAction', 'AttackAction', 'MeanFieldState',
    'DefenseActionLibrary', 'MeanFieldGameSolver', 'StackelbergMFGSolver',
]
