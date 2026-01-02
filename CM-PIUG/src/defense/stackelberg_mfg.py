"""
CM-PIUG Stackelberg-MFG Defense Module
Stackelberg-MFG 联合攻防博弈模块

实现 Algorithm 2: Stackelberg-MFG 离线求解与在线策略匹配

理论基础:
- Stackelberg博弈: 先承诺-后响应的层级对抗
- Mean Field Game: 大规模群体博弈的均值场近似
- SMFE (Stackelberg Mean Field Equilibrium): 联合均衡

核心流程:
1. 离线阶段: SMFE近似求解 (内层MFG闭合 + 外层Leader更新)
2. 在线阶段: 基于证据链的快速策略匹配
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import copy
import time


@dataclass
class DefenseAction:
    """防御动作"""
    action_id: str
    name: str
    category: str  # filter, rewrite, isolate, audit, constrain
    
    # 效果参数
    risk_reduction: float = 0.0  # 风险降低率
    coverage: Set[str] = field(default_factory=set)  # 覆盖的攻击类型
    
    # 成本参数
    latency_cost: float = 0.0     # 延迟成本
    compute_cost: float = 0.0     # 计算成本
    utility_cost: float = 0.0     # 功能损失
    false_positive_rate: float = 0.0  # 误报率
    
    @property
    def total_cost(self) -> float:
        """总部署成本"""
        return self.latency_cost + self.compute_cost + self.utility_cost
    
    def to_dict(self) -> Dict:
        return {
            'action_id': self.action_id,
            'name': self.name,
            'category': self.category,
            'risk_reduction': self.risk_reduction,
            'total_cost': self.total_cost
        }


@dataclass
class AttackAction:
    """攻击动作"""
    action_id: str
    attack_type: str
    modality: str
    
    # 攻击参数
    success_rate: float = 0.5
    perturbation_budget: float = 1.0
    detection_evasion: float = 0.5
    
    # 成本参数
    effort_cost: float = 0.1


@dataclass
class MeanFieldState:
    """均值场状态"""
    # 攻击分布特征
    attack_type_distribution: Dict[str, float] = field(default_factory=dict)
    modality_distribution: Dict[str, float] = field(default_factory=dict)
    rule_trigger_frequency: Dict[str, float] = field(default_factory=dict)
    goal_concentration: Dict[str, float] = field(default_factory=dict)
    
    # 统计量
    mean_risk_score: float = 0.0
    semantic_uncertainty: float = 0.0
    
    def to_vector(self) -> np.ndarray:
        """转换为特征向量"""
        features = []
        
        # 攻击类型分布
        for t in ['override', 'jailbreak', 'roleplay', 'exfiltration']:
            features.append(self.attack_type_distribution.get(t, 0.0))
        
        # 模态分布
        for m in ['text', 'image', 'audio']:
            features.append(self.modality_distribution.get(m, 0.0))
        
        # 目标集中度
        for g in ['privilege', 'data', 'policy', 'task']:
            features.append(self.goal_concentration.get(g, 0.0))
        
        # 统计量
        features.extend([self.mean_risk_score, self.semantic_uncertainty])
        
        return np.array(features)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'MeanFieldState':
        """从特征向量构造"""
        state = cls()
        
        # 解析攻击类型
        attack_types = ['override', 'jailbreak', 'roleplay', 'exfiltration']
        for i, t in enumerate(attack_types):
            state.attack_type_distribution[t] = vec[i]
        
        # 解析模态
        modalities = ['text', 'image', 'audio']
        for i, m in enumerate(modalities):
            state.modality_distribution[m] = vec[4 + i]
        
        # 解析目标
        goals = ['privilege', 'data', 'policy', 'task']
        for i, g in enumerate(goals):
            state.goal_concentration[g] = vec[7 + i]
        
        # 统计量
        state.mean_risk_score = vec[11]
        state.semantic_uncertainty = vec[12]
        
        return state


class DefenseActionLibrary:
    """防御动作库"""
    
    def __init__(self):
        self.actions: Dict[str, DefenseAction] = {}
        self._init_default_actions()
    
    def _init_default_actions(self):
        """初始化默认防御动作"""
        default_actions = [
            # 过滤类
            DefenseAction(
                action_id="D_FILTER_001",
                name="指令标记过滤",
                category="filter",
                risk_reduction=0.7,
                coverage={'instruction_override', 'context_pollution'},
                latency_cost=0.1,
                compute_cost=0.05,
                false_positive_rate=0.15
            ),
            DefenseAction(
                action_id="D_FILTER_002",
                name="越狱模式过滤",
                category="filter",
                risk_reduction=0.8,
                coverage={'jailbreak_attempt'},
                latency_cost=0.15,
                compute_cost=0.1,
                false_positive_rate=0.1
            ),
            
            # 重写类
            DefenseAction(
                action_id="D_REWRITE_001",
                name="提示重写净化",
                category="rewrite",
                risk_reduction=0.6,
                coverage={'instruction_override', 'roleplay_injection'},
                latency_cost=0.2,
                compute_cost=0.15,
                utility_cost=0.1
            ),
            DefenseAction(
                action_id="D_REWRITE_002",
                name="上下文隔离重写",
                category="rewrite",
                risk_reduction=0.65,
                coverage={'context_pollution'},
                latency_cost=0.25,
                compute_cost=0.2,
                utility_cost=0.15
            ),
            
            # 隔离类
            DefenseAction(
                action_id="D_ISOLATE_001",
                name="工具调用沙箱",
                category="isolate",
                risk_reduction=0.85,
                coverage={'unauthorized_tool_call', 'privilege_escalation'},
                latency_cost=0.3,
                compute_cost=0.25,
                utility_cost=0.2
            ),
            DefenseAction(
                action_id="D_ISOLATE_002",
                name="API访问隔离",
                category="isolate",
                risk_reduction=0.9,
                coverage={'sensitive_api_access', 'data_exfiltration'},
                latency_cost=0.35,
                compute_cost=0.3,
                utility_cost=0.25
            ),
            
            # 审计类
            DefenseAction(
                action_id="D_AUDIT_001",
                name="输出内容审计",
                category="audit",
                risk_reduction=0.5,
                coverage={'unsafe_content'},
                latency_cost=0.15,
                compute_cost=0.1,
                false_positive_rate=0.2
            ),
            
            # 约束类
            DefenseAction(
                action_id="D_CONSTRAIN_001",
                name="工具参数约束",
                category="constrain",
                risk_reduction=0.75,
                coverage={'tool_param_manipulation'},
                latency_cost=0.1,
                compute_cost=0.05,
                utility_cost=0.1
            ),
        ]
        
        for action in default_actions:
            self.actions[action.action_id] = action
    
    def get_action(self, action_id: str) -> Optional[DefenseAction]:
        return self.actions.get(action_id)
    
    def get_all_actions(self) -> List[DefenseAction]:
        return list(self.actions.values())
    
    def get_actions_by_category(self, category: str) -> List[DefenseAction]:
        return [a for a in self.actions.values() if a.category == category]
    
    @property
    def action_space_size(self) -> int:
        return len(self.actions)


class MeanFieldGameSolver:
    """
    均值场博弈求解器
    
    求解给定防御承诺下的攻击者均值场均衡
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 收敛阈值
        self.convergence_tol = self.config.get('mfg_inner_tol', 1e-4)
        self.max_iterations = self.config.get('mfg_max_iter', 100)
        
        # 折扣因子
        self.gamma = self.config.get('gamma', 0.95)
        
        # 攻击者效用参数
        self.attack_success_weight = self.config.get('attack_success_weight', 1.0)
        self.detection_penalty = self.config.get('detection_penalty', 0.5)
        self.congestion_factor = self.config.get('congestion_factor', 0.1)
    
    def compute_attacker_utility(self,
                                  attack_action: AttackAction,
                                  defense_policy: np.ndarray,
                                  mean_field: MeanFieldState,
                                  defense_library: DefenseActionLibrary
                                  ) -> float:
        """
        计算攻击者效用
        
        U_a(a, π_d, m) = R_attack - C_detection - C_congestion - C_effort
        """
        # 攻击收益
        attack_reward = attack_action.success_rate * self.attack_success_weight
        
        # 被检测惩罚
        detection_prob = self._compute_detection_probability(
            attack_action, defense_policy, defense_library
        )
        detection_cost = detection_prob * self.detection_penalty
        
        # 拥塞成本(同类攻击过多导致被针对)
        attack_type_freq = mean_field.attack_type_distribution.get(
            attack_action.attack_type, 0.0
        )
        congestion_cost = self.congestion_factor * attack_type_freq
        
        # 努力成本
        effort_cost = attack_action.effort_cost
        
        utility = attack_reward - detection_cost - congestion_cost - effort_cost
        return utility
    
    def _compute_detection_probability(self,
                                        attack_action: AttackAction,
                                        defense_policy: np.ndarray,
                                        defense_library: DefenseActionLibrary
                                        ) -> float:
        """计算被检测概率"""
        detection_prob = 0.0
        
        actions = defense_library.get_all_actions()
        for i, action in enumerate(actions):
            if attack_action.attack_type in action.coverage:
                detection_prob += defense_policy[i] * action.risk_reduction
        
        # 考虑攻击的规避能力
        detection_prob *= (1 - attack_action.detection_evasion)
        
        return min(detection_prob, 1.0)
    
    def compute_best_response(self,
                               defense_policy: np.ndarray,
                               mean_field: MeanFieldState,
                               attack_actions: List[AttackAction],
                               defense_library: DefenseActionLibrary
                               ) -> np.ndarray:
        """
        计算攻击者最佳响应
        
        π_a^* = argmax_{π_a} E[U_a(a, π_d, m)]
        """
        utilities = []
        for attack in attack_actions:
            util = self.compute_attacker_utility(
                attack, defense_policy, mean_field, defense_library
            )
            utilities.append(util)
        
        utilities = np.array(utilities)
        
        # Softmax策略(有界理性)
        temperature = self.config.get('temperature', 0.5)
        exp_utils = np.exp(utilities / temperature)
        attack_policy = exp_utils / np.sum(exp_utils)
        
        return attack_policy
    
    def update_mean_field(self,
                          attack_policy: np.ndarray,
                          attack_actions: List[AttackAction],
                          current_mf: MeanFieldState
                          ) -> MeanFieldState:
        """
        更新均值场分布
        
        m' = Φ(π_a, m)
        """
        new_mf = MeanFieldState()
        
        # 更新攻击类型分布
        for i, attack in enumerate(attack_actions):
            prob = attack_policy[i]
            attack_type = attack.attack_type
            new_mf.attack_type_distribution[attack_type] = \
                new_mf.attack_type_distribution.get(attack_type, 0.0) + prob
        
        # 更新模态分布
        for i, attack in enumerate(attack_actions):
            prob = attack_policy[i]
            modality = attack.modality
            new_mf.modality_distribution[modality] = \
                new_mf.modality_distribution.get(modality, 0.0) + prob
        
        # 平滑更新(与当前分布混合)
        alpha = self.config.get('mf_update_rate', 0.5)
        
        for key in new_mf.attack_type_distribution:
            new_mf.attack_type_distribution[key] = (
                alpha * new_mf.attack_type_distribution[key] +
                (1 - alpha) * current_mf.attack_type_distribution.get(key, 0.0)
            )
        
        for key in new_mf.modality_distribution:
            new_mf.modality_distribution[key] = (
                alpha * new_mf.modality_distribution[key] +
                (1 - alpha) * current_mf.modality_distribution.get(key, 0.0)
            )
        
        # 更新统计量
        new_mf.mean_risk_score = alpha * np.mean([
            a.success_rate * attack_policy[i] 
            for i, a in enumerate(attack_actions)
        ]) + (1 - alpha) * current_mf.mean_risk_score
        
        return new_mf
    
    def solve_mfg_equilibrium(self,
                               defense_policy: np.ndarray,
                               attack_actions: List[AttackAction],
                               defense_library: DefenseActionLibrary,
                               initial_mf: Optional[MeanFieldState] = None
                               ) -> Tuple[np.ndarray, MeanFieldState]:
        """
        求解均值场均衡
        
        找到满足以下条件的(π_a^*, m^*):
        1. π_a^* 是给定 m^* 下的最佳响应
        2. m^* 是 π_a^* 诱导的不动点分布
        
        Returns:
            (攻击策略, 均值场分布)
        """
        # 初始化
        if initial_mf is None:
            mean_field = MeanFieldState()
            # 均匀初始化
            n_attacks = len(attack_actions)
            for attack in attack_actions:
                mean_field.attack_type_distribution[attack.attack_type] = 1.0 / n_attacks
                mean_field.modality_distribution[attack.modality] = 1.0 / n_attacks
        else:
            mean_field = copy.deepcopy(initial_mf)
        
        attack_policy = np.ones(len(attack_actions)) / len(attack_actions)
        
        # 迭代求解
        for iteration in range(self.max_iterations):
            # 1. 计算最佳响应
            new_attack_policy = self.compute_best_response(
                defense_policy, mean_field, attack_actions, defense_library
            )
            
            # 2. 更新均值场
            new_mean_field = self.update_mean_field(
                new_attack_policy, attack_actions, mean_field
            )
            
            # 3. 检查收敛
            policy_diff = np.linalg.norm(new_attack_policy - attack_policy)
            mf_diff = np.linalg.norm(
                new_mean_field.to_vector() - mean_field.to_vector()
            )
            
            if policy_diff < self.convergence_tol and mf_diff < self.convergence_tol:
                break
            
            attack_policy = new_attack_policy
            mean_field = new_mean_field
        
        return attack_policy, mean_field


class StackelbergMFGSolver:
    """
    Stackelberg-MFG 求解器
    
    实现 Algorithm 2: 离线SMFE求解 + 在线策略匹配
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化求解器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        
        # 初始化组件
        self.defense_library = DefenseActionLibrary()
        self.mfg_solver = MeanFieldGameSolver(self.config)
        
        # 初始化攻击动作库
        self.attack_actions = self._init_attack_actions()
        
        # 求解参数
        self.outer_tol = self.config.get('mfg_outer_tol', 1e-3)
        self.outer_max_iter = self.config.get('outer_max_iter', 100)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        
        # 防御者效用参数
        self.risk_weight = self.config.get('risk_weight', 1.0)
        self.cost_weight = self.config.get('cost_weight', 0.3)
        
        # 求解结果缓存
        self.optimal_policy: Optional[np.ndarray] = None
        self.optimal_mf: Optional[MeanFieldState] = None
        
        # 策略映射表(用于在线匹配)
        self.policy_table: Dict[str, np.ndarray] = {}
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """加载配置"""
        default_config = {
            'mfg_inner_tol': 1e-4,
            'mfg_outer_tol': 1e-3,
            'mfg_max_iter': 100,
            'outer_max_iter': 100,
            'learning_rate': 0.01,
            'gamma': 0.95,
            'risk_weight': 1.0,
            'cost_weight': 0.3,
            'temperature': 0.5,
            'mf_update_rate': 0.5
        }
        
        if config_path:
            import yaml
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
            except:
                pass
        
        return default_config
    
    def _init_attack_actions(self) -> List[AttackAction]:
        """初始化攻击动作库"""
        return [
            AttackAction("A_001", "instruction_override", "text", 0.6, 0.5, 0.3),
            AttackAction("A_002", "instruction_override", "image", 0.5, 0.7, 0.5),
            AttackAction("A_003", "jailbreak_attempt", "text", 0.4, 0.8, 0.2),
            AttackAction("A_004", "jailbreak_attempt", "image", 0.35, 0.9, 0.4),
            AttackAction("A_005", "roleplay_injection", "text", 0.55, 0.4, 0.35),
            AttackAction("A_006", "context_pollution", "text", 0.5, 0.6, 0.25),
            AttackAction("A_007", "context_pollution", "image", 0.45, 0.7, 0.45),
            AttackAction("A_008", "tool_param_manipulation", "text", 0.65, 0.5, 0.2),
            AttackAction("A_009", "data_exfiltration", "text", 0.3, 0.9, 0.15),
            AttackAction("A_010", "privilege_escalation", "text", 0.35, 0.85, 0.1),
        ]
    
    def compute_defender_utility(self,
                                  defense_policy: np.ndarray,
                                  attack_policy: np.ndarray,
                                  mean_field: MeanFieldState
                                  ) -> float:
        """
        计算防御者效用
        
        U_d(π_d, π_a, m) = -Risk(m, π_a, π_d) - Cost(π_d)
        """
        # 计算剩余风险
        residual_risk = self._compute_residual_risk(
            defense_policy, attack_policy, mean_field
        )
        
        # 计算部署成本
        deployment_cost = self._compute_deployment_cost(defense_policy)
        
        # 效用 = -风险 - 成本
        utility = -self.risk_weight * residual_risk - self.cost_weight * deployment_cost
        
        return utility
    
    def _compute_residual_risk(self,
                                defense_policy: np.ndarray,
                                attack_policy: np.ndarray,
                                mean_field: MeanFieldState
                                ) -> float:
        """计算剩余风险"""
        total_risk = 0.0
        
        actions = self.defense_library.get_all_actions()
        
        for i, attack in enumerate(self.attack_actions):
            attack_prob = attack_policy[i]
            attack_success = attack.success_rate
            
            # 计算防御效果
            defense_effect = 0.0
            for j, defense in enumerate(actions):
                if attack.attack_type in defense.coverage:
                    defense_effect += defense_policy[j] * defense.risk_reduction
            
            # 剩余风险
            residual = attack_prob * attack_success * (1 - min(defense_effect, 1.0))
            total_risk += residual
        
        return total_risk
    
    def _compute_deployment_cost(self, defense_policy: np.ndarray) -> float:
        """计算部署成本"""
        total_cost = 0.0
        actions = self.defense_library.get_all_actions()
        
        for i, action in enumerate(actions):
            total_cost += defense_policy[i] * action.total_cost
        
        return total_cost
    
    def offline_solve(self, 
                      max_iterations: Optional[int] = None,
                      verbose: bool = True
                      ) -> np.ndarray:
        """
        离线阶段: SMFE近似求解
        
        Algorithm 2 离线部分:
        1. 内层MFG闭合 (固定π_d, 求解攻击者均值场均衡)
        2. 外层Leader更新 (根据均衡更新防御策略)
        
        Args:
            max_iterations: 最大迭代次数
            verbose: 是否输出日志
        
        Returns:
            最优防御策略
        """
        if max_iterations is None:
            max_iterations = self.outer_max_iter
        
        # 初始化防御策略(均匀分布)
        n_actions = self.defense_library.action_space_size
        defense_policy = np.ones(n_actions) / n_actions
        
        # 初始化均值场
        mean_field = MeanFieldState()
        
        best_utility = float('-inf')
        best_policy = defense_policy.copy()
        
        if verbose:
            print("=" * 60)
            print("CM-PIUG Stackelberg-MFG Offline Solving")
            print("=" * 60)
        
        for outer_iter in range(max_iterations):
            # ===== 内层: MFG闭合 =====
            attack_policy, mean_field = self.mfg_solver.solve_mfg_equilibrium(
                defense_policy,
                self.attack_actions,
                self.defense_library,
                mean_field
            )
            
            # ===== 外层: Leader策略更新 =====
            # 计算当前效用
            current_utility = self.compute_defender_utility(
                defense_policy, attack_policy, mean_field
            )
            
            # 梯度估计(数值梯度)
            grad = self._estimate_gradient(defense_policy, mean_field)
            
            # 策略更新
            new_policy = defense_policy + self.learning_rate * grad
            
            # 投影到单纯形
            new_policy = self._project_to_simplex(new_policy)
            
            # 收敛检查
            policy_diff = np.linalg.norm(new_policy - defense_policy)
            
            if current_utility > best_utility:
                best_utility = current_utility
                best_policy = new_policy.copy()
            
            if verbose and outer_iter % 10 == 0:
                print(f"Iter {outer_iter:3d} | Utility: {current_utility:.4f} | "
                      f"Policy diff: {policy_diff:.6f}")
            
            if policy_diff < self.outer_tol:
                if verbose:
                    print(f"Converged at iteration {outer_iter}")
                break
            
            defense_policy = new_policy
        
        # 保存结果
        self.optimal_policy = best_policy
        self.optimal_mf = mean_field
        
        # 构建策略映射表
        self._build_policy_table()
        
        if verbose:
            print("=" * 60)
            print("Optimal Defense Policy:")
            actions = self.defense_library.get_all_actions()
            for i, action in enumerate(actions):
                if best_policy[i] > 0.01:
                    print(f"  {action.name}: {best_policy[i]:.3f}")
            print(f"Final Utility: {best_utility:.4f}")
            print("=" * 60)
        
        return best_policy
    
    def _estimate_gradient(self,
                            defense_policy: np.ndarray,
                            mean_field: MeanFieldState
                            ) -> np.ndarray:
        """估计效用关于策略的梯度"""
        eps = 1e-4
        n = len(defense_policy)
        grad = np.zeros(n)
        
        for i in range(n):
            # 正向扰动
            policy_plus = defense_policy.copy()
            policy_plus[i] += eps
            policy_plus = self._project_to_simplex(policy_plus)
            
            attack_plus, mf_plus = self.mfg_solver.solve_mfg_equilibrium(
                policy_plus, self.attack_actions, self.defense_library, mean_field
            )
            util_plus = self.compute_defender_utility(policy_plus, attack_plus, mf_plus)
            
            # 负向扰动
            policy_minus = defense_policy.copy()
            policy_minus[i] -= eps
            policy_minus = self._project_to_simplex(policy_minus)
            
            attack_minus, mf_minus = self.mfg_solver.solve_mfg_equilibrium(
                policy_minus, self.attack_actions, self.defense_library, mean_field
            )
            util_minus = self.compute_defender_utility(policy_minus, attack_minus, mf_minus)
            
            # 中心差分
            grad[i] = (util_plus - util_minus) / (2 * eps)
        
        return grad
    
    def _project_to_simplex(self, vec: np.ndarray) -> np.ndarray:
        """投影到单纯形(概率分布)"""
        # 简单截断+归一化
        vec = np.maximum(vec, 0)
        total = np.sum(vec)
        if total > 0:
            return vec / total
        else:
            return np.ones_like(vec) / len(vec)
    
    def _build_policy_table(self) -> None:
        """构建策略查找表"""
        # 根据均值场特征构建离散化的策略映射
        # 这里使用简化版本:按风险等级分类
        
        risk_levels = ['low', 'medium', 'high', 'critical']
        
        for i, level in enumerate(risk_levels):
            # 根据风险等级调整策略权重
            adjusted_policy = self.optimal_policy.copy()
            
            # 高风险时增加强防御的权重
            risk_factor = (i + 1) / len(risk_levels)
            
            actions = self.defense_library.get_all_actions()
            for j, action in enumerate(actions):
                if action.risk_reduction > 0.7:
                    adjusted_policy[j] *= (1 + risk_factor)
                elif action.risk_reduction < 0.5:
                    adjusted_policy[j] *= (1 - 0.5 * risk_factor)
            
            adjusted_policy = self._project_to_simplex(adjusted_policy)
            self.policy_table[level] = adjusted_policy
    
    def online_match(self, 
                     evidence_chain: List[str],
                     risk_score: float,
                     triggered_rules: List[str]
                     ) -> Tuple[str, float]:
        """
        在线阶段: 基于证据链的快速策略匹配
        
        Algorithm 2 在线部分:
        1. 运行Algorithm 1获取证据
        2. 构造均值场特征
        3. 匹配防御动作
        
        Args:
            evidence_chain: Algorithm 1输出的证据链
            risk_score: 风险分数
            triggered_rules: 触发的规则列表
        
        Returns:
            (选择的防御动作ID, 选择概率)
        """
        # 确定风险等级
        if risk_score < 0.3:
            risk_level = 'low'
        elif risk_score < 0.5:
            risk_level = 'medium'
        elif risk_score < 0.7:
            risk_level = 'high'
        else:
            risk_level = 'critical'
        
        # 获取对应策略
        if risk_level in self.policy_table:
            policy = self.policy_table[risk_level]
        elif self.optimal_policy is not None:
            policy = self.optimal_policy
        else:
            # 默认均匀策略
            n = self.defense_library.action_space_size
            policy = np.ones(n) / n
        
        # 根据证据链调整策略
        adjusted_policy = self._adjust_policy_by_evidence(
            policy, evidence_chain, triggered_rules
        )
        
        # 选择动作(采样或argmax)
        action_idx = np.random.choice(len(adjusted_policy), p=adjusted_policy)
        action = self.defense_library.get_all_actions()[action_idx]
        
        return action.action_id, float(adjusted_policy[action_idx])
    
    def _adjust_policy_by_evidence(self,
                                    base_policy: np.ndarray,
                                    evidence_chain: List[str],
                                    triggered_rules: List[str]
                                    ) -> np.ndarray:
        """根据证据调整策略"""
        adjusted = base_policy.copy()
        
        actions = self.defense_library.get_all_actions()
        
        # 根据证据链中的关键词调整
        evidence_str = ' '.join(evidence_chain).lower()
        
        for i, action in enumerate(actions):
            # 如果证据链包含该动作覆盖的攻击类型,增加权重
            for covered_type in action.coverage:
                if covered_type.lower() in evidence_str:
                    adjusted[i] *= 1.5
        
        # 根据触发规则调整
        for rule in triggered_rules:
            rule_lower = rule.lower()
            if 'jailbreak' in rule_lower:
                # 增加越狱防御权重
                for i, action in enumerate(actions):
                    if 'jailbreak' in str(action.coverage):
                        adjusted[i] *= 1.3
        
        return self._project_to_simplex(adjusted)
    
    def get_defense_recommendation(self,
                                    risk_score: float,
                                    evidence_chain: List[str]
                                    ) -> List[Dict]:
        """
        获取防御建议
        
        Args:
            risk_score: 风险分数
            evidence_chain: 证据链
        
        Returns:
            防御建议列表 [{action_id, name, probability, reason}, ...]
        """
        recommendations = []
        
        # 获取策略
        action_id, prob = self.online_match(evidence_chain, risk_score, [])
        
        actions = self.defense_library.get_all_actions()
        
        # 确定风险等级
        if risk_score >= 0.7:
            policy = self.policy_table.get('critical', self.optimal_policy)
        elif risk_score >= 0.5:
            policy = self.policy_table.get('high', self.optimal_policy)
        else:
            policy = self.policy_table.get('medium', self.optimal_policy)
        
        if policy is None:
            policy = np.ones(len(actions)) / len(actions)
        
        # 排序并返回top-k
        sorted_indices = np.argsort(policy)[::-1]
        
        for idx in sorted_indices[:5]:
            action = actions[idx]
            if policy[idx] > 0.05:
                recommendations.append({
                    'action_id': action.action_id,
                    'name': action.name,
                    'probability': float(policy[idx]),
                    'category': action.category,
                    'risk_reduction': action.risk_reduction,
                    'cost': action.total_cost
                })
        
        return recommendations


# ==================== 便捷函数 ====================

def solve_defense_strategy(config_path: Optional[str] = None,
                           verbose: bool = True) -> StackelbergMFGSolver:
    """
    求解防御策略
    
    Args:
        config_path: 配置文件路径
        verbose: 是否输出日志
    
    Returns:
        求解器实例(包含最优策略)
    """
    solver = StackelbergMFGSolver(config_path)
    solver.offline_solve(verbose=verbose)
    return solver


def get_online_defense(solver: StackelbergMFGSolver,
                       risk_score: float,
                       evidence_chain: List[str]) -> Dict:
    """
    获取在线防御动作
    
    Args:
        solver: 已训练的求解器
        risk_score: 风险分数
        evidence_chain: 证据链
    
    Returns:
        防御建议
    """
    recommendations = solver.get_defense_recommendation(risk_score, evidence_chain)
    
    if recommendations:
        return recommendations[0]
    else:
        return {'action_id': 'none', 'name': 'No action', 'probability': 1.0}
