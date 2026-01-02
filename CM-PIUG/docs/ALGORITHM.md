# CM-PIUG 算法文档

## 1. 理论基础

### 1.1 攻击图建模

CM-PIUG 使用统一攻击图 (Unified Attack Graph) 来表示跨模态提示注入攻击：

```
AG = (V, E, τ, R, F_init, G_goal)
```

其中：
- **V**: 节点集合，表示系统状态
- **E**: 边集合，表示状态转换
- **τ**: 节点类型函数
- **R**: 规则集合
- **F_init**: 初始事实集
- **G_goal**: 攻击目标集

### 1.2 节点类型

| 类型 | 说明 | 示例 |
|------|------|------|
| INPUT | 输入层 | 用户文本、图像、音频 |
| PARSE | 解析层 | OCR结果、ASR结果 |
| CONTROL | 控制层 | 指令注入、越狱尝试 |
| EXEC | 执行层 | 工具调用、API访问 |
| GOAL | 目标层 | 数据泄露、权限提升 |

### 1.3 语义等价性

定义语义等价关系 ≈_C：

```
u ≈_C v ⟺ Entail_C(u, v) ∧ Entail_C(v, u)
```

其中 Entail_C(u, v) 表示在上下文 C 下 u 蕴含 v。

## 2. 算法1：零样本检测

### 2.1 算法描述

```
Algorithm 1: ZeroShotDetection(x, C, τ)
Input: 多模态输入 x, 上下文 C, 阈值 τ
Output: 检测标志 flag, 风险分数 Risk(x), 证据链

1. 多模态解析
   对于每个模态 m ∈ {text, image, audio}:
     F_m ← Parser_m(x_m)  // OCR/ASR解析

2. 语义归约
   F_reduced ← {}
   对于每个事实 u ∈ ∪F_m:
     找到等价类 [u]_C = {v : u ≈_C v}
     F_reduced ← F_reduced ∪ {representative([u]_C)}

3. 规则推理
   Closure ← ForwardChain(F_reduced ∪ F_sys, R)

4. 图构建
   构建攻击图 AG 基于 Closure

5. 边置信度计算
   对于每条边 (u,v) ∈ E:
     c(u,v) = (1-λ)·c_rule(u,v) + λ·c_sem(u,v)

6. 风险计算
   Risk(x) = max_{p∈Paths(AG)} PathStrength(p)

7. 证据链提取
   chain ← BacktrackPath(AG, argmax path)

8. 返回
   flag ← (Risk(x) ≥ τ)
   return flag, Risk(x), chain
```

### 2.2 时间复杂度

```
O(|F|·|R|·k + |V|·|E|)
```

其中：
- |F|: 事实数量
- |R|: 规则数量
- k: 规则平均前提数
- |V|, |E|: 图的节点和边数

### 2.3 实现细节

```python
class ZeroShotDetector:
    def detect(self, input_data: Dict) -> DetectionResult:
        # 1. 多模态解析
        facts = self._parse_multimodal(input_data)
        
        # 2. 语义归约
        reduced_facts = self._semantic_reduction(facts)
        
        # 3. 前向链推理
        closure, fired_rules = self.rule_engine.forward_chain(reduced_facts)
        
        # 4. 构建攻击图
        self._build_attack_graph(closure, fired_rules)
        
        # 5. 计算风险
        risk_score = self.attack_graph.compute_risk_score()
        
        # 6. 提取证据链
        evidence_chain = self.attack_graph.extract_evidence_chain()
        
        return DetectionResult(
            flag=risk_score >= self.threshold,
            risk_score=risk_score,
            evidence_chain=evidence_chain,
            fired_rules=fired_rules
        )
```

## 3. 算法2：Stackelberg-MFG防御

### 3.1 博弈模型

**双层Stackelberg博弈**：
- 领导者（防御方）：选择防御策略 π_d
- 跟随者（攻击方）：选择攻击策略 π_a

**目标函数**：
```
max_{π_d} U_d(π_d, BR(π_d))
s.t. BR(π_d) = argmax_{π_a} U_a(π_a, π_d)
```

### 3.2 Mean Field Game 近似

当攻击者数量趋于无穷时，使用 MFG 近似：

```
Algorithm 2: StackelbergMFGSolver

离线阶段：
1. 初始化领导者策略 π_d^0
2. 重复直到收敛:
   a. 内层循环 (MFG均衡):
      - 固定 π_d，求解 MFG(π_d)
      - 得到攻击者分布 μ 和最优响应 π_a*
   b. 外层循环 (策略更新):
      - 计算梯度 ∇U_d(π_d, π_a*)
      - 更新 π_d ← π_d + α·∇U_d
3. 返回最优策略 π_d*

在线阶段：
1. 输入: 检测结果 (evidence_chain, risk_score, fired_rules)
2. 状态映射: s = StateMapping(risk_score)
3. 策略查询: action = π_d*(s)
4. 返回推荐动作
```

### 3.3 MFG均衡求解

```python
class MeanFieldGameSolver:
    def solve(self, max_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        policy = self._initialize_policy()
        distribution = self._initialize_distribution()
        
        for _ in range(max_iterations):
            # 1. 固定分布，求最优策略 (Best Response)
            new_policy = self._best_response(distribution)
            
            # 2. 固定策略，更新分布 (Forward Equation)
            new_distribution = self._update_distribution(new_policy)
            
            # 3. 检查收敛
            if self._converged(policy, new_policy, distribution, new_distribution):
                break
            
            policy = new_policy
            distribution = new_distribution
        
        return policy, distribution
```

### 3.4 防御动作库

| ID | 类别 | 描述 | 风险降低 | 成本 |
|----|------|------|---------|------|
| D_FILTER_001 | filter | 指令标记过滤 | 70% | 0.1 |
| D_FILTER_002 | filter | 越狱模式过滤 | 80% | 0.15 |
| D_REWRITE_001 | rewrite | 提示词清洗 | 60% | 0.2 |
| D_ISOLATE_001 | isolate | 工具调用沙箱 | 85% | 0.3 |
| D_ISOLATE_002 | isolate | API访问隔离 | 90% | 0.35 |
| D_AUDIT_001 | audit | 输出审计 | 50% | 0.25 |
| D_CONSTRAIN_001 | constrain | 工具参数约束 | 75% | 0.2 |

## 4. 规则引擎

### 4.1 Horn子句规则

规则格式：
```
pre_1 ∧ pre_2 ∧ ... ∧ pre_n → post
```

### 4.2 内置规则示例

```yaml
# 输入层规则
R_INPUT_INJECT_001:
  preconditions: [has_instruction_marker, user_input]
  postcondition: instruction_injection_attempt
  confidence: 0.9

# 上下文冲突规则
R_CTX_CONFLICT_001:
  preconditions: [instruction_injection_attempt, context_conflict]
  postcondition: confirmed_injection
  confidence: 0.85

# 工具调用规则
R_TOOL_HIJACK_001:
  preconditions: [confirmed_injection, tool_call_present]
  postcondition: tool_hijack_attempt
  confidence: 0.88

# 目标达成规则
R_GOAL_LEAK_001:
  preconditions: [tool_hijack_attempt, data_access]
  postcondition: data_leak_goal
  confidence: 0.92
```

### 4.3 前向链推理

```python
def forward_chain(self, initial_facts: Set[str]) -> Tuple[Set[str], List[str]]:
    """
    前向链推理
    
    Args:
        initial_facts: 初始事实集
        
    Returns:
        (闭包, 触发的规则列表)
    """
    closure = set(initial_facts)
    fired_rules = []
    changed = True
    
    while changed:
        changed = False
        for rule in self.rules:
            # 检查前提条件是否满足
            if all(pre in closure for pre in rule.preconditions):
                # 检查结论是否已在闭包中
                if rule.postcondition not in closure:
                    closure.add(rule.postcondition)
                    fired_rules.append(rule.rule_id)
                    changed = True
    
    return closure, fired_rules
```

## 5. 性能优化

### 5.1 规则索引

使用倒排索引加速规则匹配：

```python
self.precondition_index = defaultdict(set)
for rule in self.rules:
    for pre in rule.preconditions:
        self.precondition_index[pre].add(rule.rule_id)
```

### 5.2 图剪枝

在BFS搜索中剪枝低置信度路径：

```python
def bfs_with_pruning(self, start, threshold=0.1):
    queue = [(start, 1.0)]
    while queue:
        node, path_strength = queue.pop(0)
        if path_strength < threshold:
            continue  # 剪枝
        for neighbor in self.adjacency[node]:
            edge_conf = self.get_edge_confidence(node, neighbor)
            new_strength = path_strength * edge_conf
            queue.append((neighbor, new_strength))
```

### 5.3 缓存策略

对语义等价计算结果进行缓存：

```python
@lru_cache(maxsize=1000)
def semantic_equivalence(self, text1: str, text2: str) -> bool:
    return self._compute_equivalence(text1, text2)
```

## 6. 评估指标

| 指标 | 公式 | 说明 |
|------|------|------|
| Detection F1 | 2×P×R/(P+R) | 检测的F1分数 |
| ASR | 攻击成功数/总攻击数 | 攻击成功率 |
| FPR@95TPR | FP/(FP+TN) when TPR=0.95 | 95%召回率时的误报率 |
| Defense Cost | Σ cost(action) | 防御总成本 |
| Risk Reduction | 1 - Risk_after/Risk_before | 风险降低比例 |
