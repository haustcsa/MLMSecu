"""
CM-PIUG Semantic Equivalence Module
语义等价判定模块

基于文本蕴含(Textual Entailment)与语义熵(Semantic Entropy)
实现跨模态语义对齐

理论基础:
- 上下文化蕴含: Entail_C(u, v) 
- 语义等价: u ≈_C v ⟺ Entail_C(u, v) ∧ Entail_C(v, u)
- 语义熵: SE(x) = -∑_{k} P(C_k) log P(C_k)
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import hashlib


@dataclass
class SemanticUnit:
    """语义单元"""
    unit_id: str
    content: str
    embedding: Optional[np.ndarray] = None
    source_modality: str = "text"  # text, image_ocr, audio_asr
    confidence: float = 1.0
    cluster_id: Optional[int] = None
    
    def __hash__(self):
        return hash(self.unit_id)


@dataclass
class EntailmentResult:
    """蕴含判定结果"""
    premise: str
    hypothesis: str
    entails: bool
    confidence: float
    bidirectional: bool = False  # 双向蕴含(语义等价)


class SemanticEquivalenceChecker:
    """
    语义等价检查器
    
    实现上下文化蕴含判定与语义等价归约
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # 蕴含判定阈值
        self.entailment_threshold = self.config.get('entailment_threshold', 0.7)
        
        # 语义相似度阈值
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        
        # 语义熵阈值(低于此值认为语义集中)
        self.entropy_threshold = self.config.get('entropy_threshold', 0.5)
        
        # 缓存
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._entailment_cache: Dict[Tuple[str, str], EntailmentResult] = {}
        
        # 模拟的NLI模型(实际应用中替换为真实模型)
        self._nli_model = None
        self._encoder_model = None
        
        # 是否使用真实模型
        self.use_real_model = self.config.get('use_real_model', False)
        
        if self.use_real_model:
            self._init_models()
    
    def _init_models(self):
        """初始化NLI和编码模型"""
        try:
            from sentence_transformers import SentenceTransformer
            from transformers import pipeline
            
            # 语义编码器
            self._encoder_model = SentenceTransformer(
                self.config.get('encoder_model', 'all-MiniLM-L6-v2')
            )
            
            # NLI模型
            self._nli_model = pipeline(
                "text-classification",
                model=self.config.get('nli_model', 'facebook/bart-large-mnli')
            )
        except ImportError:
            print("Warning: transformers/sentence-transformers not available, using simulation")
            self.use_real_model = False
    
    def compute_embedding(self, text: str) -> np.ndarray:
        """
        计算文本的语义嵌入
        
        Args:
            text: 输入文本
        
        Returns:
            嵌入向量
        """
        # 检查缓存
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        if self.use_real_model and self._encoder_model:
            embedding = self._encoder_model.encode(text)
        else:
            # 模拟嵌入(用于演示)
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(384)
            embedding = embedding / np.linalg.norm(embedding)
        
        self._embedding_cache[cache_key] = embedding
        return embedding
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两段文本的语义相似度
        
        Args:
            text1: 文本1
            text2: 文本2
        
        Returns:
            相似度分数 [0, 1]
        """
        emb1 = self.compute_embedding(text1)
        emb2 = self.compute_embedding(text2)
        
        # 余弦相似度
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        return float(max(0, similarity))
    
    def check_entailment(self,
                         premise: str,
                         hypothesis: str,
                         context: Optional[str] = None
                         ) -> EntailmentResult:
        """
        上下文化蕴含判定
        
        Entail_C(u, v): 在上下文C下,文本u是否蕴含v
        
        Args:
            premise: 前提文本 u
            hypothesis: 假设文本 v
            context: 上下文 C (system prompt, tool schema等)
        
        Returns:
            蕴含判定结果
        """
        # 缓存检查
        cache_key = (premise, hypothesis)
        if cache_key in self._entailment_cache:
            return self._entailment_cache[cache_key]
        
        # 构建上下文化输入
        if context:
            full_premise = f"Context: {context}\nText: {premise}"
        else:
            full_premise = premise
        
        if self.use_real_model and self._nli_model:
            # 使用真实NLI模型
            result = self._nli_model(
                f"{full_premise} [SEP] {hypothesis}",
                candidate_labels=["entailment", "neutral", "contradiction"]
            )
            
            entails = result['labels'][0] == 'entailment'
            confidence = result['scores'][0] if entails else 1 - result['scores'][0]
        else:
            # 模拟蕴含判定(基于相似度)
            similarity = self.compute_similarity(premise, hypothesis)
            
            # 添加一些启发式规则
            if any(kw in hypothesis.lower() for kw in ['ignore', 'forget', 'override']):
                if any(kw in premise.lower() for kw in ['ignore', 'forget', 'override']):
                    entails = True
                    confidence = max(similarity, 0.8)
                else:
                    entails = False
                    confidence = 0.3
            else:
                entails = similarity >= self.entailment_threshold
                confidence = similarity
        
        result = EntailmentResult(
            premise=premise,
            hypothesis=hypothesis,
            entails=entails,
            confidence=confidence
        )
        
        self._entailment_cache[cache_key] = result
        return result
    
    def check_semantic_equivalence(self,
                                   text1: str,
                                   text2: str,
                                   context: Optional[str] = None
                                   ) -> Tuple[bool, float]:
        """
        语义等价判定
        
        u ≈_C v ⟺ Entail_C(u, v) ∧ Entail_C(v, u)
        
        Args:
            text1: 文本1
            text2: 文本2
            context: 上下文
        
        Returns:
            (是否语义等价, 置信度)
        """
        # 检查双向蕴含
        ent_1_to_2 = self.check_entailment(text1, text2, context)
        ent_2_to_1 = self.check_entailment(text2, text1, context)
        
        is_equivalent = ent_1_to_2.entails and ent_2_to_1.entails
        confidence = min(ent_1_to_2.confidence, ent_2_to_1.confidence)
        
        return is_equivalent, confidence
    
    def merge_equivalent_units(self,
                               units: List[SemanticUnit],
                               context: Optional[str] = None
                               ) -> List[SemanticUnit]:
        """
        合并语义等价的单元
        
        实现等价类归约: Φ(u) → [u]_≈
        
        Args:
            units: 语义单元列表
            context: 上下文
        
        Returns:
            合并后的单元列表(每个等价类保留一个代表元)
        """
        if len(units) <= 1:
            return units
        
        # 使用Union-Find进行等价类合并
        parent = {u.unit_id: u.unit_id for u in units}
        unit_map = {u.unit_id: u for u in units}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py
        
        # 检查所有对的语义等价性
        for i, u1 in enumerate(units):
            for u2 in units[i+1:]:
                is_eq, conf = self.check_semantic_equivalence(
                    u1.content, u2.content, context
                )
                if is_eq and conf >= self.similarity_threshold:
                    union(u1.unit_id, u2.unit_id)
        
        # 收集等价类代表元
        representatives = {}
        for unit in units:
            root = find(unit.unit_id)
            if root not in representatives:
                representatives[root] = unit
            else:
                # 选择置信度更高的作为代表
                if unit.confidence > representatives[root].confidence:
                    representatives[root] = unit
        
        return list(representatives.values())


class SemanticEntropyCalculator:
    """
    语义熵计算器
    
    基于语义熵量化意义级不确定性
    SE(x) = -∑_{k} P(C_k) log P(C_k)
    """
    
    def __init__(self, 
                 equivalence_checker: SemanticEquivalenceChecker,
                 config: Optional[Dict] = None):
        self.equiv_checker = equivalence_checker
        self.config = config or {}
        
        # 熵阈值
        self.low_entropy_threshold = self.config.get('low_entropy_threshold', 0.5)
    
    def cluster_by_semantic(self,
                            candidates: List[Tuple[str, float]],
                            context: Optional[str] = None
                            ) -> Dict[int, List[Tuple[str, float]]]:
        """
        按语义等价关系聚类
        
        将候选文本按双向蕴含关系聚合到语义簇
        
        Args:
            candidates: [(文本, 权重/置信度), ...]
            context: 上下文
        
        Returns:
            {簇ID: [(文本, 权重), ...]}
        """
        if not candidates:
            return {}
        
        clusters: Dict[int, List[Tuple[str, float]]] = {}
        cluster_representatives: Dict[int, str] = {}
        next_cluster_id = 0
        
        for text, weight in candidates:
            assigned = False
            
            # 尝试分配到已有簇
            for cid, rep in cluster_representatives.items():
                is_eq, _ = self.equiv_checker.check_semantic_equivalence(
                    text, rep, context
                )
                if is_eq:
                    clusters[cid].append((text, weight))
                    assigned = True
                    break
            
            # 创建新簇
            if not assigned:
                clusters[next_cluster_id] = [(text, weight)]
                cluster_representatives[next_cluster_id] = text
                next_cluster_id += 1
        
        return clusters
    
    def compute_semantic_entropy(self,
                                  candidates: List[Tuple[str, float]],
                                  context: Optional[str] = None
                                  ) -> Tuple[float, Dict[int, float]]:
        """
        计算语义熵
        
        SE(x) = -∑_{k} P(C_k) log P(C_k)
        
        Args:
            candidates: [(文本, 权重/置信度), ...]
            context: 上下文
        
        Returns:
            (语义熵值, {簇ID: 簇概率质量})
        """
        # 聚类
        clusters = self.cluster_by_semantic(candidates, context)
        
        if not clusters:
            return 0.0, {}
        
        # 计算每个簇的概率质量
        total_weight = sum(w for text, w in candidates)
        if total_weight == 0:
            total_weight = len(candidates)  # 等权重
        
        cluster_probs = {}
        for cid, members in clusters.items():
            cluster_weight = sum(w for _, w in members)
            cluster_probs[cid] = cluster_weight / total_weight
        
        # 计算熵
        entropy = 0.0
        for prob in cluster_probs.values():
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy, cluster_probs
    
    def is_semantically_concentrated(self,
                                      candidates: List[Tuple[str, float]],
                                      context: Optional[str] = None
                                      ) -> Tuple[bool, float]:
        """
        判断候选是否语义集中
        
        当语义熵较低时,说明候选虽表述不同但意义一致
        
        Args:
            candidates: 候选列表
            context: 上下文
        
        Returns:
            (是否语义集中, 语义熵值)
        """
        entropy, _ = self.compute_semantic_entropy(candidates, context)
        is_concentrated = entropy < self.low_entropy_threshold
        return is_concentrated, entropy
    
    def select_representative(self,
                               candidates: List[Tuple[str, float]],
                               context: Optional[str] = None
                               ) -> Tuple[str, float]:
        """
        选择代表性候选
        
        当语义集中时,返回权重最高的候选作为代表
        当语义分散时,返回None表示需要多候选处理
        
        Args:
            candidates: 候选列表
            context: 上下文
        
        Returns:
            (代表文本, 语义熵)
        """
        is_concentrated, entropy = self.is_semantically_concentrated(
            candidates, context
        )
        
        if is_concentrated:
            # 选择权重最高的
            best_text, _ = max(candidates, key=lambda x: x[1])
            return best_text, entropy
        else:
            # 返回空表示需要多候选处理
            return None, entropy


class CrossModalAligner:
    """
    跨模态对齐器
    
    将不同模态输入(图像OCR、音频ASR)与原生文本对齐到统一语义表示
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.equiv_checker = SemanticEquivalenceChecker(config)
        self.entropy_calculator = SemanticEntropyCalculator(
            self.equiv_checker, config
        )
    
    def align_multimodal_inputs(self,
                                inputs: Dict[str, List[Tuple[str, float]]],
                                context: Optional[str] = None
                                ) -> List[SemanticUnit]:
        """
        对齐多模态输入
        
        Args:
            inputs: {
                'text': [(文本, 置信度), ...],
                'image_ocr': [(OCR文本, 置信度), ...],
                'audio_asr': [(ASR文本, 置信度), ...]
            }
            context: 上下文
        
        Returns:
            对齐后的语义单元列表
        """
        all_units = []
        unit_counter = 0
        
        for modality, candidates in inputs.items():
            # 计算语义熵
            is_concentrated, entropy = self.entropy_calculator.is_semantically_concentrated(
                candidates, context
            )
            
            if is_concentrated:
                # 语义集中:合并为单一单元
                best_text, _ = max(candidates, key=lambda x: x[1])
                avg_conf = np.mean([c for _, c in candidates])
                
                unit = SemanticUnit(
                    unit_id=f"sem_unit_{unit_counter}",
                    content=best_text,
                    source_modality=modality,
                    confidence=avg_conf
                )
                all_units.append(unit)
                unit_counter += 1
            else:
                # 语义分散:保留多候选
                clusters = self.entropy_calculator.cluster_by_semantic(
                    candidates, context
                )
                
                for cid, members in clusters.items():
                    rep_text, _ = max(members, key=lambda x: x[1])
                    cluster_conf = np.mean([c for _, c in members])
                    
                    unit = SemanticUnit(
                        unit_id=f"sem_unit_{unit_counter}",
                        content=rep_text,
                        source_modality=modality,
                        confidence=cluster_conf,
                        cluster_id=cid
                    )
                    all_units.append(unit)
                    unit_counter += 1
        
        # 跨模态等价合并
        merged_units = self.equiv_checker.merge_equivalent_units(
            all_units, context
        )
        
        return merged_units
    
    def compute_alignment_reliability(self,
                                      original_count: int,
                                      merged_count: int,
                                      avg_entropy: float
                                      ) -> float:
        """
        计算对齐可靠度
        
        ρ_sem: 语义归约可靠度
        
        Args:
            original_count: 原始候选数量
            merged_count: 合并后数量
            avg_entropy: 平均语义熵
        
        Returns:
            可靠度分数 [0, 1]
        """
        # 压缩比因子
        compression_factor = merged_count / max(original_count, 1)
        
        # 熵因子(低熵=高可靠)
        entropy_factor = np.exp(-avg_entropy)
        
        # 综合可靠度
        reliability = 0.6 * entropy_factor + 0.4 * (1 - compression_factor)
        
        return float(np.clip(reliability, 0, 1))
