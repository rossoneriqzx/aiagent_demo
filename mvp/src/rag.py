"""
rag.py - 标准版 RAG 子系统（适配当前 config.py & llm_client.py）

特性：
- 使用 KnowledgeBaseConfig / EmbeddingConfig / AgentConfig（完全兼容你的 config.py）
- 支持用户画像（UserProfile）与兴趣 / 专业水平分析
- 支持通用向量库 + 用户专属向量库（PersonalizedVectorStore）
- 支持 chromadb 持久化存储
- 支持 OpenAI-compatible Embedding（OPENAI / QWEN / DEEPSEEK / CUSTOM）
- 提供统一的 context 拼接接口，方便直接喂给 LLM
- 在 knowledge_base.enabled=False 时自动短路，不调用向量库
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import md5
from typing import Any, Dict, List, Optional, Tuple

from config import KnowledgeBaseConfig, EmbeddingConfig, EmbeddingProvider, AgentConfig

logger = logging.getLogger(__name__)

# ========= 基础数据结构 =========

class InteractionType(str, Enum):
    QUERY = "query"
    FEEDBACK = "feedback"
    CLICK = "click"
    VIEW = "view"
    RATING = "rating"


@dataclass
class RetrievedDocument:
    """RAG 检索结果的统一结构"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source: str = "general"  # general, user_specific, learned


@dataclass
class UserProfile:
    """动态用户画像（进程内）"""
    user_id: str
    interests: List[str] = field(default_factory=list)
    expertise_level: str = "beginner"  # beginner, intermediate, expert
    query_history: List[Dict] = field(default_factory=list)
    feedback_history: List[Dict] = field(default_factory=list)
    preferred_topics: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)

    def to_metadata(self) -> Dict[str, Any]:
        """将画像转换为元数据格式（可选写回向量库）"""
        return {
            "interests": self.interests,
            "expertise_level": self.expertise_level,
            "preferred_topics": self.preferred_topics,
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class PersonalizedResponse:
    """RAG + 画像 的输出结构"""
    documents: List[RetrievedDocument]
    user_profile: UserProfile
    enhanced_query: str
    suggested_follow_ups: List[str]


# ========= 用户画像分析器 =========

class UserProfileAnalyzer:
    """分析用户行为并更新用户画像"""

    def __init__(self):
        self.interest_keywords = self._load_interest_keywords()

    def _load_interest_keywords(self) -> Dict[str, List[str]]:
        """兴趣关键词分类（可以将来配置化）"""
        return {
            "investment": ["股票", "基金", "投资", "理财", "portfolio", "investment"],
            "technology": ["AI", "人工智能", "科技", "技术", "innovation"],
            "economics": ["经济", "宏观", "GDP", "通胀", "货币政策"],
            # 可扩展更多分类
        }

    def extract_interests_from_query(self, query: str) -> List[str]:
        """从查询中提取兴趣方向"""
        found_interests = []
        for interest, keywords in self.interest_keywords.items():
            if any(keyword in query for keyword in keywords):
                found_interests.append(interest)
        return found_interests

    def analyze_expertise_level(self, query: str, documents: List[RetrievedDocument]) -> str:
        """基于查询 & 检索结果分析用户专业水平（启发式）"""
        technical_terms = ["量化", "对冲", "衍生品", "alpha", "beta", "夏普比率"]

        technical_term_count = sum(1 for term in technical_terms if term in query)
        avg_document_complexity = sum(len(doc.text.split()) for doc in documents) / max(len(documents), 1)

        if technical_term_count >= 2 or avg_document_complexity > 200:
            return "expert"
        elif technical_term_count >= 1 or avg_document_complexity > 100:
            return "intermediate"
        else:
            return "beginner"

    def update_profile_from_interaction(
        self,
        profile: UserProfile,
        query: str,
        documents: List[RetrievedDocument],
        feedback: Optional[float] = None,
    ) -> None:
        """基于一次交互更新画像"""
        # 更新兴趣
        new_interests = self.extract_interests_from_query(query)
        profile.interests = list(set(profile.interests + new_interests))[:10]

        # 更新专业水平
        profile.expertise_level = self.analyze_expertise_level(query, documents)

        # 记录查询历史
        profile.query_history.append(
            {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "retrieved_count": len(documents),
                "feedback": feedback,
            }
        )
        if len(profile.query_history) > 100:
            profile.query_history = profile.query_history[-100:]

        profile.last_updated = datetime.now()


# ========= Embedding 客户端 =========

class EnhancedEmbeddingClient:
    """
    增强版 Embedding 客户端
    - 使用 OpenAI-compatible 接口
    - 支持 OPENAI / QWEN / DEEPSEEK / CUSTOM
    - 提供简单缓存（避免重复嵌入）
    """

    def __init__(self, cfg: EmbeddingConfig):
        self.cfg = cfg
        self._client = None
        self._cache: Dict[str, List[List[float]]] = {}
        self._max_cache_size = 1000
        self._setup_client()

    def _setup_client(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError("缺少 openai 依赖，请先 `pip install openai`") from e

        if not self.cfg.api_key:
            raise RuntimeError("EmbeddingConfig.api_key 未配置，请检查环境变量或 config.yaml")

        # 对于 QWEN / DEEPSEEK / CUSTOM，大多也走 OpenAI-compatible HTTP 接口
        self._client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url,  # 建议在 config.yaml 里显式配置
            timeout=self.cfg.timeout,
            max_retries=self.cfg.max_retries,
        )

    def _make_cache_key(self, texts: List[str]) -> str:
        h = md5("|".join(texts).encode("utf-8")).hexdigest()
        return f"{self.cfg.provider}:{self.cfg.model_name}:{h}"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量生成文本嵌入（带简单缓存）"""
        if not texts:
            return []

        provider = self.cfg.provider

        if provider not in {
            EmbeddingProvider.OPENAI,
            EmbeddingProvider.QWEN,
            EmbeddingProvider.DEEPSEEK,
            EmbeddingProvider.CUSTOM,
        }:
            # JINA / BGE 等暂未实现，避免误用
            raise NotImplementedError(
                f"Embedding provider {provider} 暂未实现。"
                "当前仅支持 OPENAI/QWEN/DEEPSEEK/CUSTOM（OpenAI-compatible）。"
            )

        cache_key = self._make_cache_key(texts)
        if cache_key in self._cache:
            return self._cache[cache_key]

        logger.debug("调用 embedding 模型: %s, 条数: %d", self.cfg.model_name, len(texts))

        try:
            resp = self._client.embeddings.create(
                model=self.cfg.model_name,
                input=texts,
            )
        except Exception as e:
            logger.error("Embedding 调用失败: %s", e)
            raise

        vectors = [item.embedding for item in resp.data]

        # 自动回填维度信息
        if self.cfg.vector_size is None and vectors:
            self.cfg.vector_size = len(vectors[0])

        # 简单缓存
        if len(self._cache) >= self._max_cache_size:
            # 随机 pop 一项，简单起见
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = vectors

        return vectors


# ========= 向量存储抽象 & 实现 =========

class BaseVectorStore:
    """向量库抽象基类"""

    def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        raise NotImplementedError

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedDocument]:
        raise NotImplementedError


class ChromadbVectorStore(BaseVectorStore):
    """基于 chromadb 的向量存储"""

    def __init__(self, kb_cfg: KnowledgeBaseConfig):
        try:
            import chromadb
        except ImportError as e:
            raise ImportError("缺少 chromadb 依赖，请先 `pip install chromadb`") from e

        self.kb_cfg = kb_cfg

        if kb_cfg.storage_path:
            self.client = chromadb.PersistentClient(path=str(kb_cfg.storage_path))
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=kb_cfg.collection_name,
            metadata={"hnsw:space": "cosine", "description": "RAG knowledge base"},
        )

    def add_documents(
        self,
        embeddings: List[List[float]],
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        if not texts:
            return

        n = len(texts)
        if ids is None:
            ids = [f"doc_{i}" for i in range(n)]
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        if not (len(embeddings) == len(ids) == len(metadatas) == len(texts)):
            raise ValueError("embeddings / texts / metadatas / ids 长度必须一致")

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedDocument]:
        if not query_embedding:
            return []

        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids = res.get("ids", [[]])[0]

        results: List[RetrievedDocument] = []
        for doc_id, text, meta, dist in zip(ids, docs, metas, dists):
            score = 1.0 / (1.0 + float(dist))
            results.append(
                RetrievedDocument(
                    id=str(doc_id),
                    text=text,
                    score=score,
                    metadata=meta or {},
                )
            )

        results.sort(key=lambda x: x.score, reverse=True)
        return results


# ========= 用户隔离向量存储 =========

class PersonalizedVectorStore:
    """支持通用 + 用户专属向量存储"""

    def __init__(self, kb_cfg: KnowledgeBaseConfig, embedding_client: EnhancedEmbeddingClient):
        self.kb_cfg = kb_cfg
        self.embedding_client = embedding_client
        self.base_store = self._create_vector_store(kb_cfg)
        self.user_stores: Dict[str, BaseVectorStore] = {}

    def _create_vector_store(self, kb_cfg: KnowledgeBaseConfig) -> BaseVectorStore:
        storage_type = (kb_cfg.storage_type or "chromadb").lower()
        if storage_type == "chromadb":
            return ChromadbVectorStore(kb_cfg)
        raise NotImplementedError(f"storage_type={storage_type} 尚未实现")

    def get_user_store(self, user_id: str) -> BaseVectorStore:
        """获取用户专属向量库（不存在时自动创建）"""
        if user_id not in self.user_stores:
            from pathlib import Path

            storage_path = (
                (self.kb_cfg.storage_path / f"users/{user_id}")
                if self.kb_cfg.storage_path
                else None
            )

            user_kb_cfg = KnowledgeBaseConfig(
                enabled=True,
                storage_type=self.kb_cfg.storage_type,
                storage_path=storage_path,
                collection_name=f"{self.kb_cfg.collection_name}_user_{user_id}",
                embedding=self.kb_cfg.embedding,
                top_k=self.kb_cfg.top_k,
                similarity_threshold=self.kb_cfg.similarity_threshold,
            )
            self.user_stores[user_id] = self._create_vector_store(user_kb_cfg)

        return self.user_stores[user_id]

    def query_combined(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[RetrievedDocument]:
        """组合查询通用知识库 + 用户专属知识库"""
        # 通用
        general_results = self.base_store.similarity_search(query_embedding, top_k)
        for doc in general_results:
            doc.source = "general"

        # 用户专属
        user_results: List[RetrievedDocument] = []
        if user_id in self.user_stores:
            user_store = self.user_stores[user_id]
            user_results = user_store.similarity_search(query_embedding, max(top_k // 2, 1))
            for doc in user_results:
                doc.source = "user_specific"

        combined = general_results + user_results
        combined.sort(key=lambda x: x.score, reverse=True)
        return combined[:top_k]

    def record_user_interaction(
        self,
        user_id: str,
        query: str,
        response: str,
        retrieved_docs: List[RetrievedDocument],
        feedback: Optional[float] = None,
        interaction_type: InteractionType = InteractionType.QUERY,
    ) -> None:
        """把一次交互作为“经验”写入用户专属向量库"""
        if not user_id or user_id == "anonymous":
            return

        user_store = self.get_user_store(user_id)

        interaction_text = f"用户查询: {query}\n助手回复: {response}"
        if feedback is not None:
            interaction_text += f"\n用户反馈: {feedback}"

        metadata = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "interaction_type": interaction_type.value,
            "feedback": feedback,
            "original_query": query[:200],
            "retrieved_docs_count": len(retrieved_docs),
            "source": "learned_interaction",
        }

        embeddings = self.embedding_client.embed_texts([interaction_text])
        user_store.add_documents(
            embeddings=embeddings,
            texts=[interaction_text],
            metadatas=[metadata],
            ids=[f"interaction_{user_id}_{int(time.time())}"],
        )
        logger.info("已记录用户 %s 的交互行为", user_id)


# ========= 增强版 RAG 管线 =========

class EnhancedRAGPipeline:
    """
    增强版 RAG 管线
    - 支持用户画像 + 个性化查询
    - 支持通用 + 用户专属向量库
    - 提供统一的 context 拼接接口给 LLM 使用
    """

    def __init__(
        self,
        kb_cfg: KnowledgeBaseConfig,
        embedding_client: EnhancedEmbeddingClient,
        vector_store: PersonalizedVectorStore,
    ):
        self.kb_cfg = kb_cfg
        self.enabled = bool(kb_cfg.enabled)
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.profile_analyzer = UserProfileAnalyzer()
        self.user_profiles: Dict[str, UserProfile] = {}

        if not self.enabled:
            logger.warning("KnowledgeBase 已禁用，RAG 将跳过向量检索，仅保留用户画像功能")

    # --- 工厂方法 ---

    @classmethod
    def from_agent_config(cls, agent_cfg: AgentConfig) -> "EnhancedRAGPipeline":
        kb_cfg = agent_cfg.knowledge_base
        embedding_client = EnhancedEmbeddingClient(kb_cfg.embedding)
        vector_store = PersonalizedVectorStore(kb_cfg, embedding_client)
        return cls(kb_cfg=kb_cfg, embedding_client=embedding_client, vector_store=vector_store)

    # --- 用户画像管理 ---

    def get_user_profile(self, user_id: str) -> UserProfile:
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        return self.user_profiles[user_id]

    def update_user_profile(
        self,
        user_id: str,
        query: str,
        documents: List[RetrievedDocument],
        feedback: Optional[float] = None,
    ) -> None:
        profile = self.get_user_profile(user_id)
        self.profile_analyzer.update_profile_from_interaction(profile, query, documents, feedback)

    # --- 查询增强 ---

    def enhance_query_with_profile(self, query: str, profile: UserProfile) -> str:
        """基于画像增强查询文本（加 hint，不改变语义）"""
        parts = [query]

        if profile.expertise_level == "expert":
            parts.append("[技术深度: 高级]")
        elif profile.expertise_level == "beginner":
            parts.append("[解释层次: 基础]")

        if profile.interests:
            interests_str = " ".join(profile.interests[:2])
            parts.append(f"[相关兴趣: {interests_str}]")

        return " ".join(parts)

    # --- 给 LLM 用的 context 构造工具 ---

    def format_documents_as_context(
        self,
        docs: List[RetrievedDocument],
        user_profile: Optional[UserProfile] = None,
        max_docs: int = 5,
        max_chars: int = 2000,
    ) -> str:
        """
        将文档列表格式化为适合 LLM 使用的上下文字符串：
        - 控制最大文档数 & 总字符数，避免 prompt 过长
        - 可选注入用户画像信息
        """
        if not docs:
            return ""

        lines: List[str] = []

        if user_profile:
            lines.append(
                f"[UserProfile] expertise_level={user_profile.expertise_level}, "
                f"interests={','.join(user_profile.interests[:3])}"
            )
            lines.append("")

        remaining_chars = max_chars
        for i, doc in enumerate(docs[:max_docs], start=1):
            header = f"[Doc {i} | source={doc.source} | score={doc.score:.3f}]"
            body = doc.text.strip()
            piece = header + "\n" + body + "\n"

            if len(piece) > remaining_chars:
                piece = piece[:remaining_chars]

            lines.append(piece)
            remaining_chars -= len(piece)
            if remaining_chars <= 0:
                break

        return "\n".join(lines).strip()

    # --- 核心检索方法 ---

    def query(
        self,
        query_text: str,
        user_id: str = "anonymous",
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        record_interaction: bool = True,
        return_context: bool = False,
        max_context_docs: int = 5,
        max_context_chars: int = 2000,
    ):
        """
        个性化检索核心接口

        Return:
            - 若 return_context=False（默认）：
                PersonalizedResponse
            - 若 return_context=True：
                (PersonalizedResponse, context_str)
        """
        profile = self.get_user_profile(user_id)

        if not query_text:
            response = PersonalizedResponse(
                documents=[],
                user_profile=profile,
                enhanced_query=query_text,
                suggested_follow_ups=[],
            )
            if return_context:
                return response, ""
            return response

        enhanced_query = self.enhance_query_with_profile(query_text, profile)

        # 知识库关闭：仅返回增强 query + 空文档
        if not self.enabled:
            logger.debug("KnowledgeBase disabled，跳过向量检索")
            response = PersonalizedResponse(
                documents=[],
                user_profile=profile,
                enhanced_query=enhanced_query,
                suggested_follow_ups=[],
            )
            if return_context:
                return response, ""
            return response

        # 生成查询向量
        query_embedding = self.embedding_client.embed_texts([enhanced_query])[0]

        if top_k is None:
            top_k = self.kb_cfg.top_k

        # 通用 + 用户专属检索
        documents = self.vector_store.query_combined(user_id, query_embedding, top_k)

        threshold = score_threshold if score_threshold is not None else self.kb_cfg.similarity_threshold
        documents = [doc for doc in documents if doc.score >= threshold]

        # 记录交互
        if record_interaction and user_id != "anonymous":
            self.vector_store.record_user_interaction(
                user_id=user_id,
                query=query_text,
                response=f"检索到 {len(documents)} 个相关文档",
                retrieved_docs=documents,
            )

        # 更新画像
        self.update_user_profile(user_id, query_text, documents)

        # 后续建议（简单示例）
        follow_ups = self._generate_follow_up_suggestions(profile, query_text, documents)

        response = PersonalizedResponse(
            documents=documents,
            user_profile=profile,
            enhanced_query=enhanced_query,
            suggested_follow_ups=follow_ups,
        )

        if return_context:
            context_str = self.format_documents_as_context(
                docs=documents,
                user_profile=profile,
                max_docs=max_context_docs,
                max_chars=max_context_chars,
            )
            return response, context_str

        return response

    def _generate_follow_up_suggestions(
        self,
        profile: UserProfile,
        query: str,
        documents: List[RetrievedDocument],
    ) -> List[str]:
        suggestions: List[str] = []

        if profile.interests:
            for interest in profile.interests[:2]:
                suggestions.append(f"关于{interest}的更多信息")

        if profile.expertise_level == "beginner" and "基础" not in query:
            suggestions.append("请用更简单的方式解释")
        elif profile.expertise_level == "expert" and "高级" not in query:
            suggestions.append("需要更深入的技术分析")

        return suggestions[:3]

    # --- 文档索引接口 ---

    def index_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        user_id: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        """批量索引文档（支持通用 / 用户专属）"""
        if not self.enabled:
            logger.warning("KnowledgeBase 已禁用，跳过 index_documents")
            return

        if not texts:
            logger.info("index_documents 收到空文本列表，跳过")
            return

        n = len(texts)
        if ids is None:
            ids = [f"doc_{i}" for i in range(n)]
        if metadatas is None:
            metadatas = [{} for _ in range(n)]

        if not (len(texts) == len(metadatas) == len(ids)):
            raise ValueError("texts / metadatas / ids 长度必须一致")

        logger.info("开始索引 %d 条文档（user_id=%s）", n, user_id)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]
            batch_metas = metadatas[start:end]
            batch_ids = ids[start:end]

            embeddings = self.embedding_client.embed_texts(batch_texts)

            if user_id:
                user_store = self.vector_store.get_user_store(user_id)
                user_store.add_documents(
                    embeddings=embeddings,
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )
            else:
                self.vector_store.base_store.add_documents(
                    embeddings=embeddings,
                    texts=batch_texts,
                    metadatas=batch_metas,
                    ids=batch_ids,
                )

        logger.info("文档索引完成")

    # --- 反馈处理 ---

    def process_feedback(
        self,
        user_id: str,
        query: str,
        retrieved_docs: List[RetrievedDocument],
        feedback_score: float,
        selected_doc_ids: Optional[List[str]] = None,
    ) -> None:
        """处理用户反馈（正向可用于学习）"""
        if user_id == "anonymous":
            return

        self.vector_store.record_user_interaction(
            user_id=user_id,
            query=query,
            response=f"用户反馈: {feedback_score}",
            retrieved_docs=retrieved_docs,
            feedback=feedback_score,
            interaction_type=InteractionType.FEEDBACK,
        )

        profile = self.get_user_profile(user_id)
        profile.feedback_history.append(
            {
                "query": query,
                "score": feedback_score,
                "timestamp": datetime.now().isoformat(),
                "selected_docs": selected_doc_ids or [],
            }
        )

        if feedback_score >= 4.0 and selected_doc_ids:
            self._learn_from_positive_feedback(user_id, query, selected_doc_ids)

    def _learn_from_positive_feedback(
        self,
        user_id: str,
        query: str,
        relevant_doc_ids: List[str],
    ) -> None:
        """从高分反馈中学习出模式"""
        learning_text = f"当用户询问 '{query}' 时，以下文档被证明是有帮助的：{relevant_doc_ids}"
        self.index_documents(
            texts=[learning_text],
            metadatas=[
                {
                    "user_id": user_id,
                    "learned_from": "positive_feedback",
                    "original_query": query,
                    "relevant_documents": relevant_doc_ids,
                    "source": "learned_pattern",
                }
            ],
            user_id=user_id,
        )


# ========= 对外工厂函数 =========

def create_enhanced_rag_pipeline(agent_cfg: AgentConfig) -> EnhancedRAGPipeline:
    """从 AgentConfig 创建增强版 RAG 管线"""
    return EnhancedRAGPipeline.from_agent_config(agent_cfg)


def create_vector_store(kb_cfg: KnowledgeBaseConfig) -> BaseVectorStore:
    """
    简单向量存储工厂（向后兼容用）
    """
    storage_type = (kb_cfg.storage_type or "chromadb").lower()
    if storage_type == "chromadb":
        return ChromadbVectorStore(kb_cfg)
    raise NotImplementedError(f"storage_type={storage_type} 尚未实现")