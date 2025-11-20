# config.py
from __future__ import annotations

import os
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Dict, Any

import yaml
from pydantic import BaseModel, Field, HttpUrl, model_validator, field_validator, ConfigDict
logger = logging.getLogger(__name__)


# ========= 基础枚举 =========

class ModelProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    AZURE = "azure"
    BEDROCK = "bedrock"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"
    BAILIAN = "bailian"

class EmbeddingProvider(str, Enum):
    """Embedding 模型提供方"""

    OPENAI = "openai"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    JINA = "jina"
    BGE = "bge"       # 比如 BAAI/bge 系列
    CUSTOM = "custom"

# ========= 子配置模型 =========

class ModelConfig(BaseModel):
    """模型配置"""
    model_config = ConfigDict(protected_namespaces=())
    provider: ModelProvider = ModelProvider.OPENAI
    model_name: str = "gpt-4.1-mini"

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens: int = 4000
    timeout: int = 30
    max_retries: int = 3

    @model_validator(mode="after")
    def fill_api_key_from_env(self) -> "ModelConfig":
        """如果没有显式配置 api_key，则从环境变量读取。"""
        if not self.api_key:
            env_map = {
                ModelProvider.OPENAI: "OPENAI_API_KEY",
                ModelProvider.QWEN: "QWEN_API_KEY",
                ModelProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                ModelProvider.CUSTOM: "LLM_API_KEY"  
            }
            # 例如 OPENAI_API_KEY / ANTHROPIC_API_KEY
            env_var = env_map.get(self.provider)
            if env_var and (value := os.getenv(env_var)):
                # 创建新实例而不是修改当前实例
                return self.model_copy(update={"api_key": value})
        return self

class EmbeddingConfig(BaseModel):
    """Embedding 模型配置（独立于对话模型）"""
    model_config = ConfigDict(protected_namespaces=())

    provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    model_name: str = "text-embedding-3-small"

    api_key: Optional[str] = None
    base_url: Optional[str] = None

    # 向量维度：可以手动指定，也可以保持 None（某些库会自动推断）
    vector_size: Optional[int] = 1536

    timeout: int = 30
    max_retries: int = 3

    @model_validator(mode="after")
    def fill_api_key_from_env(self) -> "EmbeddingConfig":
        """如果没写 api_key，就从环境变量里根据 provider 自动读取。"""
        if not self.api_key:
            env_map = {
                EmbeddingProvider.OPENAI: "OPENAI_API_KEY",
                EmbeddingProvider.QWEN: "QWEN_API_KEY",
                EmbeddingProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                EmbeddingProvider.JINA: "JINA_API_KEY",
                EmbeddingProvider.BGE: "BGE_API_KEY",
                EmbeddingProvider.CUSTOM: "EMBEDDING_API_KEY",
            }
            env_var = env_map.get(self.provider)
            if env_var:
                value = os.getenv(env_var)
                if value:
                    object.__setattr__(self, "api_key", value)
        return self


class KnowledgeBaseConfig(BaseModel):
    """知识库配置（RAG）"""

    enabled: bool = False

    # 存储后端
    storage_type: str = "chromadb"  # chromadb, pinecone, weaviate
    storage_path: Optional[Path] = None
    collection_name: str = "agent_kb"

    # Embedding 配置（关键）
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # 如果你已有旧代码用到 vector_size，可以保留这个字段，并和 embedding.vector_size 同步
    vector_size: Optional[int] = None

    # 检索参数
    top_k: int = 5
    similarity_threshold: float = 0.7

    @model_validator(mode="after")
    def default_path_and_vector_size(self) -> "KnowledgeBaseConfig":
        # 默认存储路径
        if self.enabled and self.storage_path is None:
            object.__setattr__(self, "storage_path", Path("./knowledge_base"))

        # 同步向量维度：
        # 1. 如果自己没写 vector_size，但 embedding 里有，就用 embedding 的
        if self.vector_size is None and self.embedding.vector_size is not None:
            object.__setattr__(self, "vector_size", self.embedding.vector_size)

        # 2. 如果 embedding.vector_size 没写，但 vector_size 写了，就反向填回 embedding
        if self.vector_size is not None and self.embedding.vector_size is None:
            # 注意：这里直接赋值即可，pydantic v2 默认是可变模型
            self.embedding.vector_size = self.vector_size

        return self


class MCPServerConfig(BaseModel):
    """MCP服务器配置"""

    label: str
    url: HttpUrl
    enabled: bool = True
    timeout: int = 30
    retry_attempts: int = 3
    auth_token: Optional[str] = None

    @field_validator("label")
    def validate_label(cls, v: str) -> str:
        if not v.replace("_", "").isalnum():
            raise ValueError("Label must be alphanumeric or underscore only")
        return v


class MonitoringConfig(BaseModel):
    """监控配置"""

    enabled: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    tracing_enabled: bool = False
    tracing_endpoint: Optional[str] = None

    # 性能监控
    request_timeout: int = 60
    max_concurrent_requests: int = 10


class SecurityConfig(BaseModel):
    """安全配置"""

    enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 3600  # 1小时
    allowed_domains: List[str] = Field(default_factory=list)
    content_filter_enabled: bool = True


class CacheConfig(BaseModel):
    """缓存配置"""

    enabled: bool = True
    backend: str = "redis"  # redis, memory, disk
    ttl: int = 3600  # 1小时
    max_size: int = 1000


# ========= 顶层 Agent 配置 =========

class AgentConfig(BaseModel):
    """工业级智能体配置"""

    # 基础配置
    name: str = "investment_advisor"
    version: str = "1.0.0"
    environment: str = "production"  # development / staging / production

    # 模型配置
    model: ModelConfig = Field(default_factory=ModelConfig)

    # 系统提示词
    system_prompt: str = Field(
        default=(
            "你是专业的投资顾问助手。回答要简洁专业，在必要时解释推理过程。"
            "基于提供的知识和数据给出投资建议，并明确说明风险。"
        )
    )

    # 功能模块
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # MCP配置
    use_mcp: bool = False
    mcp_servers: List[MCPServerConfig] = Field(default_factory=list)

    # 高级功能
    tools_enabled: bool = True
    streaming_enabled: bool = True
    memory_enabled: bool = True
    memory_window: int = 10  # 记忆窗口大小

    @model_validator(mode="after")
    def validate_agent(self) -> "AgentConfig":
        # 校验 name 格式
        if not self.name.replace("_", "").isalnum():
            raise ValueError("Agent name must be alphanumeric or underscore")

        if self.environment not in {"development", "staging", "production"}:
            raise ValueError("Environment must be one of: development, staging, production")

        return self


# ========= 配置管理器 =========

class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self._config: Optional[AgentConfig] = None

    # --- 对外 API ---

    def load(self, prefer_env: bool = True) -> AgentConfig:
        """修复配置加载，确保返回正确的 Pydantic 对象"""
        
        # 1. 从文件加载基础配置
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as f:
                file_data = yaml.safe_load(f) or {}
            file_config = AgentConfig.model_validate(file_data)
        else:
            file_config = AgentConfig()
        
        # 2. 从环境变量加载覆盖配置
        env_config = self.load_from_env()
        
        # 3. 合并配置（根据 prefer_env 决定优先级）
        if prefer_env:
            # 环境变量优先：先用文件配置，再用环境变量覆盖
            merged_data = file_config.model_dump()
            env_data = env_config.model_dump(exclude_unset=True)  # 只包含设置的环境变量
            # 深度合并
            merged_data = self._deep_merge(merged_data, env_data)
            final_config = AgentConfig.model_validate(merged_data)
        else:
            # 文件配置优先
            env_data = env_config.model_dump(exclude_unset=True)
            file_data = file_config.model_dump()
            merged_data = self._deep_merge(env_data, file_data)
            final_config = AgentConfig.model_validate(merged_data)
        
        self._config = final_config
        return final_config

    def _deep_merge(self, base: Dict, update: Dict) -> Dict:
        """深度合并两个字典"""
        result = base.copy()
        for key, value in update.items():
            if (key in result and isinstance(result[key], dict) 
                and isinstance(value, dict)):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    
    

    def get_config(self) -> AgentConfig:
        if self._config is None:
            return self.load()
        return self._config

    # --- 分步加载 ---

    def load_from_file(self) -> AgentConfig:
        """从YAML文件加载配置（如果不存在则使用默认 AgentConfig）。"""
        if not self.config_path.exists():
            logger.warning(f"Config file {self.config_path} not found, using defaults.")
            return AgentConfig()

        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return self._dict_to_config(data)

    def load_from_env(self) -> AgentConfig:
        """
        从环境变量加载覆盖配置（只覆盖部分常用字段）。
        如果没设置就保持默认。
        """
        base = AgentConfig()

        name = os.getenv("AGENT_NAME")
        env = os.getenv("AGENT_ENVIRONMENT")
        model_name = os.getenv("AGENT_MODEL_NAME")

        update_data: Dict[str, Any] = {}
        if name:
            update_data["name"] = name
        if env:
            update_data["environment"] = env
        if model_name:
            update_data.setdefault("model", {})
            update_data["model"]["model_name"] = model_name

        if not update_data:
            return base

        return base.model_copy(update=update_data)

    # --- 工具方法 ---

    def _dict_to_config(self, data: Dict[str, Any]) -> AgentConfig:
        """
        直接依赖 Pydantic 的嵌套解析能力：
            AgentConfig(model=..., knowledge_base=..., ...)
        """
        return AgentConfig.model_validate(data)

    def save_config(self, config: AgentConfig, path: Optional[Path] = None) -> None:
        """保存配置到 YAML 文件（便于导出环境）。"""
        save_path = Path(path) if path else self.config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)

        data = config.model_dump(mode="json")
        with save_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, allow_unicode=True)


# ========= 工厂方法 =========

def create_production_config() -> AgentConfig:
    """创建生产环境配置（可作为模板写回 YAML）"""
    return AgentConfig(
        name="professional_investment_advisor",
        environment="production",
        model=ModelConfig(
            provider=ModelProvider.OPENAI,
            model_name="gpt-4.1-mini",
            temperature=0.2,
            max_tokens=2000,
            timeout=60,
        ),
        monitoring=MonitoringConfig(
            log_level="INFO",
            tracing_enabled=True,
        ),
        security=SecurityConfig(
            rate_limit_requests=500,
            allowed_domains=["example.com", "api.financial-data.com"],
        ),
        cache=CacheConfig(
            backend="redis",
            ttl=7200,
        ),
    )
    

# ========= parse_mcp_server_arg =========

def parse_mcp_server_arg(arg: str) -> MCPServerConfig:
    """增强版 MCP 服务器参数解析 (LABEL=URL)"""
    if "=" not in arg:
        raise ValueError("MCP server must be in format LABEL=URL")

    label, url = arg.split("=", 1)
    label = label.strip()
    url = url.strip()

    if not label or not url:
        raise ValueError("MCP LABEL and URL cannot be empty")

    if not url.startswith(("http://", "https://")):
        raise ValueError("URL must start with http:// or https://")

    return MCPServerConfig(label=label, url=url)