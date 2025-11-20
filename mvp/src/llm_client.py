# llm_client.py - 国内环境优化版
"""
工业级LLM客户端 - 国内环境优化版
专注于OpenAI兼容接口，支持DeepSeek、Qwen、智谱等国内模型
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from config import ModelConfig, ModelProvider

logger = logging.getLogger(__name__)

# ========= 数据模型 =========

class LLMMessage(BaseModel):
    role: str
    content: Optional[str] = None
    name: Optional[str] = None
    tool_calls:Optional[List[Dict[str, Any]]] = None
    tool_call_id:Optional[str] = None

class LLMResponse(BaseModel):
    content: str
    model: str
    usage: Dict[str, int] = Field(default_factory=dict)
    finish_reason: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)

class StreamChunk(BaseModel):
    content: str
    is_final: bool = False
    usage: Optional[Dict[str, int]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None

# ========= 异常处理 =========

class LLMException(Exception): pass
class LLMTimeout(LLMException): pass
class LLMRateLimit(LLMException): pass
class LLMAuthenticationError(LLMException): pass
class LLMContextLengthExceeded(LLMException):
    def __init__(self, max_tokens: int, requested_tokens: int):
        super().__init__(f'context length exceeded: max={max_tokens}, requested={requested_tokens}')
        self.max_tokens = max_tokens
        self.requested_tokens = requested_tokens  # ✅ 修正拼写

# ========= 核心客户端 =========

class OpenAIClient:
    """OpenAI兼容客户端 - 支持所有国内OpenAI兼容模型"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._setup_client()
    
    def _setup_client(self) -> None:
        from openai import AsyncOpenAI
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url or self._get_default_base_url(),
            timeout=httpx.Timeout(self.config.timeout),
            max_retries=0,
        )
    
    def _get_default_base_url(self) -> str:
        """为国内模型提供默认base_url"""
        default_urls = {
            ModelProvider.DEEPSEEK: "https://api.deepseek.com/v1",
            ModelProvider.QWEN: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        }
        return default_urls.get(self.config.provider, "https://api.openai.com/v1")
    
    async def chat(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, **kwargs) -> LLMResponse:
        request_kwargs = self._build_request(messages, tools, kwargs, stream=False)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                response = await self._client.chat.completions.create(**request_kwargs)
                return self._parse_response(response)
            except Exception as e:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self._calculate_retry_delay(attempt))
                    continue
                raise self._wrap_exception(e)
    
    async def stream_chat(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        request_kwargs = self._build_request(messages, tools, kwargs, stream=True)
        
        for attempt in range(self.config.max_retries + 1):
            try:
                stream = await self._client.chat.completions.create(**request_kwargs)
                tool_calls_accumulated = []
                
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        if delta.content:
                            yield StreamChunk(content=delta.content, is_final=False)
                        if hasattr(delta, 'tool_calls') and delta.tool_calls:
                            tool_calls_accumulated.extend(delta.tool_calls)
                
                yield StreamChunk(
                    content="", 
                    is_final=True,
                    tool_calls=tool_calls_accumulated if tool_calls_accumulated else None
                )
                return
            except Exception as e:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self._calculate_retry_delay(attempt))
                    continue
                raise self._wrap_exception(e)
    
    def _build_request(self, messages: List[LLMMessage], tools: Optional[List[Dict]], kwargs: Dict, stream: bool) -> Dict:
        request = {
            "model": self.config.model_name,
            "messages": [msg.model_dump(exclude_none=True) for msg in messages],
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "stream": stream,
        }
        if tools:
            request.update({"tools": tools, "tool_choice": kwargs.get("tool_choice", "auto")})
        return request
    
    def _parse_response(self, response: Any) -> LLMResponse:
        # 主消息
        message = response.choices[0].message

        # 统一抽取 tool_calls（含 id / type / function/name/arguments）
        tool_calls: List[Dict[str, Any]] = []
        raw_tool_calls = getattr(message, "tool_calls", None)

        if raw_tool_calls:
            for tc in raw_tool_calls:
                # OpenAI SDK 类型：ChatCompletionMessageToolCall
                fn = getattr(tc, "function", None)
                if fn:
                    tool_calls.append(
                        {
                            "id": getattr(tc, "id", None),
                            "type": getattr(tc, "type", "function"),
                            "function": {
                                "name": getattr(fn, "name", "") or "",
                                "arguments": getattr(fn, "arguments", "") or "",
                            },
                        }
                    )

        return LLMResponse(
            content=message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            } if response.usage else {},
            finish_reason=response.choices[0].finish_reason,
            tool_calls=tool_calls,
        )
        
    def _wrap_exception(self, error: Exception) -> LLMException:
        error_str = str(error).lower()
        if "timeout" in error_str: return LLMTimeout(f"Timeout: {error}")
        if "rate limit" in error_str: return LLMRateLimit(f"Rate limit: {error}")
        if "authentication" in error_str: return LLMAuthenticationError(f"Auth failed: {error}")
        if "context length" in error_str:
            match = re.search(r"maximum context length is (\d+).*requested (\d+)", str(error), re.IGNORECASE)
            if match:
                max_tokens = int(match.group(1))
                requested = int(match.group(2))
            else:
                max_tokens = self.config.max_tokens
                requested = self.config.max_tokens * 2
            return LLMContextLengthExceeded(max_tokens, requested)
        return LLMException(f"LLM error: {error}")
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        return min(2 ** attempt + 1, 60)

# ========= 监控包装器 =========

class MonitoredLLMClient(OpenAIClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._metrics = {"requests": 0, "errors": 0, "tokens": 0, "duration": 0.0}
    
    async def chat(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, **kwargs) -> LLMResponse:
        start = time.time()
        self._metrics["requests"] += 1
        try:
            response = await super().chat(messages, tools, **kwargs)
            self._metrics["tokens"] += response.usage.get("total_tokens", 0)
            self._metrics["duration"] += time.time() - start
            return response
        except Exception:
            self._metrics["errors"] += 1
            raise
    
    async def stream_chat(self, messages: List[LLMMessage], tools: Optional[List[Dict]] = None, **kwargs) -> AsyncGenerator[StreamChunk, None]:
        start = time.time()
        self._metrics["requests"] += 1
        tokens_accumulated = 0
    
        try:
            # 直接使用父类已经修复的 stream_chat 方法
            async for chunk in super().stream_chat(messages, tools, **kwargs):
                # 监控逻辑：累加 token 使用量
                if chunk.usage:
                    tokens_accumulated += chunk.usage.get("total_tokens", 0)
                yield chunk
        
            # 流式结束，记录总时间和 token 数
            self._metrics["duration"] += time.time() - start
            self._metrics["tokens"] += tokens_accumulated
        
        except Exception:
            self._metrics["errors"] += 1
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        requests = max(self._metrics["requests"], 1)
        return {
            **self._metrics,
            "error_rate": self._metrics["errors"] / requests,
            "avg_duration": self._metrics["duration"] / requests,
        }

# ========= 工厂函数 =========

def create_llm_client(config: ModelConfig) -> OpenAIClient:
    """创建LLM客户端 - 专注于国内环境"""
    supported_providers = {
        ModelProvider.OPENAI,
        ModelProvider.DEEPSEEK, 
        ModelProvider.QWEN,
        ModelProvider.AZURE,
        ModelProvider.CUSTOM
    }
    
    if config.provider not in supported_providers:
        raise ValueError(f"不支持的提供商: {config.provider}。支持: {list(supported_providers)}")
    
    return OpenAIClient(config)

def create_monitored_llm_client(config: ModelConfig) -> MonitoredLLMClient:
    """创建带监控的LLM客户端"""
    return MonitoredLLMClient(config)

# ========= 快捷函数 =========

async def chat_completion(config: ModelConfig, messages: List[LLMMessage], **kwargs) -> LLMResponse:
    """快捷函数：一次性对话完成"""
    client = create_llm_client(config)
    return await client.chat(messages, **kwargs)

async def stream_completion(config: ModelConfig, messages: List[LLMMessage], **kwargs) -> AsyncGenerator[StreamChunk, None]:
    """快捷函数：流式对话完成"""
    client = create_llm_client(config)
    async for chunk in client.stream_chat(messages, **kwargs):
        yield chunk

# ========= 使用示例 =========

async def demo():
    """演示如何使用国内模型"""
    
    # DeepSeek示例
    deepseek_config = ModelConfig(
        provider=ModelProvider.DEEPSEEK,
        model_name="deepseek-chat",
        api_key="your-deepseek-api-key",
        base_url="https://api.deepseek.com/v1"  # 可省略，有默认值
    )
    
    # 通义千问示例  
    qwen_config = ModelConfig(
        provider=ModelProvider.QWEN,
        model_name="qwen-turbo", 
        api_key="your-qwen-api-key",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    
    client = create_llm_client(deepseek_config)
    messages = [LLMMessage(role="user", content="你好，请介绍一下你自己")]
    
    try:
        response = await client.chat(messages)
        print(f"DeepSeek响应: {response.content}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())