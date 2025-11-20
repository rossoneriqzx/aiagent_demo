"""
agent.py - Industrial-grade Agent Core

依赖:
- config.py: AgentConfig / ConfigManager / ModelConfig / ModelProvider
- rag.py: create_enhanced_rag_pipeline / EnhancedRAGPipeline
- llm_client.py: create_llm_client / LLMMessage / LLMResponse

特性:
- 支持可选 RAG（知识库检索）
- 支持工具调用（function calling 风格）
- 与国内 OpenAI-compatible 模型完全兼容
- 提供同步的 Agent 封装 + 异步 CLI Demo
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from config import ConfigManager, AgentConfig
from rag import create_enhanced_rag_pipeline, EnhancedRAGPipeline
from llm_client import (
    create_llm_client,
    LLMMessage,
    LLMResponse,
    StreamChunk,
)
from mcp_client import MCPClient

logger = logging.getLogger(__name__)


# 工具函数类型: name -> async callable(**kwargs) -> Any
ToolCallable = Callable[..., Awaitable[Any]]


class Agent:
    """
    核心 Agent：
    - 负责把 Config / LLM / RAG / Tools 串起来
    - 对外提供 async `chat()` / `stream_chat()` 接口
    """

    def __init__(
        self,
        cfg: AgentConfig,
        extra_tools: Optional[Dict[str, ToolCallable]] = None,
    ) -> None:
        self.cfg = cfg

        # LLM 客户端（基于 ModelConfig）
        self.llm = create_llm_client(cfg.model)

        # RAG：只有在 config 中 enabled 才启用
        self.rag: Optional[EnhancedRAGPipeline] = None
        if cfg.knowledge_base.enabled:
            try:
                self.rag = create_enhanced_rag_pipeline(cfg)
                logger.info("RAG pipeline initialized.")
            except Exception as e:
                logger.error("Failed to initialize RAG pipeline: %s", e)
                self.rag = None

        # 工具系统（可选）
        # 1）本地 Python 工具（通过 extra_tools 传入）
        self.tools: Dict[str, ToolCallable] = extra_tools or {}

        # 2）MCP 远程工具（如果在配置中启用）
        self.mcp_client: Optional[MCPClient] = None
        self._mcp_tool_names: List[str] = []
        if cfg.use_mcp and cfg.mcp_servers:
            try:
                self.mcp_client = MCPClient(cfg.mcp_servers)
                mcp_tools = self.mcp_client.build_tool_wrappers()
                if mcp_tools:
                    self.tools.update(mcp_tools)
                    self._mcp_tool_names = list(mcp_tools.keys())
                    logger.info(
                        "MCP 集成已启用，共注册 %d 个远程工具：%s",
                        len(mcp_tools),
                        self._mcp_tool_names,
                    )
            except Exception as e:
                logger.error("初始化 MCPClient 失败，将忽略 MCP 工具: %s", e)

        # 控制工具调用循环
        self.max_tool_iterations: int = 3
        self.user_histories: Dict[str, List[LLMMessage]] = {}
        
        

    # ------------------------------------------------------------------
    # 对外核心接口
    # ------------------------------------------------------------------

    async def chat(
        self,
        user_input: str,
        user_id: str = "anonymous",
        enable_rag: bool = True,
        enable_tools: bool = True,
    ) -> LLMResponse:
        """
        单轮高层封装：
        - 可选 RAG 检索
        - 可选工具调用（function calling 循环）
        - 返回最终的 LLMResponse
        """
        # 1. 构造初始 messages（含 RAG 上下文）
        messages = await self._build_initial_messages(
            user_input=user_input,
            user_id=user_id,
            enable_rag=enable_rag,
        )

        # 2. 根据配置决定是否启用工具，并构造 tools schema
        tools_schema = None
        if enable_tools and self.cfg.tools_enabled and self.tools:
            tools_schema = self._build_tool_schemas()

        # 3. 第一次 LLM 调用
        response = await self.llm.chat(
            messages,
            tools=tools_schema,
            tool_choice="auto" if tools_schema else "none",
        )

        # 4. 若开启工具调用，则执行工具循环（最多 N 轮）
        if tools_schema and response.tool_calls:
            response = await self._tool_call_loop(base_messages=messages, initial_response=response, tools_schema=tools_schema)

        return response

    async def stream_chat(
        self,
        user_input: str,
        user_id: str = "anonymous",
        enable_rag: bool = True,
        enable_tools: bool = False,
    ):
        """
        流式版本：
        - 目前只做 LLM 的流式输出
        - 工具调用在流式场景下相对复杂，这里先不做（保持简单）
        """
        messages = await self._build_initial_messages(
            user_input=user_input,
            user_id=user_id,
            enable_rag=enable_rag,
        )

        async for chunk in self.llm.stream_chat(messages):
            # 直接把 StreamChunk 往外转发
            yield chunk
    
    # ------------------------------------------------------------------
    # 内部：tools schema 构造（给 LLM 用）
    # ------------------------------------------------------------------
    def _build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        汇总本地工具 + MCP 远程工具的 schema，转换成 OpenAI tools 规范。
        """
        schemas: List[Dict[str, Any]] = []

        # 1）本地工具
        schemas.extend(self._build_local_tool_schemas())

        # 2）MCP 远程工具
        if getattr(self, "mcp_client", None) is not None:
            try:
                mcp_schemas = self.mcp_client.build_tool_schemas()
                schemas.extend(mcp_schemas)
            except Exception as e:
                logger.error("构造 MCP 工具 schema 失败: %s", e)

        return schemas

    def _build_local_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        根据已注册的本地 Python 工具自动生成 tools schema。

        - 会使用函数签名做一个简单的 JSON Schema 推断
        - 如果你希望精细控制，可以之后改成手写 schema
        """
        import inspect

        schemas: List[Dict[str, Any]] = []

        # 如果已经注册了 MCP 工具，我们只对非 MCP 的本地工具生成 schema
        mcp_tool_names = set(getattr(self, "_mcp_tool_names", []) or [])

        for name, fn in self.tools.items():
            # MCP 工具已经由 mcp_client 提供 schema，这里跳过
            if name in mcp_tool_names:
                continue

            try:
                sig = inspect.signature(fn)
                properties: Dict[str, Any] = {}
                required: List[str] = []

                for param_name, param in sig.parameters.items():
                    # 跳过 self 等
                    if param_name == "self":
                        continue

                    # 根据类型做个粗略映射
                    json_type = "string"
                    if param.annotation in (int, float):
                        json_type = "number"
                    elif param.annotation is bool:
                        json_type = "boolean"

                    properties[param_name] = {
                        "type": json_type,
                        "description": f"Parameter '{param_name}' for tool '{name}'",
                    }
                    if param.default is inspect._empty:
                        required.append(param_name)

                schema = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": (fn.__doc__ or "").strip() or f"Tool function '{name}'",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                        },
                    },
                }
                schemas.append(schema)
            except Exception as e:
                logger.warning("为工具 %s 自动生成 schema 失败: %s", name, e)

        return schemas

    # ------------------------------------------------------------------
    # 内部：构造 messages
    # ------------------------------------------------------------------

    async def _build_initial_messages(
        self,
        user_input: str,
        user_id: str,
        enable_rag: bool,
    ) -> List[LLMMessage]:
        """
        基于 system_prompt + RAG context + 用户输入 构造 messages
        """
        messages: List[LLMMessage] = []

        # 1. system prompt
        if self.cfg.system_prompt:
            messages.append(
                LLMMessage(role="system", content=self.cfg.system_prompt)
            )

        # 2. RAG 上下文（如果启用）
        if enable_rag and self.rag is not None:
            try:
                rag_resp, context = self.rag.query(
                    query_text=user_input,
                    user_id=user_id,
                    return_context=True,
                    max_context_docs=5,
                    max_context_chars=1800,
                )
                if context:
                    messages.append(
                        LLMMessage(
                            role="system",
                            content=(
                                "以下是知识库检索到的相关内容，请在回答时尽量参考：\n\n"
                                f"{context}"
                            ),
                        )
                    )
            except Exception as e:
                logger.error("RAG 查询失败，将在无知识库的情况下继续: %s", e)

        # 3. 用户输入
        messages.append(LLMMessage(role="user", content=user_input))

        return messages

    # ------------------------------------------------------------------
    # 内部：工具调用循环（function calling 风格）
    # ------------------------------------------------------------------

    async def _tool_call_loop(
        self,
        base_messages: List[LLMMessage],
        initial_response: LLMResponse,
        tools_schema: Optional[List[Dict[str, Any]]] = None,
    ) -> LLMResponse:
        """
        工具调用循环（OpenAI/Qwen 标准版）：
        每一轮：
          1) 读取上一轮的 tool_calls
          2) 把上一轮 assistant(tool_calls) 消息追加到 messages
          3) 针对每个 tool_call 生成对应的 role=tool 消息（带 tool_call_id）
          4) 再次调用 LLM，让模型整合工具结果
        满足规范要求：
          - 任意 tool 消息前，必须有一条带 tool_calls 的 assistant 消息
          - tool 消息必须带 tool_call_id 对应那条调用
        """
        messages = list(base_messages)
        last_response = initial_response

        for iteration in range(self.max_tool_iterations):
            tool_calls = last_response.tool_calls or []
            if not tool_calls:
                # 没有新的工具调用，直接返回当前回答
                return last_response

            logger.info(
                "工具调用回合 %d，检测到 %d 个 tool_calls",
                iteration + 1,
                len(tool_calls),
            )

            # 1) 把上一轮的 assistant(tool_calls) 消息追加
            messages.append(
                LLMMessage(
                    role="assistant",
                    content=last_response.content,
                    tool_calls=tool_calls or None,
                )
            )

            # 2) 执行工具，为每个 tool_call 生成一条 role=tool 的消息
            tool_messages = await self._execute_tool_calls(tool_calls)
            if tool_messages:
                messages.extend(tool_messages)

            # 3) 再次调用 LLM，让模型基于工具结果生成新内容/下一轮 tool_calls
            if tools_schema:
                last_response = await self.llm.chat(
                    messages,
                    tools=tools_schema,
                    tool_choice="auto",
                )
            else:
                last_response = await self.llm.chat(messages)

            # 这一轮如果没有新的 tool_calls，就可以返回最终回答
            if not last_response.tool_calls:
                return last_response

        logger.warning(
            "工具循环已达到最大迭代次数 %d，将返回最后一次响应。",
            self.max_tool_iterations,
        )
        return last_response


    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[LLMMessage]:
        """
        执行所有 tool_calls，为每个调用生成一条 role=tool 消息：
        - content: 工具执行结果（建议 JSON 字符串）
        - name: 工具名
        - tool_call_id: 对应的 tool_call["id"]
        """
        tool_messages: List[LLMMessage] = []

        if not self.tools:
            logger.warning("模型请求了工具调用，但当前 Agent 未注册任何工具。")
            return tool_messages

        async def _run_single(
            call_id: Optional[str],
            func_name: str,
            tool_fn: ToolCallable,
            args: Dict[str, Any],
        ) -> LLMMessage:
            try:
                result = await tool_fn(**args)
                content = json.dumps(result, ensure_ascii=False)
            except Exception as e:
                logger.error("工具 %s 执行异常: %s", func_name, e)
                content = json.dumps(
                    {"error": f"Exception during tool execution: {str(e)}"},
                    ensure_ascii=False,
                )
            # 返回一条标准 tool 消息
            return LLMMessage(
                role="tool",
                content=content,
                name=func_name,
                tool_call_id=call_id,
            )

        tasks = []

        for tc in tool_calls:
            try:
                call_id = tc.get("id")
                func = tc.get("function", {}) or {}
                func_name = func.get("name")
                raw_args = func.get("arguments", "")

                if not func_name:
                    logger.warning("tool_call 缺少函数名: %s", tc)
                    continue

                tool_fn = self.tools.get(func_name)
                if not tool_fn:
                    logger.warning("未找到工具函数: %s", func_name)
                    # 即便工具不存在，也可以返回一条工具错误消息
                    err_msg = LLMMessage(
                        role="tool",
                        content=json.dumps(
                            {"error": f"Tool '{func_name}' not registered on agent."},
                            ensure_ascii=False,
                        ),
                        name=func_name,
                        tool_call_id=call_id,
                    )
                    tool_messages.append(err_msg)
                    continue

                args = self._safe_parse_json(raw_args)
                tasks.append(_run_single(call_id, func_name, tool_fn, args))
            except Exception as e:
                logger.error("解析 tool_call 失败: %s", e)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    # 极端情况：_run_single 自身抛异常，不太可能但以防万一
                    logger.error("执行工具时发生未捕获异常: %s", r)
                else:
                    tool_messages.append(r)

        return tool_messages

    # async def _run_single_tool(
    #     self,
    #     name: str,
    #     fn: ToolCallable,
    #     args: Dict[str, Any],
    # ) -> Dict[str, Any]:
    #     """
    #     执行单个工具调用，统一包装结果格式。
    #     """
    #     try:
    #         result = await fn(**args)
    #         return {
    #             "tool": name,
    #             "success": True,
    #             "result": result,
    #         }
    #     except Exception as e:
    #         logger.error("工具 %s 执行异常: %s", name, e)
    #         return {
    #             "tool": name,
    #             "success": False,
    #             "result": f"Exception during tool execution: {e}",
    #         }

    @staticmethod
    def _safe_parse_json(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if not raw:
            return {}
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("工具参数 JSON 解析失败，原始内容: %s", raw)
                return {}
        # 其它类型直接丢空
        logger.warning("工具参数类型异常: %r", raw)
        return {}
       

# ----------------------------------------------------------------------
# 工厂函数 & CLI Demo
# ----------------------------------------------------------------------

def create_agent(extra_tools: Optional[Dict[str, ToolCallable]] = None) -> Agent:
    """
    使用 ConfigManager 自动加载配置并创建 Agent
    """
    cfg_manager = ConfigManager()
    cfg = cfg_manager.get_config()
    return Agent(cfg, extra_tools=extra_tools)


# ------- 示例工具（可选）：你可以根据需要删掉或改写 -------

async def sample_time_tool() -> Dict[str, Any]:
    """示例工具：返回当前时间"""
    import datetime

    now = datetime.datetime.now().isoformat()
    return {"now": now}


async def sample_add_tool(a: float, b: float) -> Dict[str, Any]:
    """示例工具：返回 a + b"""
    return {"a": a, "b": b, "sum": a + b}


async def interactive_cli() -> None:
    """
    简单命令行交互 Demo：
    - 支持 RAG
    - 不开启工具调用（你可以按需改成 True）
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
        # 添加配置调试信息
    cfg_manager = ConfigManager()
    cfg = cfg_manager.get_config()
    print(f"=== 配置信息 ===")
    print(f"模型提供商: {cfg.model.provider}")
    print(f"模型名称: {cfg.model.model_name}")
    print(f"API Base URL: {cfg.model.base_url}")
    print(f"API Key 前几位: {cfg.model.api_key[:10] if cfg.model.api_key else '未设置'}")
    print(f"超时时间: {cfg.model.timeout}秒")
    print("================")

    # 注册两个示例工具（如果不需要工具，可以传空 dict）
    tools = {
        "get_time": sample_time_tool,
        "add_numbers": sample_add_tool,
    }

    agent = create_agent(extra_tools=tools)

    print("=== Agent CLI ===")
    print("输入内容并回车，输入 '/exit' 退出。")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"/exit", "exit", "quit"}:
            print("Bye.")
            break

        try:
            resp = await agent.chat(
                user_input=user_input,
                user_id="cli_user",
                enable_rag=True,
                enable_tools=True,  # 如果暂时不想测工具，可以改成 False
            )
            print(f"Agent: {resp.content}")
        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    asyncio.run(interactive_cli())