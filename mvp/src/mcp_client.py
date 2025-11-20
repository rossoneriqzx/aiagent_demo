"""
mcp_client.py - Very simple HTTP-based "MCP-style" tool client

说明：
- 这是一个为当前 Agent Demo 设计的轻量版 MCP 客户端。
- 配置使用 config.MCPServerConfig（label, url, auth_token 等）。
- 假定每个 MCP server 提供以下两个 HTTP 接口（你可以按自己的服务实现）：
  1) GET  {base_url}/tools
     返回 JSON: [{"name": "...", "description": "...", "parameters": {...}}, ...]
  2) POST {base_url}/call
     请求 JSON: {"tool": "<tool name>", "arguments": {...}}
     返回 JSON: {"result": ...}

注意：
- 这不是官方 Model Context Protocol 的完整实现，只是一个「远程工具服务器」适配层。
- 你之后如果想接官方 MCP，可以把这里替换成官方 SDK 的调用逻辑，但 Agent 侧基本不用改。
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Callable, Awaitable

import httpx

from config import MCPServerConfig

logger = logging.getLogger(__name__)

# 与 agent.py 中 ToolCallable 类型保持一致
ToolCallable = Callable[..., Awaitable[Any]]


@dataclass
class MCPTool:
    """描述一个远程 MCP 工具"""
    public_name: str          # 在 LLM 里暴露的名字，例如 "search__web_search"
    description: str
    parameters: Dict[str, Any]
    server: MCPServerConfig


class MCPClient:
    """
    轻量级 MCP 客户端：
    - 启动时从所有 server 拉取工具列表
    - 为每个工具提供一个 async wrapper，供 Agent 注册到 self.tools 中
    """

    def __init__(self, servers: List[MCPServerConfig]) -> None:
        self._servers: List[MCPServerConfig] = [s for s in servers if s.enabled]
        self._tools: Dict[str, MCPTool] = {}

        if not self._servers:
            logger.info("MCPClient 初始化时未发现启用的 MCP 服务器。")
            return

        logger.info("初始化 MCPClient, servers=%s", [s.label for s in self._servers])
        self._discover_all_tools_sync()

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

    def build_tool_wrappers(self) -> Dict[str, ToolCallable]:
        """
        把 MCP 工具包装成 async 函数，供 Agent 注册到 self.tools
        """
        wrappers: Dict[str, ToolCallable] = {}

        for public_name, tool in self._tools.items():

            async def _wrapper(*, _tool: MCPTool = tool, **kwargs: Any) -> Any:
                # 用线程池执行同步 HTTP 调用，避免阻塞事件循环
                return await asyncio.to_thread(self._call_tool_sync, _tool, kwargs)

            wrappers[public_name] = _wrapper

        return wrappers

    def build_tool_schemas(self) -> List[Dict[str, Any]]:
        """
        为 LLM 构造 OpenAI tools schema
        """
        schemas: List[Dict[str, Any]] = []
        for public_name, tool in self._tools.items():
            params = tool.parameters or {
                "type": "object",
                "properties": {},
                "required": [],
            }
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": public_name,
                        "description": tool.description or f"MCP tool from server {tool.server.label}",
                        "parameters": params,
                    },
                }
            )
        return schemas

    # ------------------------------------------------------------------
    # 内部：工具发现 & 调用
    # ------------------------------------------------------------------
    def _discover_all_tools_sync(self) -> None:
        """同步拉取所有 server 的工具列表（避免在 __init__ 里用 asyncio.run）"""
        for server in self._servers:
            try:
                tools = self._fetch_server_tools_sync(server)
                for t in tools:
                    public_name = f"{server.label}__{t['name']}"
                    parameters = t.get("parameters") or {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    }
                    desc = t.get("description", "")
                    mcp_tool = MCPTool(
                        public_name=public_name,
                        description=desc,
                        parameters=parameters,
                        server=server,
                    )
                    # 后来者覆盖同名工具
                    self._tools[public_name] = mcp_tool
                logger.info(
                    "从 MCP server '%s' 发现 %d 个工具，目前总数 %d 个。",
                    server.label,
                    len(tools),
                    len(self._tools),
                )
            except Exception as e:
                logger.error("从 MCP server '%s' 拉取工具失败: %s", server.label, e)

    def _fetch_server_tools_sync(self, server: MCPServerConfig) -> List[Dict[str, Any]]:
        """
        假定远端提供 GET {base_url}/tools
        """
        base = str(server.url).rstrip("/")
        url = f"{base}/tools"
        headers = {}
        if server.auth_token:
            headers["Authorization"] = f"Bearer {server.auth_token}"

        timeout = server.timeout or 30
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(f"/tools 返回值应为 list, got={type(data)}")
            return data

    def _call_tool_sync(self, tool: MCPTool, args: Dict[str, Any]) -> Any:
        """
        假定远端提供 POST {base_url}/call
        """
        server = tool.server
        base = str(server.url).rstrip("/")
        url = f"{base}/call"
        headers = {"Content-Type": "application/json"}
        if server.auth_token:
            headers["Authorization"] = f"Bearer {server.auth_token}"

        payload = {"tool": tool.public_name.split("__", 1)[-1], "arguments": args}

        timeout = server.timeout or 30
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, content=json.dumps(payload, ensure_ascii=False))
            resp.raise_for_status()
            data = resp.json()
            # 根据你的服务设计结构，这里假定返回 {"result": ...}
            if isinstance(data, dict) and "result" in data:
                return data["result"]
            return data