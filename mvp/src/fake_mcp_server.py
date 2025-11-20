# fake_mcp_server.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict

app = FastAPI()

class CallRequest(BaseModel):
    tool: str
    arguments: Dict[str, Any]

@app.get("/tools")
def list_tools():
    # 这个返回值结构要和 mcp_client.MCPClient._fetch_server_tools_sync 里预期的一致
    return [
        {
            "name": "echo",
            "description": "Echo back the given text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to echo"}
                },
                "required": ["text"],
            },
        },
        {
            "name": "multiply",
            "description": "Multiply two numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"},
                },
                "required": ["a", "b"],
            },
        },
    ]

@app.post("/call")
def call_tool(body: CallRequest):
    if body.tool == "echo":
        return {"result": {"echo": body.arguments.get("text", "")}}
    if body.tool == "multiply":
        a = body.arguments.get("a", 0)
        b = body.arguments.get("b", 0)
        return {"result": {"a": a, "b": b, "product": a * b}}
    return {"result": {"error": f"Unknown tool {body.tool}"}}