from typing import Optional, Dict, Any

from jinja2 import Template, Undefined
from pydantic import create_model, Field
from langchain.tools import BaseTool
from langchain_core.tools import StructuredTool
from langchain_core.messages import AIMessage

def render_str(text: str, arguments: Optional[Dict]) -> str:
    args = arguments or {}
    text = Template(text, undefined=Undefined).render(**args)
    return text

def schema_to_pydantic(name: str, json_schema: Dict[str, Any]):
    """将 JSON Schema 转换成 Pydantic 模型"""
    properties = json_schema.get("properties", {})
    required = json_schema.get("required", [])

    fields = {}
    for field_name, field_schema in properties.items():
        # 类型映射
        field_type = str
        if field_schema.get("type") == "integer":
            field_type = int
        elif field_schema.get("type") == "number":
            field_type = float
        elif field_schema.get("type") == "boolean":
            field_type = bool
        elif field_schema.get("type") == "array":
            field_type = list
        elif field_schema.get("type") == "object":
            field_type = dict

        # 默认值 & 必填
        default = ... if field_name in required else None

        # 字段描述
        description = field_schema.get("description")

        # 构造字段定义
        fields[field_name] = (
            field_type,
            Field(default, description=description),
        )

    return create_model(name, **fields)

def to_tool(args_schema: Dict) -> BaseTool:
    def run(**kwargs) -> Dict:
        return kwargs
    async def arun(**kwargs) -> Dict:
        return kwargs

    return StructuredTool(
        name="extract_parameters",
        description="Extract parameters from the natural language text",
        args_schema=schema_to_pydantic("ExtractArgs", args_schema),
        func=run,
        coroutine=arun,
    )

def extract_args(msg: AIMessage) -> Dict:
    if msg.tool_calls and msg.tool_calls[0]:
        return msg.tool_calls[0]['args']
    else:
        return {}