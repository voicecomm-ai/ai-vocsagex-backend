'''
    查看MCP信息
'''

from typing import Dict, List, Optional
import json
import traceback

from fastapi import Request, Body
from pydantic import BaseModel, Field

from api.api import app, get_api_client_tag
from api.base_model import ResponseModel as ApiMcpResponseModel
from api.base_model import McpConfigModel
from core.mcp.mcp_client import MCPClient
from logger import get_logger

logger = get_logger('api')

class ApiMcpRequestModel(BaseModel):
    mcp_name: str = Field(..., description="MCP名称")
    connection: McpConfigModel = Field(..., description="MCP连接配置")

class ApiMcpToolModel(BaseModel):
    name: str = Field(..., description="工具名称")
    description: str = Field(..., description="工具描述")
    args_schema: Optional[Dict] = Field(None, description="入参schema")
    response_format: str = Field(..., description="响应格式")

class ApiMcpDataModel(BaseModel):
    mcp_name: str = Field(..., description="MCP名称")
    tools: List[ApiMcpToolModel] = Field(..., description="工具信息列表")

@app.post(
    path='/Voicecomm/VoiceSageX/Mcp/Info',
    response_model=ApiMcpResponseModel,
    response_model_exclude_none=False
)
async def handler(conn: Request, body: Dict = Body(...)):
    tag, task_id = get_api_client_tag(conn)
    logger.debug(f'{task_id} {tag}')

    # 格式校验
    try:
        logger.debug(f'{task_id} Request body:\n{json.dumps(body, indent=4, ensure_ascii=False)}')
        request = ApiMcpRequestModel.model_validate(body)
    except Exception as e:
        logger.error(f"{task_id} Fail to validate pydantic instance for:\n{traceback.format_exc()}")
        return ApiMcpResponseModel(
            code=2000,
            msg=f'{task_id}: The request body field is incorrect for:{type(e).__name__}: {str(e)}.'
        )

    # 获取工具列表
    try:
        tools = await MCPClient.get_mcp_tools(
            {
                request.mcp_name: request.connection.model_dump()
            }
        )

        tools_list = []
        for tool in tools:
            tools_list.append(
                ApiMcpToolModel(
                    name=tool.name,
                    description=tool.description,
                    args_schema=tool.args_schema.model_dump() if isinstance(tool.args_schema, BaseModel) else tool.args_schema,
                    response_format=tool.response_format,
                )
            )

        logger.debug(f"{task_id} Tool list:\n{tools_list}")

    except Exception as e:
        logger.error(f"{task_id} Fail to get tool list for:\n{traceback.format_exc()}")
        return ApiMcpResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get tool list for:{type(e).__name__}: {str(e)}.'
        )

    logger.debug(f"{task_id} Done.")

    return ApiMcpResponseModel(
        code=1000, 
        msg=f"{task_id} Success.",
        done=True,
        data=ApiMcpDataModel(
            mcp_name=request.mcp_name,
            tools=tools_list,
        ).model_dump(),
        usage=None,
    )