'''
    调用MCP
'''

from typing import Dict, Optional
import json
import traceback

from fastapi import Request, Body
from pydantic import BaseModel, Field, model_validator

from api.api import app, get_api_client_tag
from api.base_model import ResponseModel as ApiMcpResponseModel
from api.base_model import McpConfigModel
from core.mcp.mcp_client import MCPClient
from core.node.code_excutor.base_utils import check_result
from logger import get_logger

logger = get_logger('api')

class ApiMcpRequestModel(BaseModel):
    mcp_name: str = Field(..., description="MCP名称")
    connection: McpConfigModel = Field(..., description="MCP连接配置")
    tool_name: str = Field(..., description="工具名称")
    args: Optional[Dict] = Field(default_factory=dict, description="调用参数")

    @model_validator(mode='after')
    def check_args(self) -> 'ApiMcpRequestModel':
        if not self.args:
            self.args = {}
        return self

@app.post(
    path='/Voicecomm/VoiceSageX/Mcp/Invoke',
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
    except Exception as e:
        logger.error(f"{task_id} Fail to get tool list for:\n{traceback.format_exc()}")
        return ApiMcpResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to get tool list for:{type(e).__name__}: {str(e)}.'
        )

    # 调用
    try:
        target_tool = None
        for tool in tools:
            if tool.name == request.tool_name:
                target_tool = tool
                break
        
        if not target_tool:
            raise ValueError(f"Tool('{tool.name}') not found.")
        else:
            # 参数校验
            args_schema = tool.args_schema.model_dump() \
                if isinstance(tool.args_schema, BaseModel) \
                    else tool.args_schema
            check_result(request.args, args_schema)

            result = await target_tool.ainvoke(input=request.args)
            logger.debug(f"{task_id} Tool Result({type(result)}):\n{result}")

    except Exception as e:
        logger.error(f"{task_id} Fail to invoke tool for:\n{traceback.format_exc()}")
        return ApiMcpResponseModel(
            code=2000,
            msg=f'{task_id}: Fail to invoke tool for:{type(e).__name__}: {str(e)}.'
        )
    
    logger.debug(f"{task_id} Done.")

    return ApiMcpResponseModel(
        code=1000, 
        msg=f"{task_id} Success.",
        done=True,
        data={
            "result": result,
        },
        usage=None,
    )